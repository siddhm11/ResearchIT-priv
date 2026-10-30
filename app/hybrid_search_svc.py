"""
Hybrid search orchestrator — Phase 3.

Orchestrates the full pipeline:
  1. LLM rewrite (optional, via Groq)
  2. BGE-M3 encode → dense + sparse
  3. Parallel search: Qdrant dense + Zilliz sparse
  4. RRF fusion (K=60)
  5. Title-match boost (exact/substring against Turso titles)
  6. Return ranked arxiv_ids

Doc 06 confirms: RRF is correct for search (fusing different retrievers
answering the SAME query).  This is different from recommendations where
quota is correct (fusing different queries for the SAME user).

Recency rerank was removed — search relevance should not be biased toward
newer papers (that is a recommendations concern). For exact-title queries
like "attention is all you need", the recency overlay was crushing seminal
older papers below newer "X is all you need" titles.
"""
from __future__ import annotations

import asyncio
import re

from app import compute_pool
from app import runtime_stats
from app import config
from app import embed_svc
from app import qdrant_svc
from app import zilliz_svc
from app import fts_svc
from app import groq_svc
from app import turso_svc
from app import arxiv_svc
from app import features
from app import named_paper


# ── Public API ───────────────────────────────────────────────────────────────

REWRITE_KEYWORDS_SWITCH = "SEARCH_REWRITE_KEYWORDS"

# Trending papers jump the line, for a few days. A paper in the top quarter of
# this week's Hugging Face Daily Papers that the search already found (top
# TREND_POOL after fusion) may take one of the last TREND_AUDITIONS places the
# cross-encoder scores. If the cross-encoder then rates it at least
# TREND_MIN_RELEVANCE of the best score, it gains up to TREND_BOOST of the best
# score, in full for its first three days on the list and fading to nothing a
# week after (trending_shelf.freshness). At most TREND_MAX_PROMOTED papers per
# search. Search results are cached for everyone, so the switch has no preview:
# off, shadow (measure what would move, change nothing) or on (doc 06,
# 2026-10-24).
TRENDING_SWITCH = "SEARCH_TRENDING"
TREND_MIN_STRENGTH = 0.75
TREND_BOOST = 0.25           # x the best score; citations get at most 0.2
TREND_MIN_RELEVANCE = 0.5
TREND_POOL = 50
TREND_AUDITIONS = 2
TREND_MAX_PROMOTED = 2

async def search(
    query: str,
    limit: int = 10,
    use_rewrite: bool = True,
    return_meta: bool = False,
) -> list[str] | tuple[list[str], dict]:
    """
    Hybrid semantic search — returns a list of arxiv_ids ranked by
    fused relevance.

    Pipeline:
      rewrite → encode → parallel(dense, sparse) → RRF → title-boost

    Args:
        query: User's raw search query.
        limit: Number of results to return.
        use_rewrite: Whether to attempt LLM query rewriting.
        return_meta: If True, returns a tuple of (arxiv_ids, metadata_dict).

    Returns:
        list of arxiv_id strings, sorted by final score descending.
        Never raises — returns empty list on total failure.
    """
    query = query.strip()
    if not query:
        return ([], {}) if return_meta else []

    import time
    t_start = time.perf_counter()
    search_meta = {"rewritten_query": None, "groq_time_ms": 0, "groq_status": "off"}


    # ── Step 1: LLM rewrite — started now, collected after the first encode ──
    #
    # This used to `await` here, ahead of everything else, costing 161-329ms
    # (measured) before any retrieval could begin. Telemetry showed the status
    # was frequently "called, kept original", so that was often paid for a query
    # that came back unchanged.
    #
    # It now runs concurrently with encoding the original query. When the rewrite
    # is a no-op we get its latency back entirely; when it does rewrite, the
    # extra encode+search happens after, exactly as before.
    start_groq = time.perf_counter()
    rewrite_task = (
        asyncio.create_task(groq_svc.rewrite(query)) if use_rewrite else None
    )
    # Which paper a short search names ("resnet"), checked against the corpus
    # (app/named_paper.py). Runs alongside everything below; collected before
    # the final ordering.
    named_task = (
        asyncio.create_task(named_paper.resolve(query))
        if use_rewrite and named_paper.wanted(query) else None
    )

    # ── Step 2: BGE-M3 encode the original AND rewrite ──────────────────
    # Why both: The rewriter improves recall on conceptual/casual queries
    # ("when AI makes up fake facts" -> "LLM hallucination ...") but it
    # paraphrases away from literal title wording on known-item queries
    # ("attention is all you need" -> "Transformer self-attention ..."),
    # which can drop the actual famous paper out of the candidate pool
    # entirely. Searching both forms and RRF-fusing all result lists
    # gives us recall on both axes.
    # BGE-M3 inference is 285-642ms of synchronous CPU work. Called inline it
    # blocks the event loop for that whole time, which serialises every
    # concurrent request and would also stop the rewrite task above from making
    # progress. It belongs in an executor, as the cross-encoder stage already is.
    use_fts = config.SPARSE_BACKEND == "fts" and fts_svc.is_available()
    search_meta["sparse_backend"] = "fts5" if use_fts else "zilliz"
    # How many candidates to fetch before reranking
    fetch_k = limit * config.SEARCH_FETCH_K_MULTIPLIER

    # Retrieval for each query form starts the moment that form is ready.
    # Before 2026-10-14 every list waited for the Groq rewrite to come back,
    # so the rewrite's ~150-300 ms sat on the critical path of every search,
    # including the majority it leaves unchanged (prod: "encode" time equalled
    # "rewrite" time to the millisecond on 9 of 14 timed searches). The lists
    # and the fusion are identical; only the waiting moved.
    tasks: list[asyncio.Task] = []
    task_labels: list[str] = []
    t0_retrieval = time.perf_counter()

    def start_lexical(i: int, text: str) -> None:
        # Keyword retrieval does not require an embedding model. Keep it
        # alive even if one or every encode failed.
        if use_fts:
            tasks.append(asyncio.create_task(fts_svc.search_sparse(text, limit=fetch_k)))
            task_labels.append(f"fts_q{i}")

    def start_vector(i: int, pair: tuple) -> None:
        dense_vec, sparse_dict = pair
        tasks.append(asyncio.create_task(
            qdrant_svc.search_dense_merged(dense_vec.tolist(), limit=fetch_k)))
        task_labels.append(f"qdrant_q{i}")
        if not use_fts:
            tasks.append(asyncio.create_task(zilliz_svc.search_sparse(sparse_dict, limit=fetch_k)))
            task_labels.append(f"zilliz_q{i}")

    # BM25 needs only the text, so it runs alongside the encode.
    start_lexical(0, query)

    # Key encodings by their actual text; partial failures cannot misalign a
    # rewrite's vector with the original query's lexical results.
    t0_encode = time.perf_counter()
    encode_ms = 0
    encoded: dict[str, tuple] = {}
    try:
        d, s = await compute_pool.run_inference(embed_svc.encode_query, query)
        encoded[query] = (d, s)
        start_vector(0, encoded[query])
    except Exception as e:
        print(f"[hybrid_search] Encoding failed for {query!r}: {e}")
    encode_ms += int((time.perf_counter() - t0_encode) * 1000)

    # Collect the rewrite, which has been running throughout the encode above
    # and now overlaps the original query's retrieval too.
    rewritten_query = query
    if rewrite_task is not None:
        try:
            rewritten_query = await rewrite_task
            if rewritten_query and rewritten_query != query:
                search_meta["rewritten_query"] = rewritten_query
                search_meta["groq_status"] = "rewritten"
            else:
                rewritten_query = query
                word_count = len(query.strip().split())
                if word_count <= 2:
                    search_meta["groq_status"] = f"skipped (query too short: {word_count} words)"
                elif groq_svc._looks_academic(query):
                    search_meta["groq_status"] = "skipped (looks academic)"
                else:
                    search_meta["groq_status"] = "called, kept original"
        except Exception:
            rewritten_query = query  # Fallback guaranteed
            search_meta["groq_status"] = "error (fallback)"
    search_meta["groq_time_ms"] = int((time.perf_counter() - start_groq) * 1000)
    # groq_time_ms OVERLAPS encode_time_ms and retrieval_time_ms, so the stage
    # timings must not be summed to reconstruct the total.
    search_meta["groq_overlapped"] = True

    if rewritten_query != query:
        start_lexical(1, rewritten_query)
        t1 = time.perf_counter()
        try:
            d, s = await compute_pool.run_inference(embed_svc.encode_query, rewritten_query)
            encoded[rewritten_query] = (d, s)
            start_vector(1, encoded[rewritten_query])
        except Exception as e:
            print(f"[hybrid_search] Encoding failed for {rewritten_query!r}: {e}")
        encode_ms += int((time.perf_counter() - t1) * 1000)
    # Model time only. It used to include the wait for the rewrite.
    search_meta["encode_time_ms"] = encode_ms

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    search_meta["retrieval_time_ms"] = int((time.perf_counter() - t0_retrieval) * 1000)
    search_meta["n_retrieval_tasks"] = len(tasks)

    # A conversational question the rewriter changed: its literal words steer
    # keyword search off course (after the 2026-10-23 rewriter fix, the two
    # "stop" papers left in the top 10 for "how do I stop a RAG system from
    # making things up" were #2 and #3 of the original's keyword list), so only
    # the rewrite's keyword list counts. Its meaning still counts: the vector
    # list stays. Only when no paper matched every word of the original, which
    # is what tells a question from a title: every known title tested matched
    # at least one paper word for word, and the rewriter shortens titles
    # ("Direct Preference Optimization: Your Language Model is Secretly a
    # Reward Model" became "Direct Preference Optimization"), so dropping a
    # title's keyword list lost the paper.
    skipped = set()
    if rewritten_query != query and features.mode(REWRITE_KEYWORDS_SWITCH) == "on":
        orig = raw_results[task_labels.index("fts_q0")] if "fts_q0" in task_labels else None
        if isinstance(orig, list) and (not orig or orig[0].get("match") == "any"):
            skipped.add("fts_q0")
    valid_result_lists: list[list[dict]] = []
    search_meta["retrieval_sources"] = []
    for label, r in zip(task_labels, raw_results):
        if label in skipped:
            continue
        if isinstance(r, Exception):
            print(f"[hybrid_search] search task failed: {r}")
            continue
        if r:
            valid_result_lists.append(r)
            search_meta["retrieval_sources"].append(label)
    has_dense = any(s.startswith("qdrant_") for s in search_meta["retrieval_sources"])
    has_lexical = any(not s.startswith("qdrant_") for s in search_meta["retrieval_sources"])
    search_meta["retrieval_mode"] = (
        "hybrid" if has_dense and has_lexical else
        "semantic" if has_dense else "keyword" if has_lexical else "unavailable"
    )

    if not valid_result_lists:
        if named_task is not None:
            named_task.cancel()
        return ([], search_meta) if return_meta else []

    # ── Step 4: RRF fusion across all result lists ──────────────────────
    t0_rrf = time.perf_counter()
    fused = _rrf_fuse_multi(valid_result_lists, k=config.SEARCH_RRF_K)
    search_meta["rrf_time_ms"] = int((time.perf_counter() - t0_rrf) * 1000)

    if not fused:
        if named_task is not None:
            named_task.cancel()
        return ([], search_meta) if return_meta else []

    # ── Step 4.5: BGE Cross-Encoder Reranking ──────────────────────────
    t0_bge = time.perf_counter()
    bge_time_ms = 0
    trend_mode = features.mode(TRENDING_SWITCH)
    trend: dict = {}
    if trend_mode in ("shadow", "on"):
        from app import trending_shelf
        trend = {a: t for a, t in trending_shelf.trending_now().items()
                 if t["strength"] >= TREND_MIN_STRENGTH}

    if fused and config.SEARCH_BGE_RERANK:
        top_n = min(len(fused), rerank_depth())
        search_meta["rerank_depth"] = top_n
        top_candidates = fused[:top_n]
        if trend:
            extra = [it for it in fused[top_n:TREND_POOL] if it["arxiv_id"] in trend][:TREND_AUDITIONS]
            search_meta["trend_auditions"] = len(extra)
            if extra and trend_mode == "on":
                top_candidates = fused[:top_n - len(extra)] + extra
        top_ids = [item["arxiv_id"] for item in top_candidates]

        try:
            meta = await turso_svc.fetch_metadata_batch(top_ids)

            # Fetch missing from arXiv API just in case (graceful fallback)
            missing = [aid for aid in top_ids if aid not in meta]
            if missing:
                try:
                    arxiv_meta = await arxiv_svc.fetch_metadata_batch(missing)
                    meta.update(arxiv_meta)
                except Exception as e:
                    print(f"[hybrid_search] arXiv fallback in BGE rerank failed: {e}")

            # Prepare papers for reranking
            papers_to_rerank = []
            for item in top_candidates:
                aid = item["arxiv_id"]
                p_meta = meta.get(aid) or {}
                papers_to_rerank.append({
                    "arxiv_id": aid,
                    "title": p_meta.get("title") or "",
                    "abstract": p_meta.get("abstract") or "",
                })

            # Cross-encoder on the inference pool, off the event loop.
            from app import reranker_bge_svc

            reranked_papers, bge_time_ms = await compute_pool.run_inference(
                reranker_bge_svc.rerank, query, papers_to_rerank, top_n)

            # Map each cross-encoder logit to a [0, 1] relevance probability.
            # Sigmoid is monotonic, so this preserves the reranker's ordering;
            # it just gives the downstream title/citation boost a bounded,
            # well-behaved scale to work against.
            bge_scores_by_id = {}
            for p in reranked_papers:
                bge_score = p.get("bge_rerank_score", 0.0)
                try:
                    bge_prob = 1.0 / (1.0 + math.exp(-bge_score))
                except OverflowError:
                    bge_prob = 0.0 if bge_score < 0 else 1.0
                bge_scores_by_id[p["arxiv_id"]] = bge_prob

            # Narrow to the cross-encoded window.
            #
            # Previously the un-scored tail stayed in the list carrying raw RRF
            # scores while the top items carried sigmoid-scaled ones, and both
            # were sorted together.  ms-marco logits are often strongly
            # negative, so a paper the cross-encoder merely disliked collapsed
            # toward zero and sank below papers the cross-encoder had never
            # looked at.  Dropping the tail keeps every remaining score on one
            # comparable scale.
            scored = [
                it for it in top_candidates if it["arxiv_id"] in bge_scores_by_id
            ]
            if scored:
                for item in scored:
                    item["rrf_score"] = bge_scores_by_id[item["arxiv_id"]]
                scored.sort(key=lambda x: x["rrf_score"], reverse=True)
                fused = scored

        except Exception as e:
            print(f"[hybrid_search] BGE reranking stage failed: {e}")
            bge_time_ms = 0

    search_meta["bge_rerank_time_ms"] = bge_time_ms

    # ── Step 5: Title-match boost ────────────────────────────────────────
    # Use the user's ORIGINAL query (not the LLM rewrite) for title matching —
    # the user's literal text is what should match a paper title.
    t0_rerank = time.perf_counter()
    rerank_timings: dict = {}
    ranked = await _title_match_rerank(
        fused, query, top_n_for_boost=50, timings=rerank_timings,
        trend=trend, apply_trend=trend_mode == "on",
    )
    moved = {it["arxiv_id"]: {"from": it["trend_from"], "to": it["trend_to"],
                              "facts": trend[it["arxiv_id"]]["facts"]}
             for it in ranked if it.get("trend_boost") and it["trend_to"] < limit}
    if moved:
        search_meta["trending" if trend_mode == "on" else "trending_shadow"] = moved
        runtime_stats.count(f"search_trend_{trend_mode}")
    rerank_total = int((time.perf_counter() - t0_rerank) * 1000)
    search_meta["rerank_time_ms"] = rerank_total
    # Sub-timings come back via the timings dict.  They used to be stashed on
    # fused[0], but _title_match_rerank re-sorts the list before returning, so
    # the caller was popping from a different dict and always read 0 — which
    # silently folded the Turso round-trip into "rerank_compute_ms".
    turso_boost_ms = rerank_timings.get("turso_boost_fetch_ms", 0)
    search_meta["turso_boost_fetch_ms"] = turso_boost_ms
    search_meta["rerank_compute_ms"] = max(0, rerank_total - turso_boost_ms)

    # ── Step 5.5: the paper a short search names goes first ──────────────
    t0_named = time.perf_counter()
    named = await named_paper.collect(named_task, t_start)
    if named_task is not None:
        # Normally 0: the lookup finished during retrieval and reranking.
        search_meta["named_paper_wait_ms"] = int((time.perf_counter() - t0_named) * 1000)
    if named:
        if features.mode(named_paper.SWITCH) == "on":
            ranked = named_paper.pin(ranked, named)
            search_meta["named_paper"] = named
        else:
            search_meta["named_paper_shadow"] = named

    # ── Step 6: Return top results ───────────────────────────────────────
    final_results = [item["arxiv_id"] for item in ranked[:limit]]
    search_meta["total_ms"] = int((time.perf_counter() - t_start) * 1000)
    runtime_stats.record("search", search_meta)
    return (final_results, search_meta) if return_meta else final_results


def rerank_depth() -> int:
    """How many fused candidates the cross-encoder scores for this search.

    The deeper idle window when no other inference job is queued or running,
    otherwise the base window, so a burst of searches does not each pay double
    reranker time while waiting on one another. Never below the base window.
    """
    base = config.SEARCH_RERANK_TOP_N
    if compute_pool.pending() == 0:
        return max(base, config.SEARCH_RERANK_TOP_N_IDLE)
    return base


# ── RRF fusion ───────────────────────────────────────────────────────────────

def _rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of two result lists (dense + sparse).

    Kept for callers that pass exactly two lists; new code (and the
    hybrid pipeline itself) should call _rrf_fuse_multi instead.
    """
    return _rrf_fuse_multi([dense_results, sparse_results], k=k)


def _rrf_fuse_multi(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion across N result lists.

    score[paper] = sum over each list of 1/(k + rank_in_that_list)

    RRF is rank-based, so raw scores from different systems don't need
    normalization. This means we can merge dense, sparse, AND multiple
    encoded query forms (original + LLM-rewritten) without per-source
    score calibration.

    Args:
        result_lists: each list contains {'arxiv_id': str, 'score': ...}
                      sorted best-first.
        k: RRF constant (default 60).

    Returns:
        list of {'arxiv_id': str, 'rrf_score': float} sorted by rrf_score desc.
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            aid = item["arxiv_id"]
            scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)

    fused = [
        {"arxiv_id": aid, "rrf_score": score}
        for aid, score in scores.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


# ── Title-match + citation-popularity rerank ─────────────────────────────────

# Boost magnitudes are calibrated against `max_rrf` so any meaningful title
# match outranks the best non-matching candidate:
#   final = rrf_score + max_rrf * (title_boost + citation_boost)
# With boost=2.0 (exact title), the worst exact-match still beats the best
# non-match by >= max_rrf. boost=1.0 same vs. no-match.
_BOOST_EXACT_TITLE = 2.0          # query == title (after normalize)
_BOOST_SUBSTRING_TITLE = 1.0      # query is contiguous substring of title
_BOOST_HIGH_COVERAGE = 1.0        # >= 80% of query words found in title
_BOOST_MED_COVERAGE = 0.5         # >= 50% of query words found in title

# Citation-popularity boost — surfaces landmark papers even when keyword
# overlap is low. Without this, "how do transformers work in NLP" returns
# niche papers instead of "Attention Is All You Need" because RRF favors
# papers whose titles contain more query keywords.
#
# Uses log10(citations) scaled to a cap:
#   0 citations   -> 0.0 boost
#   10 citations  -> 0.03
#   100 citations -> 0.06
#   1K citations  -> 0.10
#   10K citations -> 0.13
#   100K citations-> 0.17 (near cap)
#
# Cap is deliberately small (0.2 * max_rrf) so it NUDGES but doesn't
# override title-match or strong semantic signal. A 100K-citation paper
# still loses to a perfect title match.
import math
_CITATION_BOOST_CAP = 0.2         # max boost from citations alone
_CITATION_LOG_DIVISOR = 30.0      # how many log10 units to reach the cap

# Drop any token shorter than this from coverage calculation — single-letter
# tokens ("a", "i") and tiny stop-likes inflate spurious matches.
_MIN_COVERAGE_TOKEN_LEN = 2


def _normalize_for_match(text: str) -> str:
    """Lowercase, collapse non-alnum to single spaces, strip."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _stem_plural(w: str) -> str:
    """Trim a single trailing 's' on tokens longer than 3 chars.

    Crude but cheap. Catches the 'space' vs 'spaces' problem in the
    Mamba paper title without dragging in a real stemmer dependency.
    """
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _word_set(text: str) -> set[str]:
    return {
        _stem_plural(w) for w in text.split()
        if len(w) >= _MIN_COVERAGE_TOKEN_LEN
    }


def _compute_title_boost(query_norm: str, title_raw: str) -> float:
    """Decide how much to boost a candidate based on title overlap.

    Order of checks (strongest signal first):
      1. Exact match after normalization                  -> 2.0
      2. Query is contiguous substring of normalized title -> 1.0
         (rescues "chain of thought prompting" vs
          "Chain-of-Thought Prompting Elicits Reasoning..." — punctuation
          in title was the only thing blocking the old binary substring check)
      3. Coverage: fraction of query word-stems found in title (or as
         substring of compact title — catches "multilingual" appearing
         in "Multi-Lingual" once spaces are stripped).
            >= 0.8 -> _BOOST_HIGH_COVERAGE * coverage
            >= 0.5 -> _BOOST_MED_COVERAGE * coverage
            otherwise -> 0
    """
    if not query_norm or not title_raw:
        return 0.0

    title_norm = _normalize_for_match(title_raw)
    if not title_norm:
        return 0.0

    if query_norm == title_norm:
        return _BOOST_EXACT_TITLE
    if query_norm in title_norm:
        return _BOOST_SUBSTRING_TITLE

    q_words = _word_set(query_norm)
    if not q_words:
        return 0.0

    t_words = _word_set(title_norm)
    title_compact = title_norm.replace(" ", "")

    matches = 0
    for w in q_words:
        if w in t_words:
            matches += 1
        elif len(w) >= 4 and w in title_compact:
            # Catches "multilingual" appearing within "multi lingual"
            # once whitespace is stripped from the title.
            matches += 1

    coverage = matches / len(q_words)
    if coverage >= 0.8:
        return _BOOST_HIGH_COVERAGE * coverage
    if coverage >= 0.5:
        return _BOOST_MED_COVERAGE * coverage
    return 0.0


def _compute_citation_boost(citation_count: int) -> float:
    """Log-scaled citation boost, capped at _CITATION_BOOST_CAP.

    The idea: a paper with 100K citations (like "Attention Is All You Need")
    gets a small but meaningful nudge upward even when it has zero keyword
    overlap with a beginner's query like "how do transformers work".

    The boost is small enough that a strong title match always wins, and
    a strong semantic RRF score always wins. But when two papers have
    similar RRF scores and neither has a title match, the one with 100K
    citations beats the one with 3 citations.

    Scale (log10-based):
      citations=0     -> 0.000
      citations=10    -> 0.033
      citations=100   -> 0.067
      citations=1000  -> 0.100
      citations=10000 -> 0.133
      citations=100000-> 0.167
    """
    if citation_count <= 0:
        return 0.0
    raw = math.log10(citation_count + 1) / _CITATION_LOG_DIVISOR
    return min(raw, _CITATION_BOOST_CAP)


async def _title_match_rerank(
    fused: list[dict],
    user_query: str,
    top_n_for_boost: int = 50,
    timings: dict | None = None,
    trend: dict | None = None,
    apply_trend: bool = False,
) -> list[dict]:
    """
    Boost candidates by title overlap + citation popularity.

    Two signals, both based on metadata we already fetch from Turso:

    1. Title boost (strong): exact/substring/coverage match between the
       user's ORIGINAL query and paper titles. Rescues known-item queries.

    2. Citation boost (gentle): log-scaled citation count, capped at 0.2x
       max_rrf. Rescues landmark papers for beginner queries where keyword
       overlap is low but the paper is obviously important.

    The final score is:
      final = rrf_score + max_rrf * (title_boost + citation_boost [+ trend_boost])

    3. Trend boost (temporary, see TRENDING_SWITCH): papers in `trend` that are
       at least TREND_MIN_RELEVANCE of the best score. Each records where it
       stood without it (`trend_from`) and with it (`trend_to`); it joins the
       score only when `apply_trend`.

    Safe under partial Turso failure: papers with missing metadata get
    boost=0 and rank by RRF alone.
    """
    if not fused:
        return fused

    q_norm = _normalize_for_match(user_query)
    if not q_norm:
        for item in fused:
            item["final_score"] = item["rrf_score"]
        return fused

    candidate_ids = [item["arxiv_id"] for item in fused[:top_n_for_boost]]
    titles: dict[str, str] = {}
    citations: dict[str, int] = {}
    import time as _time
    _t0_turso_boost = _time.perf_counter()
    try:
        meta = await turso_svc.fetch_metadata_batch(candidate_ids)
        titles = {aid: (m.get("title") or "") for aid, m in meta.items()}
        citations = {aid: (m.get("citation_count") or 0) for aid, m in meta.items()}
    except Exception as e:
        print(f"[hybrid_search] Metadata fetch for boost failed: {e}")
        if timings is not None:
            timings["turso_boost_fetch_ms"] = int(
                (_time.perf_counter() - _t0_turso_boost) * 1000
            )
        for item in fused:
            item["final_score"] = item["rrf_score"]
        return fused
    if timings is not None:
        timings["turso_boost_fetch_ms"] = int(
            (_time.perf_counter() - _t0_turso_boost) * 1000
        )

    max_rrf = max(item["rrf_score"] for item in fused)

    for item in fused:
        aid = item["arxiv_id"]
        t_boost = _compute_title_boost(q_norm, titles.get(aid, ""))
        c_boost = _compute_citation_boost(citations.get(aid, 0))
        item["title_boost"] = t_boost
        item["citation_boost"] = c_boost
        item["trend_boost"] = _trend_boost((trend or {}).get(aid), item["rrf_score"], max_rrf)
        item["final_score"] = item["rrf_score"] + max_rrf * (t_boost + c_boost)

    boosted = sorted((it for it in fused if it["trend_boost"]), key=lambda it: -it["trend_boost"])
    for it in boosted[TREND_MAX_PROMOTED:]:
        it["trend_boost"] = 0.0
    fused.sort(key=lambda x: x["final_score"], reverse=True)
    if boosted:
        for pos, it in enumerate(fused):
            it["trend_from"] = pos
        with_trend = sorted(fused, key=lambda x: x["final_score"] + max_rrf * x["trend_boost"], reverse=True)
        for pos, it in enumerate(with_trend):
            it["trend_to"] = pos
        if apply_trend:
            for it in fused:
                it["final_score"] += max_rrf * it["trend_boost"]
            fused = with_trend
    return fused


def _trend_boost(info: dict | None, score: float, best: float) -> float:
    """Up to TREND_BOOST for the week's most upvoted paper, half of it at the
    top-quarter line, times how fresh it still is; 0 unless the paper already
    scores TREND_MIN_RELEVANCE of the best."""
    if not info or best <= 0 or score < TREND_MIN_RELEVANCE * best:
        return 0.0
    above = (info["strength"] - TREND_MIN_STRENGTH) / (1 - TREND_MIN_STRENGTH)
    return TREND_BOOST * (0.5 + 0.5 * min(max(above, 0.0), 1.0)) * info["fresh"]
