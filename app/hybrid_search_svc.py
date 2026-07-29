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

from app import config
from app import embed_svc
from app import qdrant_svc
from app import zilliz_svc
from app import groq_svc
from app import turso_svc
from app import arxiv_svc


# ── Public API ───────────────────────────────────────────────────────────────

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
    search_meta = {"rewritten_query": None, "groq_time_ms": 0, "groq_status": "off"}

    # ── Step 1: LLM rewrite (optional, never blocks) ─────────────────────
    rewritten_query = query
    if use_rewrite:
        start_groq = time.perf_counter()
        try:
            rewritten_query = await groq_svc.rewrite(query)
            if rewritten_query != query:
                search_meta["rewritten_query"] = rewritten_query
                search_meta["groq_status"] = "rewritten"
            else:
                # Groq returned same query — either skipped by heuristic or LLM kept it
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

    # ── Step 2: BGE-M3 encode the original AND rewrite ──────────────────
    # Why both: The rewriter improves recall on conceptual/casual queries
    # ("when AI makes up fake facts" -> "LLM hallucination ...") but it
    # paraphrases away from literal title wording on known-item queries
    # ("attention is all you need" -> "Transformer self-attention ..."),
    # which can drop the actual famous paper out of the candidate pool
    # entirely. Searching both forms and RRF-fusing all result lists
    # gives us recall on both axes.
    queries_to_encode: list[str] = [query]
    if rewritten_query and rewritten_query != query:
        queries_to_encode.append(rewritten_query)

    t0_encode = time.perf_counter()
    encoded: list[tuple] = []
    for q in queries_to_encode:
        try:
            d, s = embed_svc.encode_query(q)
            encoded.append((d, s))
        except Exception as e:
            print(f"[hybrid_search] Encoding failed for {q!r}: {e}")
    search_meta["encode_time_ms"] = int((time.perf_counter() - t0_encode) * 1000)

    if not encoded:
        return ([], search_meta) if return_meta else []

    # How many candidates to fetch before reranking
    fetch_k = limit * config.SEARCH_FETCH_K_MULTIPLIER

    # ── Step 3: Parallel dense + sparse search for every encoded form ───
    # Build a flat list of search coroutines: [dense_q1, sparse_q1, dense_q2, sparse_q2, ...]
    t0_retrieval = time.perf_counter()
    tasks = []
    task_labels = []
    for i, (dense_vec, sparse_dict) in enumerate(encoded):
        tasks.append(qdrant_svc.search_dense(dense_vec.tolist(), limit=fetch_k))
        task_labels.append(f"qdrant_q{i}")
        tasks.append(zilliz_svc.search_sparse(sparse_dict, limit=fetch_k))
        task_labels.append(f"zilliz_q{i}")

    # Time each task individually
    import asyncio as _aio
    task_start = time.perf_counter()
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    search_meta["retrieval_time_ms"] = int((time.perf_counter() - t0_retrieval) * 1000)
    search_meta["n_retrieval_tasks"] = len(tasks)

    valid_result_lists: list[list[dict]] = []
    for r in raw_results:
        if isinstance(r, Exception):
            print(f"[hybrid_search] search task failed: {r}")
            continue
        if r:
            valid_result_lists.append(r)

    if not valid_result_lists:
        return ([], search_meta) if return_meta else []

    # ── Step 4: RRF fusion across all result lists ──────────────────────
    t0_rrf = time.perf_counter()
    fused = _rrf_fuse_multi(valid_result_lists, k=config.SEARCH_RRF_K)
    search_meta["rrf_time_ms"] = int((time.perf_counter() - t0_rrf) * 1000)

    if not fused:
        return ([], search_meta) if return_meta else []

    # ── Step 4.5: BGE Cross-Encoder Reranking ──────────────────────────
    t0_bge = time.perf_counter()
    bge_time_ms = 0

    if fused:
        top_n = min(len(fused), config.SEARCH_RERANK_TOP_N)
        top_candidates = fused[:top_n]
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

            # Run BGE Reranker in executor to avoid blocking the async event loop
            from app import reranker_bge_svc
            loop = asyncio.get_running_loop()

            reranked_papers, bge_time_ms = await loop.run_in_executor(
                None,
                lambda: reranker_bge_svc.rerank(query, papers_to_rerank, top_n=top_n)
            )

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
                it for it in fused[:top_n] if it["arxiv_id"] in bge_scores_by_id
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
        fused, query, top_n_for_boost=50, timings=rerank_timings
    )
    rerank_total = int((time.perf_counter() - t0_rerank) * 1000)
    search_meta["rerank_time_ms"] = rerank_total
    # Sub-timings come back via the timings dict.  They used to be stashed on
    # fused[0], but _title_match_rerank re-sorts the list before returning, so
    # the caller was popping from a different dict and always read 0 — which
    # silently folded the Turso round-trip into "rerank_compute_ms".
    turso_boost_ms = rerank_timings.get("turso_boost_fetch_ms", 0)
    search_meta["turso_boost_fetch_ms"] = turso_boost_ms
    search_meta["rerank_compute_ms"] = max(0, rerank_total - turso_boost_ms)

    # ── Step 6: Return top results ───────────────────────────────────────
    final_results = [item["arxiv_id"] for item in ranked[:limit]]
    return (final_results, search_meta) if return_meta else final_results


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
      final = rrf_score + max_rrf * (title_boost + citation_boost)

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
        item["final_score"] = item["rrf_score"] + max_rrf * (t_boost + c_boost)

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused
