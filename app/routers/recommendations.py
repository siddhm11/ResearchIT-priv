"""
Recommendations router.

GET /api/recommendations
  – Called by HTMX on page load (hx-trigger="load")
  – Returns the recommendations partial HTML

Recommendation pipeline (cascading fallback):
  Phase 2b / 4.1: Multi-interest clustering → quota fusion     (≥5 saves)
  Phase 2a:       EWMA long-term vector → single vector search  (≥3 saves)
  Phase 1:        Qdrant BEST_SCORE Recommend API with raw IDs  (≥1 save)

Phase 4 changes vs Phase 2b:
  - RRF replaced with importance-weighted quota fusion (doc 06 §3.1)
  - Hungarian matching stabilises cluster IDs across reclusters (4.2)
  - Category-level suppression filters strongly disliked topics (4.3)
"""
import asyncio
import random
import time
import uuid
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse
from app import db, qdrant_svc, arxiv_svc, turso_svc, user_state as us
from app.config import COOKIE_NAME, REC_LIMIT, REC_MIN_POSITIVES
from app.templates_env import templates
from app.recommend import profiles
from app.recommend.clustering import (
    compute_clusters,
    save_clusters_to_db,
    load_clusters_from_db,
    stabilize_cluster_ids,
    MIN_PAPERS_FOR_CLUSTERING,
)
from app.recommend.fusion import allocate_quotas, merge_quota_results
from app.recommend.reranker import rerank_candidates
# inject_exploration is no longer imported here: exploration is now drawn per
# page in _build_page() from a pre-shuffled pool, so that doc 06 §3.5's "two
# serendipitous papers per feed" holds for every page rather than once per pool.
from app.recommend.diversity import mmr_rerank

router = APIRouter(prefix="/api")

# Phase 4.5: Pipeline version tag for instrumentation.  Bump this on any
# change to the ranking logic so A/B attribution is possible.
# v7 = paginated serving; ranking itself is unchanged.
# v8 = cold-start churn: Tier 0 drops already-shown papers and fills slots
#      epsilon-greedily, so it now logs real propensities instead of 1.0.
#      Tiers 1-3 are untouched; they record impressions but do not yet use them.
_RANKER_VERSION = "v8.0_coldstart_churn"

# Minimum EWMA interactions before switching from ID-based to vector-based recs
_MIN_EWMA_INTERACTIONS = 3

# Candidate oversampling factor per cluster (fetch more than quota to handle dedup)
_OVERSAMPLE = 3

# Short-term session context: fixed supplementary pool size
_ST_SUPPLEMENT = 20

# ── Paginated feed ───────────────────────────────────────────────────────────
#
# The feed used to terminate: REC_LIMIT (10) papers plus 2 exploration picks,
# then a dead end with a "show different recommendations" button. Dismissing
# shrank it further, so triaging drained the page toward empty.
#
# Rather than re-running the pipeline per page — it reclusters, re-retrieves
# and rescores, and its exploration step is random, so page 2 would both cost
# seconds and risk repeating page 1 — the first request ranks a deep pool once
# and caches the ORDER under its query_id. Later pages are then a metadata
# fetch and nothing else.
#
# Keying on query_id rather than user_id is deliberate: query_id already exists
# to group one feed's impressions for per-feed CTR (Phase 6.5 B1), so a feed
# and its cache entry share a lifetime, and two tabs get two independent feeds.

_PAGE_SIZE = REC_LIMIT          # ranked papers per page
_FEED_POOL = 60                 # how deep MMR ranks on the first request
_N_EXPLORE = 2                  # exploration picks PER PAGE (doc 06 §3.5)

# Tier 0 gets a shallower pool than the behavioural tiers. Trending is a
# LIKE '%code%' scan plus a temp B-tree sort over 1.6M rows — the single most
# expensive query in the system, and the one that lands on brand-new users.
# Asking it for 60 rows instead of 10 made it time out entirely against Turso
# without the local sidecar. Three pages is plenty before behavioural signal
# takes over, and the sidecar makes this an index range read in production.
_TRENDING_POOL = 30

# With the sidecar present the same query is an index range read rather than a
# LIKE scan over 1.6M rows, so it can afford a much deeper pool -- and depth is
# what buys refresh runway: at _PAGE_SIZE per refresh, a 30-paper pool is spent
# after three refreshes and starts recycling, while 200 lasts twenty.
# Falls back to the shallow limit exactly where the timeout risk is real.
_TRENDING_POOL_SIDECAR = 200


def _trending_pool_size() -> int:
    try:
        from app import local_meta
        return _TRENDING_POOL_SIDECAR if local_meta.is_available() else _TRENDING_POOL
    except Exception:  # pragma: no cover - defensive
        return _TRENDING_POOL


# ── Cold-start churn ─────────────────────────────────────────────────────────
#
# eps matches the value doc 06 already earmarks for new users (§4 lists
# "epsilon-greedy exploration (eps=0.25 new users, eps=0.05 established)" under
# Phase 9). Only the exposure-randomisation half is implemented here; there is
# no bandit and nothing learns from the outcome, because that genuinely does
# need users. What is borrowed early is the part that fixes a feed which never
# changed and logged degenerate propensities.
_COLD_START_EPSILON = 0.25

# How many recent impressions to keep when the pool runs dry. Keeping roughly a
# page means the papers just served do not immediately reappear at the top,
# while everything older becomes eligible again.
_IMPRESSION_KEEP = _PAGE_SIZE

_FEED_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_FEED_CACHE_MAX = 200


def _cache_put(query_id: str, entry: dict) -> None:
    _FEED_CACHE[query_id] = entry
    _FEED_CACHE.move_to_end(query_id)
    while len(_FEED_CACHE) > _FEED_CACHE_MAX:
        _FEED_CACHE.popitem(last=False)


def _cache_get(query_id: str) -> dict | None:
    entry = _FEED_CACHE.get(query_id)
    if entry is not None:
        _FEED_CACHE.move_to_end(query_id)
    return entry


def feed_cache_stats() -> dict:
    """For diagnostics parity with the other in-process caches."""
    return {"size": len(_FEED_CACHE), "max": _FEED_CACHE_MAX}


def _take(pool: list[str], entry: dict, seen: set[str], n: int) -> list[str]:
    """Pull the next n unserved ids from `pool`.

    Tracks what has already been emitted on the entry rather than slicing by
    page index, because `seen` grows while the user reads: a paper saved on
    page 1 disappears from the pool, and index arithmetic would then silently
    skip its neighbour.
    """
    out: list[str] = []
    emitted = entry["emitted"]
    for aid in pool:
        if len(out) >= n:
            break
        if aid in emitted or aid in seen:
            continue
        out.append(aid)
        emitted.add(aid)
    return out


async def _cold_start_order(
    user_id: str, pool: list[str],
) -> tuple[list[str], dict[str, float]]:
    """Order the cold-start pool, and return each paper's selection probability.

    Two problems this solves, both measured on the deployed Space.

    1. The feed never changed. Tier 0 was a deterministic citation sort with
       exploration explicitly disabled, so a user with no interactions was
       served byte-identical papers in identical order on every refresh,
       indefinitely. `seen` did not help: it tracks saves and dismissals, so a
       reader who refreshes without clicking anything is remembered as having
       done nothing.

    2. Every impression logged propensity=1.0. A deterministic policy has no
       support over the actions it did not take, so no amount of that data can
       ever support IPS/SNIPS/DR later -- which is the whole point of the
       query_id/propensity/policy_id invariant in CLAUDE.md §3.11.

    The fix is impression memory plus epsilon-greedy slot filling:

      * papers already SHOWN to this user are dropped, so a refresh advances
        through the ranked backlog instead of reshuffling the same ten. For a
        triage feed, refresh should mean "what else have you got", not
        "shuffle" -- reordering the same papers is more disorienting than
        leaving them still.
      * each slot then takes the best remaining paper with probability 1-eps,
        or a uniform pick from the rest with probability eps. That keeps
        "best first" mostly intact while giving every paper non-zero exposure
        probability, so two users with the same categories no longer get
        identical feeds.

    epsilon-greedy rather than a Plackett-Luce / softmax policy on purpose: its
    propensities are exactly computable, in precisely the form §3.11 already
    documents ("n_explore/pool_size for exploration"). A stochastic ranking
    policy would need approximated top-k marginals, and an approximate
    propensity is a silently biased IPS estimate later.

    Returns (ordered_ids, propensity_by_id).
    """
    if not pool:
        return [], {}

    try:
        impressed = await db.get_impressed_ids(user_id)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[recs] impression lookup failed ({e}) -- serving unfiltered pool")
        impressed = set()

    fresh = [pid for pid in pool if pid not in impressed]

    # Everything on offer has been shown. Forgetting the oldest impressions is
    # the only option that keeps a feed alive -- an empty feed is a worse
    # failure than a repeat, and this user has no behavioural signal yet to
    # retrieve anything else with.
    if len(fresh) < _PAGE_SIZE:
        # How many to keep has to scale with the pool, not be a fixed page.
        # Keeping a flat 10 against a 12-paper pool leaves 2 papers -- the reset
        # starves the feed instead of refilling it. Retain only what still
        # leaves a full page free.
        keep = min(_IMPRESSION_KEEP, max(0, len(pool) - _PAGE_SIZE))
        try:
            await db.forget_oldest_impressions(user_id, keep=keep)
            impressed = await db.get_impressed_ids(user_id)
            fresh = [pid for pid in pool if pid not in impressed] or list(pool)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[recs] impression reset failed ({e})")
            fresh = list(pool)

    rng = random.Random()
    remaining = list(fresh)
    ordered: list[str] = []
    props: dict[str, float] = {}

    while remaining:
        n = len(remaining)
        if n == 1 or rng.random() >= _COLD_START_EPSILON:
            pick = 0                      # greedy: best remaining
        else:
            pick = rng.randrange(n)       # explore: uniform over the rest
        aid = remaining.pop(pick)
        ordered.append(aid)
        # P(this paper filled this slot) = (1-eps)·[it was best] + eps·(1/n).
        # Recorded once, at the slot it actually won.
        greedy_share = (1.0 - _COLD_START_EPSILON) if pick == 0 else 0.0
        props[aid] = round(greedy_share + _COLD_START_EPSILON / n, 6)

    return ordered, props


@router.get("/recommendations", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    page: int = 1,
    query_id: str | None = None,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """
    Serve one page of the feed.

    page=1 (or an unknown query_id) runs the full pipeline, ranks a deep pool
    and caches the ordering. Later pages replay that ordering, so they cost a
    metadata fetch instead of a recluster + re-retrieve + rescore.
    """
    user_id = user_id or str(uuid.uuid4())
    state = await us.ensure_loaded(user_id)
    page = max(1, page)

    def _with_cookie(resp):
        resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
        return resp

    def _empty_resp():
        return _with_cookie(templates.TemplateResponse(
            request, "partials/empty_recs.html", {"min_saves": REC_MIN_POSITIVES},
        ))

    # Continue an existing feed only when the loader asks for a later page AND
    # we still hold its ordering. A cache miss (restart, eviction) falls through
    # to a fresh feed rather than erroring — the user sees new papers, which is
    # the correct failure mode for a feed.
    entry = _cache_get(query_id) if (query_id and page > 1) else None

    if entry is None:
        query_id = str(uuid.uuid4())
        page = 1
        entry = await _build_feed(user_id, state, query_id)
        if entry is None:
            return _empty_resp()
        _cache_put(query_id, entry)

    seen = us.all_seen(user_id)
    papers, has_more = await _build_page(entry, seen)

    if not papers:
        return _empty_resp() if page == 1 else _with_cookie(
            templates.TemplateResponse(
                request, "partials/rec_page.html",
                {"papers": [], "has_more": False,
                 "next_page": page + 1, "query_id": query_id},
            ))

    # Page 1 brings the .feed wrapper; later pages are bare fragments that the
    # loader button swaps itself out for.
    template = "partials/recommendations.html" if page == 1 else "partials/rec_page.html"

    # Remember what actually reached the screen, so the next refresh advances
    # instead of repeating. Recorded here rather than in _build_feed because
    # only papers on a served page were really shown -- a ranked pool is not an
    # impression. Never fatal: a failure here costs churn, not correctness.
    try:
        await db.record_impressions(
            user_id, [p["arxiv_id"] for p in papers if p.get("arxiv_id")])
    except Exception as e:  # pragma: no cover - defensive
        print(f"[recs] impression write failed ({e})")

    return _with_cookie(templates.TemplateResponse(
        request, template,
        {
            "papers": papers,
            "has_more": has_more,
            "next_page": page + 1,
            "query_id": entry["query_id"],
            "trending": entry.get("trending", False),
        },
    ))


# ── Feed construction ────────────────────────────────────────────────────────

async def _build_feed(user_id: str, state, query_id: str) -> dict | None:
    """
    Run the tier cascade once and return a cacheable feed entry, or None when
    there is nothing to show.

    entry = {
        ranked   ordered ids — the MMR-diversified feed
        explore  shuffled leftover candidates, drawn from for serendipity
        tags     {arxiv_id: instrumentation dict}
        emitted  ids already served on some page of this feed
        position running global rank, so `position` is continuous across pages
    }
    """
    base = {
        "query_id": query_id,
        "emitted": set(),
        "position": 0,
        "trending": False,
    }

    # ── Tier 0: category trending (cold start, Phase 5) ──────────────────
    if not state.has_enough_for_recs():
        category_filter = await db.get_user_category_filter(user_id)
        if category_filter:
            trending = await turso_svc.fetch_trending_by_categories(
                category_filter, limit=_trending_pool_size(),
            )
            if trending:
                ids = [p["arxiv_id"] for p in trending if p.get("arxiv_id")]
                if ids:
                    ranked, props = await _cold_start_order(user_id, ids)
                    return {
                        **base,
                        "trending": True,
                        "ranked": ranked,
                        "explore": [],
                        "tags": {
                            aid: {
                                "ranker_version": _RANKER_VERSION,
                                "candidate_source": "trending_category_fallback",
                                "cluster_id": "",
                                "query_id": query_id,
                                "propensity": props.get(aid, 1.0),
                                "policy_id": _RANKER_VERSION,
                            } for aid in ranked
                        },
                    }
        return None

    seen = us.all_seen(user_id)

    # ── Tier 1: multi-interest clustering + quota fusion (≥5 saves) ──────
    ranked, explore, tags, _rerank_ms, _timing = await _multi_interest_recommend(
        user_id, state, seen, _FEED_POOL, query_id=query_id,
    )

    # ── Tier 2: EWMA single-vector search (≥3 saves) ─────────────────────
    if not ranked:
        ranked = await _ewma_recommend(user_id, seen, _FEED_POOL)
        explore, tags = [], {
            aid: {
                "ranker_version": _RANKER_VERSION,
                "candidate_source": "ewma_longterm",
                "cluster_id": "",
                "query_id": query_id,
                "propensity": 1.0,
                "policy_id": _RANKER_VERSION,
            } for aid in ranked
        }

    # ── Tier 3: Qdrant Recommend API (≥1 save) ───────────────────────────
    if not ranked:
        ranked = await qdrant_svc.recommend(
            positive_arxiv_ids=state.positive_list,
            negative_arxiv_ids=state.negative_list,
            seen_arxiv_ids=seen,
            limit=_FEED_POOL,
        )
        explore, tags = [], {
            aid: {
                "ranker_version": _RANKER_VERSION,
                "candidate_source": "qdrant_recommend",
                "cluster_id": "",
                "query_id": query_id,
                "propensity": 1.0,
                "policy_id": _RANKER_VERSION,
            } for aid in ranked
        }

    if not ranked:
        return None

    # Shuffled once, then drawn from in order. Doc 06 §3.5 calls for
    # SERENDIPITOUS picks — taking the pool's head instead would just serve
    # "next best by score", which is not the same thing. Shuffling here rather
    # than sampling per page keeps picks non-repeating across pages for free.
    explore = list(explore)
    random.shuffle(explore)

    return {**base, "ranked": ranked, "explore": explore, "tags": tags}


async def _build_page(entry: dict, seen: set[str]) -> tuple[list[dict], bool]:
    """Materialise the next page: _PAGE_SIZE ranked papers + _N_EXPLORE picks."""
    core = _take(entry["ranked"], entry, seen, _PAGE_SIZE)
    # Exploration rides along with ranked papers; it never carries a page on its
    # own. Without this guard the feed could report "that's everything" (has_more
    # is computed over `ranked`) and then still hand back an exploration-only
    # page to anyone who asked for the next one.
    explore = (
        _take(entry["explore"], entry, seen, _N_EXPLORE)
        if core and entry["explore"] else []
    )
    ids = core + explore
    if not ids:
        return [], False

    # Phase 3.5: Turso primary (sidecar-backed), arXiv API fallback.
    meta = await turso_svc.fetch_metadata_batch(ids)
    missing = [aid for aid in ids if aid not in meta]
    if missing:
        try:
            meta.update(await arxiv_svc.fetch_metadata_batch(missing))
        except Exception as e:
            print(f"[recommendations] arXiv fallback for {len(missing)} IDs failed: {e}")

    # Cache to SQLite so category-suppression JOINs work (Phase 4.3)
    await db.cache_turso_metadata_batch(list(meta.values()))

    explore_set = set(explore)
    # Phase 6.5 B2: probability this policy chose to show an exploration paper
    # on this page — the fraction of the pool drawn. Deterministic slots are 1.0.
    explore_propensity = (
        len(explore) / len(entry["explore"]) if entry["explore"] else 0.0
    )

    papers: list[dict] = []
    for aid in ids:
        if aid not in meta:
            continue
        tags = entry["tags"].get(aid, {})
        is_explore = aid in explore_set
        papers.append({
            **meta[aid],
            "saved": False,
            "dismissed": False,
            "ranker_version": tags.get("ranker_version", _RANKER_VERSION),
            # Serving as an exploration pick overrides the retrieval origin —
            # the same paper is only "exploration" by virtue of how it was shown.
            "candidate_source": "exploration" if is_explore
                                else tags.get("candidate_source", ""),
            "cluster_id": "" if is_explore else tags.get("cluster_id", ""),
            "query_id": entry["query_id"],
            "position": entry["position"],
            "propensity": explore_propensity if is_explore
                          else tags.get("propensity", 1.0),
            "policy_id": tags.get("policy_id", _RANKER_VERSION),
        })
        entry["position"] += 1

    has_more = any(
        aid not in entry["emitted"] and aid not in seen
        for aid in entry["ranked"]
    )
    return papers, has_more
# ── Tier 1: Multi-interest clustering + quota fusion ─────────────────────────

async def _multi_interest_recommend(
    user_id: str, state, seen: set[str], limit: int,
    *, query_id: str = "",
) -> tuple[list[str], list[str], dict[str, dict], int, dict]:
    """
    Full recommendation pipeline (Phase 2b + Phase 4 corrections):
      1. Ward clustering → identify distinct interests
      2. Quota allocation → per-cluster slot budgets (replaces RRF)
      3. Parallel per-cluster ANN searches → retrieve candidates
      4. Hungarian matching → stabilise cluster IDs across reclusters
      5. Category suppression → remove strongly disliked topics
      6. Heuristic re-ranking → score candidates
      7. MMR diversity → select top-k with diversity
      8. Exploration injection → serendipitous papers

    Returns ([], {}, 0, {}) to trigger fallback to Tier 2.
    Phase 4.5: second element is {arxiv_id: {ranker_version, candidate_source, cluster_id}}.
    """
    positives = state.positive_list
    if len(positives) < MIN_PAPERS_FOR_CLUSTERING:
        return [], [], {}, 0, {}

    try:
        # Fetch embeddings for all saved papers
        vectors = await qdrant_svc.get_paper_vectors(positives)
        if len(vectors) < MIN_PAPERS_FOR_CLUSTERING:
            return [], [], {}, 0, {}

        timing = {}  # Collect per-stage timing breakdown

        # Build aligned arrays (only papers we got vectors for)
        aligned_ids = [pid for pid in positives if pid in vectors]
        aligned_embs = np.array(
            [vectors[pid] for pid in aligned_ids], dtype=np.float32
        )

        # ── Step 1: Compute interest clusters ─────────────────────────────
        t0_cluster = time.time()
        clusters = compute_clusters(aligned_ids, aligned_embs)

        # ── Step 4.2: Stabilise cluster IDs with Hungarian matching ───────
        old_clusters_data = await load_clusters_from_db(user_id)
        if old_clusters_data:
            from app.recommend.clustering import InterestCluster
            old_clusters = []
            for row in old_clusters_data:
                # Bug B fix (Phase 6.3): prefer live vector, fall back to
                # persisted blob, skip cluster only as last resort.
                mpid = row["medoid_paper_id"]
                if mpid in vectors:
                    medoid_emb = np.array(vectors[mpid], dtype=np.float32)
                elif row.get("medoid_embedding_blob") is not None:
                    medoid_emb = np.frombuffer(
                        row["medoid_embedding_blob"], dtype=np.float32
                    ).copy()
                else:
                    # Unrecoverable — skip this stale cluster row.
                    # It will be rebuilt on the next Ward run.
                    print(
                        f"[recommendations] cluster {row['cluster_idx']}: "
                        f"medoid {mpid} unrecoverable — skipping"
                    )
                    continue

                old_clusters.append(InterestCluster(
                    cluster_idx=row["cluster_idx"],
                    medoid_paper_id=mpid,
                    medoid_embedding=medoid_emb,
                    paper_ids=[],
                    importance=row["importance"],
                ))
            if old_clusters:
                clusters = stabilize_cluster_ids(clusters, old_clusters)

        await save_clusters_to_db(user_id, clusters)
        timing["clustering_ms"] = int((time.time() - t0_cluster) * 1000)

        # Phase 6.5 B3: append snapshot for cluster history (non-blocking)
        try:
            import numpy as _np
            await db.save_cluster_snapshot(user_id, [
                {
                    "cluster_idx": c.cluster_idx,
                    "medoid_paper_id": c.medoid_paper_id,
                    "importance": c.importance,
                    "paper_ids": c.paper_ids,
                    "medoid_embedding_blob": c.medoid_embedding.astype(_np.float32).tobytes(),
                }
                for c in clusters
            ])
        except Exception as e:
            print(f"[recommendations] cluster snapshot save failed (non-fatal): {e}")

        # ── Step 2: Quota allocation ───────────────────────────────────────
        importances = [c.importance for c in clusters]
        quotas = allocate_quotas(importances, total_slots=100, min_slots=3)

        # ── Step 3: Parallel per-cluster ANN searches ─────────────────────
        t0_ann = time.time()
        st_vec = await profiles.load_profile(user_id, "short_term")

        # NOTE on latency: we previously tried passing with_vectors=True
        # to fold the candidate-vector fetch into the search call. That
        # made it *worse* on Qdrant Cloud free tier — search latency
        # ballooned from ~2s to ~40s because returning vectors triggers
        # a per-result disk read inside the search path. Keep the search
        # vector-free; vectors come from a separate cached retrieve.
        search_tasks = [
            qdrant_svc.search_by_vector_with_scores(
                query_vector=c.medoid_embedding.tolist(),
                limit=quota * _OVERSAMPLE,
                exclude_ids=seen,
            )
            for c, quota in zip(clusters, quotas)
        ]
        per_cluster_scored = await asyncio.gather(*search_tasks)

        paper_cluster_map: dict[str, int] = {}
        qdrant_score_map: dict[str, float] = {}
        for cluster, scored_results in zip(clusters, per_cluster_scored):
            for hit in scored_results:
                aid = hit["arxiv_id"]
                if aid not in paper_cluster_map:
                    paper_cluster_map[aid] = cluster.cluster_idx
                if aid not in qdrant_score_map or hit["score"] > qdrant_score_map[aid]:
                    qdrant_score_map[aid] = float(hit["score"])

        per_cluster_ids = [
            [h["arxiv_id"] for h in scored] for scored in per_cluster_scored
        ]
        candidate_ids = merge_quota_results(per_cluster_ids, quotas)

        # Supplement with short-term session context
        if st_vec is not None:
            seen_so_far = seen | set(candidate_ids)
            st_scored = await qdrant_svc.search_by_vector_with_scores(
                query_vector=st_vec.tolist(),
                limit=_ST_SUPPLEMENT,
                exclude_ids=seen_so_far,
            )
            for hit in st_scored:
                aid = hit["arxiv_id"]
                if aid not in set(candidate_ids):
                    candidate_ids.append(aid)
                    paper_cluster_map[aid] = -1  # short-term supplement
                if aid not in qdrant_score_map:
                    qdrant_score_map[aid] = float(hit["score"])

        if not candidate_ids:
            return [], [], {}, 0, {}
        timing["ann_retrieval_ms"] = int((time.time() - t0_ann) * 1000)

        # ── Step 5: Fetch candidate vectors + metadata ────────────────────
        # get_paper_vectors is now LRU-cached by arxiv_id (qdrant_svc),
        # so warm cache makes this cheap; only fresh papers pay the
        # disk-read cost.
        t0_cand_meta = time.time()
        cand_vectors = await qdrant_svc.get_paper_vectors(candidate_ids)
        cand_meta = await turso_svc.fetch_metadata_batch(candidate_ids)
        cand_missing = [cid for cid in candidate_ids if cid not in cand_meta]
        if cand_missing:
            try:
                arxiv_cand_meta = await arxiv_svc.fetch_metadata_batch(cand_missing)
                cand_meta.update(arxiv_cand_meta)
            except Exception as e:
                print(f"[recommendations] arXiv fallback for {len(cand_missing)} IDs failed: {e}")

        # Cache fetched metadata to SQLite for category suppression
        await db.cache_turso_metadata_batch(list(cand_meta.values()))

        # Only process candidates with both vectors and metadata
        valid_ids = [cid for cid in candidate_ids if cid in cand_vectors and cid in cand_meta]
        if not valid_ids:
            return candidate_ids[:limit], [], {}, 0, {}
        timing["candidate_meta_ms"] = int((time.time() - t0_cand_meta) * 1000)

        valid_embs = np.array([cand_vectors[cid] for cid in valid_ids], dtype=np.float32)
        valid_meta = [cand_meta[cid] for cid in valid_ids]

        lt_vec = await profiles.load_profile(user_id, "long_term")
        neg_vec = await profiles.load_profile(user_id, "negative")

        # ── Phase 6.1+: Prepare features 23-30 BEFORE rerank ─────────
        # Category suppression (moved from post-rerank to pre-rerank feature)
        suppressed = await db.get_suppressed_categories(user_id)
        onboarding_categories = await db.get_user_category_filter(user_id)

        # User-level interaction counts (constant across all candidates)
        user_total_saves = len(state.positive_list)
        user_total_dismissals = len(state.negative_list)

        # qdrant_score_map was built above from real cosine scores
        # (Phase 6.5 A1 — replaces the old rank-based approximation)

        qdrant_scores = np.asarray(
            [qdrant_score_map.get(cid, 0.0) for cid in valid_ids],
            dtype=np.float32,
        )

        # Per-candidate cluster importance + medoid (Phase 6.2: per-candidate)
        per_candidate_importance = np.asarray(
            [
                clusters[paper_cluster_map[cid]].importance
                if cid in paper_cluster_map and paper_cluster_map[cid] >= 0
                   and paper_cluster_map[cid] < len(clusters)
                else 0.0
                for cid in valid_ids
            ],
            dtype=np.float32,
        )

        per_candidate_medoids = np.stack(
            [
                np.asarray(
                    clusters[paper_cluster_map[cid]].medoid_embedding,
                    dtype=np.float32,
                )
                if cid in paper_cluster_map and paper_cluster_map[cid] >= 0
                   and paper_cluster_map[cid] < len(clusters)
                else np.zeros(1024, dtype=np.float32)
                for cid in valid_ids
            ],
            axis=0,
        )

        # Per-candidate suppression and onboarding flags
        is_suppressed_arr = np.asarray(
            [
                1.0 if cand_meta.get(cid, {}).get("category", "") in suppressed
                else 0.0
                for cid in valid_ids
            ],
            dtype=np.float32,
        )

        # Feature 26 compares against CATEGORY_GROUPS, which is expressed in
        # arXiv codes (cs.CL, cs.CV, ...).  It must therefore read the raw
        # `arxiv_categories` field, not `category` — the latter holds Turso's
        # friendly primary_topic label ("AI/ML"), so the old comparison against
        # a set of arXiv codes was never true and this feature was pinned to 0.
        # A paper is a match if ANY of its arXiv codes is in the user's set.
        onboarding_match_arr = np.asarray(
            [
                1.0 if (
                    onboarding_categories
                    and set(
                        (cand_meta.get(cid, {}).get("arxiv_categories") or "").split()
                    ) & onboarding_categories
                ) else 0.0
                for cid in valid_ids
            ],
            dtype=np.float32,
        )

        # ── Step 6: LightGBM re-ranking (37 features) ────────────────────
        t0_rerank = time.time()
        reranked_ids, reranked_scores, reranked_embs = rerank_candidates(
            candidate_ids=valid_ids,
            candidate_embeddings=valid_embs,
            candidate_metadata=valid_meta,
            long_term_vec=lt_vec,
            short_term_vec=st_vec,
            negative_vec=neg_vec,
            # Phase 6 additions
            qdrant_scores=qdrant_scores,
            cluster_importance=per_candidate_importance,
            cluster_medoid=per_candidate_medoids,
            is_suppressed_category=is_suppressed_arr,
            onboarding_category_match=onboarding_match_arr,
            user_total_saves=user_total_saves,
            user_total_dismissals=user_total_dismissals,
        )
        t1_rerank = time.time()
        rerank_time_ms = int((t1_rerank - t0_rerank) * 1000)

        # ── Step 4.3: Category suppression (post-rerank safety net) ───────
        # The model now sees feature 25 (is_suppressed_category), but we
        # keep a hard filter as a safety net since the model has zero
        # weight on this feature until retrained.
        if suppressed:
            kept = [
                i for i, cid in enumerate(reranked_ids)
                if cand_meta.get(cid, {}).get("category", "") not in suppressed
            ]
            if kept:
                reranked_ids = [reranked_ids[i] for i in kept]
                reranked_scores = [reranked_scores[i] for i in kept]
                reranked_embs = reranked_embs[kept]

        # ── Step 7: MMR diversity enforcement ─────────────────────────────
        t0_mmr = time.time()
        query_vec = lt_vec if lt_vec is not None else aligned_embs.mean(axis=0)
        mmr_selected = mmr_rerank(
            query_embedding=query_vec,
            candidate_embeddings=reranked_embs,
            candidate_ids=reranked_ids,
            scores=reranked_scores,
            lambda_param=0.6,
            top_k=limit,
        )
        timing["mmr_ms"] = int((time.time() - t0_mmr) * 1000)

        # ── Step 8: Split into the ranked feed and the exploration pool ───
        # Exploration is NOT injected here any more. Doc 06 §3.5 specifies two
        # serendipitous papers per FEED, and with pagination a "feed" is one
        # page — so the injection happens per page in _build_page(), against
        # this pool. Injecting once over a 60-deep pool would have both diluted
        # the ratio and stranded every exploration pick on the last page,
        # because inject_exploration appends.
        mmr_set = set(mmr_selected)
        explore_pool = [aid for aid in reranked_ids if aid not in mmr_set]

        # Phase 4.5 + 6.5: per-paper instrumentation, for the whole pool.
        # candidate_source here is the RETRIEVAL origin; papers served as an
        # exploration pick get that overridden at page-build time, since the
        # same paper is an exploration pick only by virtue of how it was served.
        paper_tags: dict[str, dict] = {}
        for aid in mmr_selected + explore_pool:
            cluster_idx = paper_cluster_map.get(aid)
            if cluster_idx == -1:
                source = "short_term_supplement"
            elif cluster_idx is not None:
                source = f"cluster_{cluster_idx}"
            else:
                source = "tier1_unknown"
            paper_tags[aid] = {
                "ranker_version": _RANKER_VERSION,
                "candidate_source": source,
                "cluster_id": str(cluster_idx) if cluster_idx is not None and cluster_idx >= 0 else "",
                "query_id": query_id,
                "propensity": 1.0,          # deterministic unless served as exploration
                "policy_id": _RANKER_VERSION,
            }

        return mmr_selected, explore_pool, paper_tags, rerank_time_ms, timing

    except Exception as e:
        print(f"[recommendations] multi-interest preprocessing failed: {e}")
        return [], [], {}, 0, {}


# ── Tier 2: EWMA single-vector search ────────────────────────────────────────

async def _ewma_recommend(
    user_id: str, seen: set[str], limit: int
) -> list[str]:
    """
    Use the long-term EWMA profile vector for vector search.

    Only activates after _MIN_EWMA_INTERACTIONS saves so the profile
    has had enough signal to be meaningful.  Returns [] to trigger fallback.
    """
    lt_count = await profiles.get_interaction_count(user_id, "long_term")
    if lt_count < _MIN_EWMA_INTERACTIONS:
        return []

    lt_vec = await profiles.load_profile(user_id, "long_term")
    if lt_vec is None:
        return []

    query_vec = lt_vec.tolist()
    return await qdrant_svc.search_by_vector(
        query_vector=query_vec,
        limit=limit,
        exclude_ids=seen,
    )
