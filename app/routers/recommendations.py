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
import uuid
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
from app.recommend.diversity import mmr_rerank, inject_exploration

router = APIRouter(prefix="/api")

# Phase 4.5: Pipeline version tag for instrumentation.  Bump this on any
# change to the ranking logic so A/B attribution is possible.
_RANKER_VERSION = "v4.1_quota_hungarian_suppression"

# Minimum EWMA interactions before switching from ID-based to vector-based recs
_MIN_EWMA_INTERACTIONS = 3

# Candidate oversampling factor per cluster (fetch more than quota to handle dedup)
_OVERSAMPLE = 3

# Short-term session context: fixed supplementary pool size
_ST_SUPPLEMENT = 20


@router.get("/recommendations", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())
    state = await us.ensure_loaded(user_id)

    def _empty_resp():
        r = templates.TemplateResponse(
            request,
            "partials/empty_recs.html",
            {"min_saves": REC_MIN_POSITIVES},
        )
        r.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
        return r

    if not state.has_enough_for_recs():
        # ── Tier 0: Category-filtered trending (Phase 5 cold-start) ──────
        # If user has onboarded with category selections but hasn't saved
        # enough papers yet, serve trending papers in their areas.
        category_filter = await db.get_user_category_filter(user_id)
        if category_filter:
            trending = await turso_svc.fetch_trending_by_categories(
                category_filter, limit=REC_LIMIT,
            )
            if trending:
                papers = []
                for paper in trending:
                    paper["saved"] = False
                    paper["dismissed"] = False
                    paper["ranker_version"] = _RANKER_VERSION
                    paper["candidate_source"] = "trending_category_fallback"
                    paper["cluster_id"] = ""
                    papers.append(paper)

                r = templates.TemplateResponse(
                    request,
                    "partials/recommendations.html",
                    {"papers": papers, "source": "recommendation", "trending": True},
                )
                r.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
                return r

        return _empty_resp()

    seen = us.all_seen(user_id)

    # Phase 4.5: paper_tags maps arxiv_id → instrumentation metadata
    # populated by whichever tier serves the result.
    paper_tags: dict[str, dict] = {}
    rec_arxiv_ids: list[str] = []

    # ── Tier 1: Multi-interest clustering + quota fusion (≥5 saves) ──────
    rec_arxiv_ids, paper_tags = await _multi_interest_recommend(
        user_id, state, seen, REC_LIMIT,
    )

    # ── Tier 2: EWMA single-vector search (≥3 saves) ──────────────────────
    if not rec_arxiv_ids:
        rec_arxiv_ids = await _ewma_recommend(user_id, seen, REC_LIMIT)
        for aid in rec_arxiv_ids:
            paper_tags[aid] = {
                "ranker_version": _RANKER_VERSION,
                "candidate_source": "ewma_longterm",
                "cluster_id": "",
            }

    # ── Tier 3: Qdrant Recommend API (≥1 save fallback) ───────────────────
    if not rec_arxiv_ids:
        rec_arxiv_ids = await qdrant_svc.recommend(
            positive_arxiv_ids=state.positive_list,
            negative_arxiv_ids=state.negative_list,
            seen_arxiv_ids=seen,
            limit=REC_LIMIT,
        )
        for aid in rec_arxiv_ids:
            paper_tags[aid] = {
                "ranker_version": _RANKER_VERSION,
                "candidate_source": "qdrant_recommend",
                "cluster_id": "",
            }

    if not rec_arxiv_ids:
        return _empty_resp()

    # Phase 3.5: Turso primary, arXiv API fallback
    meta = await turso_svc.fetch_metadata_batch(rec_arxiv_ids)
    missing = [aid for aid in rec_arxiv_ids if aid not in meta]
    if missing:
        try:
            arxiv_meta = await arxiv_svc.fetch_metadata_batch(missing)
            meta.update(arxiv_meta)
        except Exception as e:
            print(f"[recommendations] arXiv fallback for {len(missing)} IDs failed: {e}")

    # Cache to SQLite so category suppression JOINs work (Phase 4.3)
    await db.cache_turso_metadata_batch(list(meta.values()))

    papers = []
    for aid in rec_arxiv_ids:
        if aid not in meta:
            continue
        tags = paper_tags.get(aid, {})
        papers.append({
            **meta[aid],
            "saved": False,
            "dismissed": False,
            # Phase 4.5 instrumentation — embedded in card, flows back via HTMX
            "ranker_version": tags.get("ranker_version", _RANKER_VERSION),
            "candidate_source": tags.get("candidate_source", ""),
            "cluster_id": tags.get("cluster_id", ""),
        })

    resp = templates.TemplateResponse(
        request,
        "partials/recommendations.html",
        {"papers": papers},
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


# ── Tier 1: Multi-interest clustering + quota fusion ─────────────────────────

async def _multi_interest_recommend(
    user_id: str, state, seen: set[str], limit: int
) -> tuple[list[str], dict[str, dict]]:
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

    Returns ([], {}) to trigger fallback to Tier 2.
    Phase 4.5: second element is {arxiv_id: {ranker_version, candidate_source, cluster_id}}.
    """
    positives = state.positive_list
    if len(positives) < MIN_PAPERS_FOR_CLUSTERING:
        return [], {}

    try:
        # Fetch embeddings for all saved papers
        vectors = await qdrant_svc.get_paper_vectors(positives)
        if len(vectors) < MIN_PAPERS_FOR_CLUSTERING:
            return [], {}

        # Build aligned arrays (only papers we got vectors for)
        aligned_ids = [pid for pid in positives if pid in vectors]
        aligned_embs = np.array(
            [vectors[pid] for pid in aligned_ids], dtype=np.float32
        )

        # ── Step 1: Compute interest clusters ─────────────────────────────
        clusters = compute_clusters(aligned_ids, aligned_embs)

        # ── Step 4.2: Stabilise cluster IDs with Hungarian matching ───────
        old_clusters_data = await load_clusters_from_db(user_id)
        if old_clusters_data:
            from app.recommend.clustering import InterestCluster
            old_clusters = [
                InterestCluster(
                    cluster_idx=row["cluster_idx"],
                    medoid_paper_id=row["medoid_paper_id"],
                    medoid_embedding=np.array(
                        vectors[row["medoid_paper_id"]], dtype=np.float32
                    ) if row["medoid_paper_id"] in vectors else np.zeros(1024, dtype=np.float32),
                    paper_ids=[],
                    importance=row["importance"],
                )
                for row in old_clusters_data
            ]
            clusters = stabilize_cluster_ids(clusters, old_clusters)

        await save_clusters_to_db(user_id, clusters)

        # ── Step 2: Quota allocation ───────────────────────────────────────
        importances = [c.importance for c in clusters]
        quotas = allocate_quotas(importances, total_slots=100, min_slots=3)

        # ── Step 3: Parallel per-cluster ANN searches ─────────────────────
        st_vec = await profiles.load_profile(user_id, "short_term")

        search_tasks = [
            qdrant_svc.search_by_vector(
                query_vector=c.medoid_embedding.tolist(),
                limit=quota * _OVERSAMPLE,
                exclude_ids=seen,
            )
            for c, quota in zip(clusters, quotas)
        ]
        per_cluster_results = await asyncio.gather(*search_tasks)

        # Phase 4.5: Build paper → cluster mapping BEFORE merge (so we know
        # which cluster each paper was retrieved from).
        paper_cluster_map: dict[str, int] = {}
        for cluster, result_ids in zip(clusters, per_cluster_results):
            for aid in result_ids:
                if aid not in paper_cluster_map:  # first-occurrence wins
                    paper_cluster_map[aid] = cluster.cluster_idx

        # Apply quota merge (dedup globally, respect per-cluster quotas)
        candidate_ids = merge_quota_results(list(per_cluster_results), quotas)

        # Supplement with short-term session context
        if st_vec is not None:
            seen_so_far = seen | set(candidate_ids)
            st_results = await qdrant_svc.search_by_vector(
                query_vector=st_vec.tolist(),
                limit=_ST_SUPPLEMENT,
                exclude_ids=seen_so_far,
            )
            for aid in st_results:
                if aid not in set(candidate_ids):
                    candidate_ids.append(aid)
                    paper_cluster_map[aid] = -1  # short-term supplement

        if not candidate_ids:
            return [], {}

        # ── Step 5: Fetch candidate vectors + metadata ────────────────────
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
            return candidate_ids[:limit], {}

        valid_embs = np.array([cand_vectors[cid] for cid in valid_ids], dtype=np.float32)
        valid_meta = [cand_meta[cid] for cid in valid_ids]

        lt_vec = await profiles.load_profile(user_id, "long_term")
        neg_vec = await profiles.load_profile(user_id, "negative")

        # ── Step 6: Heuristic re-ranking ──────────────────────────────────
        reranked_ids, reranked_scores, reranked_embs = rerank_candidates(
            candidate_ids=valid_ids,
            candidate_embeddings=valid_embs,
            candidate_metadata=valid_meta,
            long_term_vec=lt_vec,
            short_term_vec=st_vec,
            negative_vec=neg_vec,
        )

        # ── Step 4.3: Category suppression ────────────────────────────────
        suppressed = await db.get_suppressed_categories(user_id)
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
        query_vec = lt_vec if lt_vec is not None else aligned_embs.mean(axis=0)
        mmr_selected = mmr_rerank(
            query_embedding=query_vec,
            candidate_embeddings=reranked_embs,
            candidate_ids=reranked_ids,
            scores=reranked_scores,
            lambda_param=0.6,
            top_k=limit,
        )

        # ── Step 8: Exploration injection ─────────────────────────────────
        final = inject_exploration(
            selected_ids=mmr_selected,
            all_candidate_ids=reranked_ids,
            n_explore=2,
        )
        final = final[:limit + 2]

        # Phase 4.5: Build per-paper instrumentation tags
        exploration_set = set(final) - set(mmr_selected)
        paper_tags: dict[str, dict] = {}
        for aid in final:
            cluster_idx = paper_cluster_map.get(aid)
            if aid in exploration_set:
                source = "exploration"
            elif cluster_idx == -1:
                source = "short_term_supplement"
            elif cluster_idx is not None:
                source = f"cluster_{cluster_idx}"
            else:
                source = "tier1_unknown"
            paper_tags[aid] = {
                "ranker_version": _RANKER_VERSION,
                "candidate_source": source,
                "cluster_id": str(cluster_idx) if cluster_idx is not None and cluster_idx >= 0 else "",
            }

        return final, paper_tags

    except Exception as e:
        print(f"[recommendations] multi-interest search failed: {e}")
        return [], {}


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
