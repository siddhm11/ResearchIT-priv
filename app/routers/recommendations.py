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

    # Phase 6.5 B1: one query_id per feed request for per-feed CTR analysis
    query_id = str(uuid.uuid4())

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
                for idx, paper in enumerate(trending):
                    paper["saved"] = False
                    paper["dismissed"] = False
                    paper["ranker_version"] = _RANKER_VERSION
                    paper["candidate_source"] = "trending_category_fallback"
                    paper["cluster_id"] = ""
                    paper["query_id"] = query_id
                    paper["position"] = idx
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
        user_id, state, seen, REC_LIMIT, query_id=query_id,
    )

    # ── Tier 2: EWMA single-vector search (≥3 saves) ──────────────────────
    if not rec_arxiv_ids:
        rec_arxiv_ids = await _ewma_recommend(user_id, seen, REC_LIMIT)
        for aid in rec_arxiv_ids:
            paper_tags[aid] = {
                "ranker_version": _RANKER_VERSION,
                "candidate_source": "ewma_longterm",
                "cluster_id": "",
                "query_id": query_id,
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
                "query_id": query_id,
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
    for idx, aid in enumerate(rec_arxiv_ids):
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
            # Phase 6.5 B1: query_id + position for per-feed CTR
            "query_id": tags.get("query_id", query_id),
            "position": idx,
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
    user_id: str, state, seen: set[str], limit: int,
    *, query_id: str = "",
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

        # ── Step 2: Quota allocation ───────────────────────────────────────
        importances = [c.importance for c in clusters]
        quotas = allocate_quotas(importances, total_slots=100, min_slots=3)

        # ── Step 3: Parallel per-cluster ANN searches ─────────────────────
        st_vec = await profiles.load_profile(user_id, "short_term")

        search_tasks = [
            qdrant_svc.search_by_vector_with_scores(
                query_vector=c.medoid_embedding.tolist(),
                limit=quota * _OVERSAMPLE,
                exclude_ids=seen,
            )
            for c, quota in zip(clusters, quotas)
        ]
        per_cluster_scored = await asyncio.gather(*search_tasks)

        # Build paper → cluster map AND real qdrant_score_map in one pass.
        # Phase 6.5 A1: replaces the old rank-based linear decay approximation.
        paper_cluster_map: dict[str, int] = {}
        qdrant_score_map: dict[str, float] = {}
        for cluster, scored_results in zip(clusters, per_cluster_scored):
            for hit in scored_results:
                aid = hit["arxiv_id"]
                if aid not in paper_cluster_map:  # first-occurrence wins
                    paper_cluster_map[aid] = cluster.cluster_idx
                # Keep highest cosine if a paper appears in multiple clusters
                if aid not in qdrant_score_map or hit["score"] > qdrant_score_map[aid]:
                    qdrant_score_map[aid] = float(hit["score"])

        # merge_quota_results expects list[list[str]] — extract IDs
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

        onboarding_match_arr = np.asarray(
            [
                1.0 if cand_meta.get(cid, {}).get("category", "") in onboarding_categories
                else 0.0
                for cid in valid_ids
            ],
            dtype=np.float32,
        )

        # ── Step 6: LightGBM re-ranking (37 features) ────────────────────
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
                "query_id": query_id,
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
