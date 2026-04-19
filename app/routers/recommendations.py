"""
Recommendations router.

GET /api/recommendations
  – Called by HTMX on page load (hx-trigger="load")
  – Returns the recommendations partial HTML

Recommendation pipeline (cascading fallback):
  Phase 2b: Multi-interest clustering → prefetch + RRF fusion  (≥5 saves)
  Phase 2a: EWMA long-term vector → single vector search       (≥3 saves)
  Phase 1:  Qdrant BEST_SCORE Recommend API with raw IDs       (≥1 save)
"""
import json
import uuid
import numpy as np
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse
from app import qdrant_svc, arxiv_svc, user_state as us
from app.config import COOKIE_NAME, REC_LIMIT, REC_MIN_POSITIVES
from app.templates_env import templates
from app.recommend import profiles
from app.recommend.clustering import (
    compute_clusters,
    save_clusters_to_db,
    load_clusters_from_db,
    MIN_PAPERS_FOR_CLUSTERING,
)
from app.recommend.reranker import rerank_candidates
from app.recommend.diversity import mmr_rerank, inject_exploration

router = APIRouter(prefix="/api")

# Minimum EWMA interactions before switching from ID-based to vector-based recs
_MIN_EWMA_INTERACTIONS = 3


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
        return _empty_resp()

    seen = us.all_seen(user_id)

    # ── Tier 1: Multi-interest clustering + RRF (Phase 2b, ≥5 saves) ─────
    rec_arxiv_ids = await _multi_interest_recommend(user_id, state, seen, REC_LIMIT)

    # ── Tier 2: EWMA single-vector search (Phase 2a, ≥3 saves) ───────────
    if not rec_arxiv_ids:
        rec_arxiv_ids = await _ewma_recommend(user_id, seen, REC_LIMIT)

    # ── Tier 3: Qdrant Recommend API (Phase 1 fallback, ≥1 save) ─────────
    if not rec_arxiv_ids:
        rec_arxiv_ids = await qdrant_svc.recommend(
            positive_arxiv_ids=state.positive_list,
            negative_arxiv_ids=state.negative_list,
            seen_arxiv_ids=seen,
            limit=REC_LIMIT,
        )

    if not rec_arxiv_ids:
        return _empty_resp()

    meta = await arxiv_svc.fetch_metadata_batch(rec_arxiv_ids)
    papers = [
        {**meta[aid], "saved": False, "dismissed": False}
        for aid in rec_arxiv_ids
        if aid in meta
    ]

    resp = templates.TemplateResponse(
        request,
        "partials/recommendations.html",
        {"papers": papers},
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


# ── Tier 1: Multi-interest clustering + prefetch RRF ─────────────────────────

# Per-cluster candidate limits (descending by importance)
_CLUSTER_LIMITS = [40, 30, 25, 20, 15, 15, 15]


async def _multi_interest_recommend(
    user_id: str, state, seen: set[str], limit: int
) -> list[str]:
    """
    Full recommendation pipeline (Phase 2b + 2c):
      1. Ward clustering → identify distinct interests
      2. Prefetch + RRF → retrieve ~100 candidates
      3. Heuristic re-ranking → score candidates
      4. MMR diversity → select top-k with diversity
      5. Exploration injection → 1-2 serendipitous papers

    Only activates when the user has ≥ MIN_PAPERS_FOR_CLUSTERING saves.
    Returns [] to trigger fallback to Tier 2.
    """
    positives = state.positive_list
    if len(positives) < MIN_PAPERS_FOR_CLUSTERING:
        return []

    try:
        # Fetch embeddings for all saved papers
        vectors = await qdrant_svc.get_paper_vectors(positives)
        if len(vectors) < MIN_PAPERS_FOR_CLUSTERING:
            return []

        # Build aligned arrays (only papers we got vectors for)
        aligned_ids = [pid for pid in positives if pid in vectors]
        aligned_embs = np.array(
            [vectors[pid] for pid in aligned_ids], dtype=np.float32
        )

        # ── Step 1: Compute interest clusters ─────────────────────────────
        clusters = compute_clusters(aligned_ids, aligned_embs)
        await save_clusters_to_db(user_id, clusters)

        # ── Step 2: Multi-interest retrieval via prefetch + RRF ───────────
        interest_vectors = []
        for i, cluster in enumerate(clusters):
            per_cluster_limit = _CLUSTER_LIMITS[i] if i < len(_CLUSTER_LIMITS) else 15
            interest_vectors.append(
                (cluster.medoid_embedding.tolist(), per_cluster_limit)
            )

        st_vec = await profiles.load_profile(user_id, "short_term")
        st_list = st_vec.tolist() if st_vec is not None else None

        candidate_ids = await qdrant_svc.multi_interest_search(
            interest_vectors=interest_vectors,
            short_term_vector=st_list,
            exclude_ids=seen,
            total_limit=100,  # retrieve wide, narrow with re-ranking
        )

        if not candidate_ids:
            return []

        # ── Step 3: Re-rank candidates ────────────────────────────────────
        # Fetch embeddings + metadata for candidates
        cand_vectors = await qdrant_svc.get_paper_vectors(candidate_ids)
        cand_meta = await arxiv_svc.fetch_metadata_batch(candidate_ids)

        # Only process candidates we have both vectors and metadata for
        valid_ids = [cid for cid in candidate_ids if cid in cand_vectors and cid in cand_meta]
        if not valid_ids:
            return candidate_ids[:limit]  # fallback: return raw retrieval

        valid_embs = np.array([cand_vectors[cid] for cid in valid_ids], dtype=np.float32)
        valid_meta = [cand_meta[cid] for cid in valid_ids]

        lt_vec = await profiles.load_profile(user_id, "long_term")
        neg_vec = await profiles.load_profile(user_id, "negative")

        reranked_ids, reranked_scores, reranked_embs = rerank_candidates(
            candidate_ids=valid_ids,
            candidate_embeddings=valid_embs,
            candidate_metadata=valid_meta,
            long_term_vec=lt_vec,
            short_term_vec=st_vec,
            negative_vec=neg_vec,
        )

        # ── Step 4: MMR diversity enforcement ─────────────────────────────
        query_vec = lt_vec if lt_vec is not None else aligned_embs.mean(axis=0)
        mmr_selected = mmr_rerank(
            query_embedding=query_vec,
            candidate_embeddings=reranked_embs,
            candidate_ids=reranked_ids,
            scores=reranked_scores,
            lambda_param=0.6,
            top_k=limit,
        )

        # ── Step 5: Exploration injection ─────────────────────────────────
        final = inject_exploration(
            selected_ids=mmr_selected,
            all_candidate_ids=reranked_ids,
            n_explore=2,
        )

        return final[:limit + 2]  # allow slightly over limit for exploration

    except Exception as e:
        print(f"[recommendations] multi-interest search failed: {e}")
        return []


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


