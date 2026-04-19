"""
Hybrid search orchestrator — Phase 3.

Orchestrates the full pipeline:
  1. LLM rewrite (optional, via Groq)
  2. BGE-M3 encode → dense + sparse
  3. Parallel search: Qdrant dense + Zilliz sparse
  4. RRF fusion (K=60)
  5. Recency rerank: 0.80 × RRF + 0.20 × recency
  6. Return ranked arxiv_ids

Doc 06 confirms: RRF is correct for search (fusing different retrievers
answering the SAME query).  This is different from recommendations where
quota is correct (fusing different queries for the SAME user).
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from app import config
from app import embed_svc
from app import qdrant_svc
from app import zilliz_svc
from app import groq_svc


# ── Public API ───────────────────────────────────────────────────────────────

async def search(
    query: str,
    limit: int = 10,
    use_rewrite: bool = True,
) -> list[str]:
    """
    Hybrid semantic search — returns a list of arxiv_ids ranked by
    fused relevance.

    Pipeline:
      rewrite → encode → parallel(dense, sparse) → RRF → rerank

    Args:
        query: User's raw search query.
        limit: Number of results to return.
        use_rewrite: Whether to attempt LLM query rewriting.

    Returns:
        list of arxiv_id strings, sorted by final score descending.
        Never raises — returns empty list on total failure.
    """
    query = query.strip()
    if not query:
        return []

    # ── Step 1: LLM rewrite (optional, never blocks) ─────────────────────
    search_query = query
    if use_rewrite:
        try:
            search_query = await groq_svc.rewrite(query)
        except Exception:
            search_query = query  # Fallback guaranteed

    # ── Step 2: BGE-M3 encode (dense + sparse in one pass) ───────────────
    try:
        dense_vec, sparse_dict = embed_svc.encode_query(search_query)
    except Exception as e:
        print(f"[hybrid_search] Encoding failed: {e}")
        return []

    # How many candidates to fetch before reranking
    fetch_k = limit * config.SEARCH_FETCH_K_MULTIPLIER

    # ── Step 3: Parallel dense + sparse search ───────────────────────────
    dense_results, sparse_results = await asyncio.gather(
        qdrant_svc.search_dense(dense_vec.tolist(), limit=fetch_k),
        zilliz_svc.search_sparse(sparse_dict, limit=fetch_k),
        return_exceptions=True,
    )

    # Handle individual failures gracefully
    if isinstance(dense_results, Exception):
        print(f"[hybrid_search] Dense search failed: {dense_results}")
        dense_results = []
    if isinstance(sparse_results, Exception):
        print(f"[hybrid_search] Sparse search failed: {sparse_results}")
        sparse_results = []

    if not dense_results and not sparse_results:
        return []

    # ── Step 4: RRF fusion ───────────────────────────────────────────────
    fused = _rrf_fuse(dense_results, sparse_results, k=config.SEARCH_RRF_K)

    if not fused:
        return []

    # ── Step 5: Recency rerank ───────────────────────────────────────────
    ranked = _recency_rerank(fused)

    # ── Step 6: Return top results ───────────────────────────────────────
    return [item["arxiv_id"] for item in ranked[:limit]]


# ── RRF fusion ───────────────────────────────────────────────────────────────

def _rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — merges results from dense and sparse search.

    score[paper] = 1/(k + rank_dense) + 1/(k + rank_sparse)

    RRF is rank-based, so raw scores from different systems don't need
    normalization — this is why it works for fusing Qdrant cosine scores
    with Zilliz IP scores.

    Args:
        dense_results: list of {'arxiv_id': str, 'score': float} from Qdrant
        sparse_results: list of {'arxiv_id': str, 'score': float} from Zilliz
        k: RRF constant (default 60)

    Returns:
        list of {'arxiv_id': str, 'rrf_score': float} sorted by rrf_score desc
    """
    scores: dict[str, float] = {}

    # Dense contributions (rank = position in sorted list, 1-indexed)
    for rank, item in enumerate(dense_results, start=1):
        aid = item["arxiv_id"]
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)

    # Sparse contributions
    for rank, item in enumerate(sparse_results, start=1):
        aid = item["arxiv_id"]
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)

    # Sort by fused score descending
    fused = [
        {"arxiv_id": aid, "rrf_score": score}
        for aid, score in scores.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused


# ── Recency rerank ───────────────────────────────────────────────────────────

def _recency_rerank(fused: list[dict]) -> list[dict]:
    """
    Apply recency boost to RRF scores.

    final_score = SEARCH_SEMANTIC_WEIGHT × norm_rrf + SEARCH_RECENCY_WEIGHT × recency

    Recency is estimated from the arXiv ID (YYMM format) since we don't have
    publication dates at this stage.  Papers not parseable get neutral score.

    The semantic weight (0.80) ensures RRF dominates, while recency (0.20)
    provides a mild boost to newer papers.
    """
    if not fused:
        return fused

    # Normalize RRF scores to [0, 1]
    max_rrf = max(item["rrf_score"] for item in fused)
    min_rrf = min(item["rrf_score"] for item in fused)
    rrf_range = max_rrf - min_rrf if max_rrf != min_rrf else 1.0

    now_ym = datetime.now().year * 12 + datetime.now().month

    for item in fused:
        # Normalize RRF to [0, 1]
        norm_rrf = (item["rrf_score"] - min_rrf) / rrf_range

        # Estimate recency from arXiv ID (format: YYMM.NNNNN)
        recency = 0.5  # neutral default
        aid = item["arxiv_id"]
        try:
            parts = aid.split(".")
            if len(parts) >= 2 and len(parts[0]) == 4:
                yy = int(parts[0][:2])
                mm = int(parts[0][2:4])
                year = 2000 + yy if yy < 100 else yy
                paper_ym = year * 12 + mm
                months_ago = max(0, now_ym - paper_ym)
                # Decay: recent papers get ~1.0, 10-year-old papers get ~0.0
                recency = max(0.0, 1.0 - months_ago / 120.0)
        except (ValueError, IndexError):
            pass

        item["final_score"] = (
            config.SEARCH_SEMANTIC_WEIGHT * norm_rrf
            + config.SEARCH_RECENCY_WEIGHT * recency
        )

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused
