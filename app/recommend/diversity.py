"""
MMR diversity enforcement + exploration injection.

Maximal Marginal Relevance (Carbonell & Goldstein, 1998) selects items
that are both relevant to the query AND diverse from each other.

Formula:
  MMR = argmax[λ × Sim(d_i, Q) − (1−λ) × max(Sim(d_i, d_j))]

where d_j iterates over already-selected items.

Reference: Research-MultiInterest_Recommender_Architecture.md §4
  "MMR provides practical diversity enforcement ...
   Setting λ=0.6 provides a good balance between relevance and diversity
   for academic paper discovery."
"""
from __future__ import annotations

import random
import numpy as np


def mmr_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: list[str],
    scores: list[float] | None = None,
    lambda_param: float = 0.6,
    top_k: int = 20,
) -> list[str]:
    """
    Select top_k items from candidates using Maximal Marginal Relevance.

    Args:
        query_embedding: the user's profile vector (1024-dim)
        candidate_embeddings: shape (N, 1024), embeddings for all candidates
        candidate_ids: arxiv_ids for each candidate (same order as embeddings)
        scores: optional pre-computed relevance scores (from LightGBM or RRF).
                If None, uses cosine similarity to query_embedding.
        lambda_param: balance between relevance (1.0) and diversity (0.0).
                      RFC recommends 0.6 for academic papers.
        top_k: how many items to select

    Returns:
        list of arxiv_ids in MMR-selected order

    Latency: <1ms for 100 candidates with precomputed embeddings.
    """
    n = len(candidate_ids)
    if n == 0:
        return []
    if n <= top_k:
        return list(candidate_ids)

    # Compute relevance scores
    if scores is not None:
        relevance = np.array(scores, dtype=np.float64)
        # Normalise to [0, 1]
        r_min, r_max = relevance.min(), relevance.max()
        if r_max > r_min:
            relevance = (relevance - r_min) / (r_max - r_min)
        else:
            relevance = np.ones(n, dtype=np.float64)
    else:
        # Cosine similarity to query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        cand_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
        )
        relevance = cand_norms @ query_norm

    # Precompute pairwise cosine similarity matrix
    cand_norms = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
    )
    sim_matrix = cand_norms @ cand_norms.T

    # Greedy MMR selection
    selected_indices: list[int] = []
    remaining = set(range(n))

    for _ in range(min(top_k, n)):
        best_score = -float("inf")
        best_idx = -1

        for idx in remaining:
            # Relevance term
            rel = lambda_param * relevance[idx]

            # Diversity term: max similarity to any already-selected item
            if selected_indices:
                max_sim = max(sim_matrix[idx, j] for j in selected_indices)
            else:
                max_sim = 0.0

            mmr_score = rel - (1.0 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [candidate_ids[i] for i in selected_indices]


def inject_exploration(
    selected_ids: list[str],
    all_candidate_ids: list[str],
    n_explore: int = 2,
    seed: int | None = None,
) -> list[str]:
    """
    Inject exploration papers from candidates not already selected.

    Picks n_explore random papers from the unselected pool and appends
    them to the end of the list.  This follows TikTok's 15-25% exploration
    allocation principle for breaking filter bubbles.

    Args:
        selected_ids: the MMR-selected arxiv_ids
        all_candidate_ids: the full candidate pool (before MMR selection)
        n_explore: how many exploration papers to inject
        seed: optional random seed for reproducibility

    Returns:
        selected_ids with exploration papers appended
    """
    selected_set = set(selected_ids)
    pool = [cid for cid in all_candidate_ids if cid not in selected_set]

    if not pool:
        return list(selected_ids)

    rng = random.Random(seed)
    n_pick = min(n_explore, len(pool))
    exploration = rng.sample(pool, n_pick)

    return list(selected_ids) + exploration
