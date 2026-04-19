"""
Tests for the heuristic reranker and MMR diversity enforcement.

Covers:
  - Heuristic scoring ranks relevant papers higher
  - Recency decay prefers newer papers
  - MMR selects diverse items
  - MMR handles edge cases (empty, fewer than top_k)
  - Exploration injection adds serendipitous papers
  - Feature extraction produces correct shape
"""
import pytest
import numpy as np

from app.recommend.reranker import compute_features, heuristic_score, rerank_candidates
from app.recommend.diversity import mmr_rerank, inject_exploration


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_embeddings(n: int, dim: int = 1024, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    embs = rng.randn(n, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def _make_metadata(n: int, base_year: str = "2025") -> list[dict]:
    """Generate metadata with varying dates."""
    return [
        {"published": f"{base_year}-{(i % 12) + 1:02d}-15", "arxiv_id": f"paper_{i}"}
        for i in range(n)
    ]


# ── Feature extraction tests ─────────────────────────────────────────────────

def test_feature_shape():
    """Feature matrix should have shape (N, 5) — includes negative sim (Doc 06)."""
    n = 20
    embs = _make_embeddings(n)
    meta = _make_metadata(n)
    lt = _make_embeddings(1)[0]
    st = _make_embeddings(1, seed=99)[0]

    features = compute_features(embs, meta, lt, st)
    assert features.shape == (n, 5), f"Expected (20, 5), got {features.shape}"


def test_features_without_profiles():
    """Features should still compute when profiles are None."""
    n = 10
    embs = _make_embeddings(n)
    meta = _make_metadata(n)

    features = compute_features(embs, meta, long_term_vec=None, short_term_vec=None)
    assert features.shape == (n, 5)
    # Cosine sim columns should be all zeros (LT, ST, and negative)
    assert np.allclose(features[:, 0], 0.0)
    assert np.allclose(features[:, 1], 0.0)
    assert np.allclose(features[:, 4], 0.0)


# ── Heuristic scoring tests ──────────────────────────────────────────────────

def test_heuristic_score_shape():
    """Heuristic scores should have shape (N,)."""
    n = 15
    features = np.random.randn(n, 5).astype(np.float32)  # 5 features (Doc 06)
    scores = heuristic_score(features)
    assert scores.shape == (n,)


def test_relevant_paper_scores_higher():
    """A paper with high cosine similarity should score higher than an irrelevant one."""
    lt_vec = _make_embeddings(1)[0]
    # Paper very similar to profile
    similar = lt_vec + np.random.randn(1024).astype(np.float32) * 0.01
    similar /= np.linalg.norm(similar)
    # Paper very different
    different = _make_embeddings(1, seed=999)[0]

    embs = np.array([similar, different])
    meta = [{"published": "2025-06-01"}, {"published": "2025-06-01"}]

    features = compute_features(embs, meta, lt_vec)
    scores = heuristic_score(features)
    assert scores[0] > scores[1], "Similar paper should score higher"


# ── Rerank candidates end-to-end ──────────────────────────────────────────────

def test_rerank_returns_all():
    """Reranking should return all candidate IDs in sorted order."""
    n = 10
    ids = [f"paper_{i}" for i in range(n)]
    embs = _make_embeddings(n)
    meta = _make_metadata(n)
    lt = _make_embeddings(1)[0]

    sorted_ids, scores, sorted_embs = rerank_candidates(ids, embs, meta, lt)
    assert len(sorted_ids) == n
    assert set(sorted_ids) == set(ids)


def test_rerank_scores_descending():
    """Scores should be in descending order."""
    n = 20
    ids = [f"p_{i}" for i in range(n)]
    embs = _make_embeddings(n)
    meta = _make_metadata(n)

    _, scores, _ = rerank_candidates(ids, embs, meta)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Score {i}: {scores[i]} < {scores[i+1]}"


# ── MMR diversity tests ──────────────────────────────────────────────────────

def test_mmr_selects_correct_count():
    """MMR should return exactly top_k items."""
    n = 50
    embs = _make_embeddings(n)
    ids = [f"paper_{i}" for i in range(n)]
    query = _make_embeddings(1)[0]

    result = mmr_rerank(query, embs, ids, top_k=10)
    assert len(result) == 10


def test_mmr_returns_all_if_fewer():
    """When candidates < top_k, return all."""
    embs = _make_embeddings(3)
    ids = ["a", "b", "c"]
    query = _make_embeddings(1)[0]

    result = mmr_rerank(query, embs, ids, top_k=10)
    assert len(result) == 3
    assert set(result) == {"a", "b", "c"}


def test_mmr_empty_input():
    """Empty candidates should return empty list."""
    query = _make_embeddings(1)[0]
    result = mmr_rerank(query, np.array([]).reshape(0, 1024), [], top_k=5)
    assert result == []


def test_mmr_diversity_effect():
    """MMR should prefer diverse items over very similar ones."""
    # Create 3 clusters of 5 similar papers each
    rng = np.random.RandomState(42)
    cluster_centers = [rng.randn(1024).astype(np.float32) for _ in range(3)]
    for c in cluster_centers:
        c /= np.linalg.norm(c)

    embeddings = []
    ids = []
    for ci, center in enumerate(cluster_centers):
        for j in range(5):
            noise = rng.randn(1024).astype(np.float32) * 0.02
            vec = center + noise
            vec /= np.linalg.norm(vec)
            embeddings.append(vec)
            ids.append(f"cluster{ci}_paper{j}")

    embs = np.array(embeddings)
    query = embs.mean(axis=0)  # centroid of all = equally interested

    result = mmr_rerank(query, embs, ids, lambda_param=0.5, top_k=6)

    # With diversity, we should get papers from multiple clusters
    clusters_represented = set()
    for pid in result:
        clusters_represented.add(pid.split("_")[0])

    assert len(clusters_represented) >= 2, \
        f"Expected diversity across clusters, only got: {clusters_represented}"


# ── Exploration injection tests ───────────────────────────────────────────────

def test_exploration_adds_papers():
    """Exploration should add papers not in the selected list."""
    selected = ["a", "b", "c"]
    all_candidates = ["a", "b", "c", "d", "e", "f"]

    result = inject_exploration(selected, all_candidates, n_explore=2, seed=42)
    assert len(result) == 5  # 3 selected + 2 exploration
    # First 3 should be the original selection
    assert result[:3] == selected
    # Last 2 should be from the unselected pool
    for paper in result[3:]:
        assert paper not in selected


def test_exploration_no_duplicates():
    """Exploration papers should not duplicate selected papers."""
    selected = ["a", "b"]
    all_candidates = ["a", "b", "c", "d"]

    result = inject_exploration(selected, all_candidates, n_explore=2, seed=42)
    assert len(result) == len(set(result)), "Duplicates found"


def test_exploration_handles_empty_pool():
    """If all candidates are already selected, no exploration added."""
    selected = ["a", "b", "c"]
    all_candidates = ["a", "b", "c"]

    result = inject_exploration(selected, all_candidates, n_explore=2)
    assert len(result) == 3
