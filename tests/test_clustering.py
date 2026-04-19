"""
Tests for Ward hierarchical clustering.

Covers:
  - Well-separated embeddings produce distinct clusters
  - Single cluster for < MIN_PAPERS_FOR_CLUSTERING papers
  - Medoids are actual paper IDs (not synthetic centroids)
  - Importance scores are ordered correctly
  - Cluster count respects MAX_CLUSTERS
  - DB persistence round-trip
"""
import asyncio
import pytest
import numpy as np

from app.recommend.clustering import (
    compute_clusters,
    InterestCluster,
    MIN_PAPERS_FOR_CLUSTERING,
    MAX_CLUSTERS,
    _find_medoid,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cluster_embeddings(
    n_clusters: int,
    papers_per_cluster: int,
    dim: int = 1024,
    spread: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """
    Generate well-separated embedding clusters for testing.
    Each cluster is centered on a random unit vector with small noise.
    """
    rng = np.random.RandomState(seed)
    ids = []
    embeddings = []

    for c in range(n_clusters):
        # Random cluster center (unit vector)
        center = rng.randn(dim).astype(np.float32)
        center /= np.linalg.norm(center)

        for j in range(papers_per_cluster):
            noise = rng.randn(dim).astype(np.float32) * spread
            vec = center + noise
            vec /= np.linalg.norm(vec)
            embeddings.append(vec)
            ids.append(f"paper_{c}_{j}")

    return ids, np.array(embeddings)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_well_separated_clusters_detected():
    """3 well-separated groups should produce 3 distinct clusters."""
    ids, embs = _make_cluster_embeddings(n_clusters=3, papers_per_cluster=5)
    clusters = compute_clusters(ids, embs)

    # Should detect approximately 3 clusters
    assert 2 <= len(clusters) <= 4, f"Expected ~3 clusters, got {len(clusters)}"


def test_each_cluster_has_papers():
    """Every cluster should contain at least one paper."""
    ids, embs = _make_cluster_embeddings(n_clusters=3, papers_per_cluster=5)
    clusters = compute_clusters(ids, embs)

    for c in clusters:
        assert len(c.paper_ids) > 0, f"Cluster {c.cluster_idx} has no papers"


def test_medoid_is_real_paper():
    """Medoid paper_id must be one of the papers in the cluster."""
    ids, embs = _make_cluster_embeddings(n_clusters=2, papers_per_cluster=5)
    clusters = compute_clusters(ids, embs)

    for c in clusters:
        assert c.medoid_paper_id in c.paper_ids, \
            f"Medoid {c.medoid_paper_id} not in cluster paper_ids {c.paper_ids}"
        assert c.medoid_paper_id in ids, \
            f"Medoid {c.medoid_paper_id} not in original paper list"


def test_medoid_embedding_matches_paper():
    """Medoid embedding should be the actual vector for that paper."""
    ids, embs = _make_cluster_embeddings(n_clusters=2, papers_per_cluster=5)
    clusters = compute_clusters(ids, embs)

    for c in clusters:
        medoid_idx = ids.index(c.medoid_paper_id)
        assert np.allclose(c.medoid_embedding, embs[medoid_idx], atol=1e-6), \
            "Medoid embedding doesn't match the paper's actual embedding"


def test_importance_is_sorted_descending():
    """Clusters should be returned sorted by importance (highest first)."""
    ids, embs = _make_cluster_embeddings(n_clusters=3, papers_per_cluster=5)
    clusters = compute_clusters(ids, embs)

    for i in range(len(clusters) - 1):
        assert clusters[i].importance >= clusters[i + 1].importance, \
            f"Cluster {i} importance {clusters[i].importance} < {clusters[i+1].importance}"


def test_few_papers_returns_single_cluster():
    """When papers < MIN_PAPERS_FOR_CLUSTERING, return a single catch-all cluster."""
    ids = ["p1", "p2", "p3"]
    rng = np.random.RandomState(11)
    embs = rng.randn(3, 1024).astype(np.float32)
    # Normalise
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    clusters = compute_clusters(ids, embs)
    assert len(clusters) == 1, f"Expected 1 cluster for {len(ids)} papers, got {len(clusters)}"
    assert set(clusters[0].paper_ids) == set(ids), "Single cluster should contain all papers"


def test_all_papers_accounted_for():
    """Every input paper should appear in exactly one cluster."""
    ids, embs = _make_cluster_embeddings(n_clusters=3, papers_per_cluster=5)
    clusters = compute_clusters(ids, embs)

    all_clustered = []
    for c in clusters:
        all_clustered.extend(c.paper_ids)

    assert set(all_clustered) == set(ids), "Some papers missing from clusters"
    assert len(all_clustered) == len(ids), "Some papers appear in multiple clusters"


def test_max_clusters_enforced():
    """Even with many disparate groups, cluster count should not exceed MAX_CLUSTERS."""
    # Create 10 very distinct groups
    ids, embs = _make_cluster_embeddings(n_clusters=10, papers_per_cluster=3, spread=0.01)
    clusters = compute_clusters(ids, embs)

    assert len(clusters) <= MAX_CLUSTERS, \
        f"Expected <= {MAX_CLUSTERS} clusters, got {len(clusters)}"


def test_find_medoid():
    """_find_medoid should return the index closest to the centroid."""
    embeddings = np.array([
        [0.0, 1.0, 0.0],   # far from centroid
        [0.95, 0.05, 0.0],  # closest to centroid
        [0.5, 0.5, 0.0],   # medium distance
    ], dtype=np.float32)
    centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    idx = _find_medoid(embeddings, centroid)
    assert idx == 1, f"Expected medoid idx 1, got {idx}"


# ── DB persistence test ──────────────────────────────────────────────────────

@pytest.fixture
def setup_db(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test_cluster.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    asyncio.get_event_loop().run_until_complete(db_mod.init_db())
    yield


def test_cluster_db_roundtrip(setup_db):
    """Clusters survive a save → load round-trip to SQLite."""
    from app.recommend.clustering import save_clusters_to_db, load_clusters_from_db

    ids, embs = _make_cluster_embeddings(n_clusters=2, papers_per_cluster=5)
    clusters = compute_clusters(ids, embs)

    async def _run():
        await save_clusters_to_db("user-test", clusters)
        loaded = await load_clusters_from_db("user-test")
        assert loaded is not None
        assert len(loaded) == len(clusters)

        for orig, db_row in zip(clusters, loaded):
            assert db_row["medoid_paper_id"] == orig.medoid_paper_id
            assert abs(db_row["importance"] - orig.importance) < 1e-4

    asyncio.get_event_loop().run_until_complete(_run())
