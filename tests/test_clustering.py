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
    stabilize_cluster_ids,
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
    assert len(ids) < MIN_PAPERS_FOR_CLUSTERING, "test precondition: ids must be below threshold"
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


# ── Hungarian matching / cluster ID stabilisation (Phase 4.2) ────────────────

def _make_two_cluster_pair(seed: int = 0) -> tuple[list, list]:
    """
    Build two well-separated InterestCluster lists sharing the same embedding
    space so Hungarian matching can correctly align them.

    Returns (new_clusters, old_clusters) where new_clusters[0] corresponds
    semantically to old_clusters[0].
    """
    rng = np.random.RandomState(seed)
    dim = 1024

    # Two distinct topic centers
    center_a = rng.randn(dim).astype(np.float32)
    center_a /= np.linalg.norm(center_a)
    center_b = rng.randn(dim).astype(np.float32)
    center_b /= np.linalg.norm(center_b)

    def _near(center, n=5, spread=0.001):
        # NOTE: spread is scaled small because random noise in 1024-d has
        # magnitude ~sqrt(dim)*spread, so spread=0.05 gives noise≈1.6 which
        # dominates the unit-length center. 0.001 keeps cosine sim ≥ 0.99.
        vecs = []
        for _ in range(n):
            v = center + rng.randn(dim).astype(np.float32) * spread
            v /= np.linalg.norm(v)
            vecs.append(v)
        return vecs

    medoid_a_new = _near(center_a)[0]
    medoid_b_new = _near(center_b)[0]
    medoid_a_old = _near(center_a)[0]
    medoid_b_old = _near(center_b)[0]

    old = [
        InterestCluster(cluster_idx=0, medoid_paper_id="old_a", medoid_embedding=medoid_a_old,
                        paper_ids=["old_a"], importance=5.0),
        InterestCluster(cluster_idx=1, medoid_paper_id="old_b", medoid_embedding=medoid_b_old,
                        paper_ids=["old_b"], importance=3.0),
    ]
    # new clusters have swapped order (b first, a second) → naive assignment would shuffle
    new = [
        InterestCluster(cluster_idx=0, medoid_paper_id="new_b", medoid_embedding=medoid_b_new,
                        paper_ids=["new_b"], importance=3.0),
        InterestCluster(cluster_idx=1, medoid_paper_id="new_a", medoid_embedding=medoid_a_new,
                        paper_ids=["new_a"], importance=5.0),
    ]
    return new, old


def test_stabilize_matches_semantically_equivalent_clusters():
    """
    When topic A was cluster 0 and remains cluster 0 after recluster (just
    re-ordered by importance), stabilise_cluster_ids should restore idx=0 for A.
    """
    new, old = _make_two_cluster_pair()
    # new[0] is topic B, new[1] is topic A
    # old[0] is topic A (idx=0), old[1] is topic B (idx=1)
    stabilised = stabilize_cluster_ids(new, old)

    # After stabilisation, the cluster containing "new_a" should have idx=0
    # and "new_b" should have idx=1
    idx_map = {c.medoid_paper_id: c.cluster_idx for c in stabilised}
    assert idx_map["new_a"] == 0, f"Topic A should be idx 0, got {idx_map}"
    assert idx_map["new_b"] == 1, f"Topic B should be idx 1, got {idx_map}"


def test_stabilize_preserves_all_clusters():
    """Output length must equal input length."""
    new, old = _make_two_cluster_pair()
    stabilised = stabilize_cluster_ids(new, old)
    assert len(stabilised) == len(new)


def test_stabilize_unique_indices():
    """All cluster indices in the output must be unique."""
    new, old = _make_two_cluster_pair()
    stabilised = stabilize_cluster_ids(new, old)
    indices = [c.cluster_idx for c in stabilised]
    assert len(indices) == len(set(indices)), f"Duplicate indices: {indices}"


def test_stabilize_no_old_clusters_returns_unchanged():
    """With no old clusters, return new clusters as-is."""
    new, _ = _make_two_cluster_pair()
    result = stabilize_cluster_ids(new, [])
    assert result == new


def test_stabilize_no_new_clusters_returns_empty():
    """With no new clusters, return empty list."""
    _, old = _make_two_cluster_pair()
    result = stabilize_cluster_ids([], old)
    assert result == []


def test_stabilize_rejects_unrelated_match():
    """
    Doc 06 requirement: Hungarian must NOT inherit an old cluster's identity
    when the cosine similarity is below the threshold (default 0.5).  A user's
    genuinely-new topic should get a fresh index, not steal an old NLP idx
    just because Hungarian found the "least bad" assignment.
    """
    rng = np.random.RandomState(7)
    dim = 1024

    def _rand_unit():
        v = rng.randn(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    # Two very different topics: old_topic_vec vs new_topic_vec (orthogonal-ish)
    old_vec = _rand_unit()
    new_vec = _rand_unit()
    # Force near-orthogonality so cosine sim << 0.5
    # (random 1024-dim unit vectors already average near 0, so this should hold)
    cos_sim = float(new_vec @ old_vec)
    assert abs(cos_sim) < 0.3, f"test precondition failed: cos_sim={cos_sim}"

    old = [InterestCluster(cluster_idx=5, medoid_paper_id="old_topic",
                           medoid_embedding=old_vec, paper_ids=[], importance=1.0)]
    new = [InterestCluster(cluster_idx=0, medoid_paper_id="new_topic",
                           medoid_embedding=new_vec, paper_ids=[], importance=1.0)]

    stabilised = stabilize_cluster_ids(new, old)
    # The unrelated new cluster must NOT inherit idx=5
    assert stabilised[0].cluster_idx != 5, \
        "Unrelated topic inherited old cluster's index (threshold not enforced)"


def test_stabilize_custom_threshold():
    """Custom min_cosine_sim should control matching strictness."""
    rng = np.random.RandomState(13)
    dim = 1024
    base = rng.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    # Slightly perturbed — spread=0.001 in 1024-d gives cos_sim ~ 0.9995
    perturbed = base + rng.randn(dim).astype(np.float32) * 0.001
    perturbed /= np.linalg.norm(perturbed)

    old = [InterestCluster(cluster_idx=2, medoid_paper_id="old",
                           medoid_embedding=base, paper_ids=[], importance=1.0)]
    new = [InterestCluster(cluster_idx=0, medoid_paper_id="new",
                           medoid_embedding=perturbed, paper_ids=[], importance=1.0)]

    # With default threshold 0.5, match succeeds (~0.9995 cos sim)
    default_result = stabilize_cluster_ids(new, old)
    assert default_result[0].cluster_idx == 2

    # With threshold 0.99999 (stricter than actual 0.9995 sim), match rejected
    strict_result = stabilize_cluster_ids(new, old, min_cosine_sim=0.99999)
    assert strict_result[0].cluster_idx != 2


def test_stabilize_more_new_than_old():
    """K grew from 1 → 2: matched cluster keeps idx, new gets fresh idx."""
    rng = np.random.RandomState(21)
    dim = 1024

    base = rng.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    close = base + rng.randn(dim).astype(np.float32) * 0.001
    close /= np.linalg.norm(close)
    far = rng.randn(dim).astype(np.float32)
    far /= np.linalg.norm(far)

    old = [InterestCluster(cluster_idx=0, medoid_paper_id="o",
                           medoid_embedding=base, paper_ids=[], importance=1.0)]
    new = [
        InterestCluster(cluster_idx=0, medoid_paper_id="n1",
                        medoid_embedding=close, paper_ids=[], importance=2.0),
        InterestCluster(cluster_idx=1, medoid_paper_id="n2",
                        medoid_embedding=far, paper_ids=[], importance=1.0),
    ]
    result = stabilize_cluster_ids(new, old)
    idx_map = {c.medoid_paper_id: c.cluster_idx for c in result}
    assert idx_map["n1"] == 0  # inherits old idx
    assert idx_map["n2"] != 0  # fresh idx


def test_stabilize_fewer_new_than_old():
    """K shrank from 2 → 1: the surviving cluster keeps its idx."""
    rng = np.random.RandomState(25)
    dim = 1024
    base = rng.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    other = rng.randn(dim).astype(np.float32)
    other /= np.linalg.norm(other)
    close = base + rng.randn(dim).astype(np.float32) * 0.001
    close /= np.linalg.norm(close)

    old = [
        InterestCluster(cluster_idx=7, medoid_paper_id="oA",
                        medoid_embedding=base, paper_ids=[], importance=2.0),
        InterestCluster(cluster_idx=9, medoid_paper_id="oB",
                        medoid_embedding=other, paper_ids=[], importance=1.0),
    ]
    new = [InterestCluster(cluster_idx=0, medoid_paper_id="nA",
                           medoid_embedding=close, paper_ids=[], importance=1.0)]

    result = stabilize_cluster_ids(new, old)
    assert len(result) == 1
    assert result[0].cluster_idx == 7  # inherits the matching old idx


def test_stabilize_new_cluster_gets_fresh_index():
    """
    If new_clusters has more clusters than old, the extras get fresh indices
    not conflicting with any matched index.
    """
    rng = np.random.RandomState(99)
    dim = 1024

    emb = lambda: (lambda v: v / np.linalg.norm(v))(rng.randn(dim).astype(np.float32))

    old = [
        InterestCluster(cluster_idx=0, medoid_paper_id="old_a", medoid_embedding=emb(),
                        paper_ids=[], importance=1.0),
    ]
    new = [
        InterestCluster(cluster_idx=0, medoid_paper_id="new_a", medoid_embedding=old[0].medoid_embedding.copy(),
                        paper_ids=[], importance=1.0),
        InterestCluster(cluster_idx=1, medoid_paper_id="new_brand", medoid_embedding=emb(),
                        paper_ids=[], importance=1.0),
    ]
    stabilised = stabilize_cluster_ids(new, old)
    indices = {c.medoid_paper_id: c.cluster_idx for c in stabilised}
    assert indices["new_a"] == 0, "Matched cluster should inherit old index 0"
    assert indices["new_brand"] != 0, "New unmatched cluster must not collide with idx 0"


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
