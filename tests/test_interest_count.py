"""How many interests a reader has (audit F27; doc 06, 2026-11-13).

The largest gap between Ward merge heights split single-topic readers into fake
interests and merged a main field's side interests into one group. The cut with
the best silhouette, and one interest when no cut separates anything, fixes both.
"""
import numpy as np

from app.recommend.clustering import SILHOUETTE_MIN, compute_clusters


def _reader(shape, seed=0, spread=0.02):
    """`shape[i]` papers around topic i; same-topic cosine ~0.7, like real papers."""
    rng = np.random.default_rng(seed)
    embs, ids = [], []
    for t, n in enumerate(shape):
        centre = rng.normal(size=1024)
        centre /= np.linalg.norm(centre)
        for j in range(n):
            v = centre + spread * rng.normal(size=1024)
            embs.append((v / np.linalg.norm(v)).astype(np.float32))
            ids.append(f"t{t}.{j}")
    return ids, np.array(embs)


def test_one_topic_is_one_interest():
    for seed in range(5):
        ids, embs = _reader([20], seed)
        assert len(compute_clusters(ids, embs)) == 1


def test_a_main_field_keeps_its_side_interests():
    ids, embs = _reader([16, 3, 3, 2], seed=1)
    clusters = compute_clusters(ids, embs)
    assert len(clusters) == 4
    topics = [{p.split(".")[0] for p in c.paper_ids} for c in clusters]
    assert all(len(t) == 1 for t in topics)          # no interest mixes two topics


def test_random_saves_with_no_structure_are_one_interest():
    rng = np.random.default_rng(9)
    embs = rng.normal(size=(12, 1024)).astype(np.float32)
    assert len(compute_clusters([f"p{i}" for i in range(12)], embs)) == 1
    assert 0 < SILHOUETTE_MIN < 0.5
