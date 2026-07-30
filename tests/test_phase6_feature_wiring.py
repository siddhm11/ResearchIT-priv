"""
Phase 6 feature wiring tests.

Verifies that features 23-30 carry non-zero signal when Phase 6
arguments are passed to the reranker. Catches regressions to the
"9 of 37 features" bug described in PHASE6-Reranker-Framing.md.
"""
import numpy as np
import pytest

from app.recommend.reranker import (
    compute_features,
    rerank_candidates,
    FEATURE_NAMES,
    NUM_FEATURES,
    is_model_loaded,
    get_num_trees,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_candidates(n: int = 10):
    """Generate synthetic candidate data."""
    ids = [f"2401.{1000 + i:05d}" for i in range(n)]
    embs = np.random.randn(n, 1024).astype(np.float32)
    # Normalize to unit length (mimics real BGE-M3 embeddings)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    meta = [
        {
            "arxiv_id": ids[i],
            "published": f"2025-0{(i % 9) + 1}-15",
            "update_date": f"2025-0{(i % 9) + 1}-15",
            "citation_count": (i + 1) * 10,
            "influential_citations": i,
            "category": ["cs.LG", "cs.CL", "cs.CV", "stat.ML", "cs.AI"][i % 5],
            "primary_topic": ["cs.LG", "cs.CL", "cs.CV", "stat.ML", "cs.AI"][i % 5],
            "authors": "Author A, Author B",
            "title": f"Paper {i} about ML",
        }
        for i in range(n)
    ]
    return ids, embs, meta


def _make_user_profiles(dim=1024):
    """Generate synthetic EWMA profile vectors."""
    lt = np.random.randn(dim).astype(np.float32)
    lt /= np.linalg.norm(lt)
    st = np.random.randn(dim).astype(np.float32)
    st /= np.linalg.norm(st)
    neg = np.random.randn(dim).astype(np.float32)
    neg /= np.linalg.norm(neg)
    return lt, st, neg


# ── Tests ────────────────────────────────────────────────────────────────────

class TestPhase6FeatureWiring:
    """Test that features 23-30 are non-zero when Phase 6 args are provided."""

    def test_feature_schema_count(self):
        """Schema must have exactly 37 features."""
        assert NUM_FEATURES == 37
        assert len(FEATURE_NAMES) == 37

    def test_features_2330_nonzero_with_phase6_args(self):
        """
        Bug A regression guard: when Phase 6 args are passed,
        features 23-30 must be non-zero for at least some candidates.
        """
        ids, embs, meta = _make_candidates(10)
        lt, st, neg = _make_user_profiles()

        # Build per-candidate cluster data
        cluster_importance = np.array(
            [0.7, 0.7, 0.3, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3],
            dtype=np.float32,
        )
        medoid_a = np.random.randn(1024).astype(np.float32)
        medoid_a /= np.linalg.norm(medoid_a)
        medoid_b = np.random.randn(1024).astype(np.float32)
        medoid_b /= np.linalg.norm(medoid_b)
        # Per-candidate medoids (N, 1024)
        cluster_medoid = np.stack(
            [medoid_a if i % 2 == 0 else medoid_b for i in range(10)], axis=0
        )

        is_suppressed = np.array(
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32
        )
        onboarding_match = np.array(
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32
        )

        X = compute_features(
            embs, meta, lt, st, neg,
            qdrant_scores=np.linspace(0.9, 0.5, 10).astype(np.float32),
            cluster_importance=cluster_importance,
            cluster_medoid=cluster_medoid,
            is_suppressed_category=is_suppressed,
            onboarding_category_match=onboarding_match,
            user_total_saves=15,
            user_total_dismissals=3,
            user_days_since_last_save=0.5,
            user_session_save_count=2,
        )

        assert X.shape == (10, 37), f"Got shape {X.shape}"

        # Features 23-30 must each have at least one non-zero value
        for slot in range(23, 31):
            col = X[:, slot]
            assert np.any(col != 0), (
                f"Feature {slot} ({FEATURE_NAMES[slot]}) is all zeros — "
                f"Phase 6 wiring regression"
            )

    def test_per_candidate_cluster_importance(self):
        """Feature 23 must vary per-candidate, not be broadcast scalar."""
        ids, embs, meta = _make_candidates(5)
        lt, st, neg = _make_user_profiles()

        # Two different importance values
        cluster_importance = np.array([0.8, 0.8, 0.3, 0.3, 0.8], dtype=np.float32)

        X = compute_features(
            embs, meta, lt, st, neg,
            cluster_importance=cluster_importance,
        )

        # Slot 23 should have at least 2 distinct values
        unique_vals = set(X[:, 23].tolist())
        assert len(unique_vals) >= 2, (
            f"Feature 23 should vary per-candidate, got {unique_vals}"
        )

    def test_per_candidate_medoid_distance(self):
        """Feature 24 must differ when different medoids are provided."""
        # Seeded deliberately, and BEFORE _make_candidates, because the
        # candidate embeddings are drawn randomly too and the asserted gap
        # depends on both them and the medoid.
        #
        # In 1024 dimensions the cosine between two independent random vectors
        # concentrates near zero (sigma ~ 0.031), so the asserted gap of 0.01
        # only appears on a lucky sample -- measured at roughly 1 failure in 12
        # unseeded runs. That intermittent red was mistaken for a real
        # regression more than once.
        np.random.seed(20260730)
        ids, embs, meta = _make_candidates(4)

        medoid_a = np.random.randn(1024).astype(np.float32)
        medoid_b = -medoid_a  # opposite direction → very different distances

        # Per-candidate: first 2 get medoid_a, last 2 get medoid_b
        cluster_medoid = np.stack(
            [medoid_a, medoid_a, medoid_b, medoid_b], axis=0
        )

        X = compute_features(
            embs, meta, cluster_medoid=cluster_medoid,
        )

        # Distances to opposite medoids should be meaningfully different
        dist_a = X[:2, 24].mean()
        dist_b = X[2:, 24].mean()
        assert abs(dist_a - dist_b) > 0.01, (
            f"Per-candidate medoid distances should differ: A={dist_a:.4f}, B={dist_b:.4f}"
        )

    def test_broadcast_medoid_still_works(self):
        """Single 1D medoid (Phase 6.1 legacy) should still work."""
        ids, embs, meta = _make_candidates(5)

        medoid = np.random.randn(1024).astype(np.float32)
        medoid /= np.linalg.norm(medoid)

        X = compute_features(
            embs, meta, cluster_medoid=medoid,
        )

        # Slot 24 should be non-zero (distance to medoid)
        assert np.any(X[:, 24] != 0), "Broadcast medoid should produce non-zero distances"

    def test_backward_compat_no_phase6_args(self):
        """Old caller (no Phase 6 kwargs) must still work without errors."""
        ids, embs, meta = _make_candidates(5)
        lt, st, neg = _make_user_profiles()

        sorted_ids, sorted_scores, sorted_embs = rerank_candidates(
            candidate_ids=ids,
            candidate_embeddings=embs,
            candidate_metadata=meta,
            long_term_vec=lt,
            short_term_vec=st,
            negative_vec=neg,
        )

        assert len(sorted_ids) == 5
        assert len(sorted_scores) == 5
        assert sorted_embs.shape == (5, 1024)

    def test_full_phase6_rerank_call(self):
        """Full Phase 6 call with all kwargs must produce valid output."""
        ids, embs, meta = _make_candidates(8)
        lt, st, neg = _make_user_profiles()

        sorted_ids, sorted_scores, sorted_embs = rerank_candidates(
            candidate_ids=ids,
            candidate_embeddings=embs,
            candidate_metadata=meta,
            long_term_vec=lt,
            short_term_vec=st,
            negative_vec=neg,
            qdrant_scores=np.linspace(0.9, 0.5, 8).astype(np.float32),
            cluster_importance=np.full(8, 0.6, dtype=np.float32),
            cluster_medoid=np.stack(
                [np.random.randn(1024).astype(np.float32) for _ in range(8)]
            ),
            is_suppressed_category=np.zeros(8, dtype=np.float32),
            onboarding_category_match=np.ones(8, dtype=np.float32),
            user_total_saves=20,
            user_total_dismissals=5,
        )

        assert len(sorted_ids) == 8
        assert len(sorted_scores) == 8
        assert sorted_embs.shape == (8, 1024)
        # Scores should be sorted descending
        for i in range(len(sorted_scores) - 1):
            assert sorted_scores[i] >= sorted_scores[i + 1], (
                f"Scores not sorted: {sorted_scores[i]} < {sorted_scores[i+1]}"
            )

    def test_model_accessors(self):
        """Phase 6.3 model accessors must return valid data."""
        # These should not crash regardless of model availability
        loaded = is_model_loaded()
        assert isinstance(loaded, bool)

        trees = get_num_trees()
        assert isinstance(trees, int)
        assert trees >= 0

    def test_aggregate_feature_activation(self):
        """At least 60% of feature slots should be active with full args."""
        ids, embs, meta = _make_candidates(10)
        lt, st, neg = _make_user_profiles()

        X = compute_features(
            embs, meta, lt, st, neg,
            qdrant_scores=np.linspace(0.9, 0.5, 10).astype(np.float32),
            cluster_importance=np.full(10, 0.5, dtype=np.float32),
            cluster_medoid=np.stack(
                [np.random.randn(1024).astype(np.float32) for _ in range(10)]
            ),
            is_suppressed_category=np.array([0,0,1,0,0,1,0,0,0,0], dtype=np.float32),
            onboarding_category_match=np.array([1,0,1,0,1,0,1,0,1,0], dtype=np.float32),
            user_total_saves=12,
            user_total_dismissals=3,
            user_days_since_last_save=2.0,
            user_session_save_count=1,
        )

        nonzero_rate = (X != 0).mean(axis=0)
        active_pct = (nonzero_rate > 0).mean()
        assert active_pct >= 0.6, (
            f"Only {active_pct*100:.0f}% of features active — expected ≥60%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
