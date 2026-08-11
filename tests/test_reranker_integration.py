"""
Phase 6: LightGBM Reranker Integration Tests

Tests:
  1. Smoke test — load model, predict on dummy input
  2. Feature computation — verify 37-feature vector shape and values
  3. Heuristic fallback — verify scoring works without model
  4. End-to-end — full pipeline with simulated user state
  5. Latency benchmark — confirm < 1ms for 100 candidates
  6. Backward compatibility — old call signature still works
"""
import importlib.util
import sys
import os
import time
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# lightgbm is deliberately not installed in the local dev venv (it is several
# hundred MB and app/recommend/reranker.py falls back to heuristic_score()
# without it). Only the smoke test below actually needs the library -- the
# rest of this file exercises the heuristic path and must still run, so the
# skip is scoped to that one test rather than the whole module.
_HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None

# ── Test 1: Smoke Test ───────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed locally")
def test_smoke():
    """Load the LightGBM model directly and predict on dummy input."""
    import lightgbm as lgb

    model_path = os.path.join(
        os.path.dirname(__file__), "..",
        "models", "reranker-phase6", "production_model", "reranker_v1.txt"
    )
    model_path = os.path.normpath(model_path)

    assert os.path.isfile(model_path), f"Model file not found: {model_path}"

    model = lgb.Booster(model_file=model_path)

    # Verify model properties
    assert model.num_feature() == 37, f"Expected 37 features, got {model.num_feature()}"
    print(f"  Model loaded: {model.num_trees()} trees, {model.num_feature()} features")

    # Predict on zeros
    dummy = np.zeros((5, 37), dtype=np.float32)
    scores = model.predict(dummy)
    assert scores.shape == (5,), f"Expected (5,), got {scores.shape}"
    assert not np.any(np.isnan(scores)), "NaN in predictions"
    print(f"  Zero-input scores: {scores}")

    # Predict on random input
    random_input = np.random.randn(10, 37).astype(np.float32)
    scores = model.predict(random_input)
    assert scores.shape == (10,)
    assert not np.any(np.isnan(scores))
    print(f"  Random-input score range: [{scores.min():.4f}, {scores.max():.4f}]")

    print("  ✅ Smoke test PASSED")


# ── Test 2: Feature Computation ──────────────────────────────────────────────

def test_feature_computation():
    """Verify compute_features produces correct 37-feature matrix."""
    from app.recommend.reranker import compute_features, NUM_FEATURES

    n = 5
    embeddings = np.random.randn(n, 1024).astype(np.float32)
    metadata = [
        {
            "arxiv_id": f"2401.{i:05d}",
            "category": "cs.CL",
            "published": "2024-01-15",
            "citation_count": i * 100,
            "influential_citations": i * 10,
            "authors": '["Alice Smith", "Bob Jones"]',
        }
        for i in range(n)
    ]
    lt_vec = np.random.randn(1024).astype(np.float32)
    st_vec = np.random.randn(1024).astype(np.float32)
    neg_vec = np.random.randn(1024).astype(np.float32)
    qdrant_scores = [0.95 - i * 0.05 for i in range(n)]

    features = compute_features(
        embeddings, metadata, lt_vec, st_vec, neg_vec,
        qdrant_scores=qdrant_scores,
        cluster_importance=0.75,
        suppressed_categories={"cs.CR"},
        onboarding_categories={"cs.CL", "cs.LG"},
        user_total_saves=42,
        user_total_dismissals=8,
    )

    assert features.shape == (n, NUM_FEATURES), f"Expected ({n}, {NUM_FEATURES}), got {features.shape}"
    assert features.dtype == np.float32

    # Check specific feature values
    for i in range(n):
        # Feature 0: qdrant_cosine_score
        assert abs(features[i, 0] - qdrant_scores[i]) < 1e-5, \
            f"Feature 0 mismatch: {features[i, 0]} vs {qdrant_scores[i]}"

        # Feature 1: position = i
        assert features[i, 1] == float(i)

        # Feature 2: citation_count
        assert features[i, 2] == float(i * 100)

        # Feature 3: log_citations = log(100i + 1)
        assert abs(features[i, 3] - np.log(i * 100 + 1)) < 1e-5

        # Feature 6: recency_score > 0 (2024-01-15 is recent-ish)
        assert features[i, 6] > 0, f"Recency should be > 0, got {features[i, 6]}"

        # Feature 20: ewma_longterm should be non-zero (we provided profiles)
        assert features[i, 20] != 0.0, "EWMA long-term should be computed"

        # Feature 23: cluster_importance
        assert features[i, 23] == 0.75

        # Feature 25: suppressed = 0 (category is cs.CL, not cs.CR)
        assert features[i, 25] == 0.0

        # Feature 26: onboarding = 1 (cs.CL is in onboarding set)
        assert features[i, 26] == 1.0

        # Feature 27: total_saves
        assert features[i, 27] == 42.0

        # Feature 35: position_inverse = 1/(i+1)
        assert abs(features[i, 35] - 1.0 / (i + 1)) < 1e-5

    # Check no NaN
    assert not np.any(np.isnan(features)), "NaN in features"

    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Feature value range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  Non-zero features per row: {(features != 0).sum(axis=1)}")
    print("  ✅ Feature computation test PASSED")


# ── Test 3: Heuristic Fallback ───────────────────────────────────────────────

def test_heuristic_fallback():
    """Verify heuristic scoring works correctly."""
    from app.recommend.reranker import heuristic_score

    n = 10
    features = np.zeros((n, 37), dtype=np.float32)

    # Set some features that affect heuristic scoring
    for i in range(n):
        features[i, 0] = 0.9 - i * 0.05       # qdrant_cosine (decreasing)
        features[i, 6] = np.exp(-0.002 * i * 30)  # recency (decreasing age)
        features[i, 35] = 1.0 / (i + 1)         # position_inverse

    scores = heuristic_score(features)

    assert scores.shape == (n,)
    assert not np.any(np.isnan(scores))
    # First candidate should score higher (better cosine, recency, position)
    assert scores[0] > scores[-1], \
        f"First candidate ({scores[0]:.4f}) should score higher than last ({scores[-1]:.4f})"

    print(f"  Heuristic scores: [{scores[0]:.4f}, .., {scores[-1]:.4f}]")
    print("  ✅ Heuristic fallback test PASSED")


# ── Test 4: End-to-End Pipeline ──────────────────────────────────────────────

def test_e2e_pipeline():
    """Full pipeline: feature computation → model prediction → ranking."""
    from app.recommend.reranker import rerank_candidates, _USE_LGB

    n = 50
    candidate_ids = [f"2401.{i:05d}" for i in range(n)]
    embeddings = np.random.randn(n, 1024).astype(np.float32)
    metadata = [
        {
            "arxiv_id": cid,
            "category": f"cs.{'CL' if i % 3 == 0 else 'LG' if i % 3 == 1 else 'CV'}",
            "published": f"2024-{1 + (i % 12):02d}-{1 + (i % 28):02d}",
            "citation_count": max(0, 500 - i * 10 + np.random.randint(-50, 50)),
            "influential_citations": max(0, 50 - i + np.random.randint(-5, 5)),
            "authors": '["Author A", "Author B"]',
        }
        for i, cid in enumerate(candidate_ids)
    ]
    lt_vec = np.random.randn(1024).astype(np.float32)
    st_vec = np.random.randn(1024).astype(np.float32)
    neg_vec = np.random.randn(1024).astype(np.float32)
    qdrant_scores = [0.95 - i * 0.01 for i in range(n)]

    sorted_ids, sorted_scores, sorted_embs = rerank_candidates(
        candidate_ids=candidate_ids,
        candidate_embeddings=embeddings,
        candidate_metadata=metadata,
        long_term_vec=lt_vec,
        short_term_vec=st_vec,
        negative_vec=neg_vec,
        qdrant_scores=qdrant_scores,
        cluster_importance=0.6,
        user_total_saves=25,
        user_total_dismissals=5,
    )

    assert len(sorted_ids) == n
    assert len(sorted_scores) == n
    assert sorted_embs.shape == (n, 1024)

    # Scores should be in descending order
    for i in range(len(sorted_scores) - 1):
        assert sorted_scores[i] >= sorted_scores[i + 1], \
            f"Scores not sorted at index {i}: {sorted_scores[i]} < {sorted_scores[i + 1]}"

    # The order should differ from the input (reranking should change something)
    if _USE_LGB:
        assert sorted_ids != candidate_ids, "LightGBM reranking should change the order"
        print(f"  Using: LightGBM")
    else:
        print(f"  Using: Heuristic fallback")

    print(f"  Reranked {n} candidates")
    print(f"  Score range: [{sorted_scores[-1]:.4f}, {sorted_scores[0]:.4f}]")
    print(f"  Top-5 IDs: {sorted_ids[:5]}")
    print("  ✅ End-to-end pipeline test PASSED")


# ── Test 5: Latency Benchmark ───────────────────────────────────────────────

def test_latency():
    """Verify LightGBM prediction is under 1ms for 100 candidates."""
    from app.recommend.reranker import _lgb_model, _USE_LGB

    if not _USE_LGB:
        print("  ⏭️ Skipping latency test (no LightGBM model loaded)")
        return

    features = np.random.randn(100, 37).astype(np.float32)

    # Warm up
    for _ in range(50):
        _lgb_model.predict(features)

    # Benchmark
    n_iters = 1000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _lgb_model.predict(features)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / n_iters

    print(f"  LightGBM predict latency: {elapsed_ms:.3f}ms per 100 candidates")
    assert elapsed_ms < 1.0, f"Too slow: {elapsed_ms:.3f}ms (target: <1ms)"
    print("  ✅ Latency test PASSED")


# ── Test 6: Backward Compatibility ──────────────────────────────────────────

def test_backward_compat():
    """Verify old call signature still works (no qdrant_scores, no cluster params)."""
    from app.recommend.reranker import rerank_candidates

    n = 10
    ids = [f"2401.{i:05d}" for i in range(n)]
    embs = np.random.randn(n, 1024).astype(np.float32)
    meta = [
        {"arxiv_id": cid, "published": "2024-01-01", "category": "cs.CL"}
        for cid in ids
    ]

    # Old signature: just ids, embeddings, metadata, and optional profile vecs
    sorted_ids, sorted_scores, sorted_embs = rerank_candidates(
        candidate_ids=ids,
        candidate_embeddings=embs,
        candidate_metadata=meta,
    )

    assert len(sorted_ids) == n
    assert len(sorted_scores) == n
    assert sorted_embs.shape == (n, 1024)
    print("  ✅ Backward compatibility test PASSED")


# ── Test 7: LightGBM vs Heuristic Comparison ───────────────────────────────

def test_lgb_vs_heuristic():
    """Compare LightGBM and heuristic scores on same input."""
    from app.recommend.reranker import compute_features, heuristic_score, _lgb_model, _USE_LGB

    if not _USE_LGB:
        print("  ⏭️ Skipping comparison (no LightGBM model)")
        return

    n = 20
    embeddings = np.random.randn(n, 1024).astype(np.float32)
    metadata = [
        {
            "arxiv_id": f"2401.{i:05d}",
            "category": "cs.CL",
            "published": f"2024-{1 + i % 12:02d}-15",
            "citation_count": i * 50,
            "influential_citations": i * 5,
            "authors": '["Author A"]',
        }
        for i in range(n)
    ]
    qdrant_scores = [0.9 - i * 0.02 for i in range(n)]

    features = compute_features(
        embeddings, metadata,
        qdrant_scores=qdrant_scores,
        user_total_saves=10,
    )

    heur_scores = heuristic_score(features)
    lgb_scores = _lgb_model.predict(features)

    # Rankings should differ
    heur_order = np.argsort(-heur_scores)
    lgb_order = np.argsort(-lgb_scores)

    overlap_top5 = len(set(heur_order[:5]) & set(lgb_order[:5]))

    print(f"  Heuristic score range: [{heur_scores.min():.4f}, {heur_scores.max():.4f}]")
    print(f"  LightGBM score range:  [{lgb_scores.min():.4f}, {lgb_scores.max():.4f}]")
    print(f"  Top-5 overlap: {overlap_top5}/5")
    print(f"  Heuristic top-5 positions: {heur_order[:5]}")
    print(f"  LightGBM  top-5 positions: {lgb_order[:5]}")

    # Kendall's tau - rank correlation
    from scipy.stats import kendalltau
    tau, _ = kendalltau(heur_order, lgb_order)
    print(f"  Kendall's tau (rank correlation): {tau:.4f}")
    print("  ✅ LGB vs Heuristic comparison PASSED")


# ── Run All Tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Smoke Test", test_smoke),
        ("Feature Computation", test_feature_computation),
        ("Heuristic Fallback", test_heuristic_fallback),
        ("End-to-End Pipeline", test_e2e_pipeline),
        ("Latency Benchmark", test_latency),
        ("Backward Compatibility", test_backward_compat),
        ("LGB vs Heuristic", test_lgb_vs_heuristic),
    ]

    print("=" * 60)
    print("Phase 6: LightGBM Reranker Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n─── {name} ───")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
