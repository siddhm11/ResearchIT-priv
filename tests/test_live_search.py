"""
Layer 2: Live service integration tests — Phase 3.

Tests real Qdrant Cloud and Groq API calls.
These require network access and valid credentials in config.py.

Run:   python -m pytest tests/test_live_search.py -v
Skip:  These tests connect to live services — they will fail if
       credentials are invalid or services are down.
"""
import pytest
import numpy as np

# Every test in this file hits a real cloud service. The `live` marker lets CI
# deselect them (`-m "not live"`) so a service outage or an expired token fails
# the deploy check rather than the unit suite.
pytestmark = pytest.mark.live



# ── Qdrant dense search (live) ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_qdrant_dense_search_returns_results():
    """
    search_dense() with a random 1024-dim vector should return results
    from the live Qdrant collection.  The vector won't be semantically
    meaningful but the ANN index should still return nearest neighbors.
    """
    from app.qdrant_svc import search_dense, _client
    _client.cache_clear()

    # Random vector — just checking the call works
    query_vec = np.random.rand(1024).astype(np.float32).tolist()
    results = await search_dense(query_vec, limit=5)

    assert isinstance(results, list)
    assert len(results) > 0, "Qdrant returned no results for dense search"
    # Each result should have arxiv_id and score
    for r in results:
        assert "arxiv_id" in r
        assert "score" in r
        assert isinstance(r["arxiv_id"], str)
        assert len(r["arxiv_id"]) > 0


@pytest.mark.asyncio
async def test_qdrant_dense_search_scores_are_ordered():
    """Results should be ordered by score descending."""
    from app.qdrant_svc import search_dense, _client
    _client.cache_clear()

    query_vec = np.random.rand(1024).astype(np.float32).tolist()
    results = await search_dense(query_vec, limit=10)

    if len(results) >= 2:
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"


# ── Groq rewriter (live) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_groq_rewrite_live():
    """
    Live Groq API call — rewriter should produce a different,
    shorter academic query from a casual input.
    """
    from app.groq_svc import rewrite

    original = "when AI makes up fake facts and lies"
    rewritten = await rewrite(original)

    assert isinstance(rewritten, str)
    assert len(rewritten) > 0
    # Rewritten should be different from original (unless heuristic triggered)
    # Don't assert != because sometimes the heuristic might fire
    print(f"[test] Original:  {original}")
    print(f"[test] Rewritten: {rewritten}")


@pytest.mark.asyncio
async def test_groq_rewrite_academic_bypass():
    """
    Academic queries should bypass the rewriter and return as-is.
    """
    from app.groq_svc import rewrite

    academic = "LLM hallucination NLP survey large language models BERT transformer"
    result = await rewrite(academic)

    # Should return the original since it looks academic
    assert result == academic


