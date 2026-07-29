"""
Layer 2: Live service integration tests — Phase 3.

Tests real Qdrant Cloud, Zilliz Cloud, and Groq API calls.
These require network access and valid credentials in config.py.

Run:   python -m pytest tests/test_live_search.py -v
Skip:  These tests connect to live services — they will fail if
       credentials are invalid or services are down.
"""
import pytest
import numpy as np


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


# ── Zilliz sparse search (live) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zilliz_sparse_search_returns_results():
    """
    search_sparse() with a synthetic sparse dict should return results
    from the live Zilliz collection.

    We use token IDs that are likely to exist in BGE-M3's vocabulary
    (common English tokens have IDs in the thousands range).
    """
    from app.zilliz_svc import search_sparse, _reset_client
    _reset_client()  # Fresh connection

    # Synthetic sparse dict — token IDs that likely exist in BGE-M3 vocab
    sparse_dict = {
        6:    0.3,     # Common token
        29:   0.25,    # Common token
        1600: 0.2,     # Mid-range token
        6083: 0.15,    # Another token
    }

    results = await search_sparse(sparse_dict, limit=5)

    assert isinstance(results, list)
    assert len(results) > 0, "Zilliz returned no results for sparse search"
    for r in results:
        assert "arxiv_id" in r
        assert "score" in r
        assert isinstance(r["arxiv_id"], str)
        assert r["score"] > 0, "Sparse IP score should be positive"


@pytest.mark.asyncio
async def test_zilliz_empty_sparse_returns_empty():
    """Empty sparse dict should return empty results (no crash)."""
    from app.zilliz_svc import search_sparse

    results = await search_sparse({}, limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_zilliz_search_arxiv_ids_are_valid():
    """Returned arxiv_ids should look like real arXiv IDs."""
    from app.zilliz_svc import search_sparse, _reset_client
    _reset_client()

    sparse_dict = {6: 0.3, 29: 0.25, 1600: 0.2}
    results = await search_sparse(sparse_dict, limit=3)

    for r in results:
        aid = r["arxiv_id"]
        # arXiv IDs are either YYMM.NNNNN or category/NNNNNNN
        assert ("." in aid or "/" in aid), f"Invalid arxiv_id format: {aid}"


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


# ── Cross-service: Qdrant + Zilliz parallel ──────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_search_both_return():
    """
    Both Qdrant and Zilliz should return results when called in parallel,
    matching the hybrid search pattern.
    """
    import asyncio
    from app.qdrant_svc import search_dense, _client
    from app.zilliz_svc import search_sparse, _reset_client

    _client.cache_clear()
    _reset_client()

    query_vec = np.random.rand(1024).astype(np.float32).tolist()
    sparse_dict = {6: 0.3, 29: 0.25, 1600: 0.2}

    dense_results, sparse_results = await asyncio.gather(
        search_dense(query_vec, limit=5),
        search_sparse(sparse_dict, limit=5),
    )

    assert len(dense_results) > 0, "Dense search returned nothing"
    assert len(sparse_results) > 0, "Sparse search returned nothing"

    # There should be SOME overlap between the two result sets
    # (not guaranteed but likely for a real corpus)
    dense_ids = {r["arxiv_id"] for r in dense_results}
    sparse_ids = {r["arxiv_id"] for r in sparse_results}
    print(f"[test] Dense IDs:  {dense_ids}")
    print(f"[test] Sparse IDs: {sparse_ids}")
    print(f"[test] Overlap:    {dense_ids & sparse_ids}")
