"""
Tests for hybrid search pipeline — Phase 3.

Tests RRF fusion and recency reranking logic (pure Python, no live services).
Live integration tests are separate (require BGE-M3 + Qdrant + Zilliz).
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime


# ── RRF fusion tests ─────────────────────────────────────────────────────────

class TestRRFFusion:
    """Test the RRF fusion logic in hybrid_search_svc."""

    def test_rrf_basic_merge(self):
        """Papers appearing in both lists get higher scores."""
        from app.hybrid_search_svc import _rrf_fuse

        dense = [
            {"arxiv_id": "2301.00001", "score": 0.95},
            {"arxiv_id": "2301.00002", "score": 0.90},
            {"arxiv_id": "2301.00003", "score": 0.85},
        ]
        sparse = [
            {"arxiv_id": "2301.00002", "score": 0.80},
            {"arxiv_id": "2301.00004", "score": 0.75},
            {"arxiv_id": "2301.00001", "score": 0.70},
        ]

        fused = _rrf_fuse(dense, sparse, k=60)

        # Papers in both lists should rank highest
        arxiv_ids = [f["arxiv_id"] for f in fused]
        # 2301.00001 is rank 1 in dense + rank 3 in sparse
        # 2301.00002 is rank 2 in dense + rank 1 in sparse
        # Both should be in top 2
        assert "2301.00001" in arxiv_ids[:2]
        assert "2301.00002" in arxiv_ids[:2]

        # All 4 unique papers should appear
        assert len(fused) == 4

    def test_rrf_single_source(self):
        """Works with only one source providing results."""
        from app.hybrid_search_svc import _rrf_fuse

        dense = [
            {"arxiv_id": "2301.00001", "score": 0.95},
            {"arxiv_id": "2301.00002", "score": 0.90},
        ]
        sparse = []

        fused = _rrf_fuse(dense, sparse, k=60)
        assert len(fused) == 2
        assert fused[0]["arxiv_id"] == "2301.00001"

    def test_rrf_empty_both(self):
        """Empty inputs produce empty output."""
        from app.hybrid_search_svc import _rrf_fuse

        fused = _rrf_fuse([], [], k=60)
        assert fused == []

    def test_rrf_scores_are_rank_based(self):
        """RRF scores depend on rank, not on raw scores."""
        from app.hybrid_search_svc import _rrf_fuse

        # Same papers, different raw scores — RRF should produce identical results
        dense_a = [
            {"arxiv_id": "A", "score": 0.99},
            {"arxiv_id": "B", "score": 0.50},
        ]
        dense_b = [
            {"arxiv_id": "A", "score": 0.51},
            {"arxiv_id": "B", "score": 0.50},
        ]

        fused_a = _rrf_fuse(dense_a, [], k=60)
        fused_b = _rrf_fuse(dense_b, [], k=60)

        # Same ranking → same RRF scores
        assert fused_a[0]["rrf_score"] == fused_b[0]["rrf_score"]
        assert fused_a[1]["rrf_score"] == fused_b[1]["rrf_score"]

    def test_rrf_k_parameter(self):
        """Higher K dampens rank differences."""
        from app.hybrid_search_svc import _rrf_fuse

        dense = [
            {"arxiv_id": "A", "score": 0.9},
            {"arxiv_id": "B", "score": 0.8},
        ]

        fused_k10 = _rrf_fuse(dense, [], k=10)
        fused_k100 = _rrf_fuse(dense, [], k=100)

        # Score gap should be smaller with larger K
        gap_k10 = fused_k10[0]["rrf_score"] - fused_k10[1]["rrf_score"]
        gap_k100 = fused_k100[0]["rrf_score"] - fused_k100[1]["rrf_score"]
        assert gap_k10 > gap_k100


# ── Recency rerank tests ─────────────────────────────────────────────────────

class TestRecencyRerank:
    """Test recency boosting in hybrid_search_svc."""

    def test_recency_boost_newer_papers(self):
        """Newer papers should get higher recency scores."""
        from app.hybrid_search_svc import _recency_rerank

        # Two papers with same RRF score but different ages
        fused = [
            {"arxiv_id": "2401.00001", "rrf_score": 0.5},  # Jan 2024
            {"arxiv_id": "1501.00001", "rrf_score": 0.5},  # Jan 2015
        ]

        ranked = _recency_rerank(fused)

        # Newer paper (2401) should rank higher
        assert ranked[0]["arxiv_id"] == "2401.00001"

    def test_recency_preserves_strong_rrf(self):
        """A much higher RRF score should still dominate over recency."""
        from app.hybrid_search_svc import _recency_rerank

        fused = [
            {"arxiv_id": "1501.00001", "rrf_score": 1.0},   # Old but high RRF
            {"arxiv_id": "2401.00001", "rrf_score": 0.01},   # New but low RRF
        ]

        ranked = _recency_rerank(fused)

        # High RRF should still win (0.80 weight vs 0.20 recency)
        assert ranked[0]["arxiv_id"] == "1501.00001"

    def test_recency_empty_input(self):
        """Empty input returns empty output."""
        from app.hybrid_search_svc import _recency_rerank
        assert _recency_rerank([]) == []

    def test_recency_unparseable_id(self):
        """Papers with unparseable IDs get neutral recency (0.5)."""
        from app.hybrid_search_svc import _recency_rerank

        fused = [
            {"arxiv_id": "math/0301001", "rrf_score": 0.5},
        ]

        ranked = _recency_rerank(fused)
        assert len(ranked) == 1
        assert "final_score" in ranked[0]


# ── Groq rewriter tests ─────────────────────────────────────────────────────

class TestGroqRewriter:
    """Test the query rewriter heuristics (no live API calls)."""

    def test_academic_detection_arxiv_id(self):
        """Queries with arXiv IDs should be detected as academic."""
        from app.groq_svc import _looks_academic
        assert _looks_academic("attention is all you need 1706.03762 transformer paper")

    def test_academic_detection_acronyms(self):
        """Queries with multiple acronyms should be detected."""
        from app.groq_svc import _looks_academic
        assert _looks_academic("survey of LLM hallucination in NLP tasks using BERT embeddings")

    def test_casual_query_not_academic(self):
        """Short casual queries should not be detected as academic."""
        from app.groq_svc import _looks_academic
        assert not _looks_academic("when AI makes stuff up")

    def test_rewrite_empty_query(self):
        """Empty query returns empty string."""
        import asyncio
        from app.groq_svc import rewrite
        result = asyncio.get_event_loop().run_until_complete(rewrite(""))
        assert result == ""

    def test_rewrite_fallback_no_api_key(self):
        """Without API key, returns original query."""
        import asyncio
        from app.groq_svc import rewrite

        with patch("app.config.GROQ_API_KEY", ""):
            # Reset cached client
            import app.groq_svc as gs
            gs._client = None
            result = asyncio.get_event_loop().run_until_complete(
                rewrite("when AI makes up fake facts")
            )
            assert result == "when AI makes up fake facts"


# ── Embed service tests ──────────────────────────────────────────────────────

class TestEmbedService:
    """Test embed_svc encode_query edge cases (no model loading in CI)."""

    def test_encode_empty_string(self):
        """Empty string returns zero vector and empty sparse dict."""
        from app.embed_svc import encode_query
        dense, sparse = encode_query("")
        assert dense.shape == (1024,)
        assert sparse == {}
        assert float(dense.sum()) == 0.0

    def test_encode_whitespace_only(self):
        """Whitespace-only input treated as empty."""
        from app.embed_svc import encode_query
        dense, sparse = encode_query("   ")
        assert dense.shape == (1024,)
        assert sparse == {}


# ── Search orchestrator mock tests ───────────────────────────────────────────

class TestHybridSearchOrchestrator:
    """Test the orchestrator with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        """Empty query returns empty list."""
        from app.hybrid_search_svc import search
        result = await search("")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_with_mocked_pipeline(self):
        """Full pipeline with mocked services returns ranked results."""
        import numpy as np
        from app import hybrid_search_svc

        mock_dense = np.random.rand(1024).astype(np.float32)
        mock_sparse = {100: 0.5, 200: 0.3}

        with patch.object(hybrid_search_svc.groq_svc, "rewrite", new_callable=AsyncMock, return_value="test query"), \
             patch.object(hybrid_search_svc.embed_svc, "encode_query", return_value=(mock_dense, mock_sparse)), \
             patch.object(hybrid_search_svc.qdrant_svc, "search_dense", new_callable=AsyncMock, return_value=[
                 {"arxiv_id": "2301.00001", "score": 0.95},
                 {"arxiv_id": "2301.00002", "score": 0.90},
             ]), \
             patch.object(hybrid_search_svc.zilliz_svc, "search_sparse", new_callable=AsyncMock, return_value=[
                 {"arxiv_id": "2301.00002", "score": 0.80},
                 {"arxiv_id": "2301.00003", "score": 0.70},
             ]):

            result = await hybrid_search_svc.search("test", limit=10)

        assert len(result) > 0
        assert all(isinstance(r, str) for r in result)
        # Paper appearing in both should rank high
        assert "2301.00002" in result[:2]

    @pytest.mark.asyncio
    async def test_search_dense_only_fallback(self):
        """Search works when sparse fails."""
        import numpy as np
        from app import hybrid_search_svc

        mock_dense = np.random.rand(1024).astype(np.float32)

        with patch.object(hybrid_search_svc.groq_svc, "rewrite", new_callable=AsyncMock, return_value="test"), \
             patch.object(hybrid_search_svc.embed_svc, "encode_query", return_value=(mock_dense, {})), \
             patch.object(hybrid_search_svc.qdrant_svc, "search_dense", new_callable=AsyncMock, return_value=[
                 {"arxiv_id": "2301.00001", "score": 0.95},
             ]), \
             patch.object(hybrid_search_svc.zilliz_svc, "search_sparse", new_callable=AsyncMock, return_value=[]):

            result = await hybrid_search_svc.search("test", limit=10)

        assert result == ["2301.00001"]

    @pytest.mark.asyncio
    async def test_search_sparse_only_fallback(self):
        """Search works when dense fails."""
        import numpy as np
        from app import hybrid_search_svc

        mock_dense = np.random.rand(1024).astype(np.float32)

        with patch.object(hybrid_search_svc.groq_svc, "rewrite", new_callable=AsyncMock, return_value="test"), \
             patch.object(hybrid_search_svc.embed_svc, "encode_query", return_value=(mock_dense, {100: 0.5})), \
             patch.object(hybrid_search_svc.qdrant_svc, "search_dense", new_callable=AsyncMock, return_value=[]), \
             patch.object(hybrid_search_svc.zilliz_svc, "search_sparse", new_callable=AsyncMock, return_value=[
                 {"arxiv_id": "2301.00003", "score": 0.70},
             ]):

            result = await hybrid_search_svc.search("test", limit=10)

        assert result == ["2301.00003"]

    @pytest.mark.asyncio
    async def test_search_total_failure(self):
        """Both services failing returns empty list, no crash."""
        import numpy as np
        from app import hybrid_search_svc

        mock_dense = np.random.rand(1024).astype(np.float32)

        with patch.object(hybrid_search_svc.groq_svc, "rewrite", new_callable=AsyncMock, return_value="test"), \
             patch.object(hybrid_search_svc.embed_svc, "encode_query", return_value=(mock_dense, {100: 0.5})), \
             patch.object(hybrid_search_svc.qdrant_svc, "search_dense", new_callable=AsyncMock, side_effect=Exception("down")), \
             patch.object(hybrid_search_svc.zilliz_svc, "search_sparse", new_callable=AsyncMock, side_effect=Exception("down")):

            result = await hybrid_search_svc.search("test", limit=10)

        assert result == []
