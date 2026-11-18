"""
Tests for hybrid search pipeline — Phase 3.

Tests RRF fusion and recency reranking logic (pure Python, no live services).
Live integration tests are separate (require BGE-M3 + Qdrant + the sidecar).
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime


# ── RRF fusion tests ─────────────────────────────────────────────────────────

class TestRRFFusion:
    """Test the RRF fusion logic in hybrid_search_svc."""

    def test_rrf_basic_merge(self):
        """Papers appearing in both lists get higher scores."""
        from app.hybrid_search_svc import _rrf_fuse_multi

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

        fused = _rrf_fuse_multi([dense, sparse], k=60)

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
        from app.hybrid_search_svc import _rrf_fuse_multi

        dense = [
            {"arxiv_id": "2301.00001", "score": 0.95},
            {"arxiv_id": "2301.00002", "score": 0.90},
        ]
        sparse = []

        fused = _rrf_fuse_multi([dense, sparse], k=60)
        assert len(fused) == 2
        assert fused[0]["arxiv_id"] == "2301.00001"

    def test_rrf_empty_both(self):
        """Empty inputs produce empty output."""
        from app.hybrid_search_svc import _rrf_fuse_multi

        fused = _rrf_fuse_multi([[], []], k=60)
        assert fused == []

    def test_rrf_scores_are_rank_based(self):
        """RRF scores depend on rank, not on raw scores."""
        from app.hybrid_search_svc import _rrf_fuse_multi

        # Same papers, different raw scores — RRF should produce identical results
        dense_a = [
            {"arxiv_id": "A", "score": 0.99},
            {"arxiv_id": "B", "score": 0.50},
        ]
        dense_b = [
            {"arxiv_id": "A", "score": 0.51},
            {"arxiv_id": "B", "score": 0.50},
        ]

        fused_a = _rrf_fuse_multi([dense_a, []], k=60)
        fused_b = _rrf_fuse_multi([dense_b, []], k=60)

        # Same ranking → same RRF scores
        assert fused_a[0]["rrf_score"] == fused_b[0]["rrf_score"]
        assert fused_a[1]["rrf_score"] == fused_b[1]["rrf_score"]

    def test_rrf_k_parameter(self):
        """Higher K dampens rank differences."""
        from app.hybrid_search_svc import _rrf_fuse_multi

        dense = [
            {"arxiv_id": "A", "score": 0.9},
            {"arxiv_id": "B", "score": 0.8},
        ]

        fused_k10 = _rrf_fuse_multi([dense, []], k=10)
        fused_k100 = _rrf_fuse_multi([dense, []], k=100)

        # Score gap should be smaller with larger K
        gap_k10 = fused_k10[0]["rrf_score"] - fused_k10[1]["rrf_score"]
        gap_k100 = fused_k100[0]["rrf_score"] - fused_k100[1]["rrf_score"]
        assert gap_k10 > gap_k100


# ── Title-match rerank tests ─────────────────────────────────────────────────

class TestTitleMatchRerank:
    """Test the title-match boost in hybrid_search_svc.

    Recency rerank was removed (it crushed seminal old papers like
    1706.03762 below newer "X is all you need" titles). Replaced with a
    title-match boost that promotes papers whose title matches the query.
    """

    @pytest.mark.asyncio
    async def test_exact_title_match_wins(self, monkeypatch):
        """Paper with exact-title match should rank #1 even with low RRF."""
        from app import hybrid_search_svc

        async def fake_meta(ids):
            return {
                "1706.03762": {"title": "Attention Is All You Need"},
                "2404.01183": {"title": "Positioning Is All You Need"},
            }
        monkeypatch.setattr(hybrid_search_svc.turso_svc, "fetch_metadata_batch", fake_meta)

        fused = [
            {"arxiv_id": "2404.01183", "rrf_score": 0.0317},  # higher RRF
            {"arxiv_id": "1706.03762", "rrf_score": 0.0164},  # lower RRF, exact match
        ]
        ranked = await hybrid_search_svc._title_match_rerank(
            fused, "attention is all you need"
        )
        assert ranked[0]["arxiv_id"] == "1706.03762"

    @pytest.mark.asyncio
    async def test_substring_match_beats_no_match(self, monkeypatch):
        """A substring title match outranks no-match candidates."""
        from app import hybrid_search_svc

        async def fake_meta(ids):
            return {
                "2501.05730": {"title": "Element-wise Attention Is All You Need"},
                "9999.99999": {"title": "An Unrelated Survey of Graph Theory"},
            }
        monkeypatch.setattr(hybrid_search_svc.turso_svc, "fetch_metadata_batch", fake_meta)

        fused = [
            {"arxiv_id": "9999.99999", "rrf_score": 0.05},     # higher RRF, no match
            {"arxiv_id": "2501.05730", "rrf_score": 0.01},     # lower RRF, substring match
        ]
        ranked = await hybrid_search_svc._title_match_rerank(
            fused, "attention is all you need"
        )
        assert ranked[0]["arxiv_id"] == "2501.05730"

    @pytest.mark.asyncio
    async def test_no_match_falls_back_to_rrf(self, monkeypatch):
        """When nothing matches, RRF order is preserved."""
        from app import hybrid_search_svc

        async def fake_meta(ids):
            return {
                "1234.56789": {"title": "Some Paper"},
                "9876.54321": {"title": "Another Paper"},
            }
        monkeypatch.setattr(hybrid_search_svc.turso_svc, "fetch_metadata_batch", fake_meta)

        fused = [
            {"arxiv_id": "1234.56789", "rrf_score": 0.05},
            {"arxiv_id": "9876.54321", "rrf_score": 0.01},
        ]
        ranked = await hybrid_search_svc._title_match_rerank(fused, "transformer")
        assert [r["arxiv_id"] for r in ranked] == ["1234.56789", "9876.54321"]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Empty input returns empty output."""
        from app import hybrid_search_svc
        assert await hybrid_search_svc._title_match_rerank([], "anything") == []

    @pytest.mark.asyncio
    async def test_turso_failure_falls_back_to_rrf(self, monkeypatch):
        """If Turso lookup raises, ranking falls back to pure RRF order."""
        from app import hybrid_search_svc

        async def boom(ids):
            raise RuntimeError("turso down")
        monkeypatch.setattr(hybrid_search_svc.turso_svc, "fetch_metadata_batch", boom)

        fused = [
            {"arxiv_id": "1234.56789", "rrf_score": 0.05},
            {"arxiv_id": "9876.54321", "rrf_score": 0.01},
        ]
        ranked = await hybrid_search_svc._title_match_rerank(fused, "attention")
        assert [r["arxiv_id"] for r in ranked] == ["1234.56789", "9876.54321"]
        # final_score must be set even on the fallback path
        assert all("final_score" in r for r in ranked)


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

    def test_long_casual_question_not_academic(self):
        """Ordinary lowercase words are not acronyms, so a long question gets rewritten."""
        from app.groq_svc import _looks_academic
        assert not _looks_academic("how do I stop a RAG system from making things up")
        assert not _looks_academic("when the AI makes up fake facts")

    def test_long_query_topic_words_any_case(self):
        """The topic words still count in any case; only acronyms are case-sensitive."""
        from app.groq_svc import _looks_academic
        assert _looks_academic("Mastering the game of Go with deep Neural Networks and tree search")

    def test_a_technical_phrase_is_searched_as_typed(self):
        """Live log, 7 Nov 2026: 9 of 25 rewrites only dropped "for" or "with"
        ("diffusion models for image super-resolution"), each costing a model
        call and a second retrieval for no change in meaning."""
        from app.groq_svc import _looks_academic
        for q in ["diffusion models for image super-resolution",
                  "contrastive learning for sentence embeddings",
                  "protein structure prediction with contrastive pretraining",
                  "retrieval augmented generation for scientific question answering"]:
            assert _looks_academic(q), q

    def test_questions_and_casual_phrasing_still_get_rewritten(self):
        """Every useful rewrite in the same log came from one of these."""
        from app.groq_svc import _looks_academic
        for q in ["how do I make LLMs hallucinate less", "papers for doctors",
                  "rag can you explain", "what is the alternative for attention mechanism ?",
                  "economics papers with ai ?", "ISRO mangalya launch", "make AI more safe",
                  "deepseek paper on ocr size reduction around 100x smaller"]:
            assert not _looks_academic(q), q

    def test_rewrite_empty_query(self):
        """Empty query returns empty string."""
        import asyncio
        from app.groq_svc import rewrite
        result = asyncio.run(rewrite(""))
        assert result == ""

    def test_rewrite_fallback_no_api_key(self):
        """Without API key, returns original query."""
        import asyncio
        from app.groq_svc import rewrite

        with patch("app.config.GROQ_API_KEY", ""):
            # Reset cached client
            import app.groq_svc as gs
            gs._client = None
            result = asyncio.run(
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
             patch.object(hybrid_search_svc.fts_svc, "is_available", return_value=True), \
             patch.object(hybrid_search_svc.fts_svc, "search_sparse", new_callable=AsyncMock, return_value=[
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
             patch.object(hybrid_search_svc.fts_svc, "is_available", return_value=True), \
             patch.object(hybrid_search_svc.fts_svc, "search_sparse", new_callable=AsyncMock, return_value=[]):

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
             patch.object(hybrid_search_svc.fts_svc, "is_available", return_value=True), \
             patch.object(hybrid_search_svc.fts_svc, "search_sparse", new_callable=AsyncMock, return_value=[
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
             patch.object(hybrid_search_svc.fts_svc, "is_available", return_value=True), \
             patch.object(hybrid_search_svc.fts_svc, "search_sparse", new_callable=AsyncMock, side_effect=Exception("down")):

            result = await hybrid_search_svc.search("test", limit=10)

        assert result == []



# ── After a rewrite, a question's own keyword list stops counting ─────────────

def _lexical(query_lists):
    async def search(text, limit=50):
        return query_lists[text]
    return search


@pytest.mark.asyncio
@pytest.mark.parametrize("switch,match,original_counts", [
    ("on", "any", False),     # a question: no paper had every word, so its words only steer
    ("on", "all", True),      # a title: some paper matched word for word, keep its list
    ("off", "any", True),     # switch off: unchanged behaviour
])
async def test_rewritten_question_drops_its_keyword_list(monkeypatch, switch, match, original_counts):
    import numpy as np
    from app import config, fts_svc, hybrid_search_svc
    monkeypatch.setenv("SEARCH_REWRITE_KEYWORDS", switch)
    monkeypatch.setattr(config, "SPARSE_BACKEND", "fts")
    monkeypatch.setattr(config, "SEARCH_BGE_RERANK", False)
    lists = {
        "how do I stop a RAG system from making things up":
            [{"arxiv_id": "stop-paper", "score": 9.0, "match": match}],
        "RAG hallucination mitigation":
            [{"arxiv_id": "hallucination-paper", "score": 9.0, "match": "all"}],
    }
    with patch.object(hybrid_search_svc.groq_svc, "rewrite", new_callable=AsyncMock,
                      return_value="RAG hallucination mitigation"), \
         patch.object(fts_svc, "is_available", return_value=True), \
         patch.object(fts_svc, "search_sparse", side_effect=_lexical(lists)), \
         patch.object(hybrid_search_svc.embed_svc, "encode_query",
                      return_value=(np.ones(1024, dtype=np.float32), {})), \
         patch.object(hybrid_search_svc.qdrant_svc, "search_dense_merged", new_callable=AsyncMock,
                      return_value=[]), \
         patch.object(hybrid_search_svc, "_title_match_rerank", new_callable=AsyncMock,
                      side_effect=lambda fused, q, **kw: fused):
        ids, meta = await hybrid_search_svc.search(
            "how do I stop a RAG system from making things up", limit=10, return_meta=True)
    assert "hallucination-paper" in ids
    assert ("stop-paper" in ids) is original_counts
    assert ("fts_q0" in meta["retrieval_sources"]) is original_counts


# ── What is stored with each search to tell a cold one apart ─────────────────

@pytest.mark.asyncio
async def test_a_search_records_the_gap_since_the_last_and_the_rewrite_call_alone(monkeypatch):
    import asyncio
    import numpy as np
    from app import config, fts_svc, hybrid_search_svc
    monkeypatch.setattr(config, "SPARSE_BACKEND", "fts")
    monkeypatch.setattr(config, "SEARCH_BGE_RERANK", False)
    monkeypatch.setattr(hybrid_search_svc, "_last_search_at", None)

    def slow_encode(text):
        import time
        time.sleep(0.3)                          # the rewrite returns long before this
        return np.ones(1024, dtype=np.float32), {}

    async def rewrite(q):
        await asyncio.sleep(0.01)
        return q

    with patch.object(hybrid_search_svc.groq_svc, "rewrite", side_effect=rewrite), \
         patch.object(fts_svc, "is_available", return_value=True), \
         patch.object(fts_svc, "search_sparse", side_effect=_lexical({"sparse attention for long documents": [
             {"arxiv_id": "p1", "score": 1.0, "match": "all"}]})), \
         patch.object(hybrid_search_svc.embed_svc, "encode_query", side_effect=slow_encode), \
         patch.object(hybrid_search_svc.qdrant_svc, "search_dense_merged", new_callable=AsyncMock, return_value=[]), \
         patch.object(hybrid_search_svc, "_title_match_rerank", new_callable=AsyncMock,
                      side_effect=lambda fused, q, **kw: fused):
        _, first = await hybrid_search_svc.search("sparse attention for long documents", return_meta=True)
        _, second = await hybrid_search_svc.search("sparse attention for long documents", return_meta=True)
    assert first["gap_s"] is None and second["gap_s"] == 0         # None: the first since a restart
    assert first["rewrite_call_ms"] < 200 <= first["groq_time_ms"]  # the old timer waited for the encode


# ── After the cross-encoder; landmark queries; title auditions (doc 06, 2026-11-13) ──

def _meta_from(table):
    async def fake(ids):
        return {a: table[a] for a in ids if a in table}
    return fake


@pytest.mark.asyncio
async def test_after_the_cross_encoder_word_overlap_no_longer_decides(monkeypatch):
    """Audit F16: CE p=0.95 with no title overlap used to lose to p=0.50 with 80% overlap."""
    from app import hybrid_search_svc as hs
    monkeypatch.setattr(hs.turso_svc, "fetch_metadata_batch", _meta_from({
        "a": {"title": "A study of sparse attention kernels", "citation_count": 0},
        "b": {"title": "Graph neural networks for molecules", "citation_count": 2000},
    }))
    fused = [{"arxiv_id": "a", "rrf_score": 0.50}, {"arxiv_id": "b", "rrf_score": 0.95}]
    before = await hs._title_match_rerank([dict(x) for x in fused[::-1]], "sparse attention kernels study")
    after = await hs._title_match_rerank([dict(x) for x in fused[::-1]], "sparse attention kernels study",
                                         after_ce=True)
    assert before[0]["arxiv_id"] == "a" and after[0]["arxiv_id"] == "b"


@pytest.mark.asyncio
async def test_an_exact_title_still_wins_after_the_cross_encoder(monkeypatch):
    from app import hybrid_search_svc as hs
    monkeypatch.setattr(hs.turso_svc, "fetch_metadata_batch", _meta_from({
        "1706.03762": {"title": "Attention Is All You Need", "citation_count": 100000},
        "x": {"title": "Positioning Is All You Need", "citation_count": 3},
    }))
    fused = [{"arxiv_id": "x", "rrf_score": 0.97}, {"arxiv_id": "1706.03762", "rrf_score": 0.20}]
    ranked = await hs._title_match_rerank(fused, "attention is all you need", after_ce=True)
    assert ranked[0]["arxiv_id"] == "1706.03762"


@pytest.mark.asyncio
async def test_asking_for_seminal_papers_weighs_citations(monkeypatch):
    from app import hybrid_search_svc as hs
    monkeypatch.setattr(hs.turso_svc, "fetch_metadata_batch", _meta_from({
        "new": {"title": "Efficient attention for long documents", "citation_count": 4},
        "classic": {"title": "Neural machine translation by jointly learning to align", "citation_count": 30000},
    }))
    fused = lambda: [{"arxiv_id": "new", "rrf_score": 0.80}, {"arxiv_id": "classic", "rrf_score": 0.72}]
    plain = await hs._title_match_rerank(fused(), "attention mechanism", after_ce=True)
    seminal = await hs._title_match_rerank(fused(), "seminal attention mechanism paper", after_ce=True)
    assert plain[0]["arxiv_id"] == "new" and seminal[0]["arxiv_id"] == "classic"
    assert hs.wants_landmarks("Most cited papers on RLHF") and not hs.wants_landmarks("attention for vision")


def test_a_named_title_just_below_the_window_auditions():
    """Audit F15: an exact-title paper at fused rank 11 never reached the reader."""
    from app import hybrid_search_svc as hs
    tail = [{"arxiv_id": f"t{i}"} for i in range(5)]
    titles = {"t1": "Attention Is All You Need", "t3": "Deep Residual Learning for Image Recognition",
              "t4": "Transformers for everything"}
    assert [x["arxiv_id"] for x in hs._title_auditions("Attention is all you need", tail, titles)] == ["t1"]
    assert [x["arxiv_id"] for x in hs._title_auditions("residual learning for image", tail, titles)] == ["t3"]
    # One word inside a title is not naming it.
    assert hs._title_auditions("transformers", tail, titles) == []


def test_landmark_words_are_not_required_keywords():
    from app import fts_svc
    assert fts_svc._terms("seminal paper on attention") == ["attention"]
    assert "popular" not in fts_svc._terms("most popular diffusion models")
