"""
Tests for qdrant_svc.py.
- Unit tests: ID lookup logic, cache hits
- Integration tests: actual Qdrant calls (require network)
"""
import pytest


@pytest.mark.asyncio
async def test_recommend_returns_empty_when_no_positive_ids(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    import app.qdrant_svc as qs
    # A paper with no stored vector anywhere: nothing to recommend from, no crash.
    async def no_vectors(ids, remember=True):
        return {}
    monkeypatch.setattr(qs, "get_paper_vectors", no_vectors)

    result = await qs.recommend(
        positive_arxiv_ids=["unknown_paper"],
        negative_arxiv_ids=[],
        seen_arxiv_ids=set(),
    )
    assert result == []


# ── Integration: real Qdrant calls ───────────────────────────────────────────
#
# Marked `live`: these hit the production Qdrant collection and assert on the
# contents of specific point IDs. They pass on a developer machine only because
# .env.local supplies credentials, and fail anywhere without them -- including
# CI, which has no .env.local. Run them against a deploy, not on every push.

@pytest.mark.live
@pytest.mark.asyncio
async def test_recommend_with_real_qdrant(tmp_path, monkeypatch):
    """
    End-to-end Qdrant recommend call.
    Uses a known point (0704.0002) as a positive example.
    Should return a non-empty list of arxiv IDs.
    """
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    from app.qdrant_svc import _client
    _client.cache_clear()

    from app.qdrant_svc import recommend
    recs = await recommend(
        positive_arxiv_ids=["0704.0002"],
        negative_arxiv_ids=["0704.0004"],
        seen_arxiv_ids={"0704.0002", "0704.0004"},
        limit=5,
    )
    assert isinstance(recs, list)
    assert len(recs) > 0
    # None of the seen papers should appear
    for r in recs:
        assert r not in {"0704.0002", "0704.0004"}
    # Each result should be a non-empty string (arxiv IDs vary: 1706.03762, math/0702129)
    for r in recs:
        assert isinstance(r, str) and len(r) > 0
