"""
Tests for qdrant_svc.py.
- Unit tests: ID lookup logic, cache hits
- Integration tests: actual Qdrant calls (require network)
"""
import pytest


# ── Unit: SQLite cache hit avoids Qdrant call ─────────────────────────────────

@pytest.mark.asyncio
async def test_lookup_returns_cached_ids_without_qdrant(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    # Pre-populate cache
    await db_mod.save_qdrant_id("1706.03762", 999)
    await db_mod.save_qdrant_id("0704.0002",  0)

    # Patch _scroll_by_arxiv_ids to assert it is NOT called
    import app.qdrant_svc as qs
    def should_not_be_called(ids):
        raise AssertionError("Qdrant should not be called when cache is warm")
    monkeypatch.setattr(qs, "_scroll_by_arxiv_ids", should_not_be_called)

    result = await qs.lookup_qdrant_ids(["1706.03762", "0704.0002"])
    assert result == {"1706.03762": 999, "0704.0002": 0}


@pytest.mark.asyncio
async def test_lookup_calls_qdrant_for_missing(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    import app.qdrant_svc as qs
    called_with = {}
    def fake_scroll(ids):
        called_with["ids"] = ids
        return {"1706.03762": 12345}
    monkeypatch.setattr(qs, "_scroll_by_arxiv_ids", fake_scroll)

    result = await qs.lookup_qdrant_ids(["1706.03762"])
    assert result == {"1706.03762": 12345}
    assert called_with["ids"] == ["1706.03762"]

    # Mapping should now be cached in SQLite
    cached = await db_mod.get_qdrant_id("1706.03762")
    assert cached == 12345


@pytest.mark.asyncio
async def test_recommend_returns_empty_when_no_positive_ids(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    import app.qdrant_svc as qs
    # Even if Qdrant scroll finds nothing, should not crash
    monkeypatch.setattr(qs, "_scroll_by_arxiv_ids", lambda ids: {})

    result = await qs.recommend(
        positive_arxiv_ids=["unknown_paper"],
        negative_arxiv_ids=[],
        seen_arxiv_ids=set(),
    )
    assert result == []


# ── Integration: real Qdrant calls ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_lookup_real_qdrant(tmp_path, monkeypatch):
    """Lookup a known arxiv_id in the live Qdrant collection."""
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    # Reset the lru_cache so the patched config is used
    from app.qdrant_svc import _client
    _client.cache_clear()

    from app.qdrant_svc import lookup_qdrant_ids
    # Point ID 0 has arxiv_id "0704.0002" per our earlier inspection
    result = await lookup_qdrant_ids(["0704.0002"])
    assert "0704.0002" in result
    assert isinstance(result["0704.0002"], int)


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

    # Pre-seed qdrant map so lookup doesn't need scroll
    await db_mod.save_qdrant_id("0704.0002", 0)
    await db_mod.save_qdrant_id("0704.0004", 1)

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
