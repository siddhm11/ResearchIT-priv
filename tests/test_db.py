"""
Tests for db.py using a temporary in-memory/temp database.
"""
import os
import tempfile
import pytest
import pytest_asyncio

# Override DB_PATH before importing db
@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    # Also patch the module-level constant
    import app.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    return db_path


@pytest.mark.asyncio
async def test_init_db_creates_tables(tmp_db):
    import app.db as db
    await db.init_db()
    import aiosqlite
    async with aiosqlite.connect(tmp_db) as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {r[0] for r in await cur.fetchall()}
    assert "interactions" in tables
    assert "paper_qdrant_map" in tables
    assert "paper_metadata" in tables


@pytest.mark.asyncio
async def test_log_and_retrieve_interactions(tmp_db):
    import app.db as db
    await db.init_db()
    await db.log_interaction("user1", "1706.03762", "save", source="search")
    await db.log_interaction("user1", "2302.11382", "not_interested", source="search")

    rows = await db.get_user_interactions("user1")
    assert len(rows) == 2
    paper_ids = {r["paper_id"] for r in rows}
    assert "1706.03762" in paper_ids
    assert "2302.11382" in paper_ids


@pytest.mark.asyncio
async def test_filter_interactions_by_event_type(tmp_db):
    import app.db as db
    await db.init_db()
    await db.log_interaction("u2", "aaa", "save")
    await db.log_interaction("u2", "bbb", "not_interested")
    await db.log_interaction("u2", "ccc", "click")

    saves = await db.get_user_interactions("u2", event_types=["save"])
    assert len(saves) == 1
    assert saves[0]["paper_id"] == "aaa"


@pytest.mark.asyncio
async def test_qdrant_id_roundtrip(tmp_db):
    import app.db as db
    await db.init_db()
    await db.save_qdrant_id("1706.03762", 42)
    assert await db.get_qdrant_id("1706.03762") == 42
    assert await db.get_qdrant_id("unknown") is None


@pytest.mark.asyncio
async def test_qdrant_id_batch(tmp_db):
    import app.db as db
    await db.init_db()
    await db.save_qdrant_id("a1", 10)
    await db.save_qdrant_id("a2", 20)
    result = await db.get_qdrant_ids_batch(["a1", "a2", "a3"])
    assert result == {"a1": 10, "a2": 20}


@pytest.mark.asyncio
async def test_metadata_cache_roundtrip(tmp_db):
    import app.db as db
    await db.init_db()
    paper = {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models...",
        "authors": '["Vaswani", "Shazeer"]',
        "category": "cs.CL",
        "published": "2017-06-12",
    }
    await db.cache_metadata(paper)
    cached = await db.get_cached_metadata("1706.03762")
    assert cached is not None
    assert cached["title"] == "Attention Is All You Need"
    assert cached["category"] == "cs.CL"


@pytest.mark.asyncio
async def test_metadata_cache_batch(tmp_db):
    import app.db as db
    await db.init_db()
    for i in range(3):
        await db.cache_metadata({
            "arxiv_id": f"paper{i}",
            "title": f"Title {i}",
            "abstract": "...",
            "authors": "[]",
            "category": "cs.LG",
            "published": "2023-01-01",
        })
    result = await db.get_cached_metadata_batch(["paper0", "paper2", "paper99"])
    assert "paper0" in result
    assert "paper2" in result
    assert "paper99" not in result
