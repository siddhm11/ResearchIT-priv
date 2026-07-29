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


# ── Phase 4.3: cache_turso_metadata_batch ────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_turso_metadata_batch_writes_all(tmp_db):
    """Turso dicts should be written to paper_metadata verbatim."""
    import app.db as db
    await db.init_db()
    papers = [
        {
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "Transformers.",
            "authors": '["Vaswani"]',
            "category": "cs.CL",
            "published": "2017-06-12",
            "year": 2017,
            "citation_count": 50000,
        },
        {
            "arxiv_id": "2001.00001",
            "title": "Another Paper",
            "abstract": "...",
            "authors": '["Smith"]',
            "category": "cs.CV",
            "published": "2020-01-01",
            "year": 2020,
        },
    ]
    await db.cache_turso_metadata_batch(papers)

    cached = await db.get_cached_metadata("1706.03762")
    assert cached is not None
    assert cached["title"] == "Attention Is All You Need"
    assert cached["category"] == "cs.CL"

    cached2 = await db.get_cached_metadata("2001.00001")
    assert cached2 is not None
    assert cached2["category"] == "cs.CV"


@pytest.mark.asyncio
async def test_cache_turso_metadata_batch_empty(tmp_db):
    """Empty input must not crash."""
    import app.db as db
    await db.init_db()
    await db.cache_turso_metadata_batch([])
    # No exception = success


@pytest.mark.asyncio
async def test_cache_turso_metadata_batch_skips_missing_arxiv_id(tmp_db):
    """Rows without arxiv_id should be skipped, others persisted."""
    import app.db as db
    await db.init_db()
    papers = [
        {"title": "No ID", "category": "cs.LG"},  # missing arxiv_id
        {"arxiv_id": "good.123", "title": "Good", "category": "cs.AI",
         "abstract": "", "authors": "[]", "published": "2024-01-01"},
    ]
    await db.cache_turso_metadata_batch(papers)
    cached = await db.get_cached_metadata("good.123")
    assert cached is not None
    assert cached["title"] == "Good"


@pytest.mark.asyncio
async def test_cache_turso_metadata_batch_upserts(tmp_db):
    """Second write for same arxiv_id should overwrite the first."""
    import app.db as db
    await db.init_db()
    paper_v1 = {"arxiv_id": "p1", "title": "V1", "category": "cs.LG",
                "abstract": "", "authors": "[]", "published": "2024-01-01"}
    paper_v2 = {"arxiv_id": "p1", "title": "V2", "category": "cs.CV",
                "abstract": "", "authors": "[]", "published": "2024-01-01"}
    await db.cache_turso_metadata_batch([paper_v1])
    await db.cache_turso_metadata_batch([paper_v2])
    cached = await db.get_cached_metadata("p1")
    assert cached["title"] == "V2"
    assert cached["category"] == "cs.CV"


# ── Phase 4.3: get_suppressed_categories ──────────────────────────────────────

@pytest.mark.asyncio
async def test_suppressed_empty_for_new_user(tmp_db):
    import app.db as db
    await db.init_db()
    result = await db.get_suppressed_categories("never-dismissed")
    assert result == set()


@pytest.mark.asyncio
async def test_suppressed_below_threshold_not_returned(tmp_db):
    """Two dismissals in one category (< threshold=3) should NOT suppress."""
    import app.db as db
    await db.init_db()
    # Seed metadata
    for i, aid in enumerate(["p1", "p2"]):
        await db.cache_metadata({
            "arxiv_id": aid, "title": f"t{i}", "abstract": "",
            "authors": "[]", "category": "cs.CV", "published": "2024-01-01",
        })
    # Two dismissals — below threshold=3
    await db.log_interaction("u1", "p1", "not_interested")
    await db.log_interaction("u1", "p2", "not_interested")

    result = await db.get_suppressed_categories("u1")
    assert "cs.CV" not in result


@pytest.mark.asyncio
async def test_suppressed_at_threshold_returned(tmp_db):
    """Three dismissals in same category should suppress that category."""
    import app.db as db
    await db.init_db()
    for i, aid in enumerate(["p1", "p2", "p3"]):
        await db.cache_metadata({
            "arxiv_id": aid, "title": f"t{i}", "abstract": "",
            "authors": "[]", "category": "physics.optics", "published": "2024-01-01",
        })
    for aid in ["p1", "p2", "p3"]:
        await db.log_interaction("u1", aid, "not_interested")

    result = await db.get_suppressed_categories("u1")
    assert "physics.optics" in result


@pytest.mark.asyncio
async def test_suppressed_only_counts_not_interested(tmp_db):
    """Saves should NOT count toward suppression."""
    import app.db as db
    await db.init_db()
    for aid in ["p1", "p2", "p3"]:
        await db.cache_metadata({
            "arxiv_id": aid, "title": "t", "abstract": "",
            "authors": "[]", "category": "cs.CL", "published": "2024-01-01",
        })
    # 3 saves (not dismissals) in same category
    for aid in ["p1", "p2", "p3"]:
        await db.log_interaction("u1", aid, "save")

    result = await db.get_suppressed_categories("u1")
    assert "cs.CL" not in result


@pytest.mark.asyncio
async def test_suppressed_partitions_categories(tmp_db):
    """Different categories should be independent."""
    import app.db as db
    await db.init_db()
    # 3 dismissals in cs.AI, 1 in cs.LG
    for aid in ["a1", "a2", "a3"]:
        await db.cache_metadata({
            "arxiv_id": aid, "title": "t", "abstract": "",
            "authors": "[]", "category": "cs.AI", "published": "2024-01-01",
        })
        await db.log_interaction("u1", aid, "not_interested")
    await db.cache_metadata({
        "arxiv_id": "lone", "title": "t", "abstract": "",
        "authors": "[]", "category": "cs.LG", "published": "2024-01-01",
    })
    await db.log_interaction("u1", "lone", "not_interested")

    result = await db.get_suppressed_categories("u1")
    assert "cs.AI" in result
    assert "cs.LG" not in result


@pytest.mark.asyncio
async def test_suppressed_ignores_other_users(tmp_db):
    """One user's dismissals must not affect another user's suppressions."""
    import app.db as db
    await db.init_db()
    for aid in ["p1", "p2", "p3"]:
        await db.cache_metadata({
            "arxiv_id": aid, "title": "t", "abstract": "",
            "authors": "[]", "category": "cs.CV", "published": "2024-01-01",
        })
        await db.log_interaction("userA", aid, "not_interested")

    result_a = await db.get_suppressed_categories("userA")
    result_b = await db.get_suppressed_categories("userB")
    assert "cs.CV" in result_a
    assert result_b == set()


@pytest.mark.asyncio
async def test_suppressed_empty_category_excluded(tmp_db):
    """Papers with empty category string should not produce a '' suppression."""
    import app.db as db
    await db.init_db()
    for aid in ["e1", "e2", "e3"]:
        await db.cache_metadata({
            "arxiv_id": aid, "title": "t", "abstract": "",
            "authors": "[]", "category": "", "published": "2024-01-01",
        })
        await db.log_interaction("u1", aid, "not_interested")

    result = await db.get_suppressed_categories("u1")
    assert "" not in result


@pytest.mark.asyncio
async def test_suppressed_custom_threshold(tmp_db):
    """Threshold=2 should trigger at 2 dismissals."""
    import app.db as db
    await db.init_db()
    for aid in ["x1", "x2"]:
        await db.cache_metadata({
            "arxiv_id": aid, "title": "t", "abstract": "",
            "authors": "[]", "category": "math.NT", "published": "2024-01-01",
        })
        await db.log_interaction("u1", aid, "not_interested")

    result = await db.get_suppressed_categories("u1", threshold=2)
    assert "math.NT" in result

    result_high = await db.get_suppressed_categories("u1", threshold=5)
    assert "math.NT" not in result_high


# ── Phase 4.5: Instrumentation columns ───────────────────────────────────────

@pytest.mark.asyncio
async def test_instrumentation_columns_exist(tmp_db):
    """The interactions table should have ranker_version, candidate_source, cluster_id columns."""
    import app.db as db
    import aiosqlite
    await db.init_db()
    async with aiosqlite.connect(tmp_db) as conn:
        cur = await conn.execute("PRAGMA table_info(interactions)")
        columns = {row[1] for row in await cur.fetchall()}
    assert "ranker_version" in columns
    assert "candidate_source" in columns
    assert "cluster_id" in columns


@pytest.mark.asyncio
async def test_log_interaction_stores_instrumentation_fields(tmp_db):
    """log_interaction should persist ranker_version, candidate_source, cluster_id."""
    import app.db as db
    import aiosqlite
    await db.init_db()
    await db.log_interaction(
        user_id="u1",
        paper_id="p1",
        event_type="save",
        source="recommendation",
        ranker_version="v4.1_test",
        candidate_source="cluster_0",
        cluster_id=0,
    )
    async with aiosqlite.connect(tmp_db) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT ranker_version, candidate_source, cluster_id FROM interactions WHERE paper_id = 'p1'"
        )
        row = dict(await cur.fetchone())
    assert row["ranker_version"] == "v4.1_test"
    assert row["candidate_source"] == "cluster_0"
    assert row["cluster_id"] == 0


@pytest.mark.asyncio
async def test_log_interaction_instrumentation_defaults_to_null(tmp_db):
    """Omitting instrumentation fields should store NULLs (backward compat)."""
    import app.db as db
    import aiosqlite
    await db.init_db()
    await db.log_interaction("u1", "p2", "save", source="search")
    async with aiosqlite.connect(tmp_db) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT ranker_version, candidate_source, cluster_id FROM interactions WHERE paper_id = 'p2'"
        )
        row = dict(await cur.fetchone())
    assert row["ranker_version"] is None
    assert row["candidate_source"] is None
    assert row["cluster_id"] is None


@pytest.mark.asyncio
async def test_migration_idempotent(tmp_db):
    """Calling init_db() twice must not crash (ALTER TABLE migration is safe)."""
    import app.db as db
    await db.init_db()
    await db.init_db()  # second call — migration should be idempotent
    # No exception = success


@pytest.mark.asyncio
async def test_instrumentation_exploration_tag(tmp_db):
    """Exploration papers should be stored with candidate_source='exploration'."""
    import app.db as db
    import aiosqlite
    await db.init_db()
    await db.log_interaction(
        user_id="u1",
        paper_id="explore_paper",
        event_type="save",
        source="recommendation",
        ranker_version="v4.1_quota_hungarian_suppression",
        candidate_source="exploration",
        cluster_id=None,
    )
    async with aiosqlite.connect(tmp_db) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT candidate_source, cluster_id FROM interactions WHERE paper_id = 'explore_paper'"
        )
        row = dict(await cur.fetchone())
    assert row["candidate_source"] == "exploration"
    assert row["cluster_id"] is None

