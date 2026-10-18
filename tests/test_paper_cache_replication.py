"""
Paper-level caches replicated to Turso: explanations and full abstracts.

Three traps from the redesign plan, each pinned here:
  T3  a one-second timestamp watermark skips rows in a burst, so these tables
      use an AUTOINCREMENT id instead;
  T4  one unpaged SELECT of a large table times out at boot, so they restore
      in pages;
  T7  the old explanations table keyed on cache_key alone, so it had to be
      rebuilt with an id without losing a row.
The HTTP boundary is turso_sync._execute; everything above it runs for real.
"""
import asyncio

import aiosqlite
import pytest

from app import turso_sync


@pytest.fixture
def local_db(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    path = str(tmp_path / "caches.db")
    monkeypatch.setattr(cfg, "DB_PATH", path)
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    asyncio.run(db_mod.init_db())
    return path


OLD_SCHEMA = """
CREATE TABLE paper_explanations (
    cache_key   TEXT PRIMARY KEY,
    arxiv_id    TEXT NOT NULL,
    explanation TEXT NOT NULL,
    model       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_explanations_paper ON paper_explanations(arxiv_id);
"""


async def test_the_old_explanations_table_is_rebuilt_keeping_every_row(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    path = str(tmp_path / "old.db")
    async with aiosqlite.connect(path) as conn:
        await conn.executescript(OLD_SCHEMA)
        for i in range(3):
            await conn.execute(
                "INSERT INTO paper_explanations (cache_key, arxiv_id, explanation, model, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (f"k{i}", f"2609.0000{i}", f"text {i}", "m", f"2026-10-0{i + 1} 00:00:00"))
        await conn.commit()
    monkeypatch.setattr(cfg, "DB_PATH", path)
    monkeypatch.setattr(db_mod, "DB_PATH", path)

    await db_mod.init_db()
    await db_mod.init_db()          # running twice is harmless

    async with aiosqlite.connect(path) as conn:
        cur = await conn.execute("SELECT id, cache_key, explanation FROM paper_explanations ORDER BY id")
        rows = await cur.fetchall()
        cur = await conn.execute("SELECT name FROM sqlite_master WHERE name = 'paper_explanations_old'")
        assert await cur.fetchone() is None
        cur = await conn.execute("PRAGMA index_list(paper_explanations)")
        assert "idx_explanations_paper" in {r[1] for r in await cur.fetchall()}
    assert [(r[1], r[2]) for r in rows] == [("k0", "text 0"), ("k1", "text 1"), ("k2", "text 2")]
    assert [r[0] for r in rows] == [1, 2, 3]
    # The old code path still works on the new table.
    assert await db_mod.get_explanation("k1") == "text 1"


async def test_replacing_an_explanation_gives_it_a_new_id(local_db):
    """So sync_once, which pushes ids above the watermark, sends the update."""
    import app.db as db_mod
    await db_mod.save_explanation("k", "2609.1", "first", "m")
    await db_mod.save_explanation("k", "2609.1", "second", "m")
    async with aiosqlite.connect(local_db) as conn:
        cur = await conn.execute("SELECT id, explanation FROM paper_explanations")
        assert await cur.fetchall() == [(2, "second")]


@pytest.mark.parametrize("table", ["paper_explanations", "paper_abstracts", "paper_sources",
                                   "llm_calls", "paper_briefings", "content_reports"])
def test_paper_caches_replicate_on_an_integer_id(table):
    spec = turso_sync.TABLES[table]
    assert spec["watermark"] == "id" and spec["wm_kind"] == "int"
    assert spec["cols"][0] == "id"
    assert spec.get("paged") is True
    ddl = next(d for d in turso_sync.REMOTE_DDL if f"EXISTS {table} " in d or f"EXISTS {table}(" in d)
    assert "id INTEGER PRIMARY KEY" in ddl
    # Caches upsert on a natural key, so the remote needs it UNIQUE too;
    # append-only logs have none.
    if table not in ("llm_calls", "content_reports"):
        assert "UNIQUE" in ddl


def _paged_remote(monkeypatch, n_rows):
    calls = []

    async def fake_execute(stmts, timeout=60):
        out = []
        for s in stmts:
            sql = s["sql"]
            calls.append(sql)
            if "FROM paper_abstracts WHERE id > ?" in sql:
                last = int(s["args"][0]["value"])
                ids = [i for i in range(1, n_rows + 1) if i > last][:turso_sync.RESTORE_PAGE]
                out.append([[i, f"2609.{i:05d}", f"abstract {i}", "arxiv_api", "2026-10-19"] for i in ids])
            else:
                out.append([])
        return out

    monkeypatch.setattr(turso_sync, "_execute", fake_execute)
    return calls


async def test_a_paged_table_restores_in_pages(local_db, monkeypatch):
    calls = _paged_remote(monkeypatch, 4500)
    restored = await turso_sync.restore()
    assert restored["paper_abstracts"] == 4500
    assert sum("FROM paper_abstracts" in c for c in calls) == 3
    # Unpaged tables still restore with one query each.
    assert sum("FROM user_interactions" in c for c in calls) == 1
    async with aiosqlite.connect(local_db) as conn:
        cur = await conn.execute("SELECT COUNT(*), MAX(id) FROM paper_abstracts")
        assert await cur.fetchone() == (4500, 4500)
        cur = await conn.execute("SELECT value FROM sync_state WHERE key = 'wm:paper_abstracts'")
        assert (await cur.fetchone())[0] == "4500"


async def test_an_exact_multiple_of_the_page_size_needs_one_empty_page(local_db, monkeypatch):
    calls = _paged_remote(monkeypatch, 2 * turso_sync.RESTORE_PAGE)
    restored = await turso_sync.restore()
    assert restored["paper_abstracts"] == 2 * turso_sync.RESTORE_PAGE
    assert sum("FROM paper_abstracts" in c for c in calls) == 3


async def test_a_failed_page_leaves_the_table_empty_and_reserves_ids(local_db, monkeypatch):
    seen = {"pages": 0}

    async def flaky(stmts, timeout=60):
        out = []
        for s in stmts:
            sql = s["sql"]
            if "FROM paper_abstracts WHERE id > ?" in sql:
                seen["pages"] += 1
                if seen["pages"] == 2:
                    raise RuntimeError("Turso 503")
                out.append([[i, f"2609.{i:05d}", "a", "arxiv_api", "x"] for i in range(1, turso_sync.RESTORE_PAGE + 1)])
            elif "MAX(id) FROM paper_abstracts" in sql:
                out.append([[9000]])
            else:
                out.append([])
        return out

    monkeypatch.setattr(turso_sync, "_execute", flaky)
    restored = await turso_sync.restore()
    assert "paper_abstracts" not in restored
    assert "paper_abstracts" not in turso_sync._no_push       # ids reserved, so pushing is safe
    async with aiosqlite.connect(local_db) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM paper_abstracts")
        assert (await cur.fetchone())[0] == 0
        await conn.execute("INSERT INTO paper_abstracts (arxiv_id, abstract) VALUES ('2610.1', 'x')")
        cur = await conn.execute("SELECT id FROM paper_abstracts")
        assert (await cur.fetchone())[0] == 9001


async def test_new_rows_are_pushed_above_the_watermark(local_db, monkeypatch):
    import app.db as db_mod
    pushed = []

    async def fake_execute(stmts, timeout=60):
        for s in stmts:
            if s["sql"].startswith("INSERT OR REPLACE INTO paper_explanations"):
                pushed.append(s["args"][1]["value"])
        return [[] for _ in stmts]

    monkeypatch.setattr(turso_sync, "_execute", fake_execute)
    await db_mod.save_explanation("a", "2609.1", "one", "m")
    await db_mod.save_explanation("b", "2609.2", "two", "m")
    await turso_sync.sync_once()
    await db_mod.save_explanation("a", "2609.1", "one, revised", "m")
    await turso_sync.sync_once()
    assert pushed == ["a", "b", "a"]
