"""What the feed showed and what a reader follows survive a deploy (2026-11-13).

feed_exposures, feed_impressions and collection_follows were local only, so a
new container forgot them: a returning reader saw last week's papers again,
follows vanished, and the exposure log restarted at zero. Each test pushes to
a stand-in Turso (in-memory SQLite built from REMOTE_DDL), wipes the local
database the way a deploy does, restores, and reads the result back through
the app's own functions.
"""
import asyncio
import os
import sqlite3

import pytest

from app import db, turso_sync


@pytest.fixture
def remote(tmp_path, monkeypatch):
    import app.config as cfg
    path = str(tmp_path / "local.db")
    for mod in (cfg, db):
        monkeypatch.setattr(mod, "DB_PATH", path)
    monkeypatch.setattr(turso_sync, "_boundary", {})
    asyncio.run(db.init_db())

    store = sqlite3.connect(":memory:")
    for ddl in turso_sync.REMOTE_DDL:
        store.execute(ddl)

    async def fake_execute(stmts, timeout=60):
        out = []
        for s in stmts:
            args = [turso_sync._uncell(a) for a in s.get("args", [])]
            out.append(store.execute(s["sql"], args).fetchall())
        store.commit()
        return out

    monkeypatch.setattr(turso_sync, "_execute", fake_execute)
    return store


async def _deploy() -> dict:
    """A new container: empty local database, then restore from Turso."""
    os.remove(db.DB_PATH)
    turso_sync._boundary.clear()
    await db.init_db()
    return await turso_sync.restore()


EXPOSURE = {"user_id": "r1", "paper_id": "2611.00001", "query_id": "q1", "position": 0,
            "tier": 1, "candidate_source": "cluster", "cluster_id": 3, "propensity": 1.0,
            "policy_id": "v"}


async def test_the_exposure_log_survives_a_deploy(remote):
    await db.record_exposures([EXPOSURE, {**EXPOSURE, "paper_id": "2611.00002", "position": 1}])
    pushed = await turso_sync.sync_once()
    assert pushed["feed_exposures"] == 2

    restored = await _deploy()
    assert restored["feed_exposures"] == 2
    conn = sqlite3.connect(db.DB_PATH)
    rows = conn.execute("SELECT paper_id, query_id, position, tier, cluster_id, propensity "
                        "FROM feed_exposures ORDER BY id").fetchall()
    assert rows == [("2611.00001", "q1", 0, 1, 3, 1.0), ("2611.00002", "q1", 1, 1, 3, 1.0)]
    # A new exposure after the deploy is pushed as new, not over a restored row.
    await db.record_exposures([{**EXPOSURE, "paper_id": "2611.00003"}])
    assert (await turso_sync.sync_once())["feed_exposures"] == 1
    assert remote.execute("SELECT COUNT(*) FROM feed_exposures").fetchone()[0] == 3


async def test_papers_already_shown_stay_shown_after_a_deploy(remote):
    await db.record_impressions("r1", ["2611.00001", "2611.00002"])
    await turso_sync.sync_once()
    await _deploy()
    assert await db.get_impressed_ids("r1", within_days=7) == {"2611.00001", "2611.00002"}


async def test_follows_and_unfollows_survive_a_deploy(remote):
    await db.follow_collection("r1", "rl-basics")
    await db.follow_collection("r1", "diffusion")
    await turso_sync.sync_once()
    await db.unfollow_collection("r1", "diffusion")
    # Same second as the follow: the watermark boundary must still carry it.
    await turso_sync.sync_once()

    await _deploy()
    assert await db.get_followed_slugs("r1") == {"rl-basics"}
    await db.follow_collection("r1", "diffusion")             # follow again
    await turso_sync.sync_once()
    await _deploy()
    assert await db.get_followed_slugs("r1") == {"rl-basics", "diffusion"}


async def test_an_old_follows_table_gains_the_new_columns(tmp_path, monkeypatch):
    import app.config as cfg
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE collection_follows (user_id TEXT NOT NULL, slug TEXT NOT NULL, "
                 "followed_at TEXT NOT NULL DEFAULT (datetime('now')), PRIMARY KEY (user_id, slug))")
    conn.execute("INSERT INTO collection_follows VALUES ('r1', 'old', '2026-10-01 00:00:00')")
    conn.commit()
    conn.close()
    for mod in (cfg, db):
        monkeypatch.setattr(mod, "DB_PATH", path)
    await db.init_db()
    await db.init_db()
    assert await db.get_followed_slugs("r1") == {"old"}
    row = sqlite3.connect(path).execute(
        "SELECT active, updated_at FROM collection_follows").fetchone()
    assert row == (1, "2026-10-01 00:00:00")              # so the first sync pushes it


def test_the_three_tables_are_replicated_with_matching_remote_tables():
    for table in ("feed_exposures", "feed_impressions", "collection_follows"):
        spec = turso_sync.TABLES[table]
        ddl = next(d for d in turso_sync.REMOTE_DDL if f"EXISTS {table} (" in d)
        assert all(c in ddl for c in spec["cols"])
