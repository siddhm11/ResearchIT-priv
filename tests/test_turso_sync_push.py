"""What sync_once() pushes, checked against a stand-in Turso.

The stand-in is an in-memory SQLite database built from turso_sync.REMOTE_DDL;
_execute() runs each Hrana statement against it, decoding cells with the
module's own _uncell. Nothing touches the network.
"""
import sqlite3

import numpy as np
import pytest

from app import db, turso_sync


@pytest.fixture
def remote(tmp_path, monkeypatch):
    import app.config as cfg
    path = str(tmp_path / "local.db")
    monkeypatch.setattr(cfg, "DB_PATH", path)
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(turso_sync.config, "DB_PATH", path, raising=False)
    monkeypatch.setattr(turso_sync, "_boundary", {})
    import asyncio
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


def _local(sql, args=()):
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute(sql, args)
    conn.commit()
    conn.close()


VEC = np.ones(1024, dtype=np.float32).tobytes()


def _profile(user, ts, count=1):
    _local("INSERT OR REPLACE INTO user_profiles (user_id, profile_type, vector, "
           "interaction_count, updated_at) VALUES (?, 'long_term', ?, ?, ?)",
           (user, VEC, count, ts))


async def test_same_second_writes_reach_turso(remote):
    """Audit F11: the text watermark used `>`, so reader B's profile, written in
    the same second as reader A's pushed one, was never backed up."""
    _profile("reader-A", "2026-11-07 10:00:05")
    await turso_sync.sync_once()
    _profile("reader-B", "2026-11-07 10:00:05")
    await turso_sync.sync_once()

    users = {r[0] for r in remote.execute("SELECT user_id FROM user_profiles")}
    assert users == {"reader-A", "reader-B"}


async def test_an_update_within_the_same_second_is_pushed(remote):
    _profile("reader-A", "2026-11-07 10:00:05", count=1)
    await turso_sync.sync_once()
    _profile("reader-A", "2026-11-07 10:00:05", count=2)
    await turso_sync.sync_once()

    assert remote.execute("SELECT interaction_count FROM user_profiles").fetchall() == [(2,)]


async def test_settled_rows_are_not_pushed_again(remote):
    _profile("reader-A", "2026-11-07 10:00:05")
    await turso_sync.sync_once()
    pushed = await turso_sync.sync_once()
    assert "user_profiles" not in pushed


async def test_a_reclustered_reader_does_not_keep_old_clusters_in_turso(remote):
    """save_user_clusters replaces a reader's clusters locally; the push only
    ever added, so dropped clusters lived on in Turso and came back on the next
    restore. With fresh ids per new interest (F6) they would pile up."""
    def cluster(idx):
        return {"cluster_idx": idx, "medoid_paper_id": f"p{idx}", "importance": 1.0,
                "paper_ids": f'["p{idx}"]', "medoid_embedding_blob": VEC}

    await db.save_user_clusters("reader-A", [cluster(0), cluster(1)])
    await db.save_user_clusters("reader-B", [cluster(0)])
    await turso_sync.sync_once()
    _local("UPDATE user_clusters SET computed_at = '2099-01-01 00:00:00'")  # a later build
    await db.save_user_clusters("reader-A", [cluster(2)])
    _local("UPDATE user_clusters SET computed_at = '2099-01-01 00:00:01' WHERE user_id = 'reader-A'")
    await turso_sync.sync_once()

    rows = sorted(remote.execute("SELECT user_id, cluster_idx FROM user_clusters").fetchall())
    assert rows == [("reader-A", 2), ("reader-B", 0)]
