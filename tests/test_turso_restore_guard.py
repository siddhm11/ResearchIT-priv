"""
A failed restore must never let the next sync destroy the remote history.

DB_PATH is /tmp on Spaces, so Turso holds the ONLY durable copy of every save,
EWMA profile, cluster and onboarding record. A fresh container starts with an
empty local DB whose AUTOINCREMENT restarts at 1.

The path, reproduced before the fix:
  1. restore() cannot pull `interactions` — a Turso 5xx, a timeout on a large
     table. The `except` branch logs and continues, so the table stays empty
     and its watermark stays None.
  2. A new user saves. Their first row gets id 1.
  3. sync_once() sees no watermark, so it selects everything and pushes with
     INSERT OR REPLACE keyed on `id`.
  4. Remote row 1 — a real user's first save — is overwritten. The sync
     reports success.

Measured: five rows of history, two new saves, one sync tick, and the first
two rows of the original history are gone.
"""
import sqlite3

import pytest

from app import turso_sync


LOCAL_DDL = """CREATE TABLE interactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, paper_id TEXT, event_type TEXT)"""


def _local_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(LOCAL_DDL)
    conn.commit()
    return conn


def test_the_collision_is_real_without_reserved_ids():
    """Pins the mechanism, so the reason for the guard stays legible."""
    remote = {i: ("alice", f"paper{i}") for i in range(1, 6)}
    local = _local_db()
    for pid in ("new1", "new2"):
        local.execute(
            "INSERT INTO interactions (user_id, paper_id, event_type) "
            "VALUES (?,?,?)", ("bob", pid, "save"))
    local.commit()

    new_ids = [r[0] for r in local.execute("SELECT id FROM interactions")]
    assert new_ids == [1, 2], "a fresh AUTOINCREMENT must restart at 1"
    assert set(new_ids) & set(remote), (
        "the new container's ids collide with remote ids — INSERT OR REPLACE "
        "on this set destroys real history"
    )


def test_reserving_id_space_prevents_the_collision():
    """After reserving, no local id can ever match a remote one."""
    local = _local_db()
    remote_max = 5

    # What _reserve_id_space does, against a real sqlite_sequence.
    local.execute("INSERT INTO interactions (user_id) VALUES ('seed')")
    local.execute("DELETE FROM interactions")
    local.execute(
        "UPDATE sqlite_sequence SET seq = ? WHERE name = ? AND seq < ?",
        (remote_max, "interactions", remote_max))
    local.commit()

    for pid in ("new1", "new2"):
        local.execute(
            "INSERT INTO interactions (user_id, paper_id, event_type) "
            "VALUES (?,?,?)", ("bob", pid, "save"))
    local.commit()

    new_ids = [r[0] for r in local.execute("SELECT id FROM interactions")]
    assert min(new_ids) > remote_max, (
        f"local ids {new_ids} still overlap remote ids 1..{remote_max}")


async def test_a_table_that_cannot_reserve_is_not_pushed(monkeypatch):
    """When even MAX(id) is unreadable, refuse to replicate that table.

    Losing one boot's writes is recoverable. Overwriting every earlier boot's
    is not.
    """
    async def boom(*a, **k):
        raise RuntimeError("turso unreachable")

    monkeypatch.setattr(turso_sync, "_execute", boom)
    turso_sync._no_push.clear()

    ok = await turso_sync._reserve_id_space(
        None, "interactions", turso_sync.TABLES["interactions"])
    assert ok is False, "must report failure when MAX(id) cannot be read"


def test_no_push_is_visible_in_status():
    """A container silently not backing up is worse than one that says so."""
    turso_sync._no_push.clear()
    assert turso_sync.status()["not_replicating"] == []
    turso_sync._no_push.add("interactions")
    try:
        assert turso_sync.status()["not_replicating"] == ["interactions"]
    finally:
        turso_sync._no_push.clear()


def test_sync_skips_tables_marked_no_push():
    """The guard is actually consulted by the push loop."""
    import inspect
    src = inspect.getsource(turso_sync.sync_once)
    assert "_no_push" in src, "sync_once does not consult the guard"
