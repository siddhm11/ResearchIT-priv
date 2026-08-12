"""
Tests for the restore half of app/turso_sync.py.

Why this file exists
--------------------
Hugging Face removed persistent storage for Spaces -- the `suggested_storage`
key is documented as ignored -- and DB_PATH is /tmp on an ephemeral filesystem.
So turso_sync.restore() is the only thing standing between a Space restart and
the permanent loss of every save, EWMA profile, cluster and onboarding record.
It had no test.

Nothing here touches the network. The single HTTP boundary is _execute(), so
the fake below replaces exactly that and everything above it runs for real:
the SQLite writes, the empty-table guard, the watermark advance, and the Hrana
value codec.

The codec tests are the sharp end. A user_profiles row carries a 1024-dim
float32 EWMA vector as a BLOB, and Turso returns base64 with the padding
stripped and occasionally in the URL-safe alphabet. If that decode is wrong the
restore still "succeeds" -- it just silently hands every user a corrupted
profile, which is worse than losing the row outright, because nothing errors.
"""
import base64

import numpy as np
import pytest

from app import turso_sync
from app.turso_sync import _b64decode, _cell, _uncell


# ── Hrana value codec ────────────────────────────────────────────────────────

def test_blob_roundtrip_is_byte_exact_for_a_real_profile_vector():
    """A 1024-dim float32 EWMA vector must survive _cell -> _uncell unchanged."""
    vec = np.random.default_rng(0).random(1024, dtype=np.float32)
    raw = vec.tobytes()
    assert len(raw) == 4096

    cell = _cell(raw)
    # Hrana keys blob payloads as "base64", not "value"; using "value" gets a
    # bare 400 back from Turso with no explanation.
    assert cell["type"] == "blob"
    assert "base64" in cell

    back = _uncell(cell)
    assert back == raw
    assert np.array_equal(np.frombuffer(back, dtype=np.float32), vec)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5, 4096])
def test_blob_roundtrip_at_every_base64_padding_alignment(size):
    """Lengths mod 3 in {0,1,2} produce 0, 2 and 1 padding chars respectively."""
    raw = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    raw = raw[:size]
    assert _uncell(_cell(raw)) == raw


def test_b64decode_accepts_stripped_padding():
    """Turso strips '=' padding; a plain b64decode raises Incorrect padding."""
    raw = b"researchit"  # 10 bytes -> encodes with two '=' of padding
    encoded = base64.b64encode(raw).decode()
    assert encoded.endswith("==")
    assert _b64decode(encoded.rstrip("=")) == raw


def test_b64decode_accepts_urlsafe_alphabet():
    """Payloads occasionally come back in the URL-safe alphabet ('-' and '_')."""
    # Chosen so the standard encoding contains both '+' and '/'.
    raw = bytes([0xFB, 0xEF, 0xBE, 0xFF, 0xE0])
    std = base64.b64encode(raw).decode()
    url = base64.urlsafe_b64encode(raw).decode()
    assert ("+" in std or "/" in std) and std != url
    assert _b64decode(url.rstrip("=")) == raw
    assert _b64decode(std) == raw


def test_int_and_float_cells_survive_the_roundtrip():
    """importance is REAL and interaction_count is INTEGER; both are restored."""
    assert _uncell(_cell(7)) == 7
    assert _uncell(_cell(0.625)) == pytest.approx(0.625)
    assert _uncell(_cell(None)) is None
    assert _uncell(_cell("cs.CL")) == "cs.CL"


# ── restore() ────────────────────────────────────────────────────────────────

@pytest.fixture
def local_db(tmp_path, monkeypatch):
    """A real, empty local SQLite database at a temp DB_PATH."""
    import asyncio

    import app.config as cfg
    import app.db as db_mod

    path = str(tmp_path / "restore.db")
    monkeypatch.setattr(cfg, "DB_PATH", path)
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    monkeypatch.setattr(turso_sync.config, "DB_PATH", path, raising=False)
    asyncio.run(db_mod.init_db())
    return path


def _fake_remote(monkeypatch, rows_by_table: dict[str, list[list]]):
    """Replace the one HTTP boundary with canned rows, keyed by remote table."""
    calls: list[str] = []

    async def fake_execute(stmts, timeout=60):
        out = []
        for s in stmts:
            sql = s["sql"]
            calls.append(sql)
            table = sql.rsplit(" FROM ", 1)[-1].strip() if " FROM " in sql else ""
            out.append(rows_by_table.get(table, []))
        return out

    monkeypatch.setattr(turso_sync, "_execute", fake_execute)
    return calls


async def test_restore_populates_empty_tables(local_db, monkeypatch):
    """The core promise: a blank container comes back with the user's data."""
    vec = np.random.default_rng(1).random(1024, dtype=np.float32).tobytes()
    _fake_remote(monkeypatch, {
        "user_interactions": [
            [1, "u1", "1706.03762", "save", "search", 0, "q1", "v1", "dense",
             None, 1.0, "p1", "2026-08-01T00:00:00"],
        ],
        "user_profiles": [["u1", "longterm", vec, 5, "2026-08-01T00:00:00"]],
        "user_onboarding": [["u1", '["cs.CL"]', 1, "2026-08-01", "2026-08-01"]],
    })

    restored = await turso_sync.restore()
    assert restored["interactions"] == 1
    assert restored["user_profiles"] == 1
    assert restored["user_onboarding"] == 1

    import aiosqlite
    async with aiosqlite.connect(local_db) as conn:
        cur = await conn.execute("SELECT paper_id, propensity FROM interactions")
        assert await cur.fetchone() == ("1706.03762", 1.0)

        cur = await conn.execute(
            "SELECT vector, interaction_count FROM user_profiles WHERE user_id='u1'")
        got_vec, count = await cur.fetchone()
        # The whole point: the profile is byte-identical, not merely present.
        assert bytes(got_vec) == vec
        assert count == 5


async def test_restore_never_clobbers_a_table_that_already_has_rows(local_db, monkeypatch):
    """restore() runs on every boot, so it must be a no-op for a warm container.

    If this regresses, a restart mid-session overwrites a user's live profile
    with an older snapshot from Turso.
    """
    import aiosqlite
    async with aiosqlite.connect(local_db) as conn:
        await conn.execute(
            "INSERT INTO user_onboarding (user_id, selected_categories, "
            "onboarding_completed, created_at, updated_at) VALUES "
            "('local-user', '[\"cs.LG\"]', 1, 'now', 'now')")
        await conn.commit()

    _fake_remote(monkeypatch, {
        "user_onboarding": [["remote-user", '["cs.CV"]', 1, "old", "old"]],
    })

    restored = await turso_sync.restore()
    assert "user_onboarding" not in restored

    async with aiosqlite.connect(local_db) as conn:
        cur = await conn.execute("SELECT user_id FROM user_onboarding")
        rows = [r[0] for r in await cur.fetchall()]
    assert rows == ["local-user"]


async def test_restore_advances_the_watermark_past_restored_rows(local_db, monkeypatch):
    """Restored rows came FROM Turso and must not be pushed straight back.

    Without this the first sync after every restart re-uploads the entire
    history, burning quota and growing without bound.
    """
    _fake_remote(monkeypatch, {
        "user_interactions": [
            [41, "u1", "a", "save", "search", 0, "", "", "", None, 1.0, "", "t"],
            [42, "u1", "b", "save", "search", 1, "", "", "", None, 1.0, "", "t"],
        ],
    })

    await turso_sync.restore()

    import aiosqlite
    async with aiosqlite.connect(local_db) as conn:
        wm = await turso_sync._get_wm(conn, "interactions")
    # MAX(id) over the restored rows, so the next sync selects strictly above it.
    assert wm == "42"


async def test_restore_survives_a_remote_failure_without_losing_local_data(
        local_db, monkeypatch):
    """A Turso outage at boot must degrade to 'no restore', never to a crash.

    start() calls restore() inside the app lifespan; an exception escaping here
    takes the whole Space down on boot.
    """
    async def boom(stmts, timeout=60):
        raise RuntimeError("turso unreachable")

    monkeypatch.setattr(turso_sync, "_execute", boom)

    restored = await turso_sync.restore()
    assert restored == {}


# ── Replication kill switch ──────────────────────────────────────────────────

def test_sync_can_be_disabled_while_reads_stay_live(monkeypatch):
    """TURSO_SYNC_DISABLED makes a session read-only for user data.

    Reads and replication share one credential pair, so a developer running a
    local server against real credentials -- needed to exercise search or the
    feed -- also silently gets replication, and the shutdown flush pushes that
    session's rows into production. This flag is what makes it safe to point a
    dev server at live Turso.
    """
    monkeypatch.setattr(turso_sync.config, "TURSO_URL", "https://example.turso.io")
    monkeypatch.setattr(turso_sync.config, "TURSO_DB_TOKEN", "token")

    monkeypatch.delenv("TURSO_SYNC_DISABLED", raising=False)
    assert turso_sync.enabled() is True

    monkeypatch.setenv("TURSO_SYNC_DISABLED", "1")
    assert turso_sync.enabled() is False


async def test_disabled_sync_neither_restores_nor_pushes(local_db, monkeypatch):
    """start() and stop() must both no-op, not just the periodic loop."""
    monkeypatch.setattr(turso_sync.config, "TURSO_URL", "https://example.turso.io")
    monkeypatch.setattr(turso_sync.config, "TURSO_DB_TOKEN", "token")
    monkeypatch.setenv("TURSO_SYNC_DISABLED", "1")

    called = []

    async def tripwire(stmts, timeout=60):
        called.append(stmts)
        raise AssertionError("network call made while sync was disabled")

    monkeypatch.setattr(turso_sync, "_execute", tripwire)

    await turso_sync.start()
    await turso_sync.stop()
    assert called == []
