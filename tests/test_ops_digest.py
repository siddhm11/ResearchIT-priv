"""The daily operations digest survives restarts (audit F25)."""
import asyncio
import sqlite3

import pytest

from app import db, errors, ops_digest, runtime_stats, turso_sync


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    import app.config as cfg
    path = str(tmp_path / "ops.db")
    for mod in (cfg, db):
        monkeypatch.setattr(mod, "DB_PATH", path)
    asyncio.run(db.init_db())
    ops_digest._counts.clear()
    ops_digest.recent.clear()
    return path


def _rows(path):
    return dict(sqlite3.connect(path).execute("SELECT metric, n FROM ops_daily").fetchall())


def test_a_reported_error_is_counted_per_module_and_kept_per_day(fresh):
    before = runtime_stats.counters().get("error:feed", 0)
    for _ in range(3):
        errors.report("feed", "Tier 1 failed", RuntimeError("boom"))
    assert runtime_stats.counters()["error:feed"] == before + 3
    assert ops_digest.recent[-1]["module"] == "feed" and "boom" in ops_digest.recent[-1]["message"]
    asyncio.run(ops_digest.flush())
    errors.report("feed", "again", RuntimeError("x"))
    asyncio.run(ops_digest.flush())
    assert _rows(fresh)["error:feed"] == 4            # added across flushes, not replaced


def test_times_are_gauges_not_sums(fresh):
    runtime_stats.record("search", {"total_ms": 1200})
    asyncio.run(ops_digest.flush())
    asyncio.run(ops_digest.flush())
    rows = _rows(fresh)
    assert "search_p95_ms" in rows and rows["search_p95_ms"] < 10_000


def test_the_owner_sees_the_last_days(fresh):
    ops_digest.count("boot")
    asyncio.run(ops_digest.flush())
    days = asyncio.run(ops_digest.last_days(7))
    assert list(days.values())[0]["boot"] == 1


def test_the_digest_is_replicated():
    spec = turso_sync.TABLES["ops_daily"]
    assert spec["watermark"] == "id" and spec["paged"]
    ddl = next(d for d in turso_sync.REMOTE_DDL if "EXISTS ops_daily (" in d)
    assert all(c in ddl for c in spec["cols"])
