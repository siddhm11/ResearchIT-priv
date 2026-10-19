"""
The wide feed (FEED_WIDE), the layout column on interactions, feed-control
events, and the figure warm-up worker (FIGURE_PREWARM).

The layout column is the risky part: interactions replicate to Turso, and a
column the remote doesn't have would stop every push (trap T8). So the remote
ALTER, its duplicate-column tolerance and the push/restore round trip are all
tested against the fake _execute.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app import config, db, figure_prewarm, paper_source_svc, turso_sync


async def rows(sql, *args):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        return await (await conn.execute(sql, args)).fetchall()


# ── the layout column ────────────────────────────────────────────────────────

async def test_interactions_gain_a_layout_column_and_store_only_known_values():
    await db.log_interaction("u", "2609.1", "save", layout="grid2")
    await db.log_interaction("u", "2609.2", "save", layout="list")
    await db.log_interaction("u", "2609.3", "save", layout="<script>")
    await db.log_interaction("u", "2609.4", "save")
    assert [r[0] for r in await rows("SELECT layout FROM interactions ORDER BY id")] == ["grid2", "list", None, None]


def test_save_pass_and_click_carry_the_layout():
    from app import qdrant_svc, turso_svc
    from app.main import app

    async def meta(ids):
        return {i: {"arxiv_id": i, "title": "T", "abstract": "A.", "authors": "[]"} for i in ids}

    async def nothing(*a, **k):
        return {}

    with patch.object(turso_svc, "fetch_metadata_batch", side_effect=meta), \
            patch.object(qdrant_svc, "get_paper_vectors", side_effect=nothing), TestClient(app) as c:
        c.cookies.set(config.COOKIE_NAME, "reader-1")
        c.post("/api/papers/2609.00001/save", data={"source": "recommendation", "layout": "grid2"})
        c.post("/api/papers/2609.00002/not-interested", data={"source": "recommendation", "layout": "list"})
        c.get("/p/2609.00003?qid=q1&sf=recommendation&pos=0&lay=grid2")
    got = asyncio.run(rows("SELECT paper_id, event_type, layout FROM interactions ORDER BY id"))
    assert ("2609.00001", "save", "grid2") in got
    assert ("2609.00002", "not_interested", "list") in got
    assert ("2609.00003", "click", "grid2") in got


def test_layout_is_replicated():
    assert turso_sync.TABLES["interactions"]["cols"][-1] == "layout"
    ddl = next(d for d in turso_sync.REMOTE_DDL if "user_interactions" in d and "CREATE TABLE" in d)
    assert "layout TEXT" in ddl
    assert ("user_interactions", "layout", "TEXT") in turso_sync.REMOTE_COLUMNS


async def test_the_remote_column_is_added_once_and_a_duplicate_is_fine(monkeypatch):
    sent = []

    async def fake(stmts, timeout=60):
        for s in stmts:
            sent.append(s["sql"])
            if s["sql"].startswith("ALTER TABLE"):
                raise RuntimeError("{'message': 'SQLite error: duplicate column name: layout'}")
        return [[] for _ in stmts]

    monkeypatch.setattr(turso_sync, "_execute", fake)
    await turso_sync.ensure_remote_schema()
    assert sum(s.startswith("ALTER TABLE user_interactions ADD COLUMN layout") for s in sent) == 1


async def test_any_other_alter_error_stops_replication_start(monkeypatch):
    async def fake(stmts, timeout=60):
        if any(s["sql"].startswith("ALTER TABLE") for s in stmts):
            raise RuntimeError("HTTP 500: remote unavailable")
        return [[] for _ in stmts]

    monkeypatch.setattr(turso_sync, "_execute", fake)
    with pytest.raises(RuntimeError):
        await turso_sync.ensure_remote_schema()


async def test_pushed_rows_include_the_layout(monkeypatch):
    pushed = []

    async def fake(stmts, timeout=60):
        for s in stmts:
            if s["sql"].startswith("INSERT OR REPLACE INTO user_interactions"):
                pushed.append(s["args"][-1])
        return [[] for _ in stmts]

    monkeypatch.setattr(turso_sync, "_execute", fake)
    monkeypatch.setattr(turso_sync, "_no_push", set())       # earlier tests' restores leave it set
    await db.log_interaction("u", "2609.1", "save", layout="grid2")
    await turso_sync.sync_once()
    assert pushed == [{"type": "text", "value": "grid2"}]


# ── feed controls ────────────────────────────────────────────────────────────

def test_feed_control_events_are_stored_for_known_events_only():
    from app.main import app
    with TestClient(app) as c:
        c.cookies.set(config.COOKIE_NAME, "reader-1")
        assert c.post("/api/ui-event", data={"event": "filter_on", "value": "Drone navigation", "query_id": "q"}).status_code == 204
        c.post("/api/ui-event", data={"event": "delete_everything"})
        c.cookies.clear()
        c.post("/api/ui-event", data={"event": "cardtext", "value": "abs"})
    assert asyncio.run(rows("SELECT user_id, event, value, query_id FROM ui_events")) == \
        [("reader-1", "filter_on", "Drone navigation", "q")]


def test_the_wide_feed_page_has_a_rail_only_with_its_switch(monkeypatch):
    from app.main import app

    async def done(user_id):
        return {"onboarding_completed": 1}

    with patch.object(db, "get_onboarding_state", side_effect=done), TestClient(app) as c:
        monkeypatch.setenv("FEED_WIDE", "on")
        wide = c.get("/").text
        monkeypatch.setenv("FEED_WIDE", "off")
        narrow = c.get("/").text
    assert "data-wide-feed" in wide and 'id="rail-ints"' in wide and 'class="page-wide"' in wide
    assert "data-wide-feed" not in narrow and "wrap--feed" in narrow


# ── figure warm-up ───────────────────────────────────────────────────────────

@pytest.fixture
def warmup(monkeypatch):
    calls = {"ensure": [], "thumbs": []}

    async def fake_recent(days=2):
        return [("2609.00001", "CC-BY-4.0"), ("2609.00002", "ARXIV-NONEXCLUSIVE-1.0"),
                ("2609.00003", "CC-BY-4.0"), ("2609.00004", "")]

    async def fake_ensure(arxiv_id, force=False):
        calls["ensure"].append(arxiv_id)
        if arxiv_id == "2609.00003":
            raise RuntimeError("boom")
        return SimpleNamespace(arxiv_id=arxiv_id, ok=True, figures=[{"src": f"{arxiv_id}v1/a.png"}],
                               figures_allowed=True)

    async def fake_thumb(source):
        calls["thumbs"].append(source.arxiv_id)
        return True

    monkeypatch.setattr(figure_prewarm, "recent_ids", fake_recent)
    monkeypatch.setattr(paper_source_svc, "ensure", fake_ensure)
    monkeypatch.setattr(figure_prewarm, "_thumbnail", fake_thumb)
    monkeypatch.setenv("FIGURES", "on")
    return calls


async def test_a_run_warms_new_papers_and_survives_a_failure(warmup):
    counts = await figure_prewarm.run_once()
    assert counts == {"done": 3, "skipped": 0, "failed": 1, "stopped_by_breaker": False}
    assert warmup["thumbs"] == ["2609.00001", "2609.00002", "2609.00004"]


async def test_papers_already_read_are_skipped(warmup):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute("INSERT INTO paper_sources (arxiv_id, status) VALUES ('2609.00001', 'ok')")
        await conn.commit()
    counts = await figure_prewarm.run_once()
    assert counts["skipped"] == 1 and "2609.00001" not in warmup["ensure"]


async def test_the_breaker_stops_a_run(warmup, monkeypatch):
    monkeypatch.setattr(paper_source_svc, "breaker_open", lambda: True)
    counts = await figure_prewarm.run_once()
    assert counts["stopped_by_breaker"] and warmup["ensure"] == []


async def test_a_run_handles_at_most_its_batch(warmup):
    await figure_prewarm.run_once(limit=2)
    assert len(warmup["ensure"]) == 2


async def test_it_is_off_by_default_and_only_one_runner_holds_the_lock(monkeypatch, tmp_path):
    monkeypatch.delenv("FIGURE_PREWARM", raising=False)
    await figure_prewarm.start()
    assert figure_prewarm._task is None
    monkeypatch.setattr(figure_prewarm, "LOCK_PATH", tmp_path / "prewarm.lock")
    monkeypatch.setattr(figure_prewarm, "FIRST_RUN_DELAY_S", 3600)
    monkeypatch.setenv("FIGURE_PREWARM", "on")
    import fcntl
    other = (tmp_path / "prewarm.lock").open("a")
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    await figure_prewarm.start()
    assert figure_prewarm._task is None and "lock" in figure_prewarm.status()["scheduler"]
    fcntl.flock(other, fcntl.LOCK_UN)
    other.close()
    await figure_prewarm.start()
    assert figure_prewarm._task is not None
    await figure_prewarm.stop()
    assert figure_prewarm._task is None
