"""
app/analytics.py: the search log, the AI overviews shown for searches (also
their cache), and the 90-day clearing of what people typed.
"""
import asyncio
from unittest.mock import patch

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app import analytics, config, hybrid_search_svc, turso_svc


async def rows(sql, *args):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        return await (await conn.execute(sql, args)).fetchall()


async def execute(sql, *args):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(sql, args)
        await conn.commit()


# ── the search log ───────────────────────────────────────────────────────────

async def test_a_search_is_logged_collapsed_without_a_reader():
    await analytics.log_search("  drone \n navigation ", "q-1", 10, "hybrid", "UAV vision-language navigation", 812, False)
    await analytics.log_search("transformers", "q-2", 0, "keyword", "transformers", 90, True)
    await analytics.log_search("   ", "q-3", 0, "hybrid", None, 5, False)            # nothing typed: not logged
    assert await rows("SELECT query, query_id, n_results, mode, rewritten, latency_ms, owner FROM search_log") == [
        ("drone navigation", "q-1", 10, "hybrid", "UAV vision-language navigation", 812, 0),
        ("transformers", "q-2", 0, "keyword", None, 90, 1)]               # a rewrite that changed nothing isn't kept
    cols = [r[1] for r in await rows("PRAGMA table_info(search_log)")]
    assert "user_id" not in cols


def test_the_search_route_logs_each_search_with_its_results_query_id():
    from app.main import app

    async def fake_search(q, limit=20, return_meta=False):
        return ["2609.00001", "2609.00002"], {"retrieval_mode": "keyword", "rewritten_query": None,
                                              "total_time_ms": 40}

    async def meta(ids):
        return {i: {"arxiv_id": i, "title": "T", "abstract": "A.", "authors": "[]"} for i in ids}

    with patch.object(hybrid_search_svc, "search", side_effect=fake_search), \
            patch.object(turso_svc, "fetch_metadata_batch", side_effect=meta), TestClient(app) as c:
        html = c.get("/search?q=drone+navigation", headers={"HX-Request": "true"}).text
    (query, qid, n, mode), = asyncio.run(rows("SELECT query, query_id, n_results, mode FROM search_log"))
    assert (query, n, mode) == ("drone navigation", 2, "keyword") and f"qid={qid}" in html


# ── overviews and their cache ────────────────────────────────────────────────

def overview_client(monkeypatch, calls):
    from app import groq_svc

    async def fake_summary(q, papers):
        calls.append(q)
        return f"<strong>{q}</strong> overview"

    async def meta(ids):
        return {i: {"arxiv_id": i, "title": "T", "abstract": "A."} for i in ids}

    monkeypatch.setattr(groq_svc, "generate_search_summary", fake_summary)
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", meta)
    from app.main import app
    return TestClient(app)


def test_the_same_search_over_the_same_papers_reuses_its_overview(monkeypatch):
    calls = []
    with overview_client(monkeypatch, calls) as c:
        a = c.get("/api/search/summary?q=drone+navigation&ids=2609.00001,2609.00002").text
        b = c.get("/api/search/summary?q=Drone   Navigation&ids=2609.00001,2609.00002").text
        c.get("/api/search/summary?q=drone+navigation&ids=2609.00009,2609.00002")      # other papers: new call
    assert calls == ["drone navigation", "drone navigation"] and "overview" in a and a == b
    assert asyncio.run(rows("SELECT query, paper_ids FROM search_overviews ORDER BY id")) == [
        ("drone navigation", "2609.00001,2609.00002"), ("drone navigation", "2609.00009,2609.00002")]


def test_a_week_old_overview_is_written_again(monkeypatch):
    calls = []
    with overview_client(monkeypatch, calls) as c:
        c.get("/api/search/summary?q=drones&ids=2609.00001")
        asyncio.run(execute("UPDATE search_overviews SET created_at = datetime('now', '-8 day')"))
        c.get("/api/search/summary?q=drones&ids=2609.00001")
    assert calls == ["drones", "drones"]


# ── clearing what people typed ───────────────────────────────────────────────

async def test_after_90_days_query_text_goes_and_the_counts_stay(monkeypatch):
    monkeypatch.setattr(analytics, "_tasks", [])
    for age, q in ((100, "old query"), (10, "new query")):
        await execute("INSERT INTO search_log (created_at, query, rewritten, n_results) VALUES "
                      f"(datetime('now', '-{age} day'), ?, 'rewritten', 3)", q)
        await execute("INSERT INTO search_overviews (cache_key, query, overview, created_at) VALUES "
                      f"(?, ?, 'o', datetime('now', '-{age} day'))", q, q)
        await execute("INSERT INTO llm_calls (feature, status, subject, created_at) VALUES "
                      f"('search_overview', 'ok', ?, datetime('now', '-{age} day'))", q)
        await execute("INSERT INTO llm_calls (feature, status, subject, created_at) VALUES "
                      f"('explain', 'ok', '2609.00001', datetime('now', '-{age} day'))")
    await analytics.housekeep()
    assert await rows("SELECT query, rewritten, n_results FROM search_log ORDER BY created_at") == [
        (None, None, 3), ("new query", "rewritten", 3)]
    assert await rows("SELECT query FROM search_overviews") == [("new query",)]
    assert await rows("SELECT feature, subject FROM llm_calls ORDER BY created_at, feature") == [
        ("explain", "2609.00001"), ("search_overview", None),           # papers keep their id
        ("explain", "2609.00001"), ("search_overview", "new query")]


async def test_the_clearing_runs_in_turso_too(monkeypatch):
    from app import turso_sync
    sent = []

    async def fake(stmts, timeout=60):
        sent.extend(s["sql"] for s in stmts)
        return [[] for _ in stmts]

    monkeypatch.setattr(turso_sync, "enabled", lambda: True)
    monkeypatch.setattr(turso_sync, "_execute", fake)
    await analytics.housekeep()
    assert len(sent) == 3 and all("-90 day" in s for s in sent)


def test_search_tables_replicate():
    from app import turso_sync
    assert "query" in turso_sync.TABLES["search_log"]["cols"]
    assert "overview" in turso_sync.TABLES["search_overviews"]["cols"]
    assert any("search_overviews" in d and "cache_key TEXT NOT NULL UNIQUE" in d for d in turso_sync.REMOTE_DDL)
