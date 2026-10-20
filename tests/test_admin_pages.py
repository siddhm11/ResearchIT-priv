"""
The owner's readers, searches and AI history pages (app/routers/admin.py over
app/analytics.py). Owner only; the owner's own use is left out unless ?me=1;
model-written text is shown as text.
"""
import asyncio

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app import config

TOKEN = "?preview=owner-token"


@pytest.fixture(autouse=True)
def owner(monkeypatch):
    monkeypatch.setattr(config, "PREVIEW_TOKEN", "owner-token")


def run(*statements):
    async def go():
        async with aiosqlite.connect(config.DB_PATH) as conn:
            for sql, args in statements:
                await conn.execute(sql, args)
            await conn.commit()
    asyncio.run(go())


def get(path):
    from app.main import app
    with TestClient(app) as c:
        return c.get(path)


@pytest.mark.parametrize("path", ["/admin/readers", "/admin/searches", "/admin/ai/history"])
def test_every_page_is_owner_only(path):
    assert get(path).status_code == 404
    page = get(path + TOKEN)
    assert page.status_code == 200 and 'content="noindex"' in page.text and 'aria-current="page"' in page.text


# ── readers ──────────────────────────────────────────────────────────────────

def seed_readers():
    rd = "INSERT INTO reader_days (day, user_id, owner, pages, searches, opens, saves, asks) VALUES (date('now', ?), ?, ?, ?, ?, ?, ?, ?)"
    ix = "INSERT INTO interactions (user_id, paper_id, event_type, source, timestamp) VALUES (?, '2609.00001', ?, ?, ?)"
    run((rd, ("-0 day", "alice", 0, 3, 2, 1, 1, 1)),
        (rd, ("-3 day", "alice", 0, 1, 1, 0, 0, 0)),
        (rd, ("-0 day", "bob", 0, 1, 0, 0, 0, 0)),
        (rd, ("-0 day", "me", 1, 9, 9, 9, 9, 9)),
        (ix, ("alice", "save", "search", "2026-08-10 10:00:00")),
        (ix, ("carol", "save", "search", "2026-08-11 10:00:00")),
        (ix, ("me", "save", "recommendation", "2026-08-12 10:00:00")))


def test_readers_page_counts_readers_and_leaves_the_owner_out():
    seed_readers()
    html = get("/admin/readers" + TOKEN).text
    assert "Your own use is left out." in html
    assert '<span class="ai-kpi-label">Readers today</span><span class="ai-kpi-value">2</span>' in html
    assert "1 of 2 readers (50%)" in html                       # alice searched; bob didn't
    assert '<span class="ai-kpi-label">Came back another day (30 days)</span><span class="ai-kpi-value">50%</span>' in html
    assert "Aug 2026" in html and "recommendation" not in html  # the owner's save from the feed is left out
    with_me = get("/admin/readers?me=1&preview=owner-token").text
    assert '<span class="ai-kpi-label">Readers today</span><span class="ai-kpi-value">3</span>' in with_me
    assert "recommendation" in with_me


# ── searches ─────────────────────────────────────────────────────────────────

def seed_searches():
    s = ("INSERT INTO search_log (query, query_id, n_results, mode, rewritten, latency_ms, owner) "
         "VALUES (?, ?, ?, ?, ?, ?, ?)")
    ix = "INSERT INTO interactions (user_id, paper_id, event_type, source, query_id) VALUES ('r', '2609.00001', ?, 'search', ?)"
    run((s, ("drone navigation", "q1", 10, "hybrid", "UAV vision-language navigation", 900, 0)),
        (s, ("Drone navigation", "q2", 10, "hybrid", None, 800, 0)),
        (s, ("xqzzv", "q3", 0, "keyword", None, 300, 0)),
        (s, ("owner test", "q4", 5, "hybrid", None, 100, 1)),
        (ix, ("save", "q1")), (ix, ("click", "q2")),
        ("INSERT INTO search_overviews (cache_key, query, overview) VALUES ('k', 'drone navigation', ?)",
         ("<strong>Drones</strong> navigate <script>alert(1)</script>",)))


def test_searches_page_shows_volume_queries_and_what_came_of_them():
    seed_searches()
    html = get("/admin/searches" + TOKEN).text
    assert '<span class="ai-kpi-label">Searches</span><span class="ai-kpi-value">3</span>' in html
    assert '<span class="ai-kpi-label">Found nothing</span><span class="ai-kpi-value">33%</span>' in html
    assert '<span class="ai-kpi-label">Led to a save</span><span class="ai-kpi-value">33%</span>' in html
    assert "drone navigation" in html and "UAV vision-language navigation" in html and "xqzzv" in html
    assert "owner test" not in html                                     # the owner's search is left out
    assert "<script>alert(1)</script>" not in html and "<strong>Drones" not in html   # model text as text
    assert "Drones navigate" in html


# ── AI history ───────────────────────────────────────────────────────────────

def test_history_shows_each_paper_and_search_with_what_was_written():
    run(("INSERT INTO llm_calls (feature, status, subject, tokens_in, tokens_out) VALUES ('explain', 'ok', '2609.31507', 600, 180)", ()),
        ("INSERT INTO llm_calls (feature, status, subject, tokens_in, tokens_out) VALUES ('briefing', 'ok', '2609.31507', 3300, 400)", ()),
        ("INSERT INTO llm_calls (feature, status, subject) VALUES ('briefing', 'unavailable', '2609.31507')", ()),
        ("INSERT INTO llm_calls (feature, status, subject, tokens_in, tokens_out) VALUES ('search_rewrite', 'ok', 'drone navigation', 380, 12)", ()),
        ("INSERT INTO paper_metadata (arxiv_id, title) VALUES ('2609.31507', 'SatNav')", ()),
        ("INSERT INTO paper_explanations (cache_key, arxiv_id, explanation, model) VALUES ('c', '2609.31507', 'Plain <b>words</b>.', 'm')", ()),
        ("INSERT INTO paper_briefings (cache_key, arxiv_id, status) VALUES ('b', '2609.31507', 'ok')", ()),
        ("INSERT INTO search_log (query, rewritten) VALUES ('drone navigation', 'UAV vision-language navigation')", ()))
    html = get("/admin/ai/history" + TOKEN).text
    assert 'href="/p/2609.31507">SatNav</a>' in html and "Plain words." in html and "<b>words" not in html
    assert "2 calls · written" in html and "4,480" in html          # two briefing calls; tokens summed
    assert "drone navigation" in html and "UAV vision-language navigation" in html
