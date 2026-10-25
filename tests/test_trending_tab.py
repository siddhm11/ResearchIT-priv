"""
The Trending tab (doc 06, 2026-10-25): Trending takes Collections' place in the
main bar and Collections moves into Library; /trending says which week it
shows, leads with a strip of moving topics and the reader's areas, and lists
steady topics as steady; every named topic has a page.

What must hold: the bar never links to a page that is off; a steady topic never
reads as moving; a topic page shows only named topics, keeps vector-search
papers apart from the lists and leaves out what the reader saved; the shared
topic state is never written by a request.
"""
from datetime import date
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest

from app import db, qdrant_svc, trending_shelf as ts, trending_topics as tt, user_state as us
from app.main import app
from app.templates_env import one_in

READER = "tab-reader"


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", cookies={"arxiv_user_id": READER})


def unit(axis: int) -> np.ndarray:
    v = np.zeros(1024, dtype=np.float32)
    v[axis] = 1.0
    return v


def topic(tid, name, q, slope, frm, to, axis, papers=100):
    return {"id": tid, "name": name, "q": q, "slope": slope, "from_share": frm, "to_share": to,
            "also": ["x", "y"] if name else [], "papers": papers, "coverage": 0.4,
            "rolling": [frm, (frm + to) / 2, to], "_medoid": unit(axis), "medoid_paper_id": f"m{tid}",
            "top": [{"arxiv_id": f"2606.0{tid}001", "title": f"Top of {name}", "upvotes": 300}],
            "best": [{"arxiv_id": f"2606.0{tid}001", "title": f"Top of {name}", "upvotes": 300,
                      "listed_on": "2026-06-10"},
                     {"arxiv_id": f"2606.0{tid}002", "title": f"Second of {name}", "upvotes": 90,
                      "listed_on": "2026-06-11"}]}


TOPICS = [topic(1, "World Models", 0.006, 0.50, 0.012, 0.039, 1),
          topic(2, "Reinforcement Learning", 0.04, -0.40, 0.066, 0.035, 2),
          topic(3, "Diffusion", 0.27, -0.26, 0.050, 0.051, 3),
          topic(4, "Speech", 0.92, -0.02, 0.021, 0.019, 4),
          topic(5, None, 0.99, 0.0, 0.08, 0.086, 5)]


@pytest.fixture
async def tab_world(monkeypatch):
    await db.init_db()
    us._cache.clear()
    monkeypatch.setattr(tt, "_state", {**tt._state, "topics": [dict(t) for t in TOPICS],
                                       "topic_of": {"2610.11111": 1, "2606.01001": 1, "2606.01002": 1},
                                       "first_week": "2026-04-27", "weeks": 26, "papers": 5176,
                                       "computed_at": "2026-10-25T11:50:35+00:00"})
    papers = {"2610.11111": {"arxiv_id": "2610.11111", "title": "A world model this week", "upvotes": 40,
                             "listed_on": "2026-10-22", "published_at": "2026-10-20T00:00:00+00:00",
                             "day_rank": 3, "day_size": 58, "week_percentile": .9, "week_size": 312,
                             "summary": "", "keywords": [], "categories": ["cs.RO"]},
              "2610.22222": {"arxiv_id": "2610.22222", "title": "Listed after today's grouping", "upvotes": 9,
                             "listed_on": "2026-10-23", "published_at": "2026-10-20T00:00:00+00:00",
                             "day_rank": 9, "day_size": 50, "week_percentile": .4, "week_size": 312,
                             "summary": "", "keywords": [], "categories": ["cs.CL"]}}
    monkeypatch.setattr(ts, "_state", {**ts._state, "papers": papers, "code": {},
                                       "vectors": {"2610.22222": unit(1) * .9 + unit(9) * .1}})
    monkeypatch.setenv("TRENDING_PAGE", "on")
    import app.routers.trending as tr
    tr._more_cache.clear()
    yield
    us._cache.clear()


# ── The bar and Library ──────────────────────────────────────────────────────

async def test_trending_takes_collections_place_in_both_bars(tab_world):
    async with client() as c:
        html = (await c.get("/saved")).text
    assert 'href="/trending" class="nav-link"' in html and 'href="/trending" class="botnav-item"' in html
    assert 'href="/collections" class="nav-link"' not in html
    assert 'href="/collections" class="botnav-item"' not in html
    assert 'class="library-tab"' in html and 'href="/collections" class="library-tab"' in html


async def test_without_the_trending_page_the_bar_keeps_collections(tab_world, monkeypatch):
    monkeypatch.setenv("TRENDING_PAGE", "off")
    async with client() as c:
        html = (await c.get("/saved")).text
    assert 'href="/collections" class="nav-link"' in html and 'href="/trending"' not in html
    assert 'href="/collections" class="library-tab"' not in html


async def test_library_is_current_on_collections_and_recent(tab_world):
    async with client() as c:
        for url, tab in (("/collections", "Collections"), ("/history", "Recent"), ("/saved", "Saved")):
            html = (await c.get(url)).text
            assert 'href="/saved" class="nav-link" aria-current="page"' in html, url
            assert f'aria-current="page">{tab}</a>' in html, url


# ── /trending ────────────────────────────────────────────────────────────────

async def test_the_tab_says_its_week_and_leads_with_topics_and_areas(tab_world, monkeypatch):
    await db.save_onboarding_categories(READER, ["robotics"])
    monkeypatch.setattr(ts, "window", lambda today=None: {"label": "19–23 Oct", "next": "Mon 26 Oct"})
    async with client() as c:
        html = (await c.get("/trending")).text
    assert "Lists of 19–23 Oct" in html and "next list Mon 26 Oct" in html
    chips = html[html.index('class="topic-chips"'):html.index("</nav>", html.index('class="topic-chips"'))]
    assert 'href="/trending/topics/1"' in chips and "World Models" in chips and "↑" in chips
    assert 'href="/trending/topics/2"' in chips and "↓" in chips
    assert 'href="#tp-area-robotics"' in chips and "Robotics · 1" in chips


async def test_steady_topics_show_as_steady_and_never_as_moving(tab_world):
    async with client() as c:
        html = (await c.get("/trending")).text
    steady = html[html.index('id="tp-steady"'):]
    steady = steady[:steady.index("</section>")]
    assert "Diffusion" in steady and "Speech" in steady and "World Models" not in steady
    assert steady.index("Diffusion") < steady.index("Speech"), "biggest share first"
    assert "5.1% · 1 in 20" in steady and "→" not in steady, "no then-and-now for a steady topic"
    assert "1 in 83 → 1 in 26 papers" in html, "moving topics in plain numbers too"
    assert html.count('class="topic-row topic-row--') == 4, "the unnamed topic stays out"
    assert '<a href="/trending/topics/3">Diffusion</a>' in html


def test_plain_numbers():
    assert [one_in(s) for s in (0.027, 0.055, 0.012, 0.0)] == ["1 in 37", "1 in 18", "1 in 83", "none"]


def test_the_window_names_the_lists_and_the_next_one_at_weekends(monkeypatch):
    monkeypatch.setattr(ts, "_state", {**ts._state, "papers": {
        "a": {"listed_on": "2026-10-19"}, "b": {"listed_on": "2026-10-23"}}})
    assert ts.window(date(2026, 10, 25)) == {"label": "19–23 Oct", "next": "Mon 26 Oct"}
    assert ts.window(date(2026, 10, 22)) == {"label": "19–23 Oct", "next": None}
    monkeypatch.setattr(ts, "_state", {**ts._state, "papers": {
        "a": {"listed_on": "2026-09-29"}, "b": {"listed_on": "2026-10-03"}}})
    assert ts.window(date(2026, 10, 3))["label"] == "29 Sep – 3 Oct"


# ── Topic pages ──────────────────────────────────────────────────────────────

async def test_a_topic_page_shows_its_line_lists_and_neighbours(tab_world, monkeypatch):
    search = AsyncMock(return_value=[{"arxiv_id": "2609.00001", "score": .8},
                                     {"arxiv_id": "2609.00002", "score": .7}])
    monkeypatch.setattr(qdrant_svc, "search_recent", search)

    async def meta(ids, state):
        return {a: {"arxiv_id": a, "title": "Recent " + a, "abstract": "A study.", "authors": "[]",
                    "category": "cs.LG", "published": "2026-09-01", "year": 2026, "saved": False,
                    "dismissed": False} for a in ids}

    import app.routers.collections as cr
    monkeypatch.setattr(cr, "_hydrate", meta)
    await db.log_interaction(READER, "2609.00002", "save")
    async with client() as c:
        html = (await c.get("/trending/topics/1")).text
    assert "World Models" in html and "Rising" in html and "1 in 83 → 1 in 26 papers on the lists" in html
    assert "27 Apr" in html, "the line starts at the first week"
    week = html[html.index('id="tp-week"'):html.index('id="tp-best"')]
    assert "A world model this week" in week, "grouped today"
    assert "Listed after today&#39;s grouping" in week or "Listed after today's grouping" in week, \
        "a paper listed since is placed by its nearest topic"
    assert "Top of World Models" in html and "Second of World Models" in html
    more = html[html.index('id="tp-more"'):]
    assert "2609.00001" in more and "2609.00002" not in more, "a saved paper is not offered again"
    assert "not on the lists" in more
    excluded = search.call_args.kwargs["exclude_ids"]
    assert {"2610.11111", "2606.01001"} <= excluded, "papers already on the lists are not repeated"
    assert 'href="/trending/topics/' in html[html.index('id="tp-related"'):]
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        rows = await (await conn.execute(
            "SELECT policy_id, candidate_source FROM feed_exposures WHERE user_id = ?", (READER,))).fetchall()
    assert {r[0] for r in rows} == {"trending_topic_week_v1", "trending_topic_best_v1", "trending_topic_more_v1"}
    assert ("trending_topic_more_v1", "topic_vector") in rows
    assert "query_id" not in tt.get(1)["best"][0], "a request never writes the shared topic state"


async def test_only_named_topics_have_pages(tab_world, monkeypatch):
    async with client() as c:
        assert (await c.get("/trending/topics/5")).status_code == 404, "unnamed"
        assert (await c.get("/trending/topics/99")).status_code == 404
        monkeypatch.setenv("TRENDING_PAGE", "off")
        assert (await c.get("/trending/topics/1")).status_code == 404


async def test_a_topic_page_survives_the_vector_store_failing(tab_world, monkeypatch):
    monkeypatch.setattr(qdrant_svc, "search_recent", AsyncMock(side_effect=RuntimeError("qdrant down")))
    async with client() as c:
        r = await c.get("/trending/topics/3")
    assert r.status_code == 200 and "Diffusion" in r.text and "Steady" in r.text
    assert 'id="tp-more"' not in r.text


def test_steady_label_related_and_nearest(tab_world):
    assert [t["name"] for t in tt.steady()] == ["Diffusion", "Speech"]
    assert [tt.label(t) for t in TOPICS[:4]] == ["rising", "cooling", "steady", "steady"]
    assert tt.get(5) is None and tt.get(1)["name"] == "World Models"
    v = unit(3) * .8 + unit(1) * .6
    assert [t["name"] for t in tt.related(tt.get(1))] == ["Reinforcement Learning", "Diffusion", "Speech"]
    assert tt.related(tt.get(1), n=1)[0]["name"] == "Reinforcement Learning", "never itself, never unnamed"
    assert tt.nearest(v)["id"] == 3
    assert tt.topic_of("2610.11111") == 1, "the daily grouping wins"
    assert tt.topic_of("unseen", unit(4)) == 4


async def test_search_recent_uses_only_the_recent_shard(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "qdrant_recent_configured", lambda: True)
    used = []

    class P:
        def __init__(self, aid, score):
            self.payload, self.score = {"arxiv_id": aid}, score

    def run(vec, limit, with_vectors=False):
        used.append(qdrant_svc._BACKEND.get())
        return [P("a", .9), P("a", .9), P("b", .8), P("c", .7)]

    monkeypatch.setattr(qdrant_svc, "_run_vector_search", run)
    got = await qdrant_svc.search_recent([0.0] * 1024, limit=2, exclude_ids={"b"})
    assert used == ["recent"] and [g["arxiv_id"] for g in got] == ["a", "c"]
