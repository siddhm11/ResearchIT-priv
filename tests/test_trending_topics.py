"""Rising topics in Hugging Face Daily Papers (app/trending_topics.py, /trending).

What it must guarantee: a topic counts as rising or cooling only when its
26-week share trend beats shuffled weeks after correcting for many topics; it is
shown only under a name a third of its papers carry; names never splice words
from different fields; topic ids survive a recompute; an outage keeps the last
topics; and the page shows nothing with its switch off.
"""
from collections import Counter
from datetime import date, timedelta
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest

from app import db, qdrant_svc, trending_shelf as ts, trending_topics as tt, turso_svc, user_state as us
from app.main import app

DIM = 1024


# ── Names ────────────────────────────────────────────────────────────────────

def test_phrases_stay_inside_a_field_and_skip_generic_ones():
    got = tt.phrases("World Models for Robots | world model | We study large language models.")
    assert "world model" in got and "robot" in got           # last word made singular
    assert "robots world" not in got and "model world" not in got, "never across a field"
    assert "large language model" not in got and "model" not in got, "generic alone is not a name"


def test_a_name_must_cover_the_topic_and_prefers_the_longer_phrase():
    def members(n_with, phrase_sets):
        return [set(phrase_sets[0]) if i < n_with else set(phrase_sets[1]) for i in range(100)]
    rest = [{"other"}] * 900
    # "world" is in 90 of 100 papers, "world model" in 40: the name is the longer phrase.
    topic = [({"world", "world model"} if i < 40 else {"world"}) if i < 90 else {"misc"} for i in range(100)]
    df = Counter(g for m in topic + rest for g in m)
    name, cov, _also = tt.name_topic(topic, df, 1000)
    assert name == "world model" and cov == pytest.approx(0.40)
    # Nothing carried by 30% of the topic: no name at all.
    thin = [{f"term{i % 5}"} for i in range(100)]
    df = Counter(g for m in thin + rest for g in m)
    assert tt.name_topic(thin, df, 1000)[0] is None


def test_display_name_follows_the_titles_and_prefers_the_plural():
    titles = ["Robots that learn", "Teaching robots to fold", "A robot arm", "World Models at scale"]
    assert tt.display_name("robot", titles) == "Robots"
    assert tt.display_name("world model", titles) == "World Models"
    assert tt.display_name("speech", []) == "Speech"


# ── The test ─────────────────────────────────────────────────────────────────

def test_benjamini_hochberg_matches_the_textbook_values():
    # p x m / rank, then made monotone from the largest p down.
    q = tt.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_trend_test_finds_a_planted_trend_and_not_a_flat_topic():
    rng = np.random.default_rng(3)
    topic, week = [], []
    for w in range(26):
        n = 150
        rising = int(n * (0.05 + 0.15 * w / 25))        # topic 0: 5% -> 20% of the week
        flat = int(n * 0.10)                            # topic 1: 10% every week
        topic += [0] * rising + [1] * flat + list(rng.integers(2, 8, n - rising - flat))
        week += [w] * n
    counts, slopes, q = tt.trend_test(np.array(topic), np.array(week), 8, 26, permutations=400)
    assert counts.sum() == 26 * 150
    assert slopes[0] > 0 and q[0] < 0.05
    assert q[1] > 0.05, "a topic holding its share is not a trend"


# ── One computation ──────────────────────────────────────────────────────────

CENTRES = {"robot": 0, "speech": 1, "graph": 2, "memory": 3, "protein": 4, "galaxy": 5, "compiler": 6, "music": 7}


def world(n_weeks=26, per_week=60, rising="robot", seed=5):
    """Daily Papers rows and vectors: eight tight topics, `rising` growing from
    5% to 30% of each week's list, the others sharing the rest."""
    rng = np.random.default_rng(seed)
    newest = date(2026, 10, 23)
    rows, vecs = [], {}
    for w in range(n_weeks):
        day = newest - timedelta(weeks=n_weeks - 1 - w)
        share = 0.05 + 0.25 * w / (n_weeks - 1)
        others = [c for c in CENTRES if c != rising]
        for i in range(per_week):
            name = rising if i < int(per_week * share) else others[i % len(others)]
            aid = f"26{w:02d}.{i:05d}"
            v = np.zeros(DIM, dtype=np.float32)
            v[CENTRES[name]] = 1.0
            v += rng.normal(0, 0.02, DIM).astype(np.float32)
            vecs[aid] = v
            rows.append({"arxiv_id": aid, "listed_on": day.isoformat(), "title": f"{name.title()}s: case {i}",
                         "keywords": [name], "summary": f"A {name} paper.", "github": "", "upvotes": i, "stars": 0})
    return rows, vecs


def test_compute_names_the_rising_topic_and_keeps_its_id(monkeypatch):
    monkeypatch.setattr(tt, "MIN_PAPERS", 100)
    rows, vecs = world()
    first = tt.compute(rows, vecs, k=8, permutations=300)
    by_name = {t["name"]: t for t in first["topics"]}
    robot = by_name["Robots"]
    assert robot["slope"] > 0 and robot["q"] < 0.05 and robot["to_share"] > robot["from_share"] + 0.15
    assert robot["coverage"] >= tt.NAME_COVERAGE and len(robot["rolling"]) == first["weeks"] == 26
    assert first["topic_of"][rows[-1]["arxiv_id"]] in {t["id"] for t in first["topics"]}
    again = tt.compute(rows, vecs, previous=first["topics"], k=8, permutations=300)
    assert {t["name"]: t["id"] for t in again["topics"]} == {t["name"]: t["id"] for t in first["topics"]}


def test_compute_refuses_too_few_papers():
    rows, vecs = world(n_weeks=2, per_week=10)
    with pytest.raises(ValueError):
        tt.compute(rows, vecs, k=8)


def test_topic_line_only_for_named_rising_topics(monkeypatch):
    topics = [{"id": 7, "name": "Robots", "q": 0.01, "slope": 0.5, "from_share": 0.1, "to_share": 0.5},
              {"id": 8, "name": None, "q": 0.01, "slope": 0.9, "from_share": 0.1, "to_share": 0.6},
              {"id": 9, "name": "Speech", "q": 0.01, "slope": -0.4, "from_share": 0.3, "to_share": 0.1}]
    monkeypatch.setattr(tt, "_state", {**tt._state, "topics": topics, "first_week": "2026-04-27",
                                       "topic_of": {"a": 7, "b": 8, "c": 9}})
    assert tt.topic_line("a") == {"id": 7, "name": "Robots", "from": 0.1, "to": 0.5, "since": "2026-04-27"}
    assert tt.topic_line("b") is None and tt.topic_line("c") is None and tt.topic_line("zz") is None
    assert [t["id"] for t in tt.rising()] == [7] and [t["id"] for t in tt.cooling()] == [9]


async def test_an_outage_keeps_the_last_topics(monkeypatch):
    kept = [{"id": 1, "name": "Robots", "q": 0.01, "slope": 0.5, "from_share": .1, "to_share": .5}]
    monkeypatch.setattr(tt, "_state", {**tt._state, "topics": kept})
    monkeypatch.setattr(turso_svc, "fetch_hf_daily", AsyncMock(side_effect=RuntimeError("Turso down")))
    with pytest.raises(RuntimeError):
        await tt.refresh()
    assert tt._state["topics"] is kept


async def test_refresh_reads_turso_and_qdrant_without_filling_the_cache(monkeypatch):
    monkeypatch.setattr(tt, "MIN_PAPERS", 100)
    monkeypatch.setattr(tt, "PERMUTATIONS", 200)
    monkeypatch.setattr(tt, "K", 8)
    monkeypatch.setattr(tt, "_state", {**tt._state, "topics": [], "topic_of": {}})
    rows, vecs = world()
    monkeypatch.setattr(turso_svc, "fetch_hf_daily", AsyncMock(return_value=rows))
    calls = []

    async def vectors(ids, remember=True):
        calls.append(remember)
        return {i: vecs[i] for i in ids}

    monkeypatch.setattr(qdrant_svc, "get_paper_vectors", vectors)
    stats = await tt.refresh(today=date(2026, 10, 24))
    assert stats["papers"] == len(rows) and stats["rising"] >= 1 and stats["error"] is None
    assert calls and not any(calls), "a bulk read must not push the feed's vectors out of the cache"


# ── The page ─────────────────────────────────────────────────────────────────

READER = "trending-reader"


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", cookies={"arxiv_user_id": READER})


@pytest.fixture
async def page_world(monkeypatch):
    await db.init_db()
    us._cache.clear()
    topics = [{"id": 3, "name": "Robots", "also": ["manipulation", "embodied"], "papers": 217, "q": 0.006,
               "slope": 0.55, "from_share": 0.02, "to_share": 0.057, "rolling": [0.02, 0.03, 0.05, 0.057],
               "top": [{"arxiv_id": "2610.22471", "title": "Has GPT-6-Astra Achieved Embodied Intelligence?",
                        "upvotes": 468}]},
              {"id": 4, "name": None, "also": [], "papers": 80, "q": 0.01, "slope": 0.6, "from_share": 0.01,
               "to_share": 0.05, "rolling": [0.01, 0.05], "top": []}]
    monkeypatch.setattr(tt, "_state", {**tt._state, "topics": topics, "topic_of": {"2610.00002": 3},
                                       "first_week": "2026-04-27", "weeks": 26, "papers": 5176,
                                       "computed_at": "2026-10-24T11:28:41+00:00"})
    papers = {aid: {"arxiv_id": aid, "title": "Paper " + aid, "upvotes": v, "listed_on": "2026-10-22",
                    "published_at": "2026-10-20T00:00:00+00:00", "day_rank": r, "day_size": 58,
                    "week_percentile": 1 - r / 100, "week_size": 300, "summary": "In one line.", "keywords": []}
              for r, (aid, v) in enumerate([("2610.00001", 400), ("2610.00002", 300), ("2610.00003", 5)], start=1)}
    monkeypatch.setattr(ts, "_state", {**ts._state, "papers": papers, "vectors": {},
                                       "code": {"2610.00001": ("https://github.com/a/b", 5742)}})
    monkeypatch.setenv("TRENDING_PAGE", "on")
    yield
    us._cache.clear()


async def test_page_is_absent_with_its_switch_off(page_world, monkeypatch):
    monkeypatch.setenv("TRENDING_PAGE", "off")
    async with client() as c:
        r = await c.get("/trending")
    assert r.status_code == 404


async def test_page_shows_named_rising_topics_the_week_and_logs_exposures(page_world):
    async with client() as c:
        r = await c.get("/trending")
    assert r.status_code == 200
    html = r.text
    assert "What's moving" in html and "Robots" in html and "2.0% → 5.7%" in html
    assert "Also: manipulation, embodied" in html and "Has GPT-6-Astra" in html
    assert html.count('class="topic-row topic-row--') == 1, "an unnamed topic is never shown"
    assert "Most upvoted this week" in html and "Most upvoted of 58 papers on 22 Oct" in html
    assert "#2 of 58 papers on 22 Oct" in html
    assert 'Rising topic: <a href="/trending/topics/3"><b>Robots</b></a>' in html, \
        "a paper in a rising topic says so, and links to the topic"
    assert "★ 5.7k" in html and "Hugging Face summary" in html
    assert "pol=trending_page_top_v1" in html and "shuffled 2,000 times" in html
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        rows = await (await conn.execute(
            "SELECT paper_id, policy_id FROM feed_exposures WHERE user_id = ? ORDER BY id", (READER,))).fetchall()
    assert [r[0] for r in rows] == ["2610.00001", "2610.00002", "2610.00003"]
    assert {r[1] for r in rows} == {"trending_page_top_v1"}


async def test_page_says_when_topics_failed(page_world, monkeypatch):
    monkeypatch.setattr(tt, "_state", {**tt._state, "topics": [], "computed_at": None,
                                       "error": "RuntimeError: Turso down"})
    async with client() as c:
        r = await c.get("/trending")
    assert r.status_code == 200 and "Topics can't be worked out right now" in r.text
    assert "Most upvoted this week" in r.text, "the week's papers do not depend on the topics"


async def test_page_shows_each_chosen_area_with_its_count_and_top_papers(page_world):
    """A reader's areas show without any saves, weekends included: the week's
    lists stay in place until new ones arrive (doc 06, 2026-10-25)."""
    await db.save_onboarding_categories(READER, ["robotics", "se"])
    cats = {"2610.00001": ["cs.RO"], "2610.00002": ["cs.CV", "cs.RO"], "2610.00003": ["cs.LG"]}
    for aid, c in cats.items():
        ts._state["papers"][aid]["categories"] = c
    async with client() as c:
        html = (await c.get("/trending")).text
    assert 'id="tp-area-robotics"' in html and "2 papers on this week's lists" in html
    assert 'id="tp-area-se"' not in html, "an area with no papers this week is left out"
    start = html.index('id="tp-area-robotics"')
    robotics = html[start:html.index("</section>", start)]
    assert html.index('id="tp-rising"') < start, "topics come before the area rows"
    assert "2610.00001" in robotics and "2610.00002" in robotics and "2610.00003" not in robotics
    top = html[html.index('id="tp-top"'):]
    top = top[:top.index("</section>")]
    assert "2610.00003" in top and "2610.00001" not in top, "no paper twice on the page"
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        rows = await (await conn.execute(
            "SELECT paper_id, policy_id FROM feed_exposures WHERE user_id = ? ORDER BY id", (READER,))).fetchall()
    assert rows == [("2610.00001", "trending_page_areas_v1"), ("2610.00002", "trending_page_areas_v1"),
                    ("2610.00003", "trending_page_top_v1")]
