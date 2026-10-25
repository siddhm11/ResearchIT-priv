"""The "Trending in your interests" shelf (app/trending_shelf.py, /api/trending-shelf).

What it must guarantee: papers close to one of the reader's interests or listed
in an area they picked, no interest or area filling the shelf alone, nothing
the reader decided on or that is on the feed page below, papers shown before
only after every new one, the §3.11 fields on every card and in the exposure
log, the week's most upvoted when nothing fits, and no shelf at all only when
the switch is off. The feed's first page gets a placeholder after its fourth
card, on every tier (doc 06, 2026-10-25).
"""
import re
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest

from app import db, hf_papers_svc, qdrant_svc, trending_shelf as ts, turso_svc, user_state as us
from app.main import app
from app.routers import recommendations as recs

READER = "shelf-reader"
DIM = 1024


def unit(axis: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


def near(axis: int, cos: float, spare: int) -> np.ndarray:
    """A unit vector at `cos` to unit(axis) and orthogonal to every other medoid."""
    v = cos * unit(axis) + np.sqrt(1 - cos * cos) * unit(100 + spare)
    return v.astype(np.float32)


def interests(*axes):
    return [{"cluster_idx": i, "medoid": unit(a), "label": f"topic {i}", "tone": i} for i, a in enumerate(axes)]


def paper(aid, votes=0, day="2026-10-22", cats=()):
    return {"arxiv_id": aid, "title": "Paper " + aid, "upvotes": votes,
            "published_at": "2026-10-20T00:00:00+00:00", "listed_on": day, "categories": list(cats)}


# ── Ranking ──────────────────────────────────────────────────────────────────

def test_bar_uses_the_default_below_four_saves_and_is_clamped():
    assert ts.interest_bar([0.9, 0.9, 0.9]) == ts.DEFAULT_BAR
    assert ts.interest_bar([0.40] * 10) == ts.BAR_MIN
    assert ts.interest_bar([0.95] * 10) == ts.BAR_MAX
    assert ts.interest_bar([0.58, 0.60, 0.62, 0.70]) == pytest.approx(0.595)


def test_only_on_topic_unseen_papers_and_upvotes_break_ties():
    cands = [paper("a", 1), paper("b", 100), paper("c", 900), paper("d", 5), paper("e"), paper("f")]
    vecs = {"a": near(0, .70, 1), "b": near(0, .70, 2), "c": near(0, .50, 3),   # c is below the bar
            "d": near(0, .72, 4), "e": near(0, .68, 5)}                      # f has no vector
    got = ts.rank(cands, vecs, interests(0), bar=0.60, exclude={"e"})
    assert [p["arxiv_id"] for p in got] == ["b", "d", "a"]
    assert got[0]["fit"] == pytest.approx(0.70, abs=1e-3)


def test_one_interest_cannot_fill_the_shelf():
    cands = [paper(f"a{i}") for i in range(10)] + [paper(f"b{i}") for i in range(3)]
    vecs = {f"a{i}": near(0, .85, i) for i in range(10)} | {f"b{i}": near(1, .65, 20 + i) for i in range(3)}
    got = ts.rank(cands, vecs, interests(0, 1), bar=0.60, exclude=set())
    assert sorted(p["interest"]["cluster_idx"] for p in got) == [0, 0, 0, 1, 1, 1]


def test_a_thin_interest_leaves_its_places_to_the_others():
    cands = [paper(f"a{i}") for i in range(10)] + [paper("b0")]
    vecs = {f"a{i}": near(0, .85, i) for i in range(10)} | {"b0": near(1, .65, 30)}
    got = ts.rank(cands, vecs, interests(0, 1), bar=0.60, exclude=set())
    assert len(got) == ts.SHELF_SIZE
    assert sum(p["interest"]["cluster_idx"] == 1 for p in got) == 1


def test_too_little_on_topic_shows_no_shelf():
    cands = [paper("a"), paper("b"), paper("c")]
    vecs = {"a": near(0, .80, 1), "b": near(0, .80, 2), "c": near(0, .30, 3)}
    assert ts.rank(cands, vecs, interests(0), bar=0.60, exclude=set()) == []


def test_chosen_areas_alone_fill_the_shelf_one_area_at_a_time():
    areas = ts.area_groups(["robotics", "cv", "se"])
    cands = ([paper(f"r{i}", votes=50 - i, cats=["cs.RO"]) for i in range(5)]
             + [paper(f"v{i}", votes=900 - i, cats=["cs.CV", "cs.RO"]) for i in range(5)]   # primary wins
             + [paper("s0", votes=1, cats=["cs.LG", "cs.SE"])]                              # listed counts too
             + [paper("x", votes=999, cats=["hep-th"])])                                    # in no chosen area
    got = ts.rank(cands, {}, [], bar=0.60, exclude=set(), areas=areas)
    # Round the areas, best first (vision's 900 upvotes, robotics, then software
    # engineering's one paper), until six; then best first for display.
    assert [p["arxiv_id"] for p in got] == ["v0", "v1", "v2", "r0", "r1", "s0"]
    assert [p["interest"]["label"] for p in got][-1] == "Software Engineering"


def test_a_saved_interest_claims_a_paper_before_its_area_does():
    areas = ts.area_groups(["robotics"], first_tone=1)
    cands = [paper("a", cats=["cs.RO"]), paper("b", cats=["cs.RO"]), paper("c", cats=["cs.RO"])]
    vecs = {"a": near(0, .80, 1)}
    got = {p["arxiv_id"]: p for p in ts.rank(cands, vecs, interests(0), bar=0.60, exclude=set(), areas=areas)}
    assert got["a"]["interest"]["cluster_idx"] == 0 and got["a"]["fit"] == pytest.approx(0.80, abs=1e-3)
    assert got["b"]["interest"]["label"] == "Robotics" and got["b"]["fit"] is None
    assert got["a"]["score"] > got["b"]["score"], "a save-close paper ranks above an area-only one"


def test_papers_shown_before_fill_places_after_every_new_one():
    cands = [paper(f"a{i}", votes=100 - i) for i in range(8)]
    vecs = {f"a{i}": near(0, .80, i) for i in range(8)}
    got = [p["arxiv_id"] for p in ts.rank(cands, vecs, interests(0), bar=0.60, exclude=set(),
                                           seen={"a0", "a1", "a2", "a3", "a4"})]
    assert got == ["a5", "a6", "a7", "a0", "a1", "a2"]


def test_week_by_area_counts_every_listed_paper():
    state = {"papers": {"r": paper("r", 9, cats=["cs.RO"]), "v": paper("v", 5, cats=["cs.CV", "cs.RO"]),
                        "m": paper("m", 7, cats=["cs.LG"])}, "code": {}, "vectors": {}}
    import unittest.mock as um
    with um.patch.object(ts, "_state", {**ts._state, **state}):
        rows = ts.by_area(["robotics", "se", "ml"], exclude={"r"})
    assert [(r["label"], r["count"], [p["arxiv_id"] for p in r["papers"]]) for r in rows] == [
        ("Robotics", 2, ["v"]), ("Machine Learning", 1, ["m"])]


# ── Refresh ──────────────────────────────────────────────────────────────────

async def test_refresh_keeps_the_best_upvotes_skips_a_bad_day_and_survives_an_outage(monkeypatch):
    from datetime import date
    calls = []

    async def fetch(client, *, now=None, date=None):
        calls.append(date)
        if date == "2026-10-21":
            raise httpx.HTTPError("down")
        votes = 40 if date == "2026-10-23" else 10
        return {"papers": [{"arxiv_id": "2610.00001", "title": "T", "upvotes": votes,
                            "published_at": "2026-10-20T00:00:00+00:00"}],
                "code": {"2610.00001": ("https://github.com/a/b", 3)}}

    monkeypatch.setattr(hf_papers_svc, "fetch_snapshot", fetch)
    monkeypatch.setattr(qdrant_svc, "get_paper_vectors", AsyncMock(return_value={"2610.00001": unit(0)}))
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", AsyncMock(
        return_value={"2610.00001": {"arxiv_id": "2610.00001", "arxiv_categories": "cs.RO cs.LG"}}))
    monkeypatch.setattr(ts, "_state", dict(ts._state))
    stats = await ts.refresh(client=object(), today=date(2026, 10, 23))
    assert len(calls) == ts.DAYS and stats["days"] == ts.DAYS - 1 and stats["failed_days"] == ["2026-10-21"]
    assert ts._state["papers"]["2610.00001"]["upvotes"] == 40
    assert ts._state["papers"]["2610.00001"]["categories"] == ["cs.RO", "cs.LG"]
    assert stats["with_vectors"] == 1

    async def down(client, *, now=None, date=None):
        raise httpx.HTTPError("down")

    monkeypatch.setattr(hf_papers_svc, "fetch_snapshot", down)
    with pytest.raises(RuntimeError):
        await ts.refresh(client=object(), today=date(2026, 10, 24))
    assert "2610.00001" in ts._state["papers"], "an outage must not empty the shelf"


# ── Endpoint ─────────────────────────────────────────────────────────────────

HF = [f"2610.{i:05d}" for i in range(12)]
MEMBERS = {0: ["2601.00001", "2601.00002", "2601.00003", "2601.00004", "2601.00005"],
           1: ["2602.00001", "2602.00002"]}


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", cookies={"arxiv_user_id": READER})


@pytest.fixture
async def shelf_world(monkeypatch):
    await db.init_db()
    us._cache.clear()
    await db.save_user_clusters(READER, [
        {"cluster_idx": idx, "medoid_paper_id": ids[0], "importance": 1.0,
         "paper_ids": __import__("json").dumps(ids), "medoid_embedding_blob": unit(idx).tobytes()}
        for idx, ids in MEMBERS.items()])
    member_vecs = {pid: near(idx, .75, 300 + n) for idx, ids in MEMBERS.items() for n, pid in enumerate(ids)}
    hf_vecs = {aid: near(i % 2, .70 + i / 100, 400 + i) for i, aid in enumerate(HF)}
    monkeypatch.setattr(qdrant_svc, "get_paper_vectors",
                        AsyncMock(side_effect=lambda ids: {i: member_vecs[i] for i in ids if i in member_vecs}))
    monkeypatch.setattr(ts, "_state", {**ts._state, "vectors": hf_vecs, "code": {},
                                       "papers": {aid: paper(aid, votes=i) for i, aid in enumerate(HF)}})
    monkeypatch.setenv("TRENDING_SHELF", "on")
    yield
    us._cache.clear()


def shelf_ids(response) -> list[str]:
    return re.findall(r'href="/p/([0-9.]+)\?qid=', response.text)


async def test_switch_off_renders_nothing(shelf_world, monkeypatch):
    monkeypatch.setenv("TRENDING_SHELF", "off")
    async with client() as c:
        r = await c.get("/api/trending-shelf")
    assert r.status_code == 200 and r.text == ""


async def test_shelf_carries_attribution_logs_exposures_and_moves_on(shelf_world):
    await db.log_interaction(READER, HF[11], "save")
    async with client() as c:
        first = await c.get("/api/trending-shelf")
        second = await c.get("/api/trending-shelf")
    shown = shelf_ids(first)
    assert "Trending in your interests" in first.text and len(shown) == ts.SHELF_SIZE
    assert HF[11] not in shown, "a saved paper is never on the shelf"
    assert "sf=trending" in first.text and "pol=trending_shelf_v1" in first.text and "prop=1.0" in first.text
    assert '"policy_id": "trending_shelf_v1"' in first.text and '"source": "trending"' in first.text
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        rows = await (await conn.execute(
            "SELECT paper_id, policy_id, propensity, candidate_source FROM feed_exposures ORDER BY id")).fetchall()
    assert [r[0] for r in rows[:ts.SHELF_SIZE]] == shown
    assert {(r[1], r[2], r[3]) for r in rows} == {("trending_shelf_v1", 1.0, "hf_daily")}
    again = shelf_ids(second)
    fresh = [a for a in again if a not in shown]
    assert len(again) == ts.SHELF_SIZE and fresh, "the next shelf moves on, and stays full"
    assert again[:len(fresh)] == fresh, "papers already shown come after every new one"


async def test_a_reader_with_areas_but_no_saves_gets_a_shelf_in_those_areas(shelf_world):
    await db.save_user_clusters(READER, [])
    await db.save_onboarding_categories(READER, ["robotics"])
    for i, aid in enumerate(HF):
        ts._state["papers"][aid]["categories"] = ["cs.RO"] if i < 4 else ["cs.CV"]
    async with client() as c:
        r = await c.get("/api/trending-shelf")
    assert "Trending in your areas" in r.text and "Robotics" in r.text
    assert set(shelf_ids(r)) == set(HF[:4])


async def test_a_reader_nothing_fits_gets_the_weeks_top_not_nothing(shelf_world):
    await db.save_user_clusters(READER, [])
    async with client() as c:
        r = await c.get("/api/trending-shelf")
    assert "Trending this week" in r.text
    assert shelf_ids(r) == [HF[i] for i in range(11, 5, -1)], "most upvoted first"


async def test_the_shelf_never_repeats_the_feed_page_below_it(shelf_world):
    recs._cache_put("page-q", {"user_id": READER, "tags": {}, "emitted": set(HF[6:])})
    async with client() as c:
        r = await c.get("/api/trending-shelf?after=page-q")
    assert shelf_ids(r) and not set(shelf_ids(r)) & set(HF[6:])


# ── Placement in the feed ────────────────────────────────────────────────────

POOL = [f"2603.{i:05d}" for i in range(40)]


async def metadata(aids):
    return {aid: {"arxiv_id": aid, "title": "Paper " + aid, "abstract": "A study.", "authors": "[]",
                  "category": "cs.LG", "published": "2026-03-01", "year": 2026} for aid in aids}


@pytest.fixture
async def feed_world(monkeypatch):
    await db.init_db()
    us._cache.clear()
    recs._FEED_CACHE.clear()
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", metadata)
    monkeypatch.setattr(recs, "_ewma_recommend", AsyncMock(return_value=list(POOL)))
    monkeypatch.setattr(qdrant_svc, "recommend", AsyncMock(return_value=list(POOL)))

    async def clustered(uid, state, seen, limit, *, query_id):
        tags = {a: {"candidate_source": "cluster_0", "cluster_id": 0, "propensity": 1.0,
                    "policy_id": recs._RANKER_VERSION} for a in POOL}
        return list(POOL), [], tags, 0, {}

    monkeypatch.setattr(recs, "_multi_interest_recommend", clustered)
    for pid in ("2601.00001", "2601.00002", "2601.00003"):
        await db.log_interaction(READER, pid, "save")
    yield
    us._cache.clear()
    recs._FEED_CACHE.clear()


async def test_a_new_reader_sees_the_shelf_before_the_feed(feed_world, monkeypatch):
    """Fewer than 5 saves (straight out of onboarding): trending in their areas
    first, then the feed (doc 06, 2026-10-25)."""
    monkeypatch.setenv("TRENDING_SHELF", "on")
    async with client() as c:
        page1 = (await c.get("/api/recommendations")).text
    slot = page1.index('class="shelf-slot shelf-slot--first"')
    assert slot < page1.index('data-arxiv-id="') and "first=1" in page1


async def test_the_first_shelf_says_what_it_is_for(shelf_world):
    async with client() as c:
        r = await c.get("/api/trending-shelf?first=1")
    assert "Your feed below learns from what you save." in r.text


async def test_feed_places_the_shelf_after_the_fourth_card(feed_world, monkeypatch):
    for pid in ("2601.00004", "2601.00005"):           # 5 saves: grouped into interests
        await db.log_interaction(READER, pid, "save")
    us._cache.clear()
    monkeypatch.setenv("TRENDING_SHELF", "on")
    async with client() as c:
        page1 = (await c.get("/api/recommendations")).text
        qid = next(reversed(recs._FEED_CACHE))
        page2 = (await c.get(f"/api/recommendations?page=2&query_id={qid}")).text
    cards = [m.start() for m in re.finditer(r'data-arxiv-id="', page1)]
    slot = page1.index('class="shelf-slot"')
    assert len(cards) > ts.SHELF_AFTER_CARD
    assert cards[ts.SHELF_AFTER_CARD - 1] < slot < cards[ts.SHELF_AFTER_CARD]
    assert "shelf-slot" not in page2, "only the first page carries the shelf"


async def test_placeholder_on_every_tier_and_none_with_the_switch_off(feed_world, monkeypatch):
    monkeypatch.setenv("TRENDING_SHELF", "off")
    async with client() as c:
        assert "shelf-slot" not in (await c.get("/api/recommendations")).text
    monkeypatch.setenv("TRENDING_SHELF", "on")
    monkeypatch.setattr(recs, "_multi_interest_recommend", AsyncMock(return_value=([], [], {}, 0, {})))
    recs._FEED_CACHE.clear()
    async with client() as c:
        assert "shelf-slot" in (await c.get("/api/recommendations")).text, "not only the clustered tier"


async def test_shelf_uses_the_interest_names_of_its_feed_page(shelf_world):
    recs._cache_put("feed-q", {"user_id": READER, "tags": {
        "2601.00001": {"cluster_id": "0", "cluster_label": "Language & AI · agents", "cluster_tone": 0},
        "2602.00001": {"cluster_id": "1", "cluster_label": "Language & AI · reasoning", "cluster_tone": 1}}})
    async with client() as c:
        mine = (await c.get("/api/trending-shelf?after=feed-q")).text
    assert "Language &amp; AI · agents" in mine and "Language &amp; AI · reasoning" in mine
    recs._cache_put("other-q", {"user_id": "someone-else", "tags": {
        "x": {"cluster_id": "0", "cluster_label": "Not yours", "cluster_tone": 0}}})
    async with client() as c:
        theirs = (await c.get("/api/trending-shelf?after=other-q")).text
    assert "Not yours" not in theirs
