"""A saved or disliked paper never comes back in the feed; search still shows it.

The rule (CLAUDE.md §3.4b): papers the reader decided on -- saved, or marked
not interested -- are excluded from every feed tier, on every page, forever.
Papers that were only SHOWN cool down for seven days and may then return.
Search is a lookup, not a feed: it shows decided papers, labelled.

test_discovery_refresh.py covers the happy path. These tests attack it: the
retrieval backends below deliberately IGNORE the exclusion set they are given,
decisions outnumber the bounded in-memory deques, decisions land mid-session,
and the process restarts between requests.
"""
import re
from unittest.mock import AsyncMock

import httpx
import pytest

from app import db, hybrid_search_svc, qdrant_svc, turso_svc, user_state as us
from app.main import app
from app.routers import recommendations as recs

READER = "decider"
POOL = [f"2602.{i:05d}" for i in range(200)]
SAVED = POOL[:30]            # more than REC_POSITIVE_LIMIT (20) fits in memory
DISLIKED = POOL[30:90]       # more than MAX_NEGATIVES (50) fits in memory
DECIDED = set(SAVED) | set(DISLIKED)


def ids(response) -> list[str]:
    return re.findall(r'data-arxiv-id="([^"]+)"', response.text)


def card(response, aid: str) -> str:
    """The HTML of one paper card (up to the next card)."""
    text = response.text
    start = text.index(f'data-arxiv-id="{aid}"')
    nxt = text.find('data-arxiv-id="', start + 10)
    return text[start: nxt if nxt != -1 else len(text)]


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", cookies={"arxiv_user_id": READER})


async def metadata(aids):
    return {aid: {"arxiv_id": aid, "title": "Paper " + aid,
                  "abstract": "A study of research methods.", "authors": "[]",
                  "category": "cs.LG", "published": "2026-02-01", "year": 2026}
            for aid in aids}


def restart():
    """Simulate a process restart: in-memory reader state and feed cursors vanish."""
    us._cache.clear()
    recs._FEED_CACHE.clear()


@pytest.fixture
async def world(monkeypatch):
    await db.init_db()
    restart()
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", metadata)
    monkeypatch.setattr(recs.discovery_svc, "starter_papers",
                        AsyncMock(return_value=list((await metadata(POOL)).values())))
    monkeypatch.setattr(recs, "_multi_interest_recommend", AsyncMock(return_value=([], [], {}, 0, {})))
    monkeypatch.setattr(recs, "_ewma_recommend", AsyncMock(return_value=[]))
    monkeypatch.setattr(qdrant_svc, "recommend", AsyncMock(return_value=[]))
    for pid in SAVED:
        await db.log_interaction(READER, pid, "save")
    for pid in DISLIKED:
        await db.log_interaction(READER, pid, "not_interested")
    yield
    restart()


def leaky_tier(monkeypatch, tier: int):
    """A retrieval backend that returns the WHOLE pool, decided papers first,
    ignoring the exclusion set it was handed. Only the router's own filters
    stand between these papers and the screen."""
    if tier == 1:
        async def clustered(uid, state, seen, limit, *, query_id):
            tags = {a: {"candidate_source": "cluster_0", "cluster_id": 0,
                        "propensity": 1.0, "policy_id": recs._RANKER_VERSION} for a in POOL}
            # Decided papers in the exploration pool too.
            return list(POOL), list(SAVED[:5] + DISLIKED[:5]), tags, 0, {}
        monkeypatch.setattr(recs, "_multi_interest_recommend", clustered)
    elif tier == 2:
        monkeypatch.setattr(recs, "_ewma_recommend", AsyncMock(return_value=list(POOL)))
    elif tier == 3:
        monkeypatch.setattr(qdrant_svc, "recommend", AsyncMock(return_value=list(POOL)))


async def read_feed(c, refreshes=4, pages=3) -> list[str]:
    """Refresh the feed several times and scroll a few pages each time."""
    served = []
    for _ in range(refreshes):
        r = await c.get("/api/recommendations")
        served += ids(r)
        qid = next(reversed(recs._FEED_CACHE), None)
        for page in range(2, pages + 1):
            if not qid:
                break
            served += ids(await c.get(f"/api/recommendations?page={page}&query_id={qid}"))
    return served


@pytest.mark.parametrize("tier", [1, 2, 3])
async def test_leaky_backends_never_put_a_decided_paper_on_screen(world, monkeypatch, tier):
    leaky_tier(monkeypatch, tier)
    async with client() as c:
        served = await read_feed(c)
    assert served, "the feed served nothing"
    assert not set(served) & DECIDED, f"tier {tier} re-showed {sorted(set(served) & DECIDED)[:5]}"


async def test_cold_start_tier_never_serves_decided_papers(world, monkeypatch):
    # Tier 0 applies while the reader is below the save threshold; force it.
    monkeypatch.setattr(us.UserState, "has_enough_for_recs", lambda self: False)
    async with client() as c:
        served = await read_feed(c)
    assert served
    assert not set(served) & DECIDED


async def test_exhausted_supply_retry_still_excludes_decisions(world, monkeypatch):
    """Every undecided paper was shown recently, so the feed retries without the
    seven-day cooldown. The retry relaxes impressions, never decisions."""
    leaky_tier(monkeypatch, 3)
    await db.record_impressions(READER, [p for p in POOL if p not in DECIDED])
    async with client() as c:
        served = await read_feed(c, refreshes=2)
    assert served, "exhaustion should repeat old impressions, not go empty"
    assert not set(served) & DECIDED


async def test_decision_mid_session_hides_paper_from_later_pages_and_refreshes(world, monkeypatch):
    leaky_tier(monkeypatch, 3)
    async with client() as c:
        first = await c.get("/api/recommendations")
        qid = next(reversed(recs._FEED_CACHE))
        page1 = ids(first)
        save_now, dislike_now = page1[0], page1[1]
        await c.post(f"/api/papers/{save_now}/save", data={"source": "feed"})
        await c.post(f"/api/papers/{dislike_now}/not-interested", data={"source": "feed"})
        # Expire the 7-day cooldown on everything shown, so only the decision can hide them.
        import aiosqlite
        async with aiosqlite.connect(db.DB_PATH) as conn:
            await conn.execute("UPDATE feed_impressions SET shown_at = datetime('now','-30 days')")
            await conn.commit()
        later = ids(await c.get(f"/api/recommendations?page=2&query_id={qid}"))
        later += await read_feed(c, refreshes=3)
    assert page1[2] in later, "control: an undecided paper should return after its cooldown"
    assert save_now not in later and dislike_now not in later


async def test_decisions_survive_a_restart(world, monkeypatch):
    leaky_tier(monkeypatch, 2)
    async with client() as c:
        before = await read_feed(c, refreshes=1)
        restart()
        after = await read_feed(c, refreshes=2)
    assert before and after
    assert not set(before + after) & DECIDED


async def test_unsaved_paper_is_no_longer_a_decision(world, monkeypatch):
    """Un-saving is not disliking (CLAUDE.md §3.6b): the paper leaves the
    library and the exclusion set. Locked here so a change is deliberate."""
    leaky_tier(monkeypatch, 3)
    async with client() as c:
        await c.post(f"/api/papers/{SAVED[0]}/unsave", data={"source": "saved"})
        served = await read_feed(c, refreshes=1, pages=1)
    assert SAVED[0] in served
    assert not set(served) & (DECIDED - {SAVED[0]})


# ── Search is not a feed ─────────────────────────────────────────────────────

async def test_search_shows_decided_papers_and_labels_every_saved_one(world, monkeypatch):
    """Search must not hide decisions -- the reader may be looking for a paper
    they saved months ago -- and must label it Saved even when it is older
    than the 20 saves kept in memory."""
    hits = [SAVED[0], SAVED[-1], DISLIKED[0], POOL[150]]
    monkeypatch.setattr(hybrid_search_svc, "search", AsyncMock(return_value=(hits, {})))
    async with client() as c:
        r = await c.get("/search?q=research+methods")
    assert ids(r) == hits, "search must return decided papers, in ranked order"
    for aid in (SAVED[0], SAVED[-1]):
        assert "btn-saved" in card(r, aid), f"{aid} is saved but not labelled Saved"
    assert "btn-saved" not in card(r, POOL[150])
