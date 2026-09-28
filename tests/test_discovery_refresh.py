"""Exercise refresh through HTTP and real SQLite; retrieval alone is synthetic."""
import asyncio
import re
from unittest.mock import AsyncMock

import aiosqlite
import httpx
import pytest

from app import db, qdrant_svc, turso_svc, user_state as us
from app.main import app
from app.routers import recommendations as recs


def ids(response):
    return re.findall(r'data-arxiv-id="([^"]+)"', response.text)


@pytest.fixture
async def world(monkeypatch):
    await db.init_db()
    recs._FEED_CACHE.clear()
    pool = [f"2601.{i:05d}" for i in range(90)]

    async def metadata(aids):
        return {aid: {"arxiv_id": aid, "title": "Paper " + aid,
                      "abstract": "A study of research methods.", "authors": "[]",
                      "category": "cs.AI", "published": "2026-01-01", "year": 2026}
                for aid in aids}

    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", metadata)
    monkeypatch.setattr(recs.discovery_svc, "starter_papers", AsyncMock(
        return_value=list((await metadata(pool)).values())))
    monkeypatch.setattr(recs, "_multi_interest_recommend", AsyncMock(return_value=([], [], {}, 0, {})))
    monkeypatch.setattr(recs, "_ewma_recommend", AsyncMock(return_value=[]))
    monkeypatch.setattr(qdrant_svc, "recommend", AsyncMock(return_value=[]))
    return pool


def configure_tier(monkeypatch, pool, tier):
    state = us.get_user_state("reader")
    state.loaded = True
    if tier:
        state.add_positive("saved-seed")
    if tier == 1:
        async def clustered(uid, state, seen, limit, *, query_id):
            hits = [a for a in pool if a not in seen][:limit]
            tags = {a: {"candidate_source": f"cluster_{pool.index(a) % 2}",
                        "cluster_id": pool.index(a) % 2, "propensity": 1.0,
                        "policy_id": recs._RANKER_VERSION} for a in hits}
            return hits, [], tags, 0, {}
        monkeypatch.setattr(recs, "_multi_interest_recommend", clustered)
    elif tier == 2:
        async def ewma(uid, seen, limit):
            return [a for a in pool if a not in seen][:limit]
        monkeypatch.setattr(recs, "_ewma_recommend", ewma)
    elif tier == 3:
        async def recommend(**kw):
            return [a for a in pool if a not in kw['seen_arxiv_ids']][:kw['limit']]
        monkeypatch.setattr(qdrant_svc, "recommend", recommend)


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", cookies={"arxiv_user_id": "reader"})


@pytest.mark.parametrize("tier", [0, 1, 2, 3])
async def test_three_refreshes_advance_every_tier(world, monkeypatch, tier):
    configure_tier(monkeypatch, world, tier)
    async with client() as c:
        pages = [ids(await c.get('/api/recommendations')) for _ in range(3)]
    assert all(len(p) == 10 for p in pages)
    assert len(set(sum(pages, []))) == 30
    if tier == 1:
        assert all({world.index(a) % 2 for a in page} == {0, 1} for page in pages)
    assert set(sum(pages, [])) <= await db.get_impressed_ids('reader')


async def test_concurrent_refreshes_do_not_race_the_impression_write(world, monkeypatch):
    configure_tier(monkeypatch, world, 3)
    async with client() as c:
        responses = await asyncio.gather(*(c.get('/api/recommendations') for _ in range(3)))
    pages = [ids(r) for r in responses]
    assert all(len(p) == 10 for p in pages)
    assert len(set(sum(pages, []))) == 30


async def test_exhaustion_is_labelled_and_never_resurrects_decisions(world, monkeypatch):
    pool = world[:12]
    configure_tier(monkeypatch, pool, 3)
    await db.log_interaction('reader', pool[0], 'save')
    await db.log_interaction('reader', pool[1], 'not_interested')
    await db.record_impressions('reader', pool)
    async with client() as c:
        r = await c.get('/api/recommendations')
    assert len(ids(r)) == 10
    assert not set(ids(r)) & set(pool[:2])
    assert 'Previously discovered' in r.text
    assert 'fresh matches' in r.text
    # Exhaustion must not erase the reader's history.
    assert await db.get_impressed_ids('reader') == set(pool)


async def test_impression_cooldown_expires(world, monkeypatch):
    configure_tier(monkeypatch, world, 3)
    await db.record_impressions('reader', world[:10])
    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute("UPDATE feed_impressions SET shown_at = datetime('now','-8 days')")
        await conn.commit()
    async with client() as c:
        r = await c.get('/api/recommendations')
    assert ids(r) == world[:10]


async def test_cached_cursor_is_bound_to_its_reader(world, monkeypatch):
    configure_tier(monkeypatch, world, 3)
    recs._cache_put('stolen', {'user_id': 'someone-else', 'ranked': ['private-paper']})
    async with client() as c:
        r = await c.get('/api/recommendations?page=2&query_id=stolen')
    assert r.status_code == 200
    assert 'private-paper' not in r.text
    assert len(ids(r)) == 10


async def test_pagination_does_not_repeat_first_page(world, monkeypatch):
    configure_tier(monkeypatch, world, 3)
    async with client() as c:
        first = await c.get('/api/recommendations')
        qid = next(reversed(recs._FEED_CACHE))
        second = await c.get(f'/api/recommendations?page=2&query_id={qid}')
    assert len(ids(second)) == 10
    assert not set(ids(first)) & set(ids(second))


async def test_cold_start_keeps_unseen_before_recycling(world):
    await db.record_impressions('reader', world[:10])
    ordered, _ = await recs._cold_start_order('reader', world[:12])
    assert set(ordered[:2]) == set(world[10:12])
    assert len(ordered) == 12
    assert await db.get_impressed_ids('reader') == set(world[:10])


async def test_cached_page_excludes_decisions_outside_hot_state(world, monkeypatch):
    configure_tier(monkeypatch, world, 3)
    async with client() as c:
        await c.get('/api/recommendations')
        qid = next(reversed(recs._FEED_CACHE))
        # Durable feedback must win even when absent from the bounded hot deque.
        for pid in world[10:35]:
            await db.log_interaction('reader', pid, 'save')
        response = await c.get(f'/api/recommendations?page=2&query_id={qid}')
    assert len(ids(response)) == 10
    assert not set(ids(response)) & set(world[10:35])
