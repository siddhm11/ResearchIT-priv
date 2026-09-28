"""Preferences, seed suggestions, and honest reading history in isolated storage."""
from unittest.mock import AsyncMock

import httpx
import pytest

from app import db, discovery_svc, qdrant_svc, turso_svc, user_state as us
from app.main import app


@pytest.fixture
async def client(monkeypatch):
    await db.init_db()
    async def metadata(ids):
        return {a: {'arxiv_id': a, 'title': 'Paper ' + a, 'abstract': 'Study abstract.',
                    'authors': '[]', 'category': 'cs.CL', 'published': '2026-01-01'} for a in ids}
    monkeypatch.setattr(turso_svc, 'fetch_metadata_batch', metadata)
    monkeypatch.setattr(qdrant_svc, 'get_paper_vectors', AsyncMock(return_value={}))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                               cookies={'arxiv_user_id': 'reader'}) as c:
        yield c


async def test_completed_reader_can_change_interests_without_losing_library(client):
    await db.complete_onboarding('reader')
    await db.log_interaction('reader', '2601.00001', 'save')
    page = await client.get('/interests')
    assert page.status_code == 200 and 'Research areas' in page.text
    response = await client.post('/interests', data={'categories': ['nlp', 'cv']})
    assert response.status_code == 303
    state = await db.get_onboarding_state('reader')
    assert state['selected_categories'] == ['nlp', 'cv']
    assert state['onboarding_completed'] == 1
    assert [r['paper_id'] for r in await db.get_save_history('reader')] == ['2601.00001']


@pytest.mark.parametrize('payload', [{'categories':'nlp'}, {'categories':[{}]},
                                    {'categories':['unknown']}, {'categories':['nlp']*9}, []])
async def test_invalid_interests_do_not_write_state(client, payload):
    r = await client.post('/api/onboarding/categories', json=payload)
    assert r.status_code == 422
    assert await db.get_onboarding_state('reader') is None


async def test_interest_form_limits_are_also_enforced(client):
    r = await client.post('/interests', data={'categories': ['nlp']*9})
    assert r.status_code == 422
    assert "Choose up to 8 available areas." in r.text
    assert 'action="/interests"' in r.text


async def test_starter_suggestions_use_selected_categories_and_mark_saved(client, monkeypatch):
    await db.save_onboarding_categories('reader', ['nlp'])
    await db.log_interaction('reader', '2601.00001', 'save')
    starter = AsyncMock(return_value=[{'arxiv_id':'2601.00001','title':'An NLP seed',
                                       'category':'cs.CL','year':2026}])
    monkeypatch.setattr(discovery_svc, 'starter_papers', starter)
    r = await client.get('/api/onboarding/seed-search')
    starter.assert_awaited_once_with({'cs.CL','cs.IR'}, limit=12)
    assert 'An NLP seed' in r.text and 'Saved' in r.text
    assert 'hx-post=' not in r.text


async def test_history_is_private_deduplicated_and_recent_first(client):
    for pid in ['2601.00001','2601.00002','2601.00001']:
        await db.log_interaction('reader', pid, 'click', query_id='q1')
    await db.log_interaction('other', 'private-paper', 'click')
    rows = await db.get_recent_papers('reader')
    assert [r['paper_id'] for r in rows] == ['2601.00001','2601.00002']
    r = await client.get('/history')
    assert 'not a list of completed reads' in r.text
    assert 'private-paper' not in r.text
    assert 'Paper 2601.00001' in r.text
    assert (await client.get('/history?view=bad')).status_code == 422


async def test_direct_visits_are_history_without_fake_clicks_or_profile_updates(client):
    r = await client.get('/p/2601.00001')
    assert r.status_code == 200
    rows = await db.get_user_interactions('reader')
    assert [r['event_type'] for r in rows] == ['view']
    assert [r['paper_id'] for r in await db.get_recent_papers('reader')] == ['2601.00001']
    assert (await us.ensure_loaded('reader')).positive_list == []
    assert await db.get_user_profile('reader', 'long_term') is None


async def test_opening_a_saved_paper_preserves_its_interest_signal(client):
    await db.log_interaction('reader','2601.00001','save')
    await client.get('/p/2601.00001')
    await db.log_interaction('reader','2601.00001','click')
    assert [r['paper_id'] for r in await db.get_save_history('reader')] == ['2601.00001']


async def test_unsave_survives_cache_reload_and_is_not_a_dislike(client):
    await db.log_interaction('reader', '2601.00001', 'save')
    await db.log_interaction('reader', '2601.00001', 'unsave')
    await db.log_interaction('reader', '2601.00001', 'view')
    us._cache.clear()
    state = await us.ensure_loaded('reader')
    assert not state.positive_list and not state.negative_list


async def test_library_is_not_limited_to_retrieval_deque(client):
    for i in range(25):
        await db.log_interaction('reader', f'2601.{i:05d}', 'save')
    r = await client.get('/saved')
    assert r.text.count('data-arxiv-id=') == 25


async def test_discovery_history_survives_missing_metadata(client, monkeypatch):
    await db.record_impressions('reader', ['2601.00001'])
    monkeypatch.setattr(turso_svc, 'fetch_metadata_batch', AsyncMock(side_effect=RuntimeError('offline')))
    r = await client.get('/history?view=discovered')
    assert r.status_code == 200 and 'arXiv:2601.00001' in r.text
    assert 'whether or not you opened them' in r.text


async def test_empty_history_guides_reader_back_to_feed(client):
    r = await client.get('/history')
    assert r.status_code == 200 and 'Open a paper' in r.text


async def test_balanced_starters_include_thin_categories(monkeypatch):
    monkeypatch.setattr(discovery_svc.local_meta, 'is_available', lambda: True)
    async def trending(cats, limit):
        return [{'arxiv_id': 'shared'}] + [{'arxiv_id': next(iter(cats))+str(i)} for i in range(8)]
    monkeypatch.setattr(turso_svc, 'fetch_trending_by_categories', trending)
    papers = await discovery_svc.starter_papers({'math.PR','cs.CL'}, limit=6)
    ids = [p['arxiv_id'] for p in papers]
    assert len(ids) == len(set(ids)) == 6
    assert any(a.startswith('math.PR') for a in ids[:3])
    assert any(a.startswith('cs.CL') for a in ids[:3])


async def test_without_sidecar_starters_make_one_remote_request(monkeypatch):
    monkeypatch.setattr(discovery_svc.local_meta, 'is_available', lambda: False)
    fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(turso_svc, 'fetch_trending_by_categories', fetch)
    await discovery_svc.starter_papers({'math.PR','cs.CL'}, limit=20)
    fetch.assert_awaited_once_with({'math.PR','cs.CL'}, limit=20)


async def test_direct_paper_visit_keeps_history_but_not_skip_onboarding(client, monkeypatch):
    """A shared link is often a reader's first visit; it must not count as history."""
    from app.routers import paper
    monkeypatch.setattr(paper, '_related', AsyncMock(return_value=[]))
    assert (await client.get('/')).headers['location'].endswith('/onboarding')
    assert (await client.get('/p/2305.04120')).status_code == 200
    assert [r['paper_id'] for r in await db.get_recent_papers('reader')] == ['2305.04120']
    home = await client.get('/')
    assert home.status_code == 302 and home.headers['location'].endswith('/onboarding')
    assert (await db.get_onboarding_state('reader')) is None


async def test_cookieless_paper_visit_records_nothing(monkeypatch):
    """Crawlers and link previews carry no cookie; each would mint a phantom user."""
    from app.routers import paper
    await db.init_db()
    async def metadata(ids):
        return {a: {'arxiv_id': a, 'title': 'T', 'abstract': 'A', 'authors': '[]',
                    'category': 'cs.CL', 'published': '2026-01-01'} for a in ids}
    monkeypatch.setattr(turso_svc, 'fetch_metadata_batch', metadata)
    monkeypatch.setattr(paper, '_related', AsyncMock(return_value=[]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as c:
        assert (await c.get('/p/2305.04120')).status_code == 200
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        n = (await (await conn.execute("SELECT COUNT(*) FROM interactions")).fetchone())[0]
    assert n == 0
