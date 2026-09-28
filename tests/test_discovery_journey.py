"""Preferences, seed suggestions, and honest reading history in isolated storage."""
from unittest.mock import AsyncMock

import httpx
import pytest

from app import db, qdrant_svc, turso_svc, user_state as us
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
