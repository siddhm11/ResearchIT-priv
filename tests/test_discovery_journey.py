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
