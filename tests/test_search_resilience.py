"""Lexical search must survive embedding and retriever failures."""
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app import config, hybrid_search_svc as hs


@pytest.fixture
def search_world(monkeypatch):
    monkeypatch.setattr(config, 'SPARSE_BACKEND', 'fts')
    monkeypatch.setattr(config, 'SEARCH_BGE_RERANK', False)
    monkeypatch.setattr(hs.fts_svc, 'is_available', lambda: True)
    monkeypatch.setattr(hs.groq_svc, 'rewrite', AsyncMock(return_value='rewritten topic'))
    monkeypatch.setattr(hs.qdrant_svc, 'search_dense_merged', AsyncMock(return_value=[]))
    monkeypatch.setattr(hs.zilliz_svc, 'search_sparse', AsyncMock(return_value=[]))
    monkeypatch.setattr(hs.turso_svc, 'fetch_metadata_batch', AsyncMock(return_value={}))
    monkeypatch.setattr(hs.arxiv_svc, 'fetch_metadata_batch', AsyncMock(return_value={}))
    lexical = AsyncMock(return_value=[{'arxiv_id':'2601.00001','score':1.0}])
    monkeypatch.setattr(hs.fts_svc, 'search_sparse', lexical)
    return lexical


async def test_fts_works_when_every_encode_fails(search_world, monkeypatch):
    def fail(text):
        raise RuntimeError('embedding model unavailable')
    monkeypatch.setattr(hs.embed_svc, 'encode_query', fail)
    ids, meta = await hs.search('original topic', return_meta=True)
    assert ids == ['2601.00001']
    assert meta['retrieval_mode'] == 'keyword'
    assert [c.args[0] for c in search_world.await_args_list] == ['original topic', 'rewritten topic']
    hs.qdrant_svc.search_dense_merged.assert_not_awaited()


@pytest.mark.parametrize('failed_text', ['original topic', 'rewritten topic'])
async def test_each_lexical_query_survives_partial_encoding_failure(search_world, monkeypatch, failed_text):
    def encode(text):
        if text == failed_text:
            raise RuntimeError('one encode failed')
        return np.ones(1024, dtype=np.float32), {1:0.5}
    monkeypatch.setattr(hs.embed_svc, 'encode_query', encode)
    ids, meta = await hs.search('original topic', return_meta=True)
    assert ids
    assert [c.args[0] for c in search_world.await_args_list] == ['original topic', 'rewritten topic']
    hs.qdrant_svc.search_dense_merged.assert_awaited_once()
    expected = 'qdrant_q1' if failed_text == 'original topic' else 'qdrant_q0'
    # The successful dense query is labelled by its original query identity.
    hs.qdrant_svc.search_dense_merged.return_value = [{'arxiv_id':'2601.00002','score':0.8}]
    _, meta = await hs.search('original topic', return_meta=True)
    assert expected in meta['retrieval_sources']
    assert meta['retrieval_mode'] == 'hybrid'


async def test_no_duplicate_lexical_list_when_rewrite_unchanged(search_world, monkeypatch):
    monkeypatch.setattr(hs.embed_svc, 'encode_query', lambda q: (np.ones(1024), {}))
    monkeypatch.setattr(hs.groq_svc, 'rewrite', AsyncMock(return_value='original topic'))
    await hs.search('original topic')
    search_world.assert_awaited_once_with('original topic', limit=60)


async def test_lexical_outage_keeps_dense_results(search_world, monkeypatch):
    monkeypatch.setattr(hs.embed_svc, 'encode_query', lambda q: (np.ones(1024), {}))
    search_world.side_effect = RuntimeError('fts unavailable')
    hs.qdrant_svc.search_dense_merged.return_value = [{'arxiv_id':'2601.00002','score':0.8}]
    ids, meta = await hs.search('original topic', return_meta=True)
    assert ids == ['2601.00002']
    assert meta['retrieval_mode'] == 'semantic'


async def test_without_fts_or_embeddings_returns_empty_for_router_fallback(search_world, monkeypatch):
    monkeypatch.setattr(hs.fts_svc, 'is_available', lambda: False)
    def fail(text):
        raise RuntimeError('no model')
    monkeypatch.setattr(hs.embed_svc, 'encode_query', fail)
    ids, meta = await hs.search('original topic', return_meta=True)
    assert ids == [] and meta['retrieval_mode'] == 'unavailable'
    hs.zilliz_svc.search_sparse.assert_not_awaited()
