"""
SEARCH_SHELL: a full page load of a search not in memory sends the page at
once and fetches its results over htmx (UI review, 2026-11-24: the tab stayed
blank for ~4 s while both models ran). A search in memory, an htmx request
and SEARCH_SHELL=0 render the results in the response, as before.
"""
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app import search_cache

META = {"1706.03762": {"arxiv_id": "1706.03762", "title": "Attention Is All You Need",
                       "abstract": "Transformers.", "authors": '["Vaswani"]', "category": "cs.CL",
                       "published": "2017-06-12", "year": 2017}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    import asyncio

    import app.config as cfg
    import app.db as db_mod
    import app.user_state as us
    from app.main import app
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    us._cache.clear()
    search_cache.clear()
    asyncio.run(db_mod.init_db())
    monkeypatch.setenv("SEARCH_SHELL", "1")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def searched(monkeypatch):
    import app.arxiv_svc as arxiv
    import app.hybrid_search_svc as hs
    import app.turso_svc as turso
    run = AsyncMock(return_value=(["1706.03762"], {}))
    monkeypatch.setattr(hs, "search", run)
    monkeypatch.setattr(turso, "fetch_metadata_batch", AsyncMock(return_value=META))
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", AsyncMock(return_value={}))
    return run


def test_the_page_comes_first_and_fetches_its_results(client, searched):
    page = client.get("/search?q=shell+test+one&sort=newest")
    assert page.status_code == 200 and searched.await_count == 0           # no model ran yet
    assert 'value="shell test one"' in page.text and "Attention Is All You Need" not in page.text
    assert 'data-search-deferred hx-get="/search?q=shell+test+one&amp;sort=newest"' in page.text
    assert 'id="search-loading" class="htmx-indicator htmx-indicator--block htmx-request"' in page.text
    results = client.get("/search?q=shell+test+one", headers={"HX-Request": "true"})
    assert "Attention Is All You Need" in results.text and "<html" not in results.text
    assert searched.await_count == 1


def test_a_search_in_memory_is_rendered_in_the_page(client, searched):
    client.get("/search?q=shell+test+two", headers={"HX-Request": "true"})
    again = client.get("/search?q=shell+test+two")
    assert "Attention Is All You Need" in again.text and "data-search-deferred" not in again.text
    assert searched.await_count == 1


def test_switched_off_the_page_waits_for_its_results(client, searched, monkeypatch):
    monkeypatch.setenv("SEARCH_SHELL", "0")
    page = client.get("/search?q=shell+test+three")
    assert "Attention Is All You Need" in page.text and "data-search-deferred" not in page.text


def test_an_empty_box_is_just_the_page(client, searched):
    page = client.get("/search")
    assert page.status_code == 200 and "data-search-deferred" not in page.text
    assert searched.await_count == 0
