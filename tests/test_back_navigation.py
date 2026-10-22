"""
Going back to a search (reported 2026-10-21).

/search?q=... answers htmx with the results alone and a browser with the whole
page. Nothing told the browser the two differ, so after search -> open a paper
-> Back, it could show the cached results fragment as the page: no header, no
styles, no title. The browser test is in tests/test_e2e_search.py.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import hybrid_search_svc, turso_svc


@pytest.fixture
def client():
    from app.main import app

    async def fake_search(q, limit=20, return_meta=False):
        return ["2609.00001"], {"retrieval_mode": "keyword", "rewritten_query": None, "total_time_ms": 10}

    async def meta(ids):
        return {i: {"arxiv_id": i, "title": "Drone navigation", "abstract": "A.", "authors": "[]"} for i in ids}

    with patch.object(hybrid_search_svc, "search", side_effect=fake_search), \
            patch.object(turso_svc, "fetch_metadata_batch", side_effect=meta), TestClient(app) as c:
        yield c


def test_the_results_fragment_varies_on_htmx_and_is_never_stored(client):
    r = client.get("/search?q=drone", headers={"HX-Request": "true"})
    assert "HX-Request" in r.headers["vary"] and r.headers["cache-control"] == "no-store"
    # The fragment: no page around it (its <title> only renames the tab).
    assert "<html" not in r.text.lower() and 'class="searchbar"' not in r.text


def test_the_whole_page_says_it_varies_too(client):
    r = client.get("/search?q=drone")
    assert "HX-Request" in r.headers["vary"] and "no-store" not in r.headers.get("cache-control", "")
    assert "<title>" in r.text and 'class="searchbar"' in r.text


def test_htmx_restoring_history_gets_the_whole_page(client):
    r = client.get("/search?q=drone", headers={"HX-Request": "true", "HX-History-Restore-Request": "true"})
    assert "<title>" in r.text and 'class="searchbar"' in r.text


def test_static_files_keep_their_own_caching(client):
    r = client.get("/static/app.js")
    assert "no-store" not in r.headers.get("cache-control", "") and "HX-Request" not in r.headers.get("vary", "")
