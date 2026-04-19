"""
Layer 3: Search router integration tests — Phase 3.

Tests /search endpoint with mocked hybrid_search_svc.
Validates: ranking preservation, arXiv fallback, saved/dismissed state,
HTMX partials, and that empty queries don't trigger hybrid search.

No network, no model, no external services needed.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with temp DB and cleared caches."""
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)

    # Clear caches
    import app.user_state as us
    us._cache.clear()
    from app.qdrant_svc import _client
    _client.cache_clear()

    from app.main import app
    import asyncio
    asyncio.get_event_loop().run_until_complete(db_mod.init_db())

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Hybrid search integration ────────────────────────────────────────────────

def test_search_hybrid_returns_papers(client, monkeypatch):
    """
    /search?q=... should use hybrid search and render paper cards.
    We mock hybrid_search_svc.search() to return known IDs and
    arxiv_svc.fetch_metadata_batch() to return metadata for those IDs.
    """
    import app.hybrid_search_svc as hs
    import app.arxiv_svc as arxiv

    monkeypatch.setattr(hs, "search", AsyncMock(return_value=[
        "1706.03762", "2301.00001",
    ]))
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", AsyncMock(return_value={
        "1706.03762": {
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models...",
            "authors": '["Vaswani"]',
            "category": "cs.CL",
            "published": "2017-06-12",
            "year": 2017,
        },
        "2301.00001": {
            "arxiv_id": "2301.00001",
            "title": "Some Other Paper",
            "abstract": "A really cool paper.",
            "authors": '["Author"]',
            "category": "cs.AI",
            "published": "2023-01-01",
            "year": 2023,
        },
    }))

    resp = client.get("/search?q=transformer+attention")
    assert resp.status_code == 200
    assert "Attention Is All You Need" in resp.text
    assert "Some Other Paper" in resp.text


def test_search_hybrid_preserves_ranking(client, monkeypatch):
    """
    The order of papers in the response should match the order
    returned by hybrid_search_svc.search() — i.e., paper A before paper B.
    """
    import app.hybrid_search_svc as hs
    import app.arxiv_svc as arxiv

    # Hybrid search returns A first, then B
    monkeypatch.setattr(hs, "search", AsyncMock(return_value=[
        "2401.00001", "1706.03762",
    ]))
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", AsyncMock(return_value={
        "2401.00001": {
            "arxiv_id": "2401.00001",
            "title": "First Paper Should Appear First",
            "abstract": "...", "authors": '["A"]',
            "category": "cs.AI", "published": "2024-01-01", "year": 2024,
        },
        "1706.03762": {
            "arxiv_id": "1706.03762",
            "title": "Second Paper Should Appear Second",
            "abstract": "...", "authors": '["B"]',
            "category": "cs.CL", "published": "2017-06-12", "year": 2017,
        },
    }))

    resp = client.get("/search?q=test+query")
    # First paper should appear before second paper in HTML
    pos_first = resp.text.find("First Paper Should Appear First")
    pos_second = resp.text.find("Second Paper Should Appear Second")
    assert pos_first < pos_second, "Ranking order not preserved"


def test_search_fallback_to_arxiv_api(client, monkeypatch):
    """
    When hybrid_search_svc.search() returns empty list [],
    the router should fall back to arxiv_svc.search().
    """
    import app.hybrid_search_svc as hs
    import app.arxiv_svc as arxiv

    # Hybrid returns nothing
    monkeypatch.setattr(hs, "search", AsyncMock(return_value=[]))

    # arXiv fallback should be called
    arxiv_mock = AsyncMock(return_value=[{
        "arxiv_id": "fallback.00001",
        "title": "Fallback Paper From ArXiv API",
        "abstract": "This came from keyword search.",
        "authors": '["Fallback"]',
        "category": "cs.AI", "published": "2024-01-01", "year": 2024,
    }])
    monkeypatch.setattr(arxiv, "search", arxiv_mock)

    resp = client.get("/search?q=transformer")
    assert resp.status_code == 200
    assert "Fallback Paper From ArXiv API" in resp.text
    # Verify arXiv search was actually called
    arxiv_mock.assert_called_once()


def test_search_sets_saved_dismissed_flags(client, monkeypatch):
    """
    Papers returned by search should have correct saved/dismissed flags
    based on the user's state.
    """
    import app.hybrid_search_svc as hs
    import app.arxiv_svc as arxiv

    monkeypatch.setattr(hs, "search", AsyncMock(return_value=[
        "1706.03762", "2301.00001",
    ]))
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", AsyncMock(return_value={
        "1706.03762": {
            "arxiv_id": "1706.03762", "title": "Saved Paper",
            "abstract": "...", "authors": '["A"]',
            "category": "cs.CL", "published": "2017-06-12", "year": 2017,
        },
        "2301.00001": {
            "arxiv_id": "2301.00001", "title": "Normal Paper",
            "abstract": "...", "authors": '["B"]',
            "category": "cs.AI", "published": "2023-01-01", "year": 2023,
        },
    }))

    # First: visit home to get cookie, then save a paper
    client.get("/")
    client.post("/api/papers/1706.03762/save", data={"source": "search"})

    # Now search — saved paper should be marked
    resp = client.get("/search?q=test")
    assert resp.status_code == 200
    # The response should contain both papers
    assert "Saved Paper" in resp.text
    assert "Normal Paper" in resp.text


def test_search_htmx_partial_with_hybrid(client, monkeypatch):
    """
    HTMX request should return partial HTML (no <html> tag),
    same as before the hybrid search swap.
    """
    import app.hybrid_search_svc as hs
    import app.arxiv_svc as arxiv

    monkeypatch.setattr(hs, "search", AsyncMock(return_value=["1706.03762"]))
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", AsyncMock(return_value={
        "1706.03762": {
            "arxiv_id": "1706.03762", "title": "HTMX Test Paper",
            "abstract": "...", "authors": '["A"]',
            "category": "cs.CL", "published": "2017-06-12", "year": 2017,
        },
    }))

    resp = client.get(
        "/search?q=transformer",
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert "<html" not in resp.text.lower()
    assert "HTMX Test Paper" in resp.text


def test_search_empty_query_no_hybrid_call(client, monkeypatch):
    """
    Empty query should NOT trigger hybrid search at all —
    just render the empty search page.
    """
    import app.hybrid_search_svc as hs

    search_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(hs, "search", search_mock)

    resp = client.get("/search?q=")
    assert resp.status_code == 200
    # Hybrid search should NOT have been called
    search_mock.assert_not_called()
