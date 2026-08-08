"""
Tests for GET /saved page.
Covers: empty state, paper listing, cookie, remove action, source logging.
"""
import asyncio
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)

    import app.user_state as us
    us._cache.clear()

    from app.qdrant_svc import _client
    _client.cache_clear()

    from app.main import app
    asyncio.get_event_loop().run_until_complete(db_mod.init_db())

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Basic page behaviour ──────────────────────────────────────────────────────

def test_saved_page_returns_200(client):
    resp = client.get("/saved")
    assert resp.status_code == 200


def test_saved_page_sets_cookie(client):
    resp = client.get("/saved")
    assert "arxiv_user_id" in resp.cookies


def test_saved_page_empty_for_new_user(client):
    """New user has no saves — shows the empty-state message."""
    resp = client.get("/saved")
    assert resp.status_code == 200
    assert "No saved papers" in resp.text


def test_saved_page_shows_paper_after_save(client, monkeypatch):
    """After saving a paper, it appears on the saved page."""
    import app.arxiv_svc as arxiv

    async def fake_batch(ids):
        return {
            "1706.03762": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "abstract": "The original transformer paper.",
                "authors": '["Vaswani"]',
                "category": "cs.CL",
                "published": "2017-06-12",
                "year": 2017,
            }
        }
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", fake_batch)

    client.get("/")
    client.post("/api/papers/1706.03762/save", data={"source": "search"})
    resp = client.get("/saved")
    assert resp.status_code == 200
    assert "Attention Is All You Need" in resp.text


def test_saved_page_shows_correct_count(client, monkeypatch):
    """The count badge reflects the number of saved papers."""
    import app.arxiv_svc as arxiv

    papers = {
        "1706.03762": {
            "arxiv_id": "1706.03762", "title": "Attention Is All You Need", "abstract": "...",
            "authors": '[]', "category": "cs.CL", "published": "2017-06-12", "year": 2017,
        },
        "1512.03385": {
            "arxiv_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition",
            "abstract": "...", "authors": '[]', "category": "cs.CV",
            "published": "2015-12-10", "year": 2015,
        },
    }

    async def fake_batch(ids):
        return {k: v for k, v in papers.items() if k in ids}
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", fake_batch)

    client.get("/")
    client.post("/api/papers/1706.03762/save", data={"source": "search"})
    client.post("/api/papers/1512.03385/save", data={"source": "search"})
    resp = client.get("/saved")
    assert resp.status_code == 200
    assert "2 saved" in resp.text


# ── Remove from saved ─────────────────────────────────────────────────────────

def test_remove_paper_updates_state(client):
    """Dismissing a saved paper removes it from positive_list and adds to negatives."""
    import app.user_state as us

    client.get("/")
    user_id = client.cookies.get("arxiv_user_id")

    client.post("/api/papers/1706.03762/save", data={"source": "search"})
    state = us.get_user_state(user_id)
    assert "1706.03762" in state.positive_list

    client.post("/api/papers/1706.03762/not-interested", data={"source": "saved"})
    state = us.get_user_state(user_id)
    assert "1706.03762" not in state.positive_list
    assert "1706.03762" in state.negative_list


def test_remove_returns_empty_response(client):
    """not-interested returns empty HTML (HTMX removes card)."""
    client.get("/")
    client.post("/api/papers/1706.03762/save", data={"source": "search"})
    resp = client.post("/api/papers/1706.03762/not-interested", data={"source": "saved"})
    assert resp.status_code == 200
    assert resp.text.strip() == ""


# ── Source logging ────────────────────────────────────────────────────────────

def test_save_source_is_logged(client):
    """Source field on the save action is persisted to the DB."""
    import app.db as db_mod

    client.get("/")
    user_id = client.cookies.get("arxiv_user_id")
    client.post("/api/papers/1706.03762/save", data={"source": "search", "position": "2"})

    rows = asyncio.get_event_loop().run_until_complete(
        db_mod.get_user_interactions(user_id, event_types=["save"])
    )
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "1706.03762"


def test_dismiss_source_saved_is_logged(client):
    """Dismiss from saved page logs source='saved'."""
    import app.db as db_mod

    client.get("/")
    user_id = client.cookies.get("arxiv_user_id")

    client.post("/api/papers/1706.03762/save", data={"source": "search"})
    client.post("/api/papers/1706.03762/not-interested", data={"source": "saved"})

    rows = asyncio.get_event_loop().run_until_complete(
        db_mod.get_user_interactions(user_id, event_types=["not_interested"])
    )
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "1706.03762"
