"""
Integration tests: full HTTP request/response cycle via FastAPI TestClient.
Tests the complete pipeline: search → save → recommendations.
"""
import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)

    # Clear the user_state in-process cache between tests
    import app.user_state as us
    us._cache.clear()

    # Clear qdrant client cache
    from app.qdrant_svc import _client
    _client.cache_clear()

    from app.main import app
    import asyncio
    asyncio.get_event_loop().run_until_complete(db_mod.init_db())

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Home page ─────────────────────────────────────────────────────────────────

def test_home_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ArXiv" in resp.text


def test_home_sets_user_cookie(client):
    resp = client.get("/")
    assert "arxiv_user_id" in resp.cookies


# ── Search ────────────────────────────────────────────────────────────────────

def test_search_empty_query_returns_page(client):
    resp = client.get("/search?q=")
    assert resp.status_code == 200


def test_search_htmx_returns_partial(client):
    """With HX-Request header, /search should return partial HTML (no <html> tag)."""
    resp = client.get(
        "/search?q=transformer+attention",
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    # Partial should NOT contain the full base layout
    assert "<html" not in resp.text.lower()


def test_search_real_query_returns_papers(client):
    """Real search against arXiv API — should find results."""
    resp = client.get("/search?q=transformer+attention+mechanism")
    assert resp.status_code == 200
    # Should contain at least one arxiv ID pattern in the response
    assert "arxiv.org/abs/" in resp.text


# ── Save / not-interested events ──────────────────────────────────────────────

def test_save_paper_logs_interaction(client, tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    # Use cookie from previous request for consistent user_id
    client.get("/")  # sets cookie
    user_id = client.cookies.get("arxiv_user_id")

    resp = client.post(
        "/api/papers/1706.03762/save",
        data={"source": "search", "position": "0"},
    )
    assert resp.status_code == 200
    # Response should contain the "Saved" state button
    assert "Saved" in resp.text or "saved" in resp.text.lower()


def test_not_interested_returns_empty(client):
    client.get("/")
    resp = client.post(
        "/api/papers/1706.03762/not-interested",
        data={"source": "search"},
    )
    assert resp.status_code == 200
    # Empty response so HTMX removes the card
    assert resp.text.strip() == ""


def test_save_updates_user_state(client):
    import app.user_state as us
    client.get("/")
    user_id = client.cookies.get("arxiv_user_id")

    client.post("/api/papers/1706.03762/save", data={"source": "search"})
    state = us.get_user_state(user_id)
    assert "1706.03762" in state.positive_list


def test_not_interested_updates_user_state(client):
    import app.user_state as us
    client.get("/")
    user_id = client.cookies.get("arxiv_user_id")

    client.post("/api/papers/2302.11382/not-interested", data={"source": "search"})
    state = us.get_user_state(user_id)
    assert "2302.11382" in state.negative_list


# ── Recommendations ───────────────────────────────────────────────────────────

def test_recommendations_empty_for_new_user(client):
    client.get("/")
    resp = client.get("/api/recommendations")
    assert resp.status_code == 200
    # Should show empty state message
    assert "No recommendations" in resp.text or "Save" in resp.text


def test_recommendations_after_save(client, monkeypatch):
    """
    After saving a real paper, the recommendations endpoint should
    return something (possibly recs or empty-recs if Qdrant lookup is slow).
    """
    import app.qdrant_svc as qs
    import app.db as db_mod

    # Pre-seed the Qdrant map so recommend() can find the paper
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        db_mod.save_qdrant_id("0704.0002", 0)
    )

    # Mock recommend to return a known paper ID
    async def fake_recommend(positive_arxiv_ids, negative_arxiv_ids, seen_arxiv_ids, limit):
        return ["1706.03762"]
    monkeypatch.setattr(qs, "recommend", fake_recommend)

    # Also mock metadata fetch so we don't hit Turso DB in this test
    import app.turso_svc as turso
    import app.arxiv_svc as arxiv
    async def fake_batch(ids):
        return {
            "1706.03762": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "abstract": "Transformers are great.",
                "authors": '["Vaswani"]',
                "category": "cs.CL",
                "published": "2017-06-12",
                "year": 2017,
            }
        }
    monkeypatch.setattr(turso, "fetch_metadata_batch", fake_batch)
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", AsyncMock(return_value={}))

    client.get("/")
    client.post("/api/papers/0704.0002/save", data={"source": "search"})
    resp = client.get("/api/recommendations")
    assert resp.status_code == 200
    assert "Attention Is All You Need" in resp.text


# ── Full pipeline smoke test ───────────────────────────────────────────────────

def test_quota_pipeline_preserves_minority_cluster(client, monkeypatch):
    """
    Phase 4.1 end-to-end check: with 5+ saves forming 2 distinct interests,
    the quota pipeline must surface papers from BOTH clusters in the final feed.
    This is the exact failure mode RRF was causing.
    """
    import numpy as np
    import app.qdrant_svc as qs
    import app.turso_svc as turso
    import app.arxiv_svc as arxiv
    import app.recommend.profiles as prof_mod

    # Set up cookie
    client.get("/")

    # 5 saved papers, split into two topics (3 "NLP", 2 "RL") via embeddings
    saved_ids = ["nlp_a", "nlp_b", "nlp_c", "rl_a", "rl_b"]
    rng = np.random.RandomState(42)
    nlp_center = rng.randn(1024).astype(np.float32)
    nlp_center /= np.linalg.norm(nlp_center)
    rl_center = rng.randn(1024).astype(np.float32)
    rl_center /= np.linalg.norm(rl_center)

    def _near(center):
        v = center + rng.randn(1024).astype(np.float32) * 0.05
        return (v / np.linalg.norm(v)).tolist()

    saved_vectors = {
        "nlp_a": _near(nlp_center),
        "nlp_b": _near(nlp_center),
        "nlp_c": _near(nlp_center),
        "rl_a": _near(rl_center),
        "rl_b": _near(rl_center),
    }

    # Candidate pool: 50 NLP-ish, 50 RL-ish
    candidate_vectors = {}
    nlp_candidates = [f"nlp_cand_{i}" for i in range(50)]
    rl_candidates = [f"rl_cand_{i}" for i in range(50)]
    for cid in nlp_candidates:
        candidate_vectors[cid] = _near(nlp_center)
    for cid in rl_candidates:
        candidate_vectors[cid] = _near(rl_center)

    async def fake_get_paper_vectors(ids):
        combined = {**saved_vectors, **candidate_vectors}
        return {aid: combined[aid] for aid in ids if aid in combined}

    # search_by_vector returns candidates aligned with whichever centre
    # the query is closer to
    async def fake_search_by_vector(query_vector, limit, exclude_ids=None):
        qv = np.array(query_vector, dtype=np.float32)
        qv /= np.linalg.norm(qv)
        if float(qv @ nlp_center) > float(qv @ rl_center):
            pool = nlp_candidates
        else:
            pool = rl_candidates
        exclude = exclude_ids or set()
        return [p for p in pool if p not in exclude][:limit]

    monkeypatch.setattr(qs, "get_paper_vectors", fake_get_paper_vectors)
    monkeypatch.setattr(qs, "search_by_vector", fake_search_by_vector)

    # Skip EWMA short-term lookup — returns None
    async def fake_load_profile(uid, kind):
        return None
    monkeypatch.setattr(prof_mod, "load_profile", fake_load_profile)

    async def fake_interaction_count(uid, kind):
        return 0
    monkeypatch.setattr(prof_mod, "get_interaction_count", fake_interaction_count)

    # Metadata: provide category so templates render
    async def fake_meta(ids):
        return {
            aid: {
                "arxiv_id": aid,
                "title": f"Title {aid}",
                "abstract": "...",
                "authors": "[]",
                "category": "cs.CL" if aid.startswith("nlp") else "cs.LG",
                "published": "2024-01-01",
                "year": 2024,
            }
            for aid in ids
        }
    monkeypatch.setattr(turso, "fetch_metadata_batch", fake_meta)
    from unittest.mock import AsyncMock
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", AsyncMock(return_value={}))

    # Save 5 papers to cross the MIN_PAPERS_FOR_CLUSTERING threshold
    for aid in saved_ids:
        client.post(f"/api/papers/{aid}/save", data={"source": "search"})

    resp = client.get("/api/recommendations")
    assert resp.status_code == 200

    # The response should include recs from BOTH candidate pools (quota working)
    has_nlp_rec = any(f"nlp_cand_{i}" in resp.text for i in range(50))
    has_rl_rec = any(f"rl_cand_{i}" in resp.text for i in range(50))
    assert has_nlp_rec, "No NLP cluster recs — dominant cluster failed to surface"
    assert has_rl_rec, "Minority RL cluster starved — quota fusion is not working"


def test_full_pipeline_smoke(client, monkeypatch):
    """
    1. User visits home → gets cookie
    2. Searches for 'attention transformer'
    3. Saves first result
    4. Gets recommendations (mocked Qdrant + arXiv)
    """
    import app.qdrant_svc as qs
    import app.arxiv_svc as arxiv

    saved_ids = []

    # Step 1: Home
    resp = client.get("/")
    assert resp.status_code == 200
    user_id = client.cookies.get("arxiv_user_id")
    assert user_id

    # Step 2: Search
    resp = client.get("/search?q=attention+transformer")
    assert resp.status_code == 200
    # Extract any arxiv ID from the response HTML
    import re
    ids_found = re.findall(r'\[(\d{4}\.\d{4,5})\]', resp.text)

    # Step 3: Save — use a known paper ID to avoid depending on search order
    test_paper_id = "1706.03762"
    resp = client.post(
        f"/api/papers/{test_paper_id}/save",
        data={"source": "search", "position": "0"},
    )
    assert resp.status_code == 200

    # Step 4: Recommendations (mock to avoid full Qdrant integration here)
    async def fake_rec(positive_arxiv_ids, negative_arxiv_ids, seen_arxiv_ids, limit):
        return ["2302.11382"]
    monkeypatch.setattr(qs, "recommend", fake_rec)

    import app.turso_svc as turso
    async def fake_meta(ids):
        return {
            "2302.11382": {
                "arxiv_id": "2302.11382",
                "title": "Principled Instructions Are All You Need",
                "abstract": "Better prompts.",
                "authors": '["Smith"]',
                "category": "cs.CL",
                "published": "2023-02-22",
                "year": 2023,
            }
        }
    monkeypatch.setattr(turso, "fetch_metadata_batch", fake_meta)
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", AsyncMock(return_value={}))

    resp = client.get("/api/recommendations")
    assert resp.status_code == 200
    assert "Principled Instructions" in resp.text
