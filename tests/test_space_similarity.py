"""GET /api/space/similarity with the vector shapes get_paper_vectors really returns.

get_paper_vectors yields numpy arrays, and `if va` on an array raises. The route
did exactly that, so every pair where both papers existed came back as HTTP 500
and the 3D map's compare panel fell back to "similarity unavailable".
"""
import numpy as np
import pytest

from app import qdrant_svc
from app.routers import space


@pytest.fixture
def vectors(monkeypatch):
    store = {
        "2410.24164": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "2504.16054": np.array([0.6, 0.8, 0.0], dtype=np.float32),
    }

    async def fake_vectors(ids):
        return {i: store[i] for i in ids if i in store}

    async def fake_titles(ids):
        return {i: f"title {i}" for i in ids}

    monkeypatch.setattr(qdrant_svc, "get_paper_vectors", fake_vectors)
    monkeypatch.setattr(space, "_titles_for", fake_titles)
    monkeypatch.setattr(space.config, "SPACE_SERVICE_TOKEN", "", raising=False)


async def test_similarity_of_two_known_papers(vectors):
    out = await space.space_similarity(a="2410.24164", b="2504.16054", authorization=None)
    assert out["cosine"] == pytest.approx(0.6)
    assert out["dimensions"] == 3
    assert out["a"]["found"] and out["b"]["found"]


async def test_similarity_with_a_missing_paper_is_null_not_zero(vectors):
    out = await space.space_similarity(a="2410.24164", b="9999.99999", authorization=None)
    assert out["cosine"] is None
    assert out["a"]["found"] and not out["b"]["found"]


def test_similarity_serialises_over_http(vectors):
    """Calling the function directly hid a 500: np.float32 is not JSON."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(space.router)
    r = TestClient(app).get("/api/space/similarity", params={"a": "2410.24164", "b": "2504.16054"})
    assert r.status_code == 200
    assert r.json()["cosine"] == pytest.approx(0.6)
