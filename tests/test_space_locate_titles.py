"""/api/space/locate returns each found paper's title, so the 3D map can name
a paper placed by its neighbours (it is in no spatial chunk) instead of
showing "arXiv <id>" (2026-11-27)."""
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient


def test_locate_adds_titles(monkeypatch):
    import app.map_locate_svc as mls
    import app.turso_svc as turso
    from app.main import app
    monkeypatch.setattr(mls, "locate", AsyncMock(return_value={
        "2611.00001": {"arxiv_id": "2611.00001", "position": [1.0, 2.0, 3.0], "source": "inferred"}}))
    monkeypatch.setattr(turso, "fetch_metadata_batch", AsyncMock(return_value={
        "2611.00001": {"arxiv_id": "2611.00001", "title": "A placed paper"}}))
    with TestClient(app) as c:
        out = c.get("/api/space/locate?arxiv_ids=2611.00001,2611.99999").json()
    assert out["results"][0]["title"] == "A placed paper" and out["results"][0]["source"] == "inferred"
    assert out["results"][1] is None


def test_locate_still_answers_when_titles_fail(monkeypatch):
    import app.map_locate_svc as mls
    import app.turso_svc as turso
    from app.main import app
    monkeypatch.setattr(mls, "locate", AsyncMock(return_value={
        "2611.00001": {"arxiv_id": "2611.00001", "position": [1.0, 2.0, 3.0], "source": "map"}}))
    monkeypatch.setattr(turso, "fetch_metadata_batch", AsyncMock(side_effect=RuntimeError("down")))
    with TestClient(app) as c:
        out = c.get("/api/space/locate?arxiv_ids=2611.00001").json()
    assert out["results"][0]["position"] == [1.0, 2.0, 3.0] and "title" not in out["results"][0]
