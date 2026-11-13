"""Small UI fixes from the 4 Nov audit (F14).

The corpus size comes from the sidecar instead of a number typed into three
templates; developer stage timings are for the owner only; the trending page
does not point at papers that are not there; the 404 page has a heading.
"""
from fastapi.testclient import TestClient

from app import config, local_meta
from app.templates_env import corpus_label, templates


def test_the_corpus_size_is_counted_not_typed(monkeypatch):
    monkeypatch.setattr(local_meta, "paper_count", lambda: 1_987_085)
    assert corpus_label() == "2.0M"
    monkeypatch.setattr(local_meta, "paper_count", lambda: 0)
    assert corpus_label() == "millions of"


def test_stage_timings_are_for_the_owner_only(tmp_path, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock
    import app.db as db_mod
    from app import arxiv_svc, hybrid_search_svc, turso_svc
    from app.main import app
    db_path = str(tmp_path / "t.db")
    for mod in (config, db_mod):
        monkeypatch.setattr(mod, "DB_PATH", db_path)
    asyncio.run(db_mod.init_db())
    monkeypatch.setattr(config, "PREVIEW_TOKEN", "owner-token")
    monkeypatch.setattr(hybrid_search_svc, "search", AsyncMock(return_value=(
        ["2611.00001"], {"encode_time_ms": 120, "total_time_ms": 900, "retrieval_mode": "hybrid"})))
    meta = {"2611.00001": {"arxiv_id": "2611.00001", "title": "A paper", "abstract": "About it.",
                           "authors": "[]", "category": "cs.LG", "published": "2026-11-01"}}
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", AsyncMock(return_value=meta))
    monkeypatch.setattr(arxiv_svc, "fetch_metadata_batch", AsyncMock(return_value={}))
    with TestClient(app) as c:
        reader = c.get("/search?q=timings+test+one").text
        owner = c.get("/search?q=timings+test+two&preview=owner-token").text
        owner_again = c.get("/search?q=timings+test+three").text     # the cookie now set
    assert "A paper" in reader and "Pipeline timings" not in reader
    assert "Pipeline timings" in owner or "Pipeline timings" in owner_again


def _trending(**ctx) -> str:
    from starlette.requests import Request
    req = Request({"type": "http", "method": "GET", "path": "/trending", "query_string": b"",
                   "headers": [], "server": ("test", 80), "scheme": "http", "root_path": ""})
    base = {"request": req, "rising": [], "cooling": [], "areas": [], "chips": [], "for_you": [],
            "top": [], "hn_lately": [], "basis": {"failed": True}, "window": None,
            "lasting_visible": False}
    return templates.get_template("trending.html").render(**{**base, **ctx})


def test_a_trending_failure_points_at_papers_only_when_there_are_some():
    assert "The papers above" not in _trending()
    assert "The papers above are still this week's" in _trending(
        top=[{"arxiv_id": "2611.00001", "title": "T", "authors": "A", "abstract": "x"}])


def test_the_404_page_has_a_heading():
    from app.main import app
    with TestClient(app) as c:
        page = c.get("/no-such-page-here")
    assert page.status_code == 404 and "<h1" in page.text


def test_startup_log_names_the_real_fallback():
    import inspect
    from app import main
    assert "fall back to arXiv API" not in inspect.getsource(main)
    assert config.SPARSE_BACKEND == "fts"
