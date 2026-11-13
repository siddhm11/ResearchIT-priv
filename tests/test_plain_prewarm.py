"""Plain-language lines written ahead for the papers new readers see first (audit F21)."""
import asyncio

import pytest

from app import db, groq_svc, plain_prewarm, turso_svc
from app.llm_router import Generation


@pytest.fixture
def world(tmp_path, monkeypatch):
    import app.config as cfg
    path = str(tmp_path / "plain.db")
    for mod in (cfg, db):
        monkeypatch.setattr(mod, "DB_PATH", path)
    asyncio.run(db.init_db())
    plain_prewarm._state.update(day="", written=0)
    monkeypatch.setattr(plain_prewarm, "SPACING_S", 0)
    ids = [f"2611.0000{i}" for i in range(5)]

    async def candidates():
        return ids

    async def meta(want):
        return {a: {"arxiv_id": a, "title": f"Paper {a}", "abstract": "A full abstract. " * 20} for a in want}
    monkeypatch.setattr(plain_prewarm, "candidates", candidates)
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", meta)
    calls = []

    async def explain(title, abstract, arxiv_id=None):
        calls.append(arxiv_id)
        return Generation("success", f"In short: {arxiv_id}", "model-a")
    monkeypatch.setattr(groq_svc, "explain_paper_result", explain)
    monkeypatch.setattr(groq_svc, "explanation_models", lambda: ("model-a",))
    return ids, calls


def test_each_paper_gets_one_stored_line(world):
    ids, calls = world
    assert asyncio.run(plain_prewarm.run_once()) == 5
    key = groq_svc.explain_cache_key(ids[0], "A full abstract. " * 20, model="model-a")
    assert asyncio.run(db.get_explanation(key)) == f"In short: {ids[0]}"
    assert asyncio.run(plain_prewarm.run_once()) == 0          # already explained: no second call
    assert len(calls) == 5


def test_the_daily_cap_holds(world, monkeypatch):
    monkeypatch.setattr(plain_prewarm, "DAILY_MAX", 2)
    assert asyncio.run(plain_prewarm.run_once()) == 2
    assert asyncio.run(plain_prewarm.run_once()) == 0
    assert plain_prewarm.stats()["left_today"] == 0


def test_three_failures_in_a_row_stop_the_pass(world, monkeypatch):
    ids, calls = world

    async def down(title, abstract, arxiv_id=None):
        calls.append(arxiv_id)
        return Generation("unavailable", None, None)
    monkeypatch.setattr(groq_svc, "explain_paper_result", down)
    assert asyncio.run(plain_prewarm.run_once()) == 0
    assert len(calls) == 3 and plain_prewarm.stats()["status"] == "stopped after failures"


def test_candidates_are_starter_first_pages_then_trending(monkeypatch):
    from app import config, discovery_svc, trending_shelf
    monkeypatch.setattr(config, "CATEGORY_GROUPS", {"ml": {"arxiv": ["cs.LG"]}, "cv": {"arxiv": ["cs.CV"]}})

    async def starter(shape, limit=12):
        key = next(iter(shape))
        return [{"arxiv_id": f"{key}-{i}"} for i in range(limit)]
    monkeypatch.setattr(discovery_svc, "starter_papers", starter)
    monkeypatch.setattr(trending_shelf, "trending_now", lambda: {
        "weak": {"strength": 0.2}, "strong": {"strength": 0.9}, "ml-0": {"strength": 0.5}})
    got = asyncio.run(plain_prewarm.candidates())
    assert got[:6] == [f"ml-{i}" for i in range(6)] and got[6:12] == [f"cv-{i}" for i in range(6)]
    assert got[12:] == ["strong", "weak"]                       # by strength, no repeats
