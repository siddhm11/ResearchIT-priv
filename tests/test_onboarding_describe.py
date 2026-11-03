"""
Onboarding's "In a sentence": a reader describes their work, the areas its
papers sit in are suggested, and step 2 opens on those papers.
"""
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.discovery_svc import areas_for_papers


def _paper(cats: str, title: str = "A paper") -> dict:
    return {"arxiv_id": f"id-{cats}-{title}", "title": title, "arxiv_categories": cats,
            "category": cats.split()[0]}


# ── areas_for_papers ─────────────────────────────────────────────────────────

def test_a_cross_listed_area_needs_most_of_the_papers():
    """cs.LG is cross-listed on nearly everything; counted in full, half the
    papers cross-listed would suggest Machine Learning. At half weight it
    takes about 70% of them."""
    half = [_paper("cs.CL cs.LG")] * 5 + [_paper("cs.CL")] * 5
    assert areas_for_papers(half) == ["nlp"]                          # ml 2.5 < 3.5
    most = [_paper("cs.RO cs.LG")] * 8 + [_paper("cs.RO")] * 2
    assert areas_for_papers(most) == ["robotics", "ml"]               # ml 4.0 >= 3.5


def test_a_close_second_area_is_suggested_and_a_stray_one_is_not():
    papers = [_paper("cs.CL")] * 8 + [_paper("cs.CV")] * 4 + [_paper("q-bio.NC")]
    assert areas_for_papers(papers) == ["nlp", "cv"]


def test_codes_in_one_area_count_once_per_paper():
    """cs.CL and cs.IR are both Natural Language Processing."""
    papers = [_paper("cs.CL cs.IR")] * 3
    assert areas_for_papers(papers) == ["nlp"]


def test_at_most_limit_areas_and_unknown_codes_ignored():
    papers = ([_paper("cs.CL")] * 4 + [_paper("cs.CV")] * 4 + [_paper("cs.RO")] * 4
              + [_paper("hep-th")] * 4 + [_paper("cs.CY")] * 9)
    assert len(areas_for_papers(papers)) == 3
    assert areas_for_papers([_paper("cs.CY")]) == [] and areas_for_papers([]) == []


def test_a_single_paper_still_names_its_area():
    assert areas_for_papers([_paper("astro-ph.GA")]) == ["astro"]


def test_only_the_primary_category_is_used_when_the_full_list_is_missing():
    assert areas_for_papers([{"category": "quant-ph"}]) == ["quant_ph"]


# ── /api/onboarding/describe ─────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    import asyncio

    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    from app.main import app
    asyncio.run(db_mod.init_db())
    with TestClient(app) as c:
        yield c


def _search_returns(monkeypatch, papers):
    import app.hybrid_search_svc as hs
    import app.turso_svc as turso
    search = AsyncMock(return_value=[p["arxiv_id"] for p in papers])
    monkeypatch.setattr(hs, "search", search)
    monkeypatch.setattr(turso, "fetch_metadata_batch",
                        AsyncMock(return_value={p["arxiv_id"]: p for p in papers}))
    return search


def test_describe_answers_areas_with_their_evidence(client, monkeypatch):
    papers = [_paper("cs.RO cs.CV", "Grasping from video")] + [_paper("cs.RO", f"Arm {i}") for i in range(5)]
    search = _search_returns(monkeypatch, papers)
    r = client.post("/api/onboarding/describe", json={"text": "  robots that learn to grasp from video  "})
    assert r.status_code == 200
    assert r.json() == {"areas": [{"key": "robotics", "name": "Robotics"}],
                        "matched": 6, "example": "Grasping from video"}
    # Searched as written: no LLM rewrite, so the sentence reaches no model.
    args, kwargs = search.call_args
    assert args[0] == "robots that learn to grasp from video" and kwargs["use_rewrite"] is False


def test_describe_keeps_a_long_description_to_its_limit(client, monkeypatch):
    search = _search_returns(monkeypatch, [])
    client.post("/api/onboarding/describe", json={"text": "x" * 1000})
    assert len(search.call_args.args[0]) == 300


@pytest.mark.parametrize("body", [{"text": "   "}, {"text": 7}, {}, ["robots"]])
def test_describe_needs_a_description(client, body):
    assert client.post("/api/onboarding/describe", json=body).status_code == 422


def test_describe_with_search_down_suggests_nothing(client, monkeypatch):
    import app.hybrid_search_svc as hs
    import app.turso_svc as turso
    monkeypatch.setattr(hs, "search", AsyncMock(return_value=["2401.00001"]))
    monkeypatch.setattr(turso, "fetch_metadata_batch", AsyncMock(side_effect=RuntimeError("turso down")))
    r = client.post("/api/onboarding/describe", json={"text": "robots"})
    assert r.status_code == 200 and r.json() == {"areas": [], "matched": 0, "example": ""}


def test_step_two_opens_on_the_description(client):
    html = client.post("/api/onboarding/categories",
                       json={"categories": ["robotics"], "describe": "grasping & video"}).text
    assert 'value="grasping &amp; video"' in html
    assert 'hx-get="/api/onboarding/seed-search?q=grasping%20%26%20video"' in html


def test_step_two_without_a_description_opens_on_suggestions(client):
    html = client.post("/api/onboarding/categories", json={"categories": ["robotics"]}).text
    assert 'hx-get="/api/onboarding/seed-search"' in html and 'name="q" value=""' in html
