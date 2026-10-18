"""
The briefing (app/briefing_svc.py) and its endpoint.

The model is the fake provider; the paper is the SatNav fixture, parsed and
saved as paper_source_svc would. What is pinned: passage picking, the code
checks that drop unsupported points, caching, single flight, the switch modes
and escaping.
"""
import asyncio
import json
import re
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app import arxiv_html, briefing_svc, config, llm, paper_source_svc, turso_svc

FIXTURES = Path(__file__).parent / "fixtures" / "arxiv_html"
AID = "2609.31507"
PARSED = arxiv_html.parse((FIXTURES / f"{AID}.html").read_bytes(), AID)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(config, "RESEARCHIT_ENV", "test")
    monkeypatch.setattr(config, "LLM_PROVIDER", "fake")
    llm._fake_replies.clear()
    yield
    llm._fake_replies.clear()


def save_satnav():
    asyncio.run(paper_source_svc._save_parsed(PARSED, None))


async def satnav_inputs():
    await paper_source_svc._save_parsed(PARSED, None)
    return await paper_source_svc.paragraphs(AID), await paper_source_svc.fulltext(AID)


def passages_now():
    async def go():
        paras, full = await satnav_inputs()
        return briefing_svc.select_passages(PARSED.abstract, paras, full)
    return asyncio.run(go())


def good_reply(system, user):
    """A reply built from the prompt it was given: two supported points, one invented."""
    anchors = re.findall(r'<passage anchor="([^"]+)">', user)
    intro = next(a for a in anchors if a.startswith("S1."))
    return json.dumps({
        "insufficient": False,
        "things": [
            {"lead": "118K episodes from satellite imagery.", "text": "They span 59 scenes across 18 cities.",
             "anchor": "abstract"},
            {"lead": "Built without 3D city models.", "text": "Satellite crops stand in for drone views.",
             "anchor": intro},
            {"lead": "Beats every baseline by 97%.", "text": "An invented claim.", "anchor": "abstract"},
        ],
        "how": [{"name": "Pair the maps", "text": "Satellite images aligned with map data."},
                {"name": "Write the tasks", "text": "Routes become instructions."},
                {"name": "Navigate", "text": "The agent picks one of 4 actions."}],
        "use": "Teams training drones that follow spoken directions.",
    })


# ── passages ─────────────────────────────────────────────────────────────────

def test_passages_start_with_the_abstract_and_include_the_introduction_and_captions():
    passages = passages_now()
    assert passages[0]["anchor"] == "abstract"
    anchors = [p["anchor"] for p in passages]
    assert any(a.startswith("S1.") for a in anchors)             # introduction
    assert "S1.F1" in anchors                                    # a figure caption
    assert len(anchors) == len(set(anchors))
    assert sum(len(p["text"]) for p in passages) <= briefing_svc.TOTAL_CHARS
    assert all(len(p["text"]) <= briefing_svc.PASSAGE_CHARS for p in passages)


def test_passages_are_labelled_by_section_number():
    labels = {p["anchor"]: p["label"] for p in passages_now()}
    assert labels["abstract"] == "Abstract" and labels["S1.F1"] == "Figure 1"
    assert any(v == "§1" for a, v in labels.items() if a.startswith("S1.p"))


def test_the_prompt_holds_only_the_passages_with_escaped_anchors():
    prompt = briefing_svc.build_prompt("T", [{"anchor": 'x"><b', "text": "hello", "label": "x"}])
    assert prompt == '<title>T</title>\n<passage anchor="x&quot;>&lt;b">hello</passage>'


# ── verification ─────────────────────────────────────────────────────────────

PASSAGES = [{"anchor": "abstract", "text": "We collect 118,494 tasks in 18 cities with 24,310 images.", "label": "Abstract"},
            {"anchor": "S5.p1", "text": "Success rose from 38.9% to 55.6% on real flights.", "label": "§5"}]


def thing(lead, text, anchor):
    return {"lead": lead, "text": text, "anchor": anchor}


def test_supported_points_survive_with_their_labels():
    status, b = briefing_svc.verify({"things": [thing("118,494 tasks.", "In 18 cities.", "abstract"),
                                               thing("Real flights improved.", "From 38.9% to 55.6%.", "S5.p1")]},
                                    PASSAGES)
    assert status == "ok" and [t["label"] for t in b["things"]] == ["Abstract", "§5"]


@pytest.mark.parametrize("token,ok", [("118,494", True), ("118494", True), ("118K", True), ("24K", True),
                                      ("38.9%", True), ("55.6", True), ("119K", False), ("97%", False),
                                      ("18", True), ("19", False)])
def test_number_checks(token, ok):
    assert briefing_svc.number_supported(token, " ".join(p["text"] for p in PASSAGES)) is ok


def test_a_point_with_an_unsupported_number_is_dropped():
    status, b = briefing_svc.verify({"things": [thing("118,494 tasks.", "In 18 cities.", "abstract"),
                                               thing("Huge gains.", "Up 97%.", "abstract"),
                                               thing("Real flights improved.", "From 38.9%.", "S5.p1")]},
                                    PASSAGES)
    assert status == "ok" and len(b["things"]) == 2
    assert "97%" in b["dropped"][0]["why"]


def test_a_number_must_be_in_the_cited_passage_not_just_anywhere():
    status, b = briefing_svc.verify({"things": [thing("Flights rose to 55.6%.", "See results.", "abstract"),
                                               thing("118,494 tasks.", "In 18 cities.", "abstract")]}, PASSAGES)
    assert status == "insufficient" and len(b["things"]) == 1


def test_an_anchor_that_was_not_sent_drops_the_point():
    status, b = briefing_svc.verify({"things": [thing("A point.", "Text.", "S9.p9"),
                                               thing("118,494 tasks.", "In 18 cities.", "abstract")]}, PASSAGES)
    assert status == "insufficient" and "not sent" in b["dropped"][0]["why"]


@pytest.mark.parametrize("bad", [
    {"things": [{"lead": "no anchor", "text": "x"}]},
    {"things": [thing("x" * 200, "t", "abstract")]},
    {"things": ["just a string"]},
    {"things": [thing("ok.", "ok.", "abstract")], "how": [{"name": "n"}]},
])
def test_malformed_replies_fail(bad):
    assert briefing_svc.verify(bad, PASSAGES)[0] == "failed"


def test_numbers_in_the_steps_and_use_line_are_checked_against_all_passages():
    status, b = briefing_svc.verify({
        "things": [thing("118,494 tasks.", "In 18 cities.", "abstract"), thing("Flights.", "38.9% then 55.6%.", "S5.p1")],
        "how": [{"name": "Collect", "text": "Gather 24,310 images."}, {"name": "Invent", "text": "Train for 900 hours."}],
        "use": "Useful for 7 billion people."}, PASSAGES)
    assert status == "ok" and [s["name"] for s in b["how"]] == ["Collect"] and b["use"] == ""


def test_the_models_own_refusal_is_insufficient():
    assert briefing_svc.verify({"insufficient": True}, PASSAGES)[0] == "insufficient"
    assert briefing_svc.verify(None, PASSAGES)[0] == "failed"


# ── generation and caching ───────────────────────────────────────────────────

async def test_a_briefing_is_generated_once_and_cached(fake):
    await satnav_inputs()
    prompts = []

    def reply(system, user):
        prompts.append(user)
        return good_reply(system, user)

    llm.fake_reply(reply)
    status, b = await briefing_svc.get_or_generate(AID, PARSED.title, PARSED.abstract)
    assert status == "ok" and len(b["things"]) == 2              # the invented 97% point was dropped
    assert b["things"][0]["label"] == "Abstract"
    status2, b2 = await briefing_svc.get_or_generate(AID, PARSED.title, PARSED.abstract)
    assert (status2, b2) == (status, b) and len(prompts) == 1
    assert "<passage anchor=\"abstract\">" in prompts[0] and "Bibliography stub" not in prompts[0]


async def test_concurrent_requests_share_one_generation(fake):
    await satnav_inputs()
    calls = []

    def reply(system, user):
        calls.append(1)
        return good_reply(system, user)

    llm.fake_reply(reply)
    results = await asyncio.gather(*[briefing_svc.get_or_generate(AID, PARSED.title, PARSED.abstract)
                                     for _ in range(5)])
    assert len(calls) == 1 and all(r[0] == "ok" for r in results)


async def test_a_new_prompt_version_means_a_new_cache_key(monkeypatch):
    before = briefing_svc.cache_key(AID, "v1")
    monkeypatch.setattr(briefing_svc, "PROMPT_VERSION", "b2")
    assert briefing_svc.cache_key(AID, "v1") != before


async def test_over_budget_is_not_cached(fake, monkeypatch):
    await satnav_inputs()
    monkeypatch.setattr(config, "LLM_DAILY_CALLS", 0)
    status, _ = await briefing_svc.get_or_generate(AID, PARSED.title, PARSED.abstract)
    assert status == "insufficient"
    assert await briefing_svc.cached(AID, PARSED.version) is None


# ── the endpoint ─────────────────────────────────────────────────────────────

def client_for(monkeypatch, mode):
    from app.main import app
    monkeypatch.setenv("BRIEFING", mode)
    meta = {"arxiv_id": AID, "title": PARSED.title, "abstract": PARSED.abstract, "authors": "[]"}

    async def fake_meta(ids):
        return {i: meta for i in ids if i == AID}

    return app, patch.object(turso_svc, "fetch_metadata_batch", side_effect=fake_meta)


def test_switch_off_answers_nothing_and_calls_no_model(fake, monkeypatch):
    save_satnav()
    app, meta = client_for(monkeypatch, "off")
    with meta, TestClient(app) as c:
        assert c.get(f"/api/papers/{AID}/briefing").text == ""


def test_shadow_generates_and_stores_but_shows_nothing(fake, monkeypatch):
    save_satnav()
    llm.fake_reply(good_reply)
    app, meta = client_for(monkeypatch, "shadow")
    with meta, TestClient(app) as c:
        assert c.get(f"/api/papers/{AID}/briefing").text == ""
    assert asyncio.run(briefing_svc.cached(AID, PARSED.version))[0] == "ok"


def test_on_renders_the_briefing_with_links_to_each_passage(fake, monkeypatch):
    save_satnav()
    llm.fake_reply(good_reply)
    app, meta = client_for(monkeypatch, "on")
    with meta, TestClient(app) as c:
        html = c.get(f"/api/papers/{AID}/briefing").text
    assert 'id="sec-things"' in html and 'id="sec-how"' in html and 'id="sec-use"' in html
    assert f'href="https://arxiv.org/html/{AID}v1#abstract"' in html
    assert "97%" not in html
    assert "Report a problem" in html


def test_model_text_is_escaped(fake, monkeypatch):
    save_satnav()

    def evil(system, user):
        data = json.loads(good_reply(system, user))
        data["things"][1]["text"] = "<script>alert(1)</script>"
        data["use"] = '<img src=x onerror="alert(1)">'
        return json.dumps(data)

    llm.fake_reply(evil)
    app, meta = client_for(monkeypatch, "on")
    with meta, TestClient(app) as c:
        html = c.get(f"/api/papers/{AID}/briefing").text
    assert "<script>alert(1)</script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_an_unknown_paper_never_reaches_the_model(fake, monkeypatch):
    app, meta = client_for(monkeypatch, "on")
    llm.fake_reply(good_reply)
    with meta, TestClient(app) as c:
        assert c.get("/api/papers/2609.99999/briefing").text == ""
    assert len(llm._fake_replies) == 1


def test_a_report_is_stored(monkeypatch):
    from app.main import app
    with TestClient(app) as c:
        r = c.post(f"/api/papers/{AID}/report", data={"feature": "briefing", "reason": "wrong number"})
        c.post(f"/api/papers/{AID}/report", data={"feature": "nonsense"})
    assert "Thanks" in r.text

    async def rows():
        async with aiosqlite.connect(config.DB_PATH) as conn:
            cur = await conn.execute("SELECT arxiv_id, feature, reason FROM content_reports")
            return await cur.fetchall()
    assert asyncio.run(rows()) == [(AID, "briefing", "wrong number")]


def test_briefing_and_report_routes_are_rate_limited():
    from app import rate_limit
    assert rate_limit._rule_for(f"/api/papers/{AID}/briefing")[0] == "/api/papers/*/briefing"
    assert rate_limit._rule_for(f"/api/papers/{AID}/report")[0] == "/api/papers/*/report"



@pytest.mark.parametrize("sentence,numbers", [
    ("Built without 3D city models.", []),
    ("GPT-4 and Qwen2.5-VL on x86 with F1 scores.", []),
    ("118K episodes in 18 cities, up 38.9%.", ["118K", "18", "38.9%"]),
    ("Accuracy rose by 7.17 to 27.31 points.", ["7.17", "27.31"]),
    ("It costs 1,200 dollars.", ["1,200"]),
])
def test_names_with_digits_are_not_numbers(sentence, numbers):
    assert briefing_svc._numbers(sentence) == numbers
