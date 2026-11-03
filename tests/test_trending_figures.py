"""
Trending cards with figures, and the page laid out papers first.

What must hold: a card shows a figure only from a saved source the figure
rules allow, with its credit, and otherwise the paper's own name (never an
empty box or a request to arXiv while the page renders); the week's papers are
queued for a figure most upvoted first, only when nobody has looked yet; the
organisation Hugging Face lists is read as text only; the page leads with the
week's top paper, before any topic.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app import card_data, figure_prewarm, hf_papers_svc, paper_source_svc, trending_shelf as ts
from app.main import app
from app.templates_env import byline, short_name
from tests.test_trending_tab import tab_world  # noqa: F401  (fixture)


# ── Text helpers ─────────────────────────────────────────────────────────────

def test_short_name_takes_the_name_before_the_colon():
    assert short_name("WeVisOne-VL: Unified Visual Perception from Objects to Pixels") == "WeVisOne-VL"
    assert short_name("Has GPT-6-Astra Achieved Embodied Intelligence?") == "Has GPT-6-Astra Achieved Embodied Intelligence?"
    assert short_name("Right Screen, Wrong Transition: World Models as Verifiers") == "Right Screen, Wrong Transition"
    long_head = "A " * 30 + ": tail"
    assert short_name(long_head).endswith(" …"), "a long head is no name: first six words instead"
    assert short_name("") == ""
    assert short_name("Has GPT-6-Astra Achieved Embodied Intelligence?", 0) == "", \
        "no name and no words wanted: the card shows its upvotes instead"
    assert short_name("DAPD: Dual-Anchored Policy Distillation", 0) == "DAPD"


def test_byline_prefers_the_organisation_then_the_first_author():
    assert byline({"org": "Tencent", "authors": "A. One, B. Two"}) == "Tencent"
    assert byline({"org": "", "authors": "A. One, B. Two"}) == "A. One et al."
    assert byline({"authors": '["Solo Author"]'}) == "Solo Author"
    assert byline({}) == ""


def test_extras_read_the_organisation_name_only():
    payload = [{"paper": {"id": "2610.31329v2", "ai_summary": " s ", "ai_keywords": ["k"],
                          "organization": {"name": "tencent", "fullname": "Tencent",
                                           "avatar": "https://cdn-avatars.huggingface.co/x.png"}},
                "numComments": 2},
               {"paper": {"id": "2610.24738"}, "organization": {"name": "jingdong1"}},
               {"paper": {"id": "2610.30926", "organization": "not a dict"}}]
    out = hf_papers_svc._extras(payload)
    assert out["2610.31329"]["org"] == "Tencent"
    assert out["2610.24738"]["org"] == "jingdong1", "the record's organisation when the paper has none"
    assert out["2610.30926"]["org"] == ""
    assert "avatar" not in str(out), "no third-party image URL is kept"


# ── Figures on cards ─────────────────────────────────────────────────────────

def source(arxiv_id, allowed=True, figures=None):
    return SimpleNamespace(arxiv_id=arxiv_id, figures_allowed=allowed, licence_label="CC BY 4.0",
                           figures=figures if figures is not None else
                           [{"src": f"{arxiv_id}v1/x1.png", "label": "Figure 1", "caption": "Overview",
                             "w": 800, "h": 400}])


async def test_add_figures_uses_saved_sources_only(monkeypatch):
    monkeypatch.setenv("FIGURES", "on")
    get_many = AsyncMock(return_value={"2610.31329": source("2610.31329"),
                                       "2610.24738": source("2610.24738", allowed=False)})
    monkeypatch.setattr(paper_source_svc, "get_many", get_many)
    ensure = AsyncMock()
    monkeypatch.setattr(paper_source_svc, "ensure", ensure)
    papers = [{"arxiv_id": "2610.31329", "authors": "Tianming Liang, Junyi Pan", "year": 2026},
              {"arxiv_id": "2610.24738"}, {"arxiv_id": "2610.30926"}]
    await card_data.add_figures(papers)
    fig = papers[0]["card_figure"]
    assert fig["url"].startswith("/fig/2610.31329/0-") and fig["w"] == 800
    assert fig["credit"] == "Liang et al. 2026, CC BY 4.0"
    assert papers[1]["card_figure"] is None, "a licence the rules refuse shows no figure"
    assert papers[2]["card_figure"] is None, "never read: no figure, and no fetch while rendering"
    ensure.assert_not_called()


async def test_add_figures_with_the_switch_off_or_a_failure(monkeypatch):
    papers = [{"arxiv_id": "2610.31329"}]
    monkeypatch.setenv("FIGURES", "off")
    await card_data.add_figures(papers)
    assert papers[0]["card_figure"] is None
    monkeypatch.setenv("FIGURES", "on")
    monkeypatch.setattr(paper_source_svc, "get_many", AsyncMock(side_effect=RuntimeError("db gone")))
    await card_data.add_figures(papers)
    assert papers[0]["card_figure"] is None, "a failure costs the figures, never the page"


async def test_warm_figures_queues_unread_papers_most_upvoted_first(monkeypatch):
    read = SimpleNamespace(status="ok")
    monkeypatch.setattr(paper_source_svc, "get_many", AsyncMock(return_value={"b": read, "c": read}))
    monkeypatch.setattr(paper_source_svc, "needs_fetch", lambda s: s is None)
    queued = []
    monkeypatch.setattr(figure_prewarm, "enqueue", lambda ids: queued.extend(ids) or len(ids))
    monkeypatch.setattr(ts, "WARM_MAX", 2)
    assert await ts.warm_figures(["a", "b", "c", "d", "e"]) == 2
    assert queued == ["a", "d"], "already read papers are skipped; the cap keeps the most upvoted"


async def test_warm_figures_never_raises(monkeypatch):
    monkeypatch.setattr(paper_source_svc, "get_many", AsyncMock(side_effect=RuntimeError("db gone")))
    assert await ts.warm_figures(["a"]) == 0


# ── The page ─────────────────────────────────────────────────────────────────

def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_the_page_leads_with_the_top_paper_and_its_figure(tab_world, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FIGURES", "on")
    monkeypatch.setattr(paper_source_svc, "get_many",
                        AsyncMock(return_value={"2610.11111": source("2610.11111")}))
    ts._state["papers"]["2610.11111"]["org"] = "Tencent"
    async with client() as c:
        html = (await c.get("/trending")).text
    top = html[html.index('id="tp-top"'):]
    assert html.index('id="tp-top"') < html.index('id="tp-rising"'), "papers before topics"
    lead = top[top.index("tc--lead"):]
    lead = lead[:lead.index("</li>")]
    assert "A world model this week" in lead and "/fig/2610.11111/0-" in lead
    assert "Figure 1 · " in lead and "CC BY 4.0" in lead, "every shown figure carries its credit"
    assert "Tencent" in lead
    second = top[top.index("Listed after today"):]
    assert "tc--lead" not in second[:second.index("</li>")]
    card = top[top.rindex("<li", 0, top.index("Listed after today")):]
    assert "is-bare" in card and "▲ 9" in card, "no figure and no name: the cover shows the upvotes"


async def test_moving_topics_keep_the_shared_state_clean(tab_world, monkeypatch):  # noqa: F811
    from app import trending_topics as tt
    monkeypatch.setenv("FIGURES", "on")
    monkeypatch.setattr(paper_source_svc, "get_many",
                        AsyncMock(return_value={"2606.01001": source("2606.01001")}))
    async with client() as c:
        html = (await c.get("/trending")).text
    moving = html[html.index('id="tp-rising"'):html.index('id="tp-steady"')]
    assert "/fig/2606.01001/0-" in moving and "↑ Rising" in moving and "↓ Cooling" in moving
    assert all("card_figure" not in p for t in tt._state["topics"] for p in t["top"]), \
        "figures go on copies, never on the shared topic state"
