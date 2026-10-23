"""
The redesigned feed card (FEED_CARD_V2) and app/card_data.py.

The guarantees that matter more than looks: the card's links and Save/Pass
values are exactly today's (§3.11 instrumentation), enrichment never changes
order or ids and makes no outside call, figures follow the licence policy, and
"In plain terms" appears only when one is cached for the abstract shown.
"""
import asyncio
import json
import random
import re
import time
from html.parser import HTMLParser
from unittest.mock import patch

import aiosqlite
import httpx
import pytest
from fastapi.testclient import TestClient

from app import card_data, config, groq_svc, http_client
from app.templates_env import templates

PAPER = {
    "arxiv_id": "2609.31507", "title": "SatNav", "abstract": "Urban UAV navigation needs memory. We build SatNav.",
    "authors": '["Jiajun Jiang", "Chunliang Hua", "Zichun Chen"]', "year": 2026, "category": "cs.CV",
    "query_id": "q-1", "position": 4, "candidate_source": "cluster_1", "cluster_id": 1,
    "propensity": 0.25, "policy_id": "v11_rescaled_pubdate", "ranker_version": "v11",
    "cluster_label": "Drone navigation", "cluster_tone": 2,
}
FIG = {"anchor": "S1.F1", "label": "Figure 1", "caption": "Overview.", "src": "2609.31507v1/overview.png", "w": 476, "h": 235}


def render(card, paper, source="recommendation"):
    return templates.env.get_template(f"partials/{card}.html").render(paper=paper, source=source)


def href(html):
    return re.search(r'class="card-title"\s+href="([^"]+)"', html).group(1)


def save_vals(html):
    return re.search(r"hx-vals='([^']+)'", html).group(1)


def test_links_and_save_values_match_todays_card():
    old, new = render("paper_card", PAPER), render("paper_card_v2", PAPER)
    assert href(old) == href(new)
    assert "sf=recommendation" in href(new).replace("&amp;", "&")
    assert json.loads(save_vals(old)) == json.loads(save_vals(new))


def test_the_hooks_app_js_relies_on_are_kept():
    html = render("paper_card_v2", PAPER)
    for hook in ('class="card pc', 'id="paper-2609.31507"', 'data-arxiv-id="2609.31507"',
                 "id=\"actions-2609.31507\"", 'id="why-2609.31507"'):
        assert hook in html


class _Clickables(HTMLParser):
    """Links and buttons a reader sees, not counting the closed menu's items."""
    def __init__(self):
        super().__init__()
        self.depth_in_menu, self.stack, self.count = 0, [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        in_menu = "menu" in classes and tag == "div"
        self.stack.append(in_menu)
        if in_menu:
            self.depth_in_menu += 1
        elif tag in ("a", "button") and not self.depth_in_menu and "card-why" not in classes:
            self.count += 1

    def handle_endtag(self, tag):
        if self.stack and self.stack.pop():
            self.depth_in_menu -= 1


def test_a_card_has_at_most_eight_things_to_click():
    full = {**PAPER, "card_plain": "Plain words.", "card_code": {"repo": "https://github.com/a/b", "stars": 3},
            "card_figure": {"url": "/fig/x/0-00000000.webp", "w": 4, "h": 2, "label": "Figure 1",
                            "caption": "c", "credit": "Jiang et al. 2026, CC BY 4.0"}}
    p = _Clickables()
    p.feed(render("paper_card_v2", full))
    assert p.count <= 8, p.count


def test_without_a_cached_summary_the_card_shows_the_abstract_only():
    html = render("paper_card_v2", PAPER)
    assert "data-cardtab" not in html and "We build SatNav." in html


# ── enrichment ───────────────────────────────────────────────────────────────

async def save_source(licence="CC-BY-4.0", repo=None):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO paper_sources (arxiv_id, status, version, licence, figures_json, parser_version, "
            "code_repo, code_stars) VALUES (?, 'ok', 'v1', ?, ?, 1, ?, ?)",
            (PAPER["arxiv_id"], licence, json.dumps([FIG]), repo, 12 if repo else None))
        await conn.commit()


@pytest.fixture
def no_network(monkeypatch):
    def refuse(request):
        raise AssertionError(f"outside call: {request.url}")
    trap = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    monkeypatch.setattr(http_client, "get_client", lambda: trap)


async def test_enrichment_adds_figure_code_and_cached_summary(monkeypatch, no_network):
    monkeypatch.setenv("FIGURES", "on")
    await save_source(repo="https://github.com/Eku127/SatNav")
    from app import db
    await db.save_explanation(groq_svc.explain_cache_key(PAPER["arxiv_id"], PAPER["abstract"]),
                              PAPER["arxiv_id"], "Plain words.", "m")
    (card,) = await card_data.enrich([PAPER])
    assert card["card_plain"] == "Plain words."
    assert card["card_code"] == {"repo": "https://github.com/Eku127/SatNav", "stars": 12}
    assert card["card_figure"]["url"].startswith("/fig/2609.31507/0-")
    assert card["card_figure"]["credit"] == "Jiang et al. 2026, CC BY 4.0"


async def test_a_summary_written_for_a_different_abstract_is_not_shown(no_network):
    from app import db
    await db.save_explanation(groq_svc.explain_cache_key(PAPER["arxiv_id"], "an older, cut abstract"),
                              PAPER["arxiv_id"], "Stale words.", "m")
    (card,) = await card_data.enrich([PAPER])
    assert card["card_plain"] is None


async def test_figures_follow_their_switch_and_the_licence_policy(monkeypatch, no_network):
    await save_source(licence="ARXIV-NONEXCLUSIVE-1.0")
    assert (await card_data.enrich([PAPER]))[0]["card_figure"] is None          # FIGURES off
    monkeypatch.setenv("FIGURES", "on")
    monkeypatch.setenv("FIGURE_LICENCES", "CC-BY-4.0")
    assert (await card_data.enrich([PAPER]))[0]["card_figure"] is None          # licence outside the policy
    monkeypatch.setenv("FIGURE_LICENCES", "all")
    assert (await card_data.enrich([PAPER]))[0]["card_figure"] is not None


async def test_enrichment_keeps_order_ids_and_instrumentation(no_network):
    rng = random.Random(3)
    papers = [{**PAPER, "arxiv_id": f"2609.{i:05d}", "position": i, "propensity": rng.random()} for i in range(12)]
    rng.shuffle(papers)
    before = json.dumps(papers)
    out = await card_data.enrich(papers)
    assert json.dumps(papers) == before                                  # inputs untouched
    keys = ("arxiv_id", "query_id", "position", "candidate_source", "cluster_id", "propensity", "policy_id")
    assert [[p[k] for k in keys] for p in out] == [[p[k] for k in keys] for p in papers]


async def test_enrichment_is_quick_for_a_page_of_twelve(no_network):
    papers = [{**PAPER, "arxiv_id": f"2609.{i:05d}"} for i in range(12)]
    t0 = time.perf_counter()
    await card_data.enrich(papers)
    assert time.perf_counter() - t0 < 0.2


# ── through the feed route ───────────────────────────────────────────────────

def feed_html(monkeypatch, mode):
    from app.main import app
    from app.routers import recommendations as recs
    monkeypatch.setenv("FEED_CARD_V2", mode)

    async def fake_build_feed(user_id, state, query_id, **kw):
        return {"user_id": user_id, "query_id": query_id, "ranked": [PAPER["arxiv_id"]], "tags": {}}

    async def fake_build_page(entry, seen):
        return [dict(PAPER)], False

    with patch.object(recs, "_build_feed", side_effect=fake_build_feed), \
            patch.object(recs, "_build_page", side_effect=fake_build_page), TestClient(app) as c:
        return c.get("/api/recommendations").text


def test_the_feed_serves_v2_cards_only_with_the_switch_on(monkeypatch):
    assert 'class="card pc' in feed_html(monkeypatch, "on")
    assert 'class="card pc' not in feed_html(monkeypatch, "off")


def test_v2_feed_cards_never_trigger_a_summary(monkeypatch):
    called = []

    async def spy(title, abstract, arxiv_id=None):
        called.append(1)
        return "x"

    monkeypatch.setattr(groq_svc, "explain_paper", spy)
    html = feed_html(monkeypatch, "on")
    assert "/explain" not in html and called == []
