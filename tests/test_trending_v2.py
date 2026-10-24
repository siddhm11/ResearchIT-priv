"""Trending v2 on the cards: why a paper is trending, its rising topic, and the
search step-up's tag (app/trending_shelf.py, app/hybrid_search_svc.py,
partials/trending_card.html, partials/paper_card.html).

What must hold: the "why" line is counts from the lists; the topic line and the
/trending link appear only with TRENDING_PAGE on; a search result carries the
trending tag only when the step-up really moved it (SEARCH_TRENDING=on, not
shadow); and the classic card is byte-for-byte unchanged otherwise, because
every page uses it.
"""
from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import hybrid_search_svc as hs, trending_shelf as ts, turso_svc
from app.templates_env import _short_count, templates


# ── Why it is trending ───────────────────────────────────────────────────────

def test_trend_facts_say_where_the_paper_placed():
    first = {"listed_on": "2026-09-30", "day_rank": 1, "day_size": 92, "week_percentile": 1.0, "week_size": 404}
    assert ts.trend_facts(first) == ["Most upvoted of 92 papers on 30 Sep", "top 1% of this week's 404"]
    third = {**first, "day_rank": 3, "week_percentile": 0.80}
    assert ts.trend_facts(third) == ["#3 of 92 papers on 30 Sep", "top 20% of this week's 404"]
    middling = {**first, "day_rank": 40, "week_percentile": 0.5}
    assert ts.trend_facts(middling) == ["#40 of 92 papers on 30 Sep"], "no week claim below the top quarter"
    small_week = {**first, "week_size": 12}
    assert ts.trend_facts(small_week) == ["Most upvoted of 92 papers on 30 Sep"]


def test_freshness_is_full_for_three_days_and_gone_at_a_week():
    today = date(2026, 10, 24)
    days = [ts.freshness(f"2026-10-{24 - age:02d}", today) for age in range(9)]
    assert days[:3] == [1.0, 1.0, 1.0] and days[7] == 0.0 and days[8] == 0.0
    assert days[3] > days[4] > days[5] > days[6] > 0


def test_short_counts():
    assert [_short_count(n) for n in (950, 5742, 10_000, 113_000, 1_300_000)] == ["950", "5.7k", "10k", "113k", "1.3M"]
    assert _short_count("x") == "" and _short_count(None) == ""


# ── The step-up in search ────────────────────────────────────────────────────

def test_trend_boost_scales_with_place_and_freshness_and_needs_relevance():
    best = 10.0
    top = {"strength": 1.0, "fresh": 1.0}
    assert hs._trend_boost(top, best, best) == pytest.approx(hs.TREND_BOOST)
    assert hs._trend_boost({**top, "strength": hs.TREND_MIN_STRENGTH}, best, best) == pytest.approx(hs.TREND_BOOST / 2)
    assert hs._trend_boost({**top, "fresh": 0.5}, best, best) == pytest.approx(hs.TREND_BOOST / 2)
    assert hs._trend_boost(top, best * hs.TREND_MIN_RELEVANCE * 0.99, best) == 0.0, "an off-topic paper never moves"
    assert hs._trend_boost(None, best, best) == 0.0


PAPER = {"arxiv_id": "2609.33439", "title": "Raven", "abstract": "A harness of harnesses.", "authors": '["A B"]',
         "year": 2026, "category": "cs.AI", "query_id": "q-1", "position": 0, "propensity": 1.0,
         "policy_id": "search_v2_independent_lexical"}


def render(paper):
    return templates.env.get_template("partials/paper_card.html").render(paper=paper, source="search")


def test_the_classic_card_is_unchanged_without_a_trend():
    assert render(PAPER) == render({**PAPER, "trending": None})
    tagged = render({**PAPER, "trending": {"fact": "Most upvoted of 92 papers on 30 Sep"}})
    assert '<span class="trend-tag">Trending on Hugging Face · Most upvoted of 92 papers on 30 Sep</span>' in tagged


def search_html(meta):
    from app.main import app

    async def fake_search(q, limit=20, return_meta=False):
        return ["2609.33439", "2609.00002"], dict(meta)

    async def details(ids):
        return {i: {**PAPER, "arxiv_id": i, "title": f"Paper {i}"} for i in ids}

    with patch.object(hs, "search", side_effect=fake_search), \
            patch.object(turso_svc, "fetch_metadata_batch", side_effect=details), TestClient(app) as c:
        return c.get(f"/search?q=harness{len(meta)}{list(meta)[:1]}", headers={"HX-Request": "true"}).text


def test_search_tags_only_what_the_step_up_moved():
    moved = {"2609.33439": {"from": 12, "to": 3, "facts": ["Most upvoted of 92 papers on 30 Sep"]}}
    on = search_html({"trending": moved})
    assert on.count('class="trend-tag"') == 1 and "Most upvoted of 92 papers on 30 Sep" in on
    assert "trend-tag" not in search_html({"trending_shadow": moved}), "shadow mode changes nothing on screen"
