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

from app import trending_shelf as ts
from app.templates_env import _short_count


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
