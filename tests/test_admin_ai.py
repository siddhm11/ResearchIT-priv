"""
The AI usage dashboard (/admin/ai): owner only, built from app/ai_usage.py.

What matters: nobody without the preview token sees it, its numbers are the
ones ai_usage recorded, the insights fire on the conditions they describe, and
the chart's columns and table view agree.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app import ai_usage, config, groq_svc
from app.groq_svc import Completion
from app.routers import admin


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(config, "PREVIEW_TOKEN", "owner-token")
    monkeypatch.setattr(ai_usage, "_pending", {})
    monkeypatch.setattr(groq_svc, "_windows", {})


def get(path="/admin/ai", cookies=None):
    from app.main import app
    with TestClient(app) as c:
        for k, v in (cookies or {}).items():
            c.cookies.set(k, v)
        return c.get(path)


@pytest.mark.parametrize("path", ["/admin/ai", "/admin/ai?preview=wrong", "/admin/ai?preview="])
def test_without_the_preview_token_it_does_not_exist(owner, path):
    assert get(path).status_code == 404


def test_without_any_preview_token_configured_nobody_sees_it(owner, monkeypatch):
    monkeypatch.setattr(config, "PREVIEW_TOKEN", "")
    assert get("/admin/ai?preview=").status_code == 404


def test_the_token_opens_it_and_sets_the_cookie_for_next_time(owner):
    from app import features
    r = get("/admin/ai?preview=owner-token")
    assert r.status_code == 200 and "AI usage" in r.text and 'name="robots" content="noindex"' in r.text
    assert get("/admin/ai", cookies={features.PREVIEW_COOKIE: features._cookie_value()}).status_code == 200


def test_the_page_shows_what_was_recorded(owner):
    async def seed():
        for _ in range(3):
            await ai_usage.log_call("briefing", Completion("x", "qwen/qwen3.8-27b", False, 3300, 400, 0), latency_ms=1500)
        await ai_usage.log_call("search_overview", Completion("x", "openai/gpt-oss-20b", False, 1300, 140, 1))
        ai_usage.count("briefing", "cache_hit", 9)
        ai_usage.count("briefing", "cache_miss", 3)
        await ai_usage.flush()
    asyncio.run(seed())
    html = get("/admin/ai?preview=owner-token").text
    assert ">4</span>" in html                                   # model calls today
    assert "{:,}".format(3 * 3700 + 1440) in html                # tokens today
    assert "75%" in html and "of 12" in html                     # briefing cache: 9 of 12
    assert "Briefing came from the cache 75% of the time today." in html
    assert "not cached" in html                                  # search has no cache yet
    assert "Show as a table" in html


def test_an_empty_history_says_so_instead_of_drawing_nothing(owner):
    html = get("/admin/ai?preview=owner-token").text
    assert "No model calls in the last 14 days yet." in html and "No AI calls yet today." in html


# ── the chart ────────────────────────────────────────────────────────────────

def day(d, **tokens):
    full = {f: tokens.get(f, 0) for f in ai_usage.FEATURES}
    return {"day": d, "tokens": full, "total": sum(full.values())}


def test_columns_scale_to_a_round_top_and_stack_in_feature_order():
    c = admin.chart([day("2026-10-19", briefing=3000, search_overview=1400), day("2026-10-20", explain=800)])
    assert [t["label"] for t in c["ticks"]] == ["0", "1.2K", "2.5K", "3.8K", "5K"]
    first, second = c["columns"]
    assert first["height"] == 88.0 and second["height"] == 16.0
    assert [s["feature"] for s in first["segments"]] == ["search_overview", "briefing"]   # FEATURES order
    assert first["label"] == "19 Oct" and not c["empty"]


@pytest.mark.parametrize("top,expected", [(0, 1000), (7, 10), (120, 200), (2400, 2500), (79_198, 100_000)])
def test_the_scale_top_is_a_round_number(top, expected):
    assert admin._nice(top) == expected


# ── insights ─────────────────────────────────────────────────────────────────

def features_with(**over):
    base = {f: {"calls": 0, "ok": 0, "failed": 0, "unavailable": 0, "unparsable": 0, "over_budget": 0,
                "fallback": 0, "truncated": 0, "tokens_in": 0, "tokens_out": 0, "avg_ms": None, "counts": {}}
            for f in ai_usage.FEATURES}
    for f, v in over.items():
        base[f].update(v)
    return base


def levels(insights):
    return [(i["level"], i["text"]) for i in insights]


def test_a_model_near_its_daily_tokens_is_critical_and_a_tight_minute_a_warning():
    models = {"m": {"share_of_daily_tokens": 0.85, "window": {"tokens_limit": 8000, "tokens_left": 900, "age_s": 10}}}
    got = levels(ai_usage._insights(features_with(briefing={"calls": 1, "ok": 1}), models, {}))
    assert ("critical", "m has used 85% of today's token limit.") in got
    assert any(lv == "warning" and "nearly out of tokens for the minute" in t for lv, t in got)


def test_backup_answers_over_a_tenth_are_flagged_only_with_enough_calls():
    few = features_with(explain={"calls": 3, "ok": 3, "fallback": 2})
    many = features_with(explain={"calls": 10, "ok": 10, "fallback": 3})
    assert not any("backup model" in t for _, t in levels(ai_usage._insights(few, {}, {})))
    assert any("30% of answers came from a backup model" in t for _, t in levels(ai_usage._insights(many, {}, {})))


def test_uncached_search_using_most_tokens_is_a_warning():
    got = levels(ai_usage._insights(features_with(), {}, {"search_overview": 55, "briefing": 45}))
    assert got[0][0] == "warning" and "isn't cached" in got[0][1]


def test_a_quiet_healthy_day_says_nothing_needs_attention():
    got = levels(ai_usage._insights(features_with(briefing={"calls": 2, "ok": 2}), {}, {"briefing": 100}))
    assert got[-1] == ("good", "Nothing needs attention today.")
