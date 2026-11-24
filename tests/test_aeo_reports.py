"""
Pages built to be cited (docs/AEO-PLAN.md phase 3) and the visibility
dashboard (phase 6):

  * topic pages live at a slug that survives restarts; the numeric address 301s,
  * each month's trend report is stored, frozen once the month is over, and
    served as a page and a CSV; neither sets a cookie or writes exposures,
  * Lasting as CSV and RSS,
  * the sitemap lists reports and slug addresses,
  * /admin/visibility sums the crawler and AI-referral counts, owner only.
"""
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import aiosqlite
import httpx
import numpy as np
import pytest

from app import bots, config, db, features, hn_svc, ops_digest, qdrant_svc, trend_reports, visibility
from app import trending_shelf as ts
from app import trending_topics as tt
from app.main import app
from app.routers import reports as reports_router

GOOGLEBOT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"


def unit(axis: int) -> np.ndarray:
    v = np.zeros(1024, dtype=np.float32)
    v[axis] = 1.0
    return v


def topic(tid, name, q, slope, frm, to, papers=100):
    return {"id": tid, "name": name, "q": q, "slope": slope, "from_share": frm, "to_share": to,
            "also": ["x", "y"] if name else [], "papers": papers, "coverage": 0.4,
            "rolling": [frm, (frm + to) / 2, to], "_medoid": unit(tid), "medoid_paper_id": f"m{tid}",
            "top": [{"arxiv_id": f"2606.0{tid}001", "title": f"Top of {name}", "upvotes": 300}],
            "best": [{"arxiv_id": f"2606.0{tid}001", "title": f"Top of {name}", "upvotes": 300,
                      "listed_on": "2026-06-10"}]}


def topics():
    out = [topic(1, "World Models", 0.006, 0.50, 0.012, 0.039, papers=300),
           topic(2, "Reinforcement Learning", 0.04, -0.40, 0.066, 0.035),
           topic(3, "Diffusion", 0.27, -0.26, 0.050, 0.051),
           topic(4, None, 0.99, 0.0, 0.08, 0.086)]
    tt.assign_slugs(out)
    return out


def client(cookie=None, ua=None):
    headers = {"User-Agent": ua} if ua else {}
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers,
                             cookies={config.COOKIE_NAME: cookie} if cookie else {})


@pytest.fixture
async def world(monkeypatch):
    await db.init_db()
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute("DELETE FROM trend_snapshots")
        await conn.commit()
    monkeypatch.setattr(tt, "_state", {**tt._state, "topics": topics(), "topic_of": {},
                                       "first_week": "2026-05-25", "weeks": 26, "papers": 5176,
                                       "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    monkeypatch.setattr(ts, "_state", {**ts._state, "papers": {}, "code": {}, "vectors": {}})
    monkeypatch.setattr(qdrant_svc, "search_recent", AsyncMock(return_value=[]))
    monkeypatch.setenv("TRENDING_PAGE", "on")
    import app.routers.trending as tr
    tr._more_cache.clear()


async def rows(sql, *args):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        return await (await conn.execute(sql, args)).fetchall()


# ── Slugs ────────────────────────────────────────────────────────────────────

def test_slugs_come_from_names_and_never_collide():
    assert tt.slugify("Vision-Language-Action Models") == "vision-language-action-models"
    assert tt.slugify("  3D Gaussian Splatting! ") == "3d-gaussian-splatting"
    ts_ = [{"name": "Robots", "papers": 50}, {"name": "Robots", "papers": 90}, {"name": None, "papers": 99},
           {"name": "!!!", "papers": 10}]
    tt.assign_slugs(ts_)
    assert ts_[1]["slug"] == "robots" and ts_[0]["slug"] == "robots-2"      # the bigger keeps the plain slug
    assert ts_[2]["slug"] is None and ts_[3]["slug"] == "topic"
    year = [{"name": "2024", "papers": 5}]
    tt.assign_slugs(year)
    assert year[0]["slug"] == "topic-2024"


def test_compute_gives_every_named_topic_a_slug():
    rng = np.random.default_rng(1)
    rows_, vectors = [], {}
    for i in range(600):
        aid = f"2605.{i:05d}"
        rows_.append({"arxiv_id": aid, "title": f"Paper about world models {i}" if i % 2 else f"Robot arms {i}",
                      "keywords": [], "summary": "", "upvotes": i % 7,
                      "listed_on": f"2026-{5 + i % 5:02d}-{1 + i % 27:02d}"})
        vectors[aid] = rng.normal(size=16) + (5 if i % 2 else -5)
    out = tt.compute(rows_, vectors, k=4, permutations=20)
    for t in out["topics"]:
        assert (t["slug"] is None) == (t["name"] is None)
    named = [t["slug"] for t in out["topics"] if t["slug"]]
    assert len(named) == len(set(named))


async def test_topic_page_lives_at_its_slug_and_the_id_redirects(world):
    async with client() as c:
        r = await c.get("/trending/topics/world-models")
        assert r.status_code == 200 and "World Models" in r.text
        assert f'<link rel="canonical" href="{config.PUBLIC_URL}/trending/topics/world-models"' in r.text
        old = await c.get("/trending/topics/1?utm_source=x")
        assert old.status_code == 301 and old.headers["location"] == "/trending/topics/world-models?utm_source=x"
        assert (await c.get("/trending/topics/no-such-topic")).status_code == 404
        assert (await c.get("/trending/topics/4")).status_code == 404            # unnamed topics never show
        assert (await c.get("/trending/topics/%C2%B2")).status_code == 404       # a Unicode digit is not an id


async def test_trending_links_topics_by_slug(world):
    async with client() as c:
        html = (await c.get("/trending")).text
    assert 'href="/trending/topics/world-models"' in html
    assert 'href="/trending/topics/1"' not in html
    assert 'href="/trending/reports"' in html


# ── Monthly reports: storage ─────────────────────────────────────────────────

def test_snapshot_keeps_named_topics_in_report_order():
    snap = trend_reports.snapshot(topics(), {"computed_at": "2026-11-24T10:00:00+00:00",
                                             "first_week": "2026-05-25", "weeks": 26, "papers": 5176},
                                  tt.Q_MAX, tt.label)
    assert [(t["name"], t["label"]) for t in snap["topics"]] == [
        ("World Models", "rising"), ("Reinforcement Learning", "cooling"), ("Diffusion", "steady")]
    assert snap["topics"][0]["slug"] == "world-models" and snap["topics"][0]["top"][0]["arxiv_id"] == "2606.01001"
    json.dumps(snap)                                                         # storable as is


async def test_only_the_current_month_is_ever_written(world):
    now = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
    snap = {"computed_at": "2026-11-24T10:00:00+00:00", "topics": [], "first_week": "2026-05-25"}
    assert await trend_reports.save(snap, now) == "2026-11"
    assert await trend_reports.save({**snap, "papers": 9}, now) == "2026-11"   # same month: updated
    assert (await trend_reports.get("2026-11"))["papers"] == 9
    # A computation stamped in October, saved in November, must not reopen October.
    assert await trend_reports.save({**snap, "computed_at": "2026-10-31T23:00:00+00:00"}, now) is None
    assert await trend_reports.get("2026-10") is None
    # And once December comes, November's row stays as it was.
    december = datetime(2026, 12, 1, 1, tzinfo=timezone.utc)
    assert await trend_reports.save({**snap, "computed_at": "2026-11-30T23:59:00+00:00", "papers": 1},
                                    december) is None
    assert (await trend_reports.get("2026-11"))["papers"] == 9
    assert await trend_reports.months() == ["2026-11"]


def test_months_are_validated_and_finality_is_by_calendar():
    from datetime import date
    assert trend_reports.is_month("2026-11") and not trend_reports.is_month("2026-13")
    assert not trend_reports.is_month("lasting") and not trend_reports.is_month("2026-1")
    assert not trend_reports.is_month("\uff12\uff10\uff12\uff16-11")                  # full-width digits
    assert trend_reports.is_final("2026-10", date(2026, 11, 1))
    assert not trend_reports.is_final("2026-11", date(2026, 11, 30))


async def test_the_refresh_writes_this_months_report(world, monkeypatch):
    await trend_reports.record()
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    snap = await trend_reports.get(month)
    assert snap and [t["name"] for t in snap["topics"]][0] == "World Models"


async def test_a_failed_write_never_raises(world, monkeypatch):
    monkeypatch.setattr(trend_reports, "save", AsyncMock(side_effect=RuntimeError("disk")))
    await trend_reports.record()


def test_csv_has_one_row_per_topic():
    snap = trend_reports.snapshot(topics(), {"computed_at": "2026-11-24T10:00:00+00:00"}, tt.Q_MAX, tt.label)
    table = list(csv.DictReader(io.StringIO(trend_reports.to_csv("2026-11", snap))))
    assert [r["topic"] for r in table] == ["World Models", "Reinforcement Learning", "Diffusion"]
    assert table[0]["trend"] == "rising" and table[0]["change_points"] == "2.70"
    assert table[0]["top_paper_arxiv_id"] == "2606.01001"


# ── Monthly reports: pages ───────────────────────────────────────────────────

async def test_the_report_page_answers_first_and_leaves_no_trace(world):
    await trend_reports.record()
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    label = trend_reports.month_label(month)
    async with client() as c:
        r = await c.get(f"/trending/{month}")
    assert r.status_code == 200 and "set-cookie" not in r.headers
    assert f"Rising and cooling AI research topics, {label}" in r.text
    lead = re.search(r'<p class="trend-lead">(.*?)</p>', r.text, re.S).group(1)
    assert lead.startswith(f"{label} so far: 1 topic rising and 1 cooling.") and "World Models rose fastest, from 1 in 83 to 1 in 26" in lead
    assert "recomputed daily" in r.text and f'href="/trending/{month}.csv"' in r.text
    assert 'href="/trending/topics/world-models"' in r.text
    data = json.loads(re.search(r'ld\+json">(.*?)</script>', r.text, re.S).group(1))
    assert data["@type"] == "Dataset" and data["distribution"][0]["contentUrl"].endswith(f"/trending/{month}.csv")
    assert await rows("SELECT * FROM feed_exposures") == []


async def test_a_past_report_says_it_is_final(world):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        snap = trend_reports.snapshot(topics(), {"computed_at": "2026-10-31T09:00:00+00:00",
                                                 "first_week": "2026-05-04", "weeks": 26, "papers": 5000},
                                      tt.Q_MAX, tt.label)
        await conn.execute("INSERT INTO trend_snapshots VALUES (?, ?, ?, ?)",
                           ("2026-10", snap["computed_at"], json.dumps(snap), "2026-10-31 09:00:00"))
        await conn.commit()
    async with client() as c:
        r = await c.get("/trending/2026-10")
        csv_r = await c.get("/trending/2026-10.csv")
        index = await c.get("/trending/reports")
    assert "Final report: computed 31 October 2026" in r.text
    assert "October 2026: 1 topic rising and 1 cooling. World Models rose fastest" in r.text
    assert csv_r.status_code == 200 and csv_r.headers["content-type"].startswith("text/csv")
    assert csv_r.text.splitlines()[0].startswith("month,topic,slug,trend")
    assert 'href="/trending/2026-10"' in index.text


async def test_report_addresses_that_are_not_reports_are_404(world):
    async with client() as c:
        assert (await c.get("/trending/2026-13")).status_code == 404
        assert (await c.get("/trending/2019-01")).status_code == 404             # no such report
        assert (await c.get("/trending/whatever")).status_code == 404
        assert (await c.get("/trending/2019-01.csv")).status_code == 404


async def test_reports_hide_with_their_switch(world, monkeypatch):
    await trend_reports.record()
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    monkeypatch.setenv("TRENDING_PAGE", "off")
    async with client() as c:
        assert (await c.get(f"/trending/{month}")).status_code == 404
        assert (await c.get("/trending/reports")).status_code == 404


async def test_lasting_is_still_routed_to_its_own_page(world, monkeypatch):
    monkeypatch.setenv("HN_LASTING", "on")
    async with client() as c:
        r = await c.get("/trending/lasting")
    assert r.status_code == 200 and "Hacker News keeps coming back to" in r.text   # not the report route's 404


def test_report_lead_with_no_trends_names_the_largest_change_as_failing_the_test():
    snap = {"topics": [{"label": "steady", "name": "Robots", "from_share": 0.044, "to_share": 0.064},
                       {"label": "steady", "name": "Vision-Language Models", "from_share": 0.063, "to_share": 0.034}],
            "first_week": "2026-05-25", "weeks": 26, "papers": 10}
    lead = reports_router.report_lead(snap, "2026-11", final=True)
    assert lead.startswith("November 2026: no topic's share")
    assert lead.endswith("The largest change, Vision-Language Models (fell from 1 in 16 to 1 in 29 papers), "
                         "did not pass the test.")
    assert reports_router.report_lead({**snap, "topics": []}, "2026-11", True).endswith("would move it.")


def test_report_pages_are_typed_for_counting():
    assert bots.page_type("/trending/2026-11") == "report"
    assert bots.page_type("/trending/reports") == "report"
    assert bots.page_type("/trending/lasting.csv") == "lasting"
    assert bots.page_type("/trending") == "trending"


# ── Lasting as data ──────────────────────────────────────────────────────────

def lasting_cards():
    def card(rank, aid, title, joined):
        return {"rank": rank, "arxiv_id": aid, "title": title, "year": 2017, "field": "ai",
                "hn": {"people": 40, "months": 30, "threads": 55, "span": "2017–2026"},
                "first": 1497000000, "last": 1763000000, "joined": joined}
    return [card(1, "1706.03762", "Attention Is All You Need", 1500000000),
            card(2, "2005.11401", "Retrieval &\n   <Generation>", 1760000000),
            card(3, "1312.6114", "Auto-Encoding Variational Bayes", 0)]


async def test_lasting_csv_and_rss(world, monkeypatch):
    monkeypatch.setenv("HN_LASTING", "on")
    monkeypatch.setattr(hn_svc, "lasting_cards", lasting_cards)
    async with client() as c:
        r = await c.get("/trending/lasting.csv")
        feed = await c.get("/trending/lasting.rss")
    assert r.status_code == 200 and "set-cookie" not in r.headers
    table = list(csv.DictReader(io.StringIO(r.text)))
    assert [t["arxiv_id"] for t in table] == ["1706.03762", "2005.11401", "1312.6114"]
    assert table[0]["people"] == "40" and table[0]["first_linked"] == "2017-06-09"
    assert table[1]["title"] == "Retrieval & <Generation>"                   # LaTeX line breaks folded
    assert table[2]["became_lasting"] == "" and table[0]["url"] == f"{config.PUBLIC_URL}/p/1706.03762"
    assert feed.headers["content-type"].startswith("application/rss+xml")
    channel = ET.fromstring(feed.text).find("channel")
    titles = [i.find("title").text for i in channel.findall("item")]
    assert titles == ["Retrieval & <Generation>", "Attention Is All You Need"]   # newest first, escaped, never-joined left out


async def test_lasting_data_hides_with_its_switch(world, monkeypatch):
    monkeypatch.setenv("HN_LASTING", "off")
    async with client() as c:
        assert (await c.get("/trending/lasting.csv")).status_code == 404
        assert (await c.get("/trending/lasting.rss")).status_code == 404


# ── Sitemap ──────────────────────────────────────────────────────────────────

async def test_the_sitemap_lists_reports_and_slug_addresses(world):
    await trend_reports.record()
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    from app.routers import seo
    seo._cache.update(at=0.0, body="")
    async with client(ua=GOOGLEBOT) as c:
        body = (await c.get("/sitemap.xml")).text
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = {u.find("s:loc", ns).text for u in ET.fromstring(body).findall("s:url", ns)}
    base = config.PUBLIC_URL
    assert {f"{base}/trending/{month}", f"{base}/trending/reports", f"{base}/trending/topics/world-models"} <= locs
    assert f"{base}/trending/topics/1" not in locs
    lastmod = {u.find("s:loc", ns).text: u.find("s:lastmod", ns).text for u in ET.fromstring(body).findall("s:url", ns)
               if u.find("s:lastmod", ns) is not None}
    assert lastmod[f"{base}/trending/{month}"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def test_reports_leave_the_sitemap_when_their_page_is_off(world, monkeypatch):
    await trend_reports.record()
    monkeypatch.setenv("TRENDING_PAGE", "preview")
    from app.routers import seo
    seo._cache.update(at=0.0, body="")
    async with client(ua=GOOGLEBOT) as c:
        body = (await c.get("/sitemap.xml")).text
    assert "/trending/reports" not in body


# ── /admin/visibility ────────────────────────────────────────────────────────

def test_visibility_sums_by_crawler_and_window():
    days = {"2026-11-24": {"bot:search:Googlebot:paper": 5, "bot:search:Googlebot:topic": 2,
                           "bot:user_fetch:ChatGPT-User:paper": 1, "ai_ref:chatgpt:paper": 2,
                           "bot_status:training:404": 3, "indexnow:200": 1, "search_ms": 900},
            "2026-11-10": {"bot:search:Googlebot:paper": 10, "ai_ref:perplexity:trending": 1}}
    out = visibility.summarise(days, "2026-11-24")
    g = next(r for r in out["crawlers"] if r["name"] == "Googlebot")
    assert (g["week"], g["month"]) == (7, 17) and g["top_pages"].startswith("paper 15")
    assert out["kpis"] == {"search_week": 7, "user_fetch_week": 1, "training_week": 0,
                           "referrals_week": 2, "referrals_month": 3}
    assert out["errors"] == [{"cls": "training", "status": "404", "n": 3}]
    assert out["indexnow"] == {"200": 1}
    assert len(out["days"]) == visibility.DAYS and out["days"][-1] == {
        "day": "2026-11-24", "values": {"search": 7, "user_fetch": 1}}


async def test_visibility_dashboard_is_owner_only(world, monkeypatch):
    monkeypatch.setattr(ops_digest, "_counts", {})
    async with client() as c:
        assert (await c.get("/admin/visibility")).status_code == 404
    monkeypatch.setattr(features, "has_preview", lambda request: True)
    ops_digest.count("bot:user_fetch:Claude-User:paper")
    async with client() as c:
        r = await c.get("/admin/visibility")
    assert r.status_code == 200 and "Claude-User" in r.text and "noindex" in r.text
