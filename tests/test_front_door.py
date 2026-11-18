"""
The front door (doc 06, 2026-11-17).

8 of the first 18 outside readers never reached a feed: / sent them straight
to a form. Now / shows a front page with search and the week's papers; a
library opens in another browser from a private link; a search with no close
match says so; search engines get robots.txt, a sitemap, https links and a
preview picture; onboarding marks the areas whose older papers are partial.
"""
import re
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock

import httpx
import pytest

from app import config, db, trending_shelf as ts, user_state as us
from app.main import app

WEEK = [f"2611.{i:05d}" for i in range(9)]


def client(cookie: str | None = None, **headers):
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers)
    if cookie:      # the domain the app's Set-Cookie lands on, so a switch replaces it
        c.cookies.set("arxiv_user_id", cookie, domain="test.local")
    return c


def week_paper(aid: str, votes: int) -> dict:
    return {"arxiv_id": aid, "title": "Paper " + aid, "upvotes": votes,
            "published_at": "2026-11-10T00:00:00+00:00", "listed_on": "2026-11-12", "categories": ["cs.LG"]}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Library pages look papers up; these tests never reach Turso or arXiv."""
    from app import arxiv_svc, turso_svc

    async def meta(ids):
        return {a: {"arxiv_id": a, "title": "Paper " + a, "abstract": "About it.", "authors": "[]",
                    "category": "cs.LG", "published": "2024-01-01"} for a in ids}
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", meta)
    monkeypatch.setattr(arxiv_svc, "fetch_metadata_batch", AsyncMock(return_value={}))


@pytest.fixture
def week(monkeypatch):
    monkeypatch.setattr(ts, "_state", {**ts._state, "code": {}, "vectors": {},
                                       "papers": {a: week_paper(a, 100 - i) for i, a in enumerate(WEEK)}})


async def save(user: str, pid: str, event: str = "save"):
    await db.log_interaction(user_id=user, paper_id=pid, event_type=event, source="test")


# ── The front page ───────────────────────────────────────────────────────────

async def test_a_new_visitor_gets_the_front_page_not_a_form(week):
    async with client() as c:
        r = await c.get("/")
    assert r.status_code == 200
    page = r.text
    assert 'class="wl-title"' in page and 'action="/search"' in page
    assert 'href="/onboarding"' in page and "Personalise your feed" in page
    assert "arxiv_user_id" in r.headers.get("set-cookie", "")
    # The week's papers, most upvoted first, as real cards with Save buttons.
    shown = re.findall(r'id="actions-([0-9.]+)"', page)
    assert shown == WEEK[:7]
    assert page.count('hx-post="/api/papers/') >= 7


async def test_front_page_cards_carry_their_ranking_fields_and_are_logged(week):
    async with client("visitor-1") as c:
        page = (await c.get("/")).text
    href = re.search(r'href="(/p/2611\.00000\?[^"]+)"', page).group(1)
    assert "sf=welcome" in href and "pol=welcome_top_v1" in href and "pos=0" in href and "prop=1.0" in href
    import aiosqlite
    async with aiosqlite.connect(config.DB_PATH) as conn:
        rows = await (await conn.execute(
            "SELECT paper_id, position, policy_id, propensity FROM feed_exposures WHERE user_id = 'visitor-1' "
            "ORDER BY position")).fetchall()
    assert [r[0] for r in rows] == WEEK[:7]
    assert {r[2] for r in rows} == {"welcome_top_v1"} and {r[3] for r in rows} == {1.0}


async def test_front_page_without_this_weeks_papers_still_works(monkeypatch):
    monkeypatch.setattr(ts, "_state", {**ts._state, "papers": {}})
    async with client() as c:
        r = await c.get("/")
    assert r.status_code == 200 and 'class="wl-title"' in r.text and "Most upvoted this week" not in r.text


async def test_a_visitor_who_started_onboarding_is_invited_to_finish(week):
    await db.save_onboarding_categories("half-done", ["nlp"])
    async with client("half-done") as c:
        page = (await c.get("/")).text
    assert "Finish setting up your feed" in page


async def test_a_save_from_the_front_page_opens_the_feed_next_time(week):
    async with client("saver") as c:
        assert 'class="wl-title"' in (await c.get("/")).text
        r = await c.post(f"/api/papers/{WEEK[0]}/save", data={"source": "welcome"})
        assert r.status_code == 200
        home = (await c.get("/")).text
    assert 'class="wl-title"' not in home and 'id="rec-section"' in home
    assert (await db.get_onboarding_state("saver"))["onboarding_completed"]


async def test_a_finished_onboarding_still_gets_the_feed(week):
    await db.complete_onboarding("done")
    async with client("done") as c:
        home = (await c.get("/")).text
    assert 'id="rec-section"' in home and 'class="wl-title"' not in home


async def test_onboarding_can_go_back_and_marks_partial_areas():
    async with client("ob") as c:
        page = (await c.get("/onboarding")).text
    assert 'class="ob-back"' in page
    for key, note in config.AREA_COVERAGE_NOTES.items():
        card = re.search(rf'data-key="{key}".*?</button>', page, re.S).group(0)
        assert note in card
    nlp = re.search(r'data-key="nlp".*?</button>', page, re.S).group(0)
    assert 'class="cov"' not in nlp


# ── Search engines and link previews ─────────────────────────────────────────

async def test_robots_txt_is_text_and_points_at_the_sitemap():
    async with client() as c:
        r = await c.get("/robots.txt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "Disallow: /search" in r.text and "Disallow: /api/" in r.text and "Disallow: /l/" in r.text
    assert f"Sitemap: {config.PUBLIC_URL}/sitemap.xml" in r.text
    assert "<html" not in r.text


async def test_sitemap_lists_public_pages_and_papers_and_survives_a_failing_source(week, monkeypatch):
    from app import collections_svc, hn_svc
    from app.routers import seo
    monkeypatch.setattr(seo, "_cache", {"at": 0.0, "body": ""})
    monkeypatch.setattr(collections_svc, "load_all", lambda: [{"slug": "rag"}, {"slug": "agents"}])
    monkeypatch.setattr(collections_svc, "anchor_ids", lambda slug: ["2005.11401"] if slug == "rag" else [])

    def broken():
        raise RuntimeError("hn not loaded")
    monkeypatch.setattr(hn_svc, "lasting_cards", broken)
    await db.save_explanation("k1", "1706.03762", "In plain terms.", "m")
    async with client() as c:
        r = await c.get("/sitemap.xml")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/xml")
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [e.text for e in ET.fromstring(r.text).findall("s:url/s:loc", ns)]
    base = config.PUBLIC_URL
    assert locs[0] == base + "/" and base + "/trending" in locs
    assert base + "/collections/rag" in locs and base + "/collections/agents" in locs
    assert base + f"/p/{WEEK[0]}" in locs and base + "/p/2005.11401" in locs and base + "/p/1706.03762" in locs
    assert len(locs) == len(set(locs)) and all(u.startswith("https://") for u in locs)


async def test_sitemap_is_capped(week, monkeypatch):
    from app.routers import seo
    monkeypatch.setattr(seo, "SITEMAP_MAX", 6)
    assert len(await seo.paths()) == 6


async def test_shared_links_say_https_and_carry_a_picture(week):
    async with client() as c:
        page = (await c.get("/")).text
    assert f'<meta property="og:url" content="{config.PUBLIC_URL}/"' in page
    assert f'<link rel="canonical" href="{config.PUBLIC_URL}/"' in page
    assert f'<meta property="og:image" content="{config.PUBLIC_URL}/static/og.png"' in page
    assert 'name="twitter:card" content="summary_large_image"' in page
    assert config.PUBLIC_URL.startswith("https://")
    import os
    assert os.path.getsize("app/static/og.png") > 10_000


async def test_personal_and_search_pages_are_not_indexed(monkeypatch):
    from app import hybrid_search_svc
    # A result, so the search never falls back to arXiv's own API.
    monkeypatch.setattr(hybrid_search_svc, "search", AsyncMock(return_value=(["2611.00001"], {})))
    async with client("p") as c:
        for path in ("/saved", "/history", "/library/devices", "/search?q=x"):
            page = (await c.get(path)).text
            assert '<meta name="robots" content="noindex"' in page, path
            assert 'rel="canonical"' not in page, path


def test_a_paper_link_shows_its_first_raster_figure():
    from app.routers.paper import share_image
    assert share_image([]) is None
    assert share_image([{"url": "/fig/1/0-a.svg"}]) is None
    assert share_image([{"url": "/fig/1/0-a.svg"}, {"url": "/fig/1/1-b.webp"}]) == config.PUBLIC_URL + "/fig/1/1-b.webp"


# ── A search with no close match ─────────────────────────────────────────────

async def _search(monkeypatch, ce_top, named=None):
    from app import hybrid_search_svc, turso_svc
    meta = {"total_time_ms": 900, "retrieval_mode": "hybrid"}
    if ce_top is not None:
        meta["ce_top"] = ce_top
    if named:
        meta["named_paper"] = {"arxiv_id": named}
    monkeypatch.setattr(hybrid_search_svc, "search", AsyncMock(return_value=(["2611.00001", "2611.00002"], meta)))
    papers = {a: {"arxiv_id": a, "title": "T " + a, "abstract": "A.", "authors": "[]", "category": "cs.LG",
                  "published": "2026-11-01"} for a in ("2611.00001", "2611.00002")}
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", AsyncMock(return_value=papers))
    async with client("searcher", **{"HX-Request": "true"}) as c:
        return (await c.get("/search?q=recipe+for+butter+chicken")).text


async def test_a_weak_search_says_so_and_skips_the_overview(monkeypatch):
    html = await _search(monkeypatch, 0.04)
    assert "None of these is a close match" in html
    assert 'id="ai-summary-container"' not in html
    assert "T 2611.00001" in html          # the nearest papers are still shown


async def test_a_good_search_has_no_notice(monkeypatch):
    html = await _search(monkeypatch, 0.62)
    assert "close match" not in html and 'id="ai-summary-container"' in html


async def test_no_notice_without_a_score_or_when_a_paper_was_named(monkeypatch):
    assert "close match" not in await _search(monkeypatch, None)
    assert "close match" not in await _search(monkeypatch, 0.02, named="2611.00001")


async def test_the_notice_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_WEAK_MATCH", 0.0)
    assert "close match" not in await _search(monkeypatch, 0.02)


# ── Your library on another device ───────────────────────────────────────────

async def make_link(user: str) -> str:
    async with client(user) as c:
        r = await c.post("/api/library-link")
        assert r.status_code == 303 and r.headers["location"].startswith("/library/devices")
        page = (await c.get("/library/devices")).text
    url = re.search(r'id="devlink-url" value="([^"]+)"', page).group(1)
    assert url.startswith(config.PUBLIC_URL + "/l/")
    return url.rsplit("/", 1)[1]


async def test_a_new_browser_opens_the_library_from_its_link():
    await save("owner", "2401.00001")
    token = await make_link("owner")
    async with client() as phone:
        r = await phone.get(f"/l/{token}")
        assert r.status_code == 303 and r.headers["location"] == "/saved?opened=1"
        assert phone.cookies.get("arxiv_user_id") == "owner"
        page = (await phone.get("/saved?opened=1")).text
    assert "Your library is open in this browser." in page


async def test_a_new_link_replaces_the_old_one_and_off_means_off():
    old = await make_link("owner")
    new = await make_link("owner")
    assert old != new and len(new) >= 20
    async with client() as c:
        gone = await c.get(f"/l/{old}")
        assert gone.status_code == 404 and "This link no longer works" in gone.text
        assert (await c.get(f"/l/{new}")).status_code == 303
    async with client("owner") as c:
        await c.post("/api/library-link/revoke")
        page = (await c.get("/library/devices")).text
    assert "Make my library link" in page
    async with client() as c:
        assert (await c.get(f"/l/{new}")).status_code == 404


async def test_a_browser_with_its_own_saves_is_asked_then_merged():
    await save("owner", "2401.00001")
    await save("owner", "2401.00009", "not_interested")
    await save("laptop", "2401.00002")
    await save("laptop", "2401.00009")         # the owner passed on this one: the owner's decision stays
    await db.follow_collection("laptop", "rag")
    token = await make_link("owner")
    async with client("laptop") as c:
        ask = await c.get(f"/l/{token}")
        assert ask.status_code == 200 and "Open this library here?" in ask.text
        assert c.cookies.get("arxiv_user_id") == "laptop"       # nothing switched yet
        r = await c.post(f"/l/{token}")
        assert r.status_code == 303 and r.headers["location"] == "/saved?opened=1&added=1"
        assert c.cookies.get("arxiv_user_id") == "owner"
    feedback = {f["paper_id"]: f["event_type"] for f in await db.get_current_feedback("owner")}
    assert feedback == {"2401.00001": "save", "2401.00002": "save", "2401.00009": "not_interested"}
    assert "rag" in await db.get_followed_slugs("owner")
    us._cache.clear()
    assert "2401.00002" in (await us.ensure_loaded("owner")).positive_list


async def test_another_site_cannot_switch_a_library():
    await save("victim", "2401.00003")
    token = await make_link("attacker")
    async with client("victim", **{"sec-fetch-site": "cross-site"}) as c:
        assert (await c.post(f"/l/{token}")).status_code == 403
        assert (await c.post("/api/library-link")).status_code == 403
    async with client("victim", origin="https://evil.example") as c:
        assert (await c.post(f"/l/{token}")).status_code == 403
    assert await db.get_current_feedback("attacker") == []
    assert await db.get_library_link("victim") is None


async def test_opening_your_own_link_changes_nothing():
    token = await make_link("owner")
    async with client("owner") as c:
        r = await c.get(f"/l/{token}")
    assert r.headers["location"] == "/saved?opened=here"


async def test_library_links_replicate_and_are_rate_limited():
    import aiosqlite
    from app import rate_limit, turso_sync
    spec = turso_sync.TABLES["library_links"]
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cols = {r[1] for r in await (await conn.execute("PRAGMA table_info(library_links)")).fetchall()}
    assert set(spec["cols"]) <= cols and spec["watermark"] == "updated_at"
    assert any("library_links" in ddl for ddl in turso_sync.REMOTE_DDL)
    assert rate_limit._rule_for("/l/abc")[0] == "/l"
    assert rate_limit._rule_for("/lasting") is None
