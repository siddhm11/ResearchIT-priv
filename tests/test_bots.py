"""
Crawlers, answer engines and link previews (app/bots.py; docs/AEO-PLAN.md).

What has to hold:
  * a bot gets the same page a reader gets (no cloaking),
  * but leaves no trace in reader data: no cookie, no click, view or exposure,
  * and never starts a model call,
  * while a person arriving at a public page gets a reader id so their first
    click matches an exposure (CLAUDE.md §3.11).
Plus the crawl basics: one address per paper, HEAD answered, 404s unindexed,
the summary in the HTML, the sitemap's dates, About and Privacy.
"""
import json
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import httpx
import pytest

from app import analytics, bots, config, db, groq_svc, ops_digest, qdrant_svc, turso_svc
from app import trending_shelf as ts
from app.main import app

GPTBOT = "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; GPTBot/1.2; +https://openai.com/gptbot)"
CLAUDE_USER = "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; Claude-User/1.0; +Claude-User@anthropic.com)"
GOOGLEBOT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
SLACK = "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)"
SAFARI = ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
          "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1")
CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/131.0.0.0 Safari/537.36")
PID = "2005.11401"


# ── classification ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("ua,expected", [
    (GPTBOT, ("training", "GPTBot")),
    (CLAUDE_USER, ("user_fetch", "Claude-User")),
    ("Mozilla/5.0 (compatible; ChatGPT-User/1.0; +https://openai.com/bot)", ("user_fetch", "ChatGPT-User")),
    ("Mozilla/5.0 (compatible; OAI-SearchBot/1.0; +https://openai.com/searchbot)", ("search", "OAI-SearchBot")),
    ("Mozilla/5.0 (compatible; Claude-SearchBot/1.0)", ("search", "Claude-SearchBot")),
    ("Mozilla/5.0 (compatible; PerplexityBot/1.0)", ("search", "PerplexityBot")),
    ("Mozilla/5.0 (compatible; Perplexity-User/1.0)", ("user_fetch", "Perplexity-User")),
    (GOOGLEBOT, ("search", "Googlebot")),
    ("Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)", ("search", "bingbot")),
    ("Mozilla/5.0 (compatible; ClaudeBot/1.0; +claudebot@anthropic.com)", ("training", "ClaudeBot")),
    ("CCBot/2.0 (https://commoncrawl.org/faq/)", ("training", "CCBot")),
    (SLACK, ("preview", "Slackbot")),
    ("Twitterbot/1.0", ("preview", "Twitterbot")),
    ("facebookexternalhit/1.1", ("preview", "facebookexternalhit")),
    ("Mozilla/5.0 (compatible; SemrushBot/7~bl)", ("other", "other")),
    ("python-requests/2.32", ("other", "other")),
    ("", ("other", "empty")),
    (None, ("other", "empty")),
    (SAFARI, None),
    (CHROME, None),
])
def test_user_agents_are_classed(ua, expected):
    assert bots.identify(ua) == expected


def test_a_hosted_claude_browser_counts_as_a_user_fetch():
    # Anthropic-hosted browsers add the token to an ordinary Chrome string.
    assert bots.classify(CHROME + " Claude-Agent/1.0") == "user_fetch"


@pytest.mark.parametrize("path,kind", [
    ("/", "home"), ("/p/2005.11401", "paper"), ("/trending", "trending"), ("/trending/lasting", "lasting"),
    ("/trending/topics/3", "topic"), ("/collections/rag-retrieval", "collection"), ("/sitemap.xml", "crawl_file"),
    ("/about", "about"), ("/search", "other"),
])
def test_page_types(path, kind):
    assert bots.page_type(path) == kind


def _req(ua, path="/p/1", referer=None, utm=None, cookie=None):
    headers = {"user-agent": ua}
    if referer:
        headers["referer"] = referer
    return SimpleNamespace(method="GET", url=SimpleNamespace(path=path), headers=headers,
                           cookies={config.COOKIE_NAME: cookie} if cookie else {},
                           query_params={"utm_source": utm} if utm else {})


def test_ai_referrals_by_utm_or_referer():
    assert bots.ai_referral(_req(SAFARI, utm="chatgpt.com")) == "chatgpt"
    assert bots.ai_referral(_req(SAFARI, referer="https://claude.ai/chat/abc")) == "claude"
    assert bots.ai_referral(_req(SAFARI, referer="https://www.perplexity.ai/search?q=x")) == "perplexity"
    assert bots.ai_referral(_req(SAFARI, referer="https://news.ycombinator.com/item?id=1")) is None
    assert bots.ai_referral(_req(SAFARI, referer="https://notclaude.ai.example.com/")) is None


def test_bots_are_counted_apart_and_never_as_readers(monkeypatch):
    monkeypatch.setattr(ops_digest, "_counts", {})
    monkeypatch.setattr(analytics, "_readers", {})
    analytics.observe(_req(CLAUDE_USER, "/p/2005.11401", cookie="someone"), 200, "text/html")
    analytics.observe(_req(GOOGLEBOT, "/trending"), 200, "text/html")
    analytics.observe(_req(GPTBOT, "/nope"), 404, "text/html")
    analytics.observe(_req(SAFARI, "/p/2005.11401", referer="https://chatgpt.com/", cookie="r1"), 200, "text/html")
    metrics = {m: n for (_d, m), n in ops_digest._counts.items()}
    assert metrics == {"bot:user_fetch:Claude-User:paper": 1, "bot:search:Googlebot:trending": 1,
                       "bot_status:training:404": 1, "ai_ref:chatgpt:paper": 1}
    assert [u for (_d, u) in analytics._readers] == ["r1"]


# ── routes ───────────────────────────────────────────────────────────────────

def client(ua=SAFARI, cookie=None):
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                          headers={"user-agent": ua}, follow_redirects=False)
    if cookie:
        c.cookies.set(config.COOKIE_NAME, cookie, domain="test.local")
    return c


@pytest.fixture(autouse=True)
async def world(monkeypatch):
    await db.init_db()

    async def meta(ids):
        return {a: {"arxiv_id": a, "title": "Retrieval-Augmented Generation", "abstract": "An abstract about RAG.",
                    "authors": '["Patrick Lewis", "Ethan Perez"]', "category": "cs.CL",
                    "published": "2020-05-22"} for a in ids}
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", meta)
    monkeypatch.setattr(qdrant_svc, "get_paper_vectors", AsyncMock(return_value={}))
    monkeypatch.setenv("PAPER_PAGE_V2", "on")


async def rows(sql, *args):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        return await (await conn.execute(sql, args)).fetchall()


async def test_a_crawler_gets_the_page_but_leaves_no_trace():
    async with client(GPTBOT) as c:
        r = await c.get(f"/p/{PID}?qid=q1&pos=0&src=x&pol=trending_page_top_v1")
    assert r.status_code == 200 and "Retrieval-Augmented Generation" in r.text
    assert "set-cookie" not in r.headers
    assert await rows("SELECT * FROM interactions") == []


async def test_a_person_from_a_ranked_link_still_logs_the_click():
    async with client(SAFARI, cookie="reader") as c:
        r = await c.get(f"/p/{PID}?qid=q1&pos=2&pol=trending_page_top_v1&prop=1")
    assert r.status_code == 200 and "set-cookie" in r.headers
    assert await rows("SELECT event_type, query_id, position FROM interactions") == [("click", "q1", 2)]


async def test_a_written_summary_is_in_the_html_and_a_missing_one_is_not_faked(monkeypatch):
    async with client(GPTBOT) as c:
        before = (await c.get(f"/p/{PID}")).text
    assert ">Writing a plain-language summary" not in before          # drawn by CSS from an attribute, not page text
    assert 'hx-get="/api/papers/2005.11401/explain?v=2"' in before
    model = groq_svc.explanation_models()[0]
    await db.save_explanation(groq_svc.explain_cache_key(PID, "An abstract about RAG.", model=model), PID,
                              "It lets a language model look facts up before it answers.", model)
    async with client(GPTBOT) as c:
        after = (await c.get(f"/p/{PID}")).text
    assert "It lets a language model look facts up before it answers." in after
    assert "/explain?v=2" not in after


async def test_a_crawler_never_starts_a_summary(monkeypatch):
    called = AsyncMock()
    monkeypatch.setattr(groq_svc, "explain_paper_result", called)
    async with client(GOOGLEBOT) as c:
        r = await c.get(f"/api/papers/{PID}/explain?v=2")
    assert r.text == "" and not called.await_count


async def test_the_paper_page_says_what_it_is_in_schema_org():
    async with client(GPTBOT) as c:
        html = (await c.get(f"/p/{PID}")).text
    blob = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(blob)
    assert data["@type"] == "ScholarlyArticle" and data["sameAs"] == f"https://arxiv.org/abs/{PID}"
    assert [a["name"] for a in data["author"]] == ["Patrick Lewis", "Ethan Perez"]
    assert data["datePublished"] == "2020-05-22" and data["url"].endswith(f"/p/{PID}")


@pytest.mark.parametrize("path", [f"/p/{PID}v2", f"/abs/{PID}", f"/abs/{PID}v3", f"/paper/{PID}", f"/p/abs/{PID}"])
async def test_borrowed_and_versioned_addresses_fold_into_one(path):
    async with client() as c:
        r = await c.get(path)
    assert r.status_code == 301 and r.headers["location"] == f"/p/{PID}"


async def test_old_style_ids_keep_their_slash():
    async with client() as c:
        r = await c.get("/abs/hep-th/9711200v1")
    assert r.headers["location"] == "/p/hep-th/9711200"


async def test_head_is_answered_like_get():
    async with client(GPTBOT) as c:
        r = await c.head("/robots.txt")
        g = await c.get("/robots.txt")
    assert r.status_code == 200 and r.content == b""
    assert int(r.headers["content-length"]) == len(g.content)


async def test_a_missing_page_is_not_offered_for_indexing():
    async with client(GPTBOT) as c:
        r = await c.get("/no-such-page")
    assert r.status_code == 404
    assert 'rel="canonical"' not in r.text and 'name="robots" content="noindex"' in r.text


async def test_about_and_privacy_exist_and_link_from_every_page():
    async with client(GPTBOT) as c:
        about, privacy = await c.get("/about"), await c.get("/privacy")
    assert about.status_code == 200 and "free reading feed and search engine for arXiv" in about.text
    assert json.loads(re.search(r'ld\+json">(.*?)</script>', about.text, re.S).group(1))["isAccessibleForFree"]
    assert privacy.status_code == 200 and "arxiv_user_id" in privacy.text and "Groq" in privacy.text
    for page in (about, privacy):
        assert 'href="/collections"' in page.text and 'href="/privacy"' in page.text


async def test_the_front_page_writes_no_exposures_for_a_crawler(monkeypatch):
    monkeypatch.setattr(ts, "_state", {**ts._state, "code": {}, "vectors": {}, "papers": {
        "2611.00001": {"arxiv_id": "2611.00001", "title": "A paper", "upvotes": 9,
                       "published_at": "2026-11-10T00:00:00+00:00", "listed_on": "2026-11-12",
                       "categories": ["cs.LG"]}}})
    async with client(GOOGLEBOT) as c:
        r = await c.get("/")
    assert r.status_code == 200 and "A paper" in r.text and "set-cookie" not in r.headers
    assert 'href="/collections"' in r.text
    assert await rows("SELECT * FROM feed_exposures") == []
    async with client(SAFARI) as c:
        r = await c.get("/")
    assert "set-cookie" in r.headers and len(await rows("SELECT * FROM feed_exposures")) == 1


async def test_collections_set_no_cookie_for_a_crawler_and_describe_themselves():
    async with client(GPTBOT) as c:
        r = await c.get("/collections/rag-retrieval")
    assert r.status_code == 200 and "set-cookie" not in r.headers
    data = json.loads(re.search(r'ld\+json">(.*?)</script>', r.text, re.S).group(1))
    assert data["@type"] == "ItemList" and data["numberOfItems"] >= 6
    assert data["itemListElement"][0]["url"].startswith(config.PUBLIC_URL + "/p/")
    assert "A personal reading feed for arXiv" not in re.search(r'<meta name="description" content="([^"]*)"',
                                                                 r.text).group(1)


async def test_the_sitemap_dates_only_what_it_knows():
    model = groq_svc.explanation_models()[0]
    await db.save_explanation(groq_svc.explain_cache_key(PID, "x", model=model), PID, "A summary.", model)
    from app.routers import seo
    seo._cache.update(at=0.0, body="")
    async with client(GOOGLEBOT) as c:
        body = (await c.get("/sitemap.xml")).text
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = {u.find("s:loc", ns).text: (u.find("s:lastmod", ns).text if u.find("s:lastmod", ns) is not None else None)
            for u in ET.fromstring(body).findall("s:url", ns)}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", urls[f"{config.PUBLIC_URL}/p/{PID}"])
    assert urls[f"{config.PUBLIC_URL}/collections"] is None
    assert f"{config.PUBLIC_URL}/about" in urls and f"{config.PUBLIC_URL}/privacy" in urls


async def test_indexnow_key_file(monkeypatch):
    async with client(GOOGLEBOT) as c:
        assert (await c.get("/abc123.txt")).status_code == 404
        monkeypatch.setattr(config, "INDEXNOW_KEY", "abc123")
        r = await c.get("/abc123.txt")
        assert r.status_code == 200 and r.text == "abc123"
        assert (await c.get("/other.txt")).status_code == 404


async def test_verification_tags_render_only_when_set(monkeypatch):
    from app.templates_env import templates
    async with client(GOOGLEBOT) as c:
        assert "google-site-verification" not in (await c.get("/about")).text
        monkeypatch.setitem(templates.env.globals, "google_site_verification", "tok-1")
        assert '<meta name="google-site-verification" content="tok-1" />' in (await c.get("/about")).text


# ── IndexNow ─────────────────────────────────────────────────────────────────

async def test_indexnow_sends_todays_changes_only_when_keyed(monkeypatch):
    from app import http_client, indexnow
    from app.routers import seo
    today = __import__("time").strftime("%Y-%m-%d", __import__("time").gmtime())
    monkeypatch.setattr(seo, "entries", AsyncMock(return_value=[("/trending", today), ("/collections", None),
                                                                ("/p/1", "2020-01-01")]))
    sent = []

    class Client:
        async def post(self, url, json, timeout, headers):
            sent.append((url, json))
            return SimpleNamespace(status_code=202, text="")
    monkeypatch.setattr(http_client, "get_client", lambda: Client())
    monkeypatch.setattr(config, "INDEXNOW_KEY", "")
    assert await indexnow.ping(await indexnow.changed_urls()) is None and sent == []
    monkeypatch.setattr(config, "INDEXNOW_KEY", "k123")
    assert await indexnow.ping(await indexnow.changed_urls()) == 202
    url, body = sent[0]
    assert url == indexnow.ENDPOINT and body["urlList"] == [config.PUBLIC_URL + "/trending"]
    assert body["keyLocation"] == f"{config.PUBLIC_URL}/k123.txt" and body["host"] in config.PUBLIC_URL


@pytest.mark.parametrize("ua", [
    "Mozilla/5.0 (Linux; Android 9; CUBOT P30) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/131.0.0.0 Safari/537.36",
])
def test_people_whose_browser_string_looks_odd_are_still_people(ua):
    assert bots.identify(ua) is None


@pytest.mark.parametrize("ua", ["Mozilla/5.0 (compatible; AhrefsBot/7.0)", "MJ12bot/v1.4.8",
                                "Mozilla/5.0 (compatible; DotBot/1.2)", "some crawler 1.0"])
def test_unknown_bots_are_still_bots(ua):
    assert bots.classify(ua) == "other"


async def test_a_redirect_keeps_the_ranked_link_so_its_click_is_logged():
    async with client(SAFARI, cookie="reader") as c:
        r = await c.get(f"/p/{PID}v2?qid=q9&pos=1&pol=p1")
        assert r.status_code == 301 and r.headers["location"] == f"/p/{PID}?qid=q9&pos=1&pol=p1"
        await c.get(r.headers["location"])
    assert await rows("SELECT event_type, query_id FROM interactions") == [("click", "q9")]


def test_old_ids_with_a_subject_class_lose_it():
    from app.routers.paper import canonical_arxiv_id
    assert canonical_arxiv_id("math.AG/0101001v2") == "math/0101001"
    assert canonical_arxiv_id("solv-int/9901001") == "solv-int/9901001"
    assert canonical_arxiv_id("2005.11401") == "2005.11401"


async def test_a_head_request_from_a_browser_logs_nothing():
    async with client(SAFARI, cookie="reader") as c:
        r = await c.head(f"/p/{PID}?qid=q1&pos=0")
    assert r.status_code == 200 and r.content == b""
    assert await rows("SELECT * FROM interactions") == []


async def test_a_crawler_with_a_cookie_still_writes_no_exposures(monkeypatch):
    monkeypatch.setenv("TRENDING_PAGE", "on")
    monkeypatch.setattr(ts, "_state", {**ts._state, "code": {}, "vectors": {}, "papers": {
        "2611.00001": {"arxiv_id": "2611.00001", "title": "A paper", "upvotes": 9,
                       "published_at": "2026-11-10T00:00:00+00:00", "listed_on": "2026-11-12",
                       "categories": ["cs.LG"]}}})
    async with client(GOOGLEBOT, cookie="someone") as c:
        r = await c.get("/trending")
    assert r.status_code == 200 and "set-cookie" not in r.headers
    assert await rows("SELECT * FROM feed_exposures") == []
    async with client(SAFARI) as c:
        r = await c.get("/trending")
    assert "set-cookie" in r.headers and await rows("SELECT * FROM feed_exposures") != []


async def test_a_crawler_never_queues_arxiv_fetches(monkeypatch):
    from app import abstract_svc, paper_source_svc
    queued = []
    monkeypatch.setattr(abstract_svc, "is_truncated", lambda a: True)
    monkeypatch.setattr(abstract_svc, "request", lambda a: queued.append(("abstract", a)))
    monkeypatch.setattr(paper_source_svc, "schedule", lambda a: queued.append(("source", a)))
    async with client(GPTBOT) as c:
        await c.get(f"/p/{PID}")
    assert queued == []
    async with client(SAFARI) as c:
        await c.get(f"/p/{PID}")
    assert ("source", PID) in queued


async def test_compare_lists_ourselves_and_sources_every_row():
    from app.routers import pages
    async with client(GPTBOT) as c:
        r = await c.get("/compare")
    assert r.status_code == 200 and "Where ResearchIT falls short" in r.text
    assert all(row["source"] for row in pages.COMPARE)
    assert r.text.count("<tr") == len(pages.COMPARE) + 1
