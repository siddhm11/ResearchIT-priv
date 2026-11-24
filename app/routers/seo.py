"""
/robots.txt and /sitemap.xml (doc 06, 2026-11-17).

/robots.txt used to be the HTML 404 page and there was no sitemap, so a search
engine had nothing to go on but links. Crawlers are kept off what is personal
or costly (search runs two models per query on 2 CPUs; /api/ is fragments)
and pointed at a few thousand pages worth finding: the public pages, every
collection, and the papers people actually look at here (this week's lists,
the papers Hacker News keeps returning to, collection papers, and papers that
already have a plain-language summary). Not all 2 million: a crawler working
through those would compete with readers for the same CPUs.

The sitemap is built in memory at most every SITEMAP_TTL_S and never fails:
a source that errors is left out.
"""
from __future__ import annotations

import time
from xml.sax.saxutils import escape

import aiosqlite
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response

from app import config, errors

router = APIRouter()

SITEMAP_MAX = 5000
SITEMAP_TTL_S = 6 * 3600
_cache: dict = {"at": 0.0, "body": ""}

ROBOTS = """User-agent: *
Disallow: /api/
Disallow: /search
Disallow: /saved
Disallow: /history
Disallow: /interests
Disallow: /onboarding
Disallow: /admin
Disallow: /healthz
Disallow: /l/
Disallow: /library/
Allow: /

Sitemap: {base}/sitemap.xml
"""

# Pages that exist only when written (app/routers/pages.py); listed when present.
STATIC_EXTRA = ("/about", "/compare", "/privacy")


@router.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt() -> PlainTextResponse:
    return PlainTextResponse(ROBOTS.format(base=config.PUBLIC_URL),
                             headers={"Cache-Control": "public, max-age=86400"})


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    """Browsers and search engines ask for /favicon.ico whatever the page says;
    it was the HTML 404 page (UI review, 2026-11-24)."""
    return FileResponse("app/static/favicon.ico", media_type="image/x-icon",
                        headers={"Cache-Control": "public, max-age=604800"})


@router.get("/apple-touch-icon.png", include_in_schema=False)
@router.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def touch_icon() -> FileResponse:
    return FileResponse("app/static/apple-touch-icon.png", media_type="image/png",
                        headers={"Cache-Control": "public, max-age=604800"})


async def _explained(limit: int) -> list[tuple[str, str]]:
    """(arxiv_id, day its newest summary was written), newest first."""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        rows = await (await conn.execute(
            "SELECT arxiv_id, MAX(created_at) FROM paper_explanations GROUP BY arxiv_id "
            "ORDER BY MAX(id) DESC LIMIT ?", (limit,))).fetchall()
    return [(r[0], (r[1] or "")[:10]) for r in rows]


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


async def entries() -> list[tuple[str, str | None]]:
    """(path, lastmod) for every sitemap URL, most important first, at most SITEMAP_MAX.

    lastmod is given only when it is true: the day a page that changes daily
    was built (the trend pages, Lasting), or the day a paper's summary was
    written. A paper page otherwise only repeats arXiv's metadata, so it gets
    no date rather than an invented one.
    """
    from app import collections_svc, hn_svc, trending_shelf, trending_topics
    today = _today()
    out: dict[str, str | None] = {"/": None, "/trending": today, "/trending/lasting": today,
                                  "/collections": None}
    for path in STATIC_EXTRA:
        out[path] = None
    try:
        from app import features, trend_reports, trending_topics
        if features.mode(trending_topics.SWITCH) == "on":     # the pages 404 otherwise
            dated = await trend_reports.dates()
            months = sorted(dated, reverse=True)
            if months:
                out["/trending/reports"] = dated[months[0]] or None
            for m in months:
                # The day it was last computed: today for this month, the
                # month's last run for a past one, which never changes again.
                out[f"/trending/{m}"] = dated[m] or None
    except Exception as e:
        errors.report("seo", "sitemap source trend reports failed", e)
    papers: dict[str, str | None] = {}
    sources = (
        ("collections", lambda: [f"/collections/{c['slug']}" for c in collections_svc.load_all()]),
        ("topics", lambda: [trending_topics.path(t) for t in
                            trending_topics.rising() + trending_topics.cooling() + trending_topics.steady()]),
        ("trending", lambda: [p["arxiv_id"] for p in trending_shelf.week_top(200)]),
        ("lasting", lambda: [c["arxiv_id"] for c in hn_svc.lasting_cards()]),
        ("collection papers", lambda: [a for c in collections_svc.load_all()
                                       for a in collections_svc.anchor_ids(c["slug"])]),
    )
    for name, get in sources:
        try:
            got = get()
        except Exception as e:
            errors.report("seo", f"sitemap source {name} failed", e)
            continue
        if name == "collections":
            out.update((p, None) for p in got)
        elif name == "topics":
            out.update((p, today) for p in got)
        else:
            for a in got:
                if a:
                    papers.setdefault(a, None)
    try:
        for a, day in await _explained(SITEMAP_MAX):
            papers[a] = day or None
    except Exception as e:
        errors.report("seo", "sitemap source explained papers failed", e)
    from app.routers.paper import canonical_arxiv_id
    for a, day in papers.items():
        out.setdefault(f"/p/{canonical_arxiv_id(a)}", day)      # never a URL that redirects
    return list(out.items())[:SITEMAP_MAX]


async def paths() -> list[str]:
    """Every path in the sitemap, most important first."""
    return [p for p, _ in await entries()]


@router.get("/sitemap.xml")
async def sitemap_xml() -> Response:
    from app import trending_shelf
    now = time.time()
    body = _cache["body"]
    if not body or now - _cache["at"] > SITEMAP_TTL_S:
        urls = "".join(
            f"<url><loc>{escape(config.PUBLIC_URL + p)}</loc>"
            + (f"<lastmod>{day}</lastmod>" if day else "") + "</url>"
            for p, day in await entries())
        body = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>\n')
        # Just after a restart the week's lists are still loading: serve this
        # one, but build again next time rather than keep it for hours.
        if trending_shelf.week_top(1):
            _cache.update(at=now, body=body)
    return Response(body, media_type="application/xml",
                    headers={"Cache-Control": "public, max-age=3600"})


# ── Borrowed addresses ───────────────────────────────────────────────────────
# Swapping arxiv.org for this site in an arXiv link should land on the paper,
# and old or guessed paper URLs should never 404 (AI fetchers 404 on ~35% of
# fetches, mostly such URLs). One address per paper: /p/{id}.

@router.get("/abs/{arxiv_id:path}")
@router.get("/paper/{arxiv_id:path}")
async def paper_alias(arxiv_id: str, request: Request) -> RedirectResponse:
    from app.routers.paper import canonical_arxiv_id
    query = request.url.query
    return RedirectResponse(f"/p/{canonical_arxiv_id(arxiv_id)}" + (f"?{query}" if query else ""),
                            status_code=301)


@router.get("/pdf/{arxiv_id:path}")
async def pdf_alias(arxiv_id: str) -> RedirectResponse:
    """The PDF lives on arXiv; this only saves typing the host."""
    from app.routers.paper import canonical_arxiv_id
    return RedirectResponse(f"https://arxiv.org/pdf/{canonical_arxiv_id(arxiv_id)}", status_code=302)


# ── IndexNow ─────────────────────────────────────────────────────────────────
# Bing (which feeds Copilot and is one of ChatGPT's sources), Yandex and others
# accept a ping for new and changed URLs; the key file proves we own the host.
# Unset INDEXNOW_KEY turns all of it off. The daily ping is app/indexnow.py.

@router.get("/{key}.txt", include_in_schema=False)
async def indexnow_key(key: str) -> PlainTextResponse:
    if config.INDEXNOW_KEY and key == config.INDEXNOW_KEY:
        return PlainTextResponse(config.INDEXNOW_KEY)
    raise HTTPException(status_code=404)
