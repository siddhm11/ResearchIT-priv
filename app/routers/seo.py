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
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response

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

STATIC_PATHS = ("/", "/trending", "/trending/lasting", "/collections")


@router.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt() -> PlainTextResponse:
    return PlainTextResponse(ROBOTS.format(base=config.PUBLIC_URL),
                             headers={"Cache-Control": "public, max-age=86400"})


async def _explained_ids(limit: int) -> list[str]:
    async with aiosqlite.connect(config.DB_PATH) as conn:
        rows = await (await conn.execute(
            "SELECT arxiv_id FROM paper_explanations GROUP BY arxiv_id ORDER BY MAX(id) DESC LIMIT ?",
            (limit,))).fetchall()
    return [r[0] for r in rows]


async def paths() -> list[str]:
    """Every path in the sitemap, most important first, at most SITEMAP_MAX."""
    from app import collections_svc, hn_svc, trending_shelf
    out = list(STATIC_PATHS)
    papers: list[str] = []
    sources = (
        ("collections", lambda: [f"/collections/{c['slug']}" for c in collections_svc.load_all()]),
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
            out += got
        else:
            papers += got
    try:
        papers += await _explained_ids(SITEMAP_MAX)
    except Exception as e:
        errors.report("seo", "sitemap source explained papers failed", e)
    out += [f"/p/{a}" for a in dict.fromkeys(a for a in papers if a)]
    return list(dict.fromkeys(out))[:SITEMAP_MAX]


@router.get("/sitemap.xml")
async def sitemap_xml() -> Response:
    from app import trending_shelf
    now = time.time()
    body = _cache["body"]
    if not body or now - _cache["at"] > SITEMAP_TTL_S:
        urls = "".join(f"<url><loc>{escape(config.PUBLIC_URL + p)}</loc></url>" for p in await paths())
        body = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>\n')
        # Just after a restart the week's lists are still loading: serve this
        # one, but build again next time rather than keep it for hours.
        if trending_shelf.week_top(1):
            _cache.update(at=now, body=body)
    return Response(body, media_type="application/xml",
                    headers={"Cache-Control": "public, max-age=3600"})
