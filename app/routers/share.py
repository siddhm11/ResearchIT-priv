"""GET /og/...: the share card a page names as its og:image (app/share_cards.py).

    /og/p/<arxiv_id>.png       a paper
    /og/topic/<slug>.png       a trending topic
    /og/list/<slug>.png        a reading list
    /og/lasting.png            Lasting on Hacker News
    /og/trending.png           the Trending page

Asked for by link-preview bots (Slackbot, Twitterbot, facebookexternalhit,
Discordbot, WhatsApp) the moment someone pastes a link, so: data already held
only, no model call, no arXiv fetch, no cookie, no reader row (CLAUDE.md §3.11).
A paper's figure is used when its thumbnail is already in the figure cache.
Any failure answers with the site's own picture rather than a broken image.
"""
from __future__ import annotations

import asyncio
import hashlib

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

from app import config, hn_svc, paper_source_svc, share_cards, trending_shelf, trending_topics, turso_svc
from app import collections_svc
from app.routers import figures
from app.templates_env import _tojson_parse

router = APIRouter()

# Bumped when the card design changes, so platforms that cache by URL refetch.
CARD_VERSION = "1"
_sem: asyncio.Semaphore | None = None


def card_url(kind: str, key: str = "") -> str:
    """The absolute URL a page puts in og:image."""
    path = f"/og/{kind}/{key}.png" if key else f"/og/{kind}.png"
    return f"{config.PUBLIC_URL}{path}?v={CARD_VERSION}"


def _fallback() -> Response:
    from pathlib import Path
    return FileResponse(Path(__file__).resolve().parents[1] / "static" / "og.png", media_type="image/png",
                        headers={"Cache-Control": "public, max-age=600"})


def _png(data: bytes, long: bool) -> Response:
    # A day when the card is complete; an hour when a figure may still arrive.
    return Response(data, media_type="image/png",
                    headers={"Cache-Control": f"public, max-age={86400 if long else 3600}"})


async def _draw(key: str, fn, **kw) -> bytes:
    """One card at a time per process: drawing is ~80 ms of CPU, and a burst of
    preview bots must not crowd out page requests."""
    global _sem
    hit = share_cards.cache.get(key)
    if hit is not None:
        return hit
    if _sem is None:
        _sem = asyncio.Semaphore(1)
    async with _sem:
        hit = share_cards.cache.get(key)
        if hit is None:
            hit = await asyncio.to_thread(fn, **kw)
            share_cards.cache.put(key, hit)
    return hit


def authors_of(paper: dict) -> list[str]:
    raw = paper.get("authors") or ""
    authors = _tojson_parse(raw or "[]") if isinstance(raw, str) else raw
    if not authors and isinstance(raw, str) and raw:
        authors = [a.strip() for a in raw.split(";" if ";" in raw else ",") if a.strip()]
    return [str(a) for a in authors or []]


def first_sentence(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    for end in (". ", "? ", "! "):
        i = text.find(end)
        if 40 <= i <= limit:
            return text[:i + 1]
    return text[:limit].rsplit(" ", 1)[0] + "…" if len(text) > limit else text


async def _saved_figure(arxiv_id: str) -> bytes | None:
    """The first raster figure's thumbnail, if it is already in the cache."""
    try:
        source = await paper_source_svc.get(arxiv_id)
    except Exception:          # a card without a figure is still a card
        return None
    if source is None:
        return None
    for n in range(min(len(source.figures), 12)):
        url = figures.figure_url(source, n)
        if not url or not url.endswith(".webp"):
            continue
        path = figures._cache_path(arxiv_id, url.rsplit("/", 1)[1])
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError:
            return None        # the lead figure isn't made yet: no figure rather than a later one
    return None


@router.get("/og/p/{arxiv_id:path}.png", include_in_schema=False)
async def paper_card(arxiv_id: str) -> Response:
    if not paper_source_svc.ID_PATTERN.match(arxiv_id):
        return _fallback()
    try:
        paper = (await turso_svc.fetch_metadata_batch([arxiv_id])).get(arxiv_id)
        if not paper:
            return _fallback()
        from app.routers.paper import cached_explanation
        hook = first_sentence(await cached_explanation(paper) or "") or first_sentence(paper.get("abstract") or "")
        fig = await _saved_figure(arxiv_id)
        year = str(paper.get("year") or (paper.get("published") or "")[:4] or "")
        key = "p:" + hashlib.sha1(f"{arxiv_id}|{hook}|{bool(fig)}|{paper.get('citation_count')}".encode()).hexdigest()
        data = await _draw(key, share_cards.paper_card, title=paper.get("title") or f"arXiv:{arxiv_id}",
                           authors=authors_of(paper), category=paper.get("category") or "", year=year,
                           cited=paper.get("citation_count") or None, arxiv_id=arxiv_id, hook=hook, figure=fig)
        return _png(data, long=bool(fig))
    except Exception as e:
        print(f"[share] paper card {arxiv_id} failed: {type(e).__name__}: {e}")
        return _fallback()


@router.get("/og/topic/{slug}.png", include_in_schema=False)
async def topic_card(slug: str) -> Response:
    topic = trending_topics.by_slug(slug)
    if topic is None:
        return _fallback()
    try:
        label = {"rising": "Rising topic", "cooling": "Cooling topic"}.get(trending_topics.label(topic), "Steady topic")
        since = trending_topics.coverage().get("first_week") or ""
        since = _day(since)
        papers = [p["title"] for p in (topic.get("best") or topic.get("top") or [])]
        key = "t:" + hashlib.sha1(repr((slug, label, topic["from_share"], topic["to_share"], papers[:3])).encode()).hexdigest()
        data = await _draw(key, share_cards.topic_card, name=topic["name"], label=label,
                           from_share=topic["from_share"], to_share=topic["to_share"],
                           rolling=list(topic.get("rolling") or []), papers=papers, since=since)
        return _png(data, long=True)
    except Exception as e:
        print(f"[share] topic card {slug} failed: {type(e).__name__}: {e}")
        return _fallback()


def _day(iso: str) -> str:
    """'2026-06-01' as '1 Jun'."""
    from datetime import date
    try:
        d = date.fromisoformat(str(iso)[:10])
    except ValueError:
        return ""
    return f"{d.day} {d:%b}"


@router.get("/og/list/{slug}.png", include_in_schema=False)
async def list_card(slug: str) -> Response:
    col = collections_svc.get(slug)
    if col is None:
        return _fallback()
    try:
        ids = [a["id"] for a in col["anchors"]]
        meta = await turso_svc.fetch_metadata_batch(ids[:6])
        titles = [meta[i]["title"] for i in ids[:6] if i in meta and meta[i].get("title")]
        key = "l:" + hashlib.sha1(repr((slug, col["title"], titles[:3])).encode()).hexdigest()
        data = await _draw(key, share_cards.list_card, title=col["title"],
                           kicker=f"Reading list · {len(ids)} papers",
                           blurb=first_sentence(col.get("blurb") or ""), items=titles)
        return _png(data, long=True)
    except Exception as e:
        print(f"[share] list card {slug} failed: {type(e).__name__}: {e}")
        return _fallback()


@router.get("/og/lasting.png", include_in_schema=False)
async def lasting_card() -> Response:
    try:
        stats = hn_svc.stats()
        cards = hn_svc.lasting_cards()
        if not cards:
            return _fallback()
        total = stats.get("lasting_all") or stats.get("lasting") or len(cards)
        titles = [c.get("title") or c["arxiv_id"] for c in cards[:3]]
        key = "h:" + hashlib.sha1(repr((total, titles)).encode()).hexdigest()
        data = await _draw(key, share_cards.list_card, title="The papers Hacker News keeps coming back to",
                           kicker=f"Lasting on Hacker News · {total:,} papers",
                           blurb="Linked by at least 3 different people in at least 3 separate months, since 2007.",
                           items=titles)
        return _png(data, long=True)
    except Exception as e:
        print(f"[share] lasting card failed: {type(e).__name__}: {e}")
        return _fallback()


@router.get("/og/trending.png", include_in_schema=False)
async def trending_card() -> Response:
    try:
        rising = trending_topics.rising()
        if rising:
            t = rising[0]
            papers = [p["title"] for p in (t.get("best") or t.get("top") or [])]
            key = "r:" + hashlib.sha1(repr((t["name"], t["from_share"], t["to_share"], papers[:3])).encode()).hexdigest()
            data = await _draw(key, share_cards.topic_card, name=t["name"], label="Rising fastest",
                               from_share=t["from_share"], to_share=t["to_share"],
                               rolling=list(t.get("rolling") or []), papers=papers,
                               since=_day(trending_topics.coverage().get("first_week") or ""))
            return _png(data, long=False)
        week = trending_shelf.week_top(3)
        if not week:
            return _fallback()
        titles = [p.get("title") or p["arxiv_id"] for p in week]
        key = "w:" + hashlib.sha1(repr(titles).encode()).hexdigest()
        data = await _draw(key, share_cards.list_card, title="Most upvoted AI papers this week",
                           kicker="Hugging Face Daily Papers", blurb="", items=titles)
        return _png(data, long=False)
    except Exception as e:
        print(f"[share] trending card failed: {type(e).__name__}: {e}")
        return _fallback()
