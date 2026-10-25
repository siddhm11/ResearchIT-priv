"""
GET /api/trending-shelf: the "Trending in your interests" shelf (app/trending_shelf.py).

Requested by the feed's first page once its cards are in, so the feed never
waits on it. The shelf is there whenever the week has papers: papers close to
the reader's saves or listed in their chosen areas, then, for a reader nothing
fits, the week's most upvoted under "Trending this week" (doc 06, 2026-10-25).
It answers with an empty body only when the switch is off for this request or
no list could be fetched; htmx then swaps the placeholder for nothing.

A ranked surface like the feed, so it follows CLAUDE.md §3.11: one query_id per
shelf, propensity 1.0 (the ranking is deterministic) and its own policy_id, on
every card, in the exposure log and through the Save round trip.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse

from datetime import date as Date

from app import db, errors, features, trending_shelf, trending_topics
from app import user_state as us
from app.config import COOKIE_NAME
from app.routers import recommendations
from app.templates_env import templates

router = APIRouter()


@router.get("/api/trending-shelf", response_class=HTMLResponse)
async def get_trending_shelf(
    request: Request,
    after: str = "",
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    if not user_id or not features.visible(trending_shelf.SWITCH, request):
        return HTMLResponse("")
    await us.ensure_loaded(user_id)
    # Never: papers the reader decided on, and the feed page this shelf sits on.
    exclude = us.all_seen(user_id) | {r["paper_id"] for r in await db.get_current_feedback(user_id)}
    entry = recommendations._cache_get(after) if after else None
    if entry and entry.get("user_id") == user_id:
        exclude |= set(entry.get("emitted") or ())
    # Shown before: after everything new, but still on the shelf. Leaving them
    # out emptied it for a reader who had seen the few that fit (25 Oct: 7 of
    # 312 fitted two saves, the first shelf showed 6, the next had 1 and hid).
    try:
        seen = await db.get_impressed_ids(user_id)
    except Exception as e:  # pragma: no cover - defensive
        errors.report("trending", "impression lookup failed", e)
        seen = set()
    papers = await trending_shelf.shelf_for(user_id, exclude, labels=_feed_labels(after, user_id),
                                            seen=seen, fallback=True)
    if not papers:
        return HTMLResponse("")
    personal = any(p.get("interest") for p in papers)
    page = features.visible(trending_topics.SWITCH, request)
    if page:
        _add_topics(papers)

    query_id = str(uuid.uuid4())
    for position, p in enumerate(papers):
        p.update(query_id=query_id, position=position, propensity=1.0,
                 policy_id=trending_shelf.POLICY_ID, ranker_version=trending_shelf.POLICY_ID,
                 candidate_source=trending_shelf.CANDIDATE_SOURCE,
                 cluster_id=(p.get("interest") or {}).get("cluster_idx"))
    ids = [p["arxiv_id"] for p in papers]
    try:
        # Shown is shown: the feed won't repeat these, and the next shelf moves on.
        await db.record_impressions(user_id, ids)
        await db.record_exposures([
            {"user_id": user_id, "paper_id": p["arxiv_id"], "query_id": query_id,
             "position": p["position"], "tier": None, "candidate_source": p["candidate_source"],
             "cluster_id": p["cluster_id"], "propensity": 1.0, "policy_id": p["policy_id"]}
            for p in papers])
    except Exception as e:  # pragma: no cover - analytics must not cost the shelf
        errors.report("trending", "shelf exposure write failed", e)
    return templates.TemplateResponse(request, "partials/trending_shelf.html",
                                      {"papers": papers, "query_id": query_id, "personal": personal,
                                       "source": trending_shelf.SOURCE, "trending_page": page})


def _add_topics(papers: list[dict]) -> None:
    """The rising topic each paper belongs to, if any (trending_topics.topic_line)."""
    for p in papers:
        line = trending_topics.topic_line(p["arxiv_id"])
        if line:
            line["since_label"] = f"{Date.fromisoformat(line['since']):%B}"
        p["topic"] = line


# ── /trending: the week's papers and the rising topics ───────────────────────

PAGE_SOURCE = "trending_page"
PAGE_FOR_YOU = 9
PAGE_PER_AREA = 3
PAGE_TOP = 9                # three full rows of the wide grid
SPARK_W, SPARK_H = 120, 32


def _spark(rolling: list[float], top: float) -> str:
    """SVG polyline points for a topic's weekly share, on one scale for the page."""
    n = max(len(rolling) - 1, 1)
    return " ".join(f"{i * SPARK_W / n:.1f},{SPARK_H - 2 - v / top * (SPARK_H - 4):.1f}"
                    for i, v in enumerate(rolling))


def _topic_view(t: dict, top: float) -> dict:
    return {"id": t["id"], "name": t["name"], "also": t["also"], "papers": t["papers"],
            "from": t["from_share"], "to": t["to_share"], "spark": _spark(t["rolling"], top),
            "top": t["top"]}


@router.get("/trending", response_class=HTMLResponse)
async def trending_page(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    if not features.visible(trending_topics.SWITCH, request):
        return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
    decided: set[str] = set()
    for_you: list[dict] = []
    areas: list[dict] = []
    if user_id:
        await us.ensure_loaded(user_id)
        decided = {r["paper_id"] for r in await db.get_current_feedback(user_id)}
        # Not the shelf's "shown is shown": this page lists the whole week, so
        # only papers the reader saved or turned down are left out.
        for_you = await trending_shelf.shelf_for(user_id, decided, size=PAGE_FOR_YOU)
        # Each area the reader picked: how many of the week's papers it has, and
        # its most upvoted ones not already above.
        areas = trending_shelf.by_area(await trending_shelf.reader_areas(user_id),
                                       exclude=decided | {p["arxiv_id"] for p in for_you},
                                       n=PAGE_PER_AREA)
    shown = {p["arxiv_id"] for p in for_you} | {p["arxiv_id"] for a in areas for p in a["papers"]}
    top = trending_shelf.week_top(PAGE_TOP, exclude=decided | shown)
    _add_topics(for_you + [p for a in areas for p in a["papers"]] + top)

    rising, cooling = trending_topics.rising(), trending_topics.cooling()
    scale = max([v for t in rising + cooling for v in t["rolling"]] or [0.01])
    query_id = str(uuid.uuid4())
    sections = (("for_you", for_you, "trending_page_interests_v1"),
                *((f"area_{a['area']}", a["papers"], "trending_page_areas_v1") for a in areas),
                ("top", top, "trending_page_top_v1"))
    exposures, position = [], 0
    for _name, papers, policy in sections:
        for p in papers:
            p.update(query_id=query_id, position=position, propensity=1.0, policy_id=policy,
                     ranker_version=policy, candidate_source=trending_shelf.CANDIDATE_SOURCE,
                     cluster_id=(p.get("interest") or {}).get("cluster_idx"))
            exposures.append({"user_id": user_id, "paper_id": p["arxiv_id"], "query_id": query_id,
                              "position": position, "tier": None, "candidate_source": p["candidate_source"],
                              "cluster_id": p["cluster_id"], "propensity": 1.0, "policy_id": policy})
            position += 1
    if user_id and exposures:
        try:
            await db.record_exposures(exposures)
        except Exception as e:  # pragma: no cover - analytics must not cost the page
            errors.report("trending", "page exposure write failed", e)
    return templates.TemplateResponse(request, "trending.html", {
        "for_you": for_you, "areas": areas, "top": top, "query_id": query_id, "source": PAGE_SOURCE,
        "rising": [_topic_view(t, scale) for t in rising],
        "cooling": [_topic_view(t, scale) for t in cooling],
        "basis": trending_topics.coverage(), "spark_w": SPARK_W, "spark_h": SPARK_H})


def _feed_labels(after: str, user_id: str) -> dict[int, tuple[str, int]]:
    """Interest names as the feed page below `after` shows them.

    The feed names its interests from each build's papers, so names can shift
    between builds. Taking them from the page the shelf sits on keeps the two
    in agreement on screen.
    """
    entry = recommendations._cache_get(after) if after else None
    if not entry or entry.get("user_id") != user_id:
        return {}
    out: dict[int, tuple[str, int]] = {}
    for tags in entry.get("tags", {}).values():
        cid, label = tags.get("cluster_id"), tags.get("cluster_label")
        if label and str(cid).lstrip("-").isdigit():
            out.setdefault(int(cid), (label, int(tags.get("cluster_tone", 0))))
    return out
