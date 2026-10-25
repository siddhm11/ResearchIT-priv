"""
GET /api/trending-shelf: the "Trending in your interests" shelf (app/trending_shelf.py).
GET /trending: the Trending tab. GET /trending/topics/<id>: one topic.

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

import time
import uuid

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse

from datetime import date as Date

from app import db, errors, features, qdrant_svc, runtime_stats, trending_shelf, trending_topics
from app import user_state as us
from app.config import COOKIE_NAME
from app.routers import recommendations
from app.templates_env import templates

router = APIRouter()


@router.get("/api/trending-shelf", response_class=HTMLResponse)
async def get_trending_shelf(
    request: Request,
    after: str = "",
    first: int = 0,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    if not user_id or not features.visible(trending_shelf.SWITCH, request):
        return HTMLResponse("")
    t0 = time.perf_counter()
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
    # What the shelf may call itself: close to your saves, in your areas only,
    # or nothing fitted and these are the week's top.
    kind = ("interests" if any(p.get("fit") is not None for p in papers)
            else "areas" if any(p.get("interest") for p in papers) else "week")
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
    runtime_stats.record("trending_shelf", {"request_ms": int((time.perf_counter() - t0) * 1000)})
    return templates.TemplateResponse(request, "partials/trending_shelf.html",
                                      {"papers": papers, "query_id": query_id, "kind": kind,
                                       "first": bool(first), "source": trending_shelf.SOURCE,
                                       "trending_page": page})


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
            "top": t["top"], "label": trending_topics.label(t)}


def _chips(rising: list[dict], cooling: list[dict], areas: list[dict]) -> list[dict]:
    """The strip under the page title: topics that moved, then the reader's areas."""
    return ([{"text": t["name"], "href": f"/trending/topics/{t['id']}", "kind": "up"} for t in rising]
            + [{"text": t["name"], "href": f"/trending/topics/{t['id']}", "kind": "down"} for t in cooling]
            + [{"text": f"{a['label']} · {a['count']}", "href": f"#tp-area-{a['area']}", "kind": "area"}
               for a in areas])


@router.get("/trending", response_class=HTMLResponse)
async def trending_page(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    if not features.visible(trending_topics.SWITCH, request):
        return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
    t0 = time.perf_counter()
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

    rising, cooling, steady = trending_topics.rising(), trending_topics.cooling(), trending_topics.steady()
    scale = max([v for t in rising + cooling + steady for v in t["rolling"]] or [0.01])
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
    runtime_stats.record("trending_page", {"request_ms": int((time.perf_counter() - t0) * 1000)})
    return templates.TemplateResponse(request, "trending.html", {
        "for_you": for_you, "areas": areas, "top": top, "query_id": query_id, "source": PAGE_SOURCE,
        "rising": [_topic_view(t, scale) for t in rising],
        "cooling": [_topic_view(t, scale) for t in cooling],
        "steady": [_topic_view(t, scale) for t in steady],
        "chips": _chips(rising, cooling, areas), "window": trending_shelf.window(),
        "basis": trending_topics.coverage(), "spark_w": SPARK_W, "spark_h": SPARK_H})


# ── /trending/topics/<id>: one topic ─────────────────────────────────────────
#
# What a topic is (its phrases, size and 26-week line), its papers on this
# week's lists, its most upvoted over the 26 weeks, recent arXiv papers closest
# to it beyond Hugging Face's lists, and its nearest topics. The "more like
# this" list is the same retrieval a followed topic would use in the feed, so
# this page is where that is seen and judged first (doc 06, 2026-10-25).

TOPIC_SOURCE = "trending_topic"
TOPIC_MORE = 10
TOPIC_MORE_TTL_S = 6 * 3600
BIG_W, BIG_H = 640, 120
_more_cache: dict[tuple[int, str], tuple[float, list[dict]]] = {}


def _day_label(iso: str | None) -> str:
    if not iso:
        return ""
    d = Date.fromisoformat(iso)
    return f"{d.day} {d:%b}"


def _big_spark(rolling: list[float]) -> str:
    top = max(rolling or [0.01]) or 0.01
    n = max(len(rolling) - 1, 1)
    return " ".join(f"{i * BIG_W / n:.1f},{BIG_H - 4 - v / top * (BIG_H - 12):.1f}"
                    for i, v in enumerate(rolling))


async def _more_like(topic: dict) -> list[dict]:
    """Recent arXiv papers closest to the topic's middle paper that are not
    among its papers on the lists. One search a topic, kept for 6 hours."""
    key = (topic["id"], trending_topics.coverage()["computed_at"] or "")
    hit = _more_cache.get(key)
    if hit and time.time() - hit[0] < TOPIC_MORE_TTL_S:
        return hit[1]
    medoid = topic.get("_medoid")
    if medoid is None:
        return []
    listed = {a for a, t in trending_topics._state["topic_of"].items() if t == topic["id"]}
    hits = await qdrant_svc.search_recent([float(x) for x in medoid], limit=TOPIC_MORE * 3,
                                          exclude_ids=listed)
    _more_cache[key] = (time.time(), hits)
    return hits


@router.get("/trending/topics/{topic_id}", response_class=HTMLResponse)
async def topic_page(
    topic_id: int,
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    topic = trending_topics.get(topic_id) if features.visible(trending_topics.SWITCH, request) else None
    if topic is None:
        return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
    t0 = time.perf_counter()
    state = await us.ensure_loaded(user_id) if user_id else None
    decided = {r["paper_id"] for r in await db.get_current_feedback(user_id)} if user_id else set()
    saved = set(state.positive_list) if state else set()

    # This week: the shelf's papers in this topic, by the daily grouping or,
    # for papers listed since, by their nearest topic.
    week = [p for p in trending_shelf._state["papers"].values()
            if p["arxiv_id"] not in decided
            and trending_topics.topic_of(p["arxiv_id"], trending_shelf._state["vectors"].get(p["arxiv_id"])) == topic_id]
    week = [trending_shelf._for_card(dict(p)) for p in
            sorted(week, key=lambda p: (-p["upvotes"], p["arxiv_id"]))[:trending_shelf.SHELF_SIZE]]
    # Copies: these dicts live in the shared topic state, and attribution is per request.
    best = [dict(b) for b in topic.get("best") or topic["top"] if b["arxiv_id"] not in {p["arxiv_id"] for p in week}]
    for b in best:
        day = Date.fromisoformat(b["listed_on"]) if b.get("listed_on") else None
        b["listed_label"] = f"{day.day} {day:%b}" if day else ""

    t_more = time.perf_counter()
    try:
        hits = await _more_like(topic)
    except Exception as e:  # the topic's own lists do not depend on the vector store
        errors.report("trending", "topic search failed", e)
        hits = []
    more_ms = int((time.perf_counter() - t_more) * 1000)
    shown = {p["arxiv_id"] for p in week} | {b["arxiv_id"] for b in best} | decided | saved
    more_ids = [h["arxiv_id"] for h in hits if h["arxiv_id"] not in shown][:TOPIC_MORE]
    from app.routers.collections import _hydrate
    meta = await _hydrate(more_ids, state)
    more = [meta[a] for a in more_ids if a in meta]

    query_id = str(uuid.uuid4())
    sections = (("week", week, "trending_topic_week_v1", trending_shelf.CANDIDATE_SOURCE),
                ("best", best, "trending_topic_best_v1", trending_shelf.CANDIDATE_SOURCE),
                ("more", more, "trending_topic_more_v1", "topic_vector"))
    exposures, position = [], 0
    for _name, papers, policy, source in sections:
        for p in papers:
            p.update(query_id=query_id, position=position, propensity=1.0, policy_id=policy,
                     ranker_version=policy, candidate_source=source, cluster_id=None)
            exposures.append({"user_id": user_id, "paper_id": p["arxiv_id"], "query_id": query_id,
                              "position": position, "tier": None, "candidate_source": source,
                              "cluster_id": None, "propensity": 1.0, "policy_id": policy})
            position += 1
    if user_id and exposures:
        try:
            await db.record_exposures(exposures)
        except Exception as e:  # pragma: no cover - analytics must not cost the page
            errors.report("trending", "topic exposure write failed", e)
    basis = trending_topics.coverage()
    runtime_stats.record("trending_topic", {"request_ms": int((time.perf_counter() - t0) * 1000),
                                            "more_ms": more_ms})
    return templates.TemplateResponse(request, "topic.html", {
        "t": topic, "label": trending_topics.label(topic), "week": week, "best": best, "more": more,
        "related": trending_topics.related(topic), "query_id": query_id, "source": TOPIC_SOURCE,
        "spark": _big_spark(topic["rolling"]), "big_w": BIG_W, "big_h": BIG_H,
        "since": _day_label(basis.get("first_week")),
        "basis": basis})


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
