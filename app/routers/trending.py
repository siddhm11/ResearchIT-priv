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

from datetime import date as Date, datetime, timezone

from app import card_data, db, errors, features, hn_svc, qdrant_svc, runtime_stats, trending_shelf, trending_topics
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
    await card_data.add_figures(papers, request)
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
PAGE_TOP = 10               # the lead, then three full rows of the grid
SPARK_W, SPARK_H = 120, 32


def _spark(rolling: list[float], top: float) -> str:
    """SVG polyline points for a topic's weekly share, on one scale for the page."""
    n = max(len(rolling) - 1, 1)
    return " ".join(f"{i * SPARK_W / n:.1f},{SPARK_H - 2 - v / top * (SPARK_H - 4):.1f}"
                    for i, v in enumerate(rolling))


def _topic_view(t: dict, top: float) -> dict:
    return {"id": t["id"], "name": t["name"], "also": t["also"], "papers": t["papers"],
            "from": t["from_share"], "to": t["to_share"], "spark": _spark(t["rolling"], top),
            # Copies: these dicts live in the shared topic state, and figures are per request.
            "top": [dict(p) for p in t["top"]], "label": trending_topics.label(t)}


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
    shown |= {p["arxiv_id"] for p in top}
    hn_lately = await _hn_lately(user_id, decided | shown) if features.visible(hn_svc.SWITCH_FRESH, request) else []

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
    exposures += _instrument(hn_lately, user_id, HN_LATELY_POLICY, HN_FRESH_SOURCE, query_id, start=position)
    if user_id and exposures:
        try:
            await db.record_exposures(exposures)
        except Exception as e:  # pragma: no cover - analytics must not cost the page
            errors.report("trending", "page exposure write failed", e)
    moving = [_topic_view(t, scale) for t in rising + cooling]
    await card_data.add_figures([p for _n, papers, _p in sections for p in papers]
                                + [p for t in moving for p in t["top"]], request)
    runtime_stats.record("trending_page", {"request_ms": int((time.perf_counter() - t0) * 1000)})
    return templates.TemplateResponse(request, "trending.html", {
        "for_you": for_you, "areas": areas, "top": top, "query_id": query_id, "source": PAGE_SOURCE,
        "rising": moving[:len(rising)], "cooling": moving[len(rising):],
        "steady": [_topic_view(t, scale) for t in steady],
        "chips": _chips(rising, cooling, areas), "window": trending_shelf.window(),
        "basis": trending_topics.coverage(), "since": _day_label(trending_topics.coverage().get("first_week")),
        "spark_w": SPARK_W, "spark_h": SPARK_H, "hn_lately": hn_lately, "hn_days": hn_svc.FRESH_DAYS,
        "lasting_visible": features.visible(hn_svc.SWITCH_LASTING, request)})


# ── "On Hacker News lately" on /trending ─────────────────────────────────────
#
# Papers HN discussed in the last hn_svc.FRESH_DAYS days: linked by at least
# two people, or submitted as a story that reached 20 points. A second source
# beside Hugging Face's lists, and the only one for most fields outside ML:
# of 145 papers HN discussed between August and November 2026, 33 were on
# Hugging Face's lists (doc 06, 2026-11-03). The reader's closest come first,
# then the rest by how many people linked them.

HN_LATELY = 6
HN_FRESH_SOURCE = "hn_fresh"
HN_LATELY_POLICY = "hn_fresh_page_v1"


async def _hn_lately(user_id: str | None, exclude: set[str]) -> list[dict]:
    fresh = [dict(c) for c in hn_svc.fresh_cards() if c["arxiv_id"] not in exclude]
    mine = await _reader_rank(user_id, fresh, exclude, HN_LATELY)
    picked = {p["arxiv_id"] for p in mine}
    return (mine + [c for c in fresh if c["arxiv_id"] not in picked])[:HN_LATELY]


# ── /trending/lasting: the papers Hacker News keeps coming back to ───────────
#
# app/hn_svc.py counts every HN link to arXiv since 2007; a paper is "lasting"
# when at least 3 people linked it in at least 3 separate months, a month
# counting only when someone new linked it. Two views of one list, never both
# on a page (a paper is on the page once): "Closest to you", the lasting papers
# nearest the reader's interests and areas by trending_shelf.rank (people on HN
# as the popularity term), and "All lasting", the whole list in order, by area.
# A ranked surface, so §3.11 fields ride on every row and in the exposure log.

LASTING_SOURCE = "hn_lasting"
LASTING_FOR_YOU = 30
LASTING_PAGE = 25


async def _reader_rank(user_id: str | None, candidates: list[dict], exclude: set[str], size: int) -> list[dict]:
    """trending_shelf.rank over HN candidates for one reader ([] without interests or areas)."""
    if not user_id:
        return []
    interests, bar, areas = await trending_shelf.reader_groups(user_id)
    return trending_shelf.rank(candidates, hn_svc.vectors(), interests, bar, exclude, size,
                               areas=areas, votes_key="people")


def _instrument(papers: list[dict], user_id: str | None, policy: str, source: str,
                query_id: str, start: int = 0) -> list[dict]:
    """§3.11 fields on each paper; the exposure rows to record."""
    out = []
    for i, p in enumerate(papers):
        pos = start + i
        p.update(query_id=query_id, position=pos, propensity=1.0, policy_id=policy, ranker_version=policy,
                 candidate_source=source, cluster_id=(p.get("interest") or {}).get("cluster_idx"))
        out.append({"user_id": user_id, "paper_id": p["arxiv_id"], "query_id": query_id, "position": pos,
                    "tier": None, "candidate_source": source, "cluster_id": p["cluster_id"],
                    "propensity": 1.0, "policy_id": policy})
    return out


@router.get("/trending/lasting", response_class=HTMLResponse)
async def lasting_page(
    request: Request,
    view: str = "",
    sort: str = "default",
    year: str = "",
    field: str = "",
    page: int = 1,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    if not features.visible(hn_svc.SWITCH_LASTING, request):
        return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
    t0 = time.perf_counter()
    state = await us.ensure_loaded(user_id) if user_id else None
    decided = {r["paper_id"] for r in await db.get_current_feedback(user_id)} if user_id else set()
    saved = frozenset(state.positives) if state else frozenset()
    this_year = datetime.now(timezone.utc).year

    # The controls, as DAIR.AI's Hacker Papers offers them; anything unknown
    # falls back to the default rather than erroring. "all" is the old name of
    # the whole list.
    views = dict(hn_svc.VIEWS)
    view = "lasting" if view == "all" else view
    sort = sort if sort in dict(hn_svc.SORTS) else "default"
    buckets = hn_svc.year_buckets(this_year)
    year = year if year in {k for k, *_ in buckets} else ""
    field = field if field in dict(hn_svc.FIELDS) else ""

    cards = hn_svc.lasting_cards()
    # Closest to you is the default whenever the reader has interests or areas
    # that some lasting papers fit. The filters narrow what it ranks.
    pool = hn_svc.browse(cards, year=year, field=field, this_year=this_year)
    closest = await _reader_rank(user_id, pool, decided, LASTING_FOR_YOU)
    if view not in views and view != "you":
        view = "you" if closest else "lasting"
    if view == "you" and not closest:
        view = "lasting"

    pages, total = 1, 0
    if view == "you":
        shown = closest
        total = len(closest)
    else:
        everything = hn_svc.browse(cards, view, sort, year, field, saved, this_year)
        total = len(everything)
        pages = max(1, -(-total // LASTING_PAGE))
        page = min(max(page, 1), pages)
        shown = everything[(page - 1) * LASTING_PAGE: page * LASTING_PAGE]
    for p in shown:
        p["saved"] = p["arxiv_id"] in saved
    by_field = {k: len(hn_svc.browse(cards, view if view != "you" else "lasting", "default", year, k, saved, this_year))
                for k, _ in hn_svc.FIELDS}

    query_id = str(uuid.uuid4())
    policy = {"you": "hn_lasting_for_you_v1", "lasting": "hn_lasting_list_v1"}.get(view, f"hn_lasting_{view}_v1")
    exposures = _instrument(shown, user_id, policy, LASTING_SOURCE, query_id,
                            start=0 if view == "you" else (page - 1) * LASTING_PAGE)
    if user_id and exposures:
        try:
            await db.record_exposures(exposures)
        except Exception as e:  # pragma: no cover - analytics must not cost the page
            errors.report("trending", "lasting exposure write failed", e)

    current = {"view": view, "sort": sort, "year": year, "field": field}

    def link(**change) -> str:
        """This page with some controls changed; defaults are left out, and any
        change goes back to page 1."""
        q = {**current, **change}
        parts = [f"{k}={v}" for k, v in q.items()
                 if v and not (k == "sort" and v == "default") and not (k == "view" and v == ("you" if closest else "lasting"))]
        if change.get("page", 1) > 1:
            parts.append(f"page={change['page']}")
        return "/trending/lasting" + ("?" + "&".join(parts) if parts else "")

    stats = hn_svc.stats()
    runtime_stats.record("hn_lasting", {"request_ms": int((time.perf_counter() - t0) * 1000)})
    return templates.TemplateResponse(request, "lasting.html", {
        **current, "papers": shown, "page": page, "pages": pages, "count": total,
        "views": hn_svc.VIEWS, "sorts": hn_svc.SORTS, "years": buckets, "fields": hn_svc.FIELDS,
        "by_field": by_field, "has_for_you": bool(closest), "signed": bool(user_id), "link": link,
        "gem_max": hn_svc.GEM_MAX_CITATIONS, "gem_age": hn_svc.GEM_MIN_AGE,
        # One interest's name on every row says nothing; it shows when rows differ.
        "show_interest": len({(p.get("interest") or {}).get("label") for p in shown}) > 1,
        "total": stats["lasting"], "total_all": stats["lasting_all"], "mentions": stats["mentions"],
        "loaded": bool(stats["loaded_at"]), "query_id": query_id, "source": LASTING_SOURCE,
        "trending_page": features.visible(trending_topics.SWITCH, request)})


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
    await card_data.add_figures(week, request)
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
