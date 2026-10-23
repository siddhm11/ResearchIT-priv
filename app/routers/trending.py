"""
GET /api/trending-shelf: the "Trending in your interests" shelf (app/trending_shelf.py).

Requested by the feed's first page once its cards are in, so the feed never
waits on it. Answers with an empty body whenever there is nothing to show: the
switch is off for this request, the reader has no interests yet, or fewer than
MIN_SHOWN papers clear their bar. htmx swaps the placeholder for that empty
body, so the shelf leaves no trace.

A ranked surface like the feed, so it follows CLAUDE.md §3.11: one query_id per
shelf, propensity 1.0 (the ranking is deterministic) and its own policy_id, on
every card, in the exposure log and through the Save round trip.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse

from app import db, errors, features, trending_shelf
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
    # The feed's own exclusions: decided papers, and anything already shown,
    # which includes the page this shelf sits on (its impressions are written
    # before the page is returned).
    exclude = us.all_seen(user_id) | {r["paper_id"] for r in await db.get_current_feedback(user_id)}
    try:
        exclude |= await db.get_impressed_ids(user_id)
    except Exception as e:  # pragma: no cover - defensive
        errors.report("trending", "impression lookup failed", e)
    papers = await trending_shelf.shelf_for(user_id, exclude, labels=_feed_labels(after, user_id))
    if not papers:
        return HTMLResponse("")

    query_id = str(uuid.uuid4())
    for position, p in enumerate(papers):
        p.update(query_id=query_id, position=position, propensity=1.0,
                 policy_id=trending_shelf.POLICY_ID, ranker_version=trending_shelf.POLICY_ID,
                 candidate_source=trending_shelf.CANDIDATE_SOURCE,
                 cluster_id=p["interest"]["cluster_idx"])
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
                                      {"papers": papers, "query_id": query_id,
                                       "source": trending_shelf.SOURCE})


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
