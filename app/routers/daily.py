"""GET /day and /day/<yyyy-mm-dd>: one day's Hugging Face Daily Papers list
(app/daily_pages.py). A ranked public page: a person's first visit gets a
reader id so its exposures are recorded; a crawler gets none (§3.11)."""
from __future__ import annotations

import re
import uuid
from datetime import date

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import bots, card_data, daily_pages, db, errors, features, trending_topics
from app.config import COOKIE_NAME
from app.routers import share
from app.routers.trending import _with_reader
from app.templates_env import templates

router = APIRouter()
POLICY = "daily_page_v1"
SOURCE = "hf_daily"
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def long_day(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d:%A} {d.day} {d:%B %Y}"


@router.get("/day")
async def latest_day(request: Request):
    days = daily_pages.available()
    if not days or not features.visible(trending_topics.SWITCH, request):
        return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
    return RedirectResponse(f"/day/{days[0]}", status_code=302)


@router.get("/day/{day}", response_class=HTMLResponse)
async def day_page(day: str, request: Request,
                   user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    papers = daily_pages.papers(day) if _DAY.match(day) else []
    if not papers or not features.visible(trending_topics.SWITCH, request):
        return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
    older, newer = daily_pages.neighbours(day)
    reader = None if bots.is_bot(request) else (user_id or str(uuid.uuid4()))
    query_id = str(uuid.uuid4())
    for i, p in enumerate(papers):
        p.update(position=i, query_id=query_id)
    if reader:
        try:
            await db.record_exposures([
                {"user_id": reader, "paper_id": p["arxiv_id"], "query_id": query_id, "position": p["position"],
                 "tier": None, "candidate_source": SOURCE, "cluster_id": None, "propensity": 1.0,
                 "policy_id": POLICY} for p in papers])
        except Exception as e:  # pragma: no cover - analytics must not cost the page
            errors.report("daily", "exposure write failed", e)
    await card_data.add_figures(papers[:3], request)
    label = long_day(day)
    top = papers[0]
    description = (f"The {len(papers)} papers on Hugging Face Daily Papers for {label}, most upvoted first. "
                   f"Top: “{top['title']}” with {top['upvotes']} upvotes.")
    return _with_reader(user_id, reader, templates.TemplateResponse(request, "day.html", {
        "day": day, "label": label, "papers": papers, "older": older, "newer": newer,
        "query_id": query_id, "policy": POLICY, "source": SOURCE,
        "og_title": f"Hugging Face Daily Papers, {label}", "og_description": description,
        "og_image": share.card_url("day", day)}))
