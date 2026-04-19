"""
Saved papers router.

GET /saved
  – Shows all papers the user has currently saved (positive_list)
  – Metadata fetched via arXiv API + SQLite cache
"""
import uuid
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse
from app import arxiv_svc, user_state as us
from app.config import COOKIE_NAME
from app.templates_env import templates

router = APIRouter()


@router.get("/saved", response_class=HTMLResponse)
async def saved_papers(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())
    state = await us.ensure_loaded(user_id)

    saved_ids = state.positive_list  # most-recent first, mutual-exclusion already applied

    papers = []
    if saved_ids:
        meta = await arxiv_svc.fetch_metadata_batch(saved_ids)
        papers = [
            {**meta[aid], "saved": True, "dismissed": False}
            for aid in saved_ids
            if aid in meta
        ]

    resp = templates.TemplateResponse(
        request,
        "saved.html",
        {
            "papers": papers,
            "count": len(papers),
        },
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp
