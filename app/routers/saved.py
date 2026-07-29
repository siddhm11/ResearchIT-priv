"""
Saved papers router.

GET /saved
  – Shows all papers the user has currently saved (positive_list)
  – Metadata fetched via Turso DB (Phase 3.5), arXiv API fallback
"""
import uuid
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse
from app import arxiv_svc, db, turso_svc, user_state as us
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
        # Phase 3.5: Turso primary, arXiv API fallback
        meta = await turso_svc.fetch_metadata_batch(saved_ids)
        missing = [aid for aid in saved_ids if aid not in meta]
        if missing:
            try:
                arxiv_meta = await arxiv_svc.fetch_metadata_batch(missing)
                meta.update(arxiv_meta)
            except Exception as e:
                print(f"[saved] arXiv fallback for {len(missing)} IDs failed: {e}")
        # Phase 4.3: Cache to SQLite so dismissal category JOINs work
        await db.cache_turso_metadata_batch(list(meta.values()))

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
