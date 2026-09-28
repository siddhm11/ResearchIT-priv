"""
Saved papers router.

GET /saved
  – Shows all papers the user has currently saved (durable feedback history)
  – Metadata fetched via Turso DB (Phase 3.5), arXiv API fallback
"""
import uuid
from typing import Literal
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse
from app import arxiv_svc, db, turso_svc
from app.config import COOKIE_NAME
from app.templates_env import templates

router = APIRouter()


@router.get("/history", response_class=HTMLResponse)
async def reading_history(
    request: Request,
    view: Literal["opened", "discovered"] = "opened",
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """Help a reader return to a paper without claiming it was read."""
    user_id = user_id or str(uuid.uuid4())
    rows = await db.get_recent_papers(user_id, opened=view == "opened", limit=50)
    ids = [r["paper_id"] for r in rows]
    meta = await db.get_cached_metadata_batch(ids) if ids else {}
    missing = [aid for aid in ids if aid not in meta]
    if missing:
        try:
            meta.update(await turso_svc.fetch_metadata_batch(missing))
        except Exception as e:
            print(f"[history] metadata unavailable: {e}")
    feedback = await db.get_current_feedback(user_id)
    saved_ids = {r["paper_id"] for r in feedback if r["event_type"] == "save"}
    dismissed_ids = {r["paper_id"] for r in feedback if r["event_type"] == "not_interested"}
    papers = [{
        **meta.get(row["paper_id"], {"title": "arXiv:" + row["paper_id"],
                                    "abstract": "Metadata is temporarily unavailable.",
                                    "authors": "[]", "category": "", "published": ""}),
        "arxiv_id": row["paper_id"], "last_seen": row["last_seen"],
        "saved": row["paper_id"] in saved_ids,
        "dismissed": row["paper_id"] in dismissed_ids,
    } for row in rows]
    resp = templates.TemplateResponse(request, "history.html", {"papers": papers, "view": view})
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.get("/saved", response_class=HTMLResponse)
async def saved_papers(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())

    saved_ids = [row["paper_id"] for row in await db.get_current_feedback(user_id)
                 if row["event_type"] == "save"]

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
