"""
Event router — logs user interactions and updates the hot cache.

POST /api/papers/{paper_id}/save
POST /api/papers/{paper_id}/not-interested
"""
import asyncio
import uuid
import numpy as np
from fastapi import APIRouter, Request, Cookie, Form
from fastapi.responses import HTMLResponse
from app import db, user_state as us, qdrant_svc
from app.config import COOKIE_NAME
from app.templates_env import templates
from app.recommend import profiles

router = APIRouter(prefix="/api/papers")


@router.post("/{paper_id}/save", response_class=HTMLResponse)
async def save_paper(
    paper_id: str,
    request: Request,
    source: str = Form(default="search"),
    position: int = Form(default=0),
    query_id: str = Form(default=""),
    ranker_version: str = Form(default=""),
    candidate_source: str = Form(default=""),
    cluster_id: str = Form(default=""),
    propensity: float = Form(default=0.0),
    policy_id: str = Form(default=""),
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())

    await db.log_interaction(
        user_id=user_id,
        paper_id=paper_id,
        event_type="save",
        source=source,
        position=position or None,
        query_id=query_id or None,
        ranker_version=ranker_version or None,
        candidate_source=candidate_source or None,
        cluster_id=int(cluster_id) if cluster_id else None,
        propensity=propensity if propensity > 0 else None,
        policy_id=policy_id or None,
    )

    us.record_positive(user_id, paper_id)
    # The point-id warm that used to live here is gone: get_paper_vectors now
    # resolves arxiv_id -> vector in one filtered call, so the arxiv_id -> point_id
    # cache it populated is no longer on any read path. _update_profile_on_save
    # warms the in-process vector cache, which is the one that still matters.
    asyncio.create_task(_update_profile_on_save(user_id, paper_id))

    # The response replaces the whole actions row, so it has to carry enough
    # context to re-render every control in it — including "Why this?", which
    # is derived from candidate_source. Without these the button silently
    # disappeared the moment a user saved the paper.
    resp = templates.TemplateResponse(
        request,
        "partials/action_buttons.html",
        {
            "paper_id": paper_id,
            "saved": True,
            "dismissed": False,
            "source": source,
            "position": position,
            "query_id": query_id,
            "ranker_version": ranker_version,
            "candidate_source": candidate_source,
            "cluster_id": cluster_id,
            "propensity": propensity,
            "policy_id": policy_id,
        },
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.post("/{paper_id}/not-interested", response_class=HTMLResponse)
async def not_interested(
    paper_id: str,
    request: Request,
    source: str = Form(default="search"),
    position: int = Form(default=0),
    query_id: str = Form(default=""),
    ranker_version: str = Form(default=""),
    candidate_source: str = Form(default=""),
    cluster_id: str = Form(default=""),
    propensity: float = Form(default=0.0),
    policy_id: str = Form(default=""),
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())

    await db.log_interaction(
        user_id=user_id,
        paper_id=paper_id,
        event_type="not_interested",
        source=source,
        position=position or None,
        query_id=query_id or None,
        ranker_version=ranker_version or None,
        candidate_source=candidate_source or None,
        cluster_id=int(cluster_id) if cluster_id else None,
        propensity=propensity if propensity > 0 else None,
        policy_id=policy_id or None,
    )

    us.record_negative(user_id, paper_id)
    asyncio.create_task(_update_profile_on_dismiss(user_id, paper_id))

    resp = HTMLResponse(content="")
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


# ── Background EWMA profile update helpers ────────────────────────────────────

async def _update_profile_on_save(user_id: str, paper_id: str) -> None:
    """Background task: fetch paper embedding and update EWMA profiles."""
    try:
        vectors = await qdrant_svc.get_paper_vectors([paper_id])
        if paper_id not in vectors:
            return
        embedding = np.array(vectors[paper_id], dtype=np.float32)
        await profiles.update_on_save(user_id, embedding)
    except Exception as e:
        print(f"[events] EWMA save update failed for {paper_id}: {e}")


async def _update_profile_on_dismiss(user_id: str, paper_id: str) -> None:
    """Background task: fetch paper embedding and update negative profile."""
    try:
        vectors = await qdrant_svc.get_paper_vectors([paper_id])
        if paper_id not in vectors:
            return
        embedding = np.array(vectors[paper_id], dtype=np.float32)
        await profiles.update_on_dismiss(user_id, embedding)
    except Exception as e:
        print(f"[events] EWMA dismiss update failed for {paper_id}: {e}")
