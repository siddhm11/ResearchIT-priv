"""
Your library on another device (doc 06, 2026-11-17).

A reader is a browser cookie: saves, interests and follows belong to it, so a
new phone or cleared cookies started over with nothing. This gives each reader
one private link that opens the same library in any browser, with no account
and no email:

    GET  /library/devices          the link, with Copy and Share; make a new one or turn it off
    POST /api/library-link         make a new link (the old one stops working)
    POST /api/library-link/revoke  turn the link off
    GET  /l/{token}                open a library from its link
    POST /l/{token}                confirm opening it in a browser that has its own saves

The token is 128 random bits and is not the user id, which appears in logs and
dashboards; a link that leaks is replaced in one click. Opening a link in a
browser that already has saves asks first, then adds that browser's saved
papers and follows to the linked library, so nothing a reader did is lost.
That POST must come from this site (Sec-Fetch-Site / Origin), so another page
cannot switch a visitor's library behind their back. Rows replicate to Turso
(turso_sync "library_links"), so links survive a deploy.
"""
from __future__ import annotations

import secrets
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import config, db
from app import user_state as us
from app.config import COOKIE_MAX_AGE, COOKIE_NAME
from app.templates_env import templates

router = APIRouter()

MERGE_SOURCE = "library_merge"


def link_url(token: str) -> str:
    return f"{config.PUBLIC_URL}/l/{token}"


def _with_cookie(resp, user_id: str):
    resp.set_cookie(COOKIE_NAME, user_id, max_age=COOKIE_MAX_AGE, httponly=True)
    return resp


async def _saved_ids(user_id: str) -> list[str]:
    """Saved paper ids, oldest first."""
    return [r["paper_id"] for r in reversed(await db.get_current_feedback(user_id))
            if r["event_type"] == "save"]


def _same_site(request: Request) -> bool:
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        return site in ("same-origin", "none")
    origin = request.headers.get("origin")
    if origin is None:          # older browsers send neither on a same-site form
        return True
    return urlparse(origin).netloc == request.headers.get("host", "")


async def merge_into(from_uid: str, to_uid: str) -> int:
    """Add from_uid's saved papers and follows to to_uid. Returns papers added.

    A paper the target already saved or passed on keeps the target's decision.
    Each added save updates the target's profile like a save from a card."""
    from app.routers.events import _spawn, _update_profile_on_save
    await us.ensure_loaded(to_uid)
    decided = {r["paper_id"] for r in await db.get_current_feedback(to_uid)}
    added = 0
    for pid in await _saved_ids(from_uid):
        if pid in decided:
            continue
        await db.log_interaction(user_id=to_uid, paper_id=pid, event_type="save", source=MERGE_SOURCE)
        us.record_positive(to_uid, pid)
        _spawn(_update_profile_on_save(to_uid, pid))
        added += 1
    for slug in await db.get_followed_slugs(from_uid):
        await db.follow_collection(to_uid, slug)
    if added:
        onboarding = await db.get_onboarding_state(to_uid)
        if not onboarding or not onboarding["onboarding_completed"]:
            await db.complete_onboarding(to_uid)
    return added


@router.get("/library/devices", response_class=HTMLResponse)
async def devices_page(request: Request, user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    user_id = user_id or str(uuid.uuid4())
    token = await db.get_library_link(user_id)
    resp = templates.TemplateResponse(request, "library_devices.html", {
        "link": link_url(token) if token else None,
        "count": len(await _saved_ids(user_id)),
        "library_tab": "devices",
    })
    return _with_cookie(resp, user_id)


@router.post("/api/library-link")
async def make_link(request: Request, user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    if not _same_site(request):
        return HTMLResponse("This link can only be made from ResearchIT itself.", status_code=403)
    user_id = user_id or str(uuid.uuid4())
    await db.create_library_link(user_id, secrets.token_urlsafe(16))
    return _with_cookie(RedirectResponse("/library/devices#link", status_code=303), user_id)


@router.post("/api/library-link/revoke")
async def revoke_link(request: Request, user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    if not _same_site(request):
        return HTMLResponse("This can only be done from ResearchIT itself.", status_code=403)
    if user_id:
        await db.revoke_library_link(user_id)
    return RedirectResponse("/library/devices", status_code=303)


@router.get("/l/{token}", response_class=HTMLResponse)
async def open_link(token: str, request: Request,
                    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    target = await db.resolve_library_link(token)
    if target is None:
        resp = templates.TemplateResponse(request, "library_open.html", {"state": "gone"}, status_code=404)
        return _with_cookie(resp, user_id) if user_id else resp
    if user_id == target:
        return RedirectResponse("/saved?opened=here", status_code=303)
    mine = await _saved_ids(user_id) if user_id else []
    if not mine:
        # Nothing here to lose: open the library straight away.
        return _with_cookie(RedirectResponse("/saved?opened=1", status_code=303), target)
    resp = templates.TemplateResponse(request, "library_open.html", {
        "state": "confirm", "token": token, "mine": len(mine),
        "theirs": len(await _saved_ids(target)),
    })
    return _with_cookie(resp, user_id)


@router.post("/l/{token}", response_class=HTMLResponse)
async def confirm_open(token: str, request: Request,
                       user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    if not _same_site(request):
        return HTMLResponse("Open the link again and confirm on ResearchIT's own page.", status_code=403)
    target = await db.resolve_library_link(token)
    if target is None:
        return templates.TemplateResponse(request, "library_open.html", {"state": "gone"}, status_code=404)
    added = await merge_into(user_id, target) if user_id and user_id != target else 0
    return _with_cookie(RedirectResponse(f"/saved?opened=1&added={added}", status_code=303), target)
