"""
Onboarding router — Phase 5 Cold-Start.

GET  /onboarding                    → render wizard (redirect to / if done)
POST /api/onboarding/categories     → save selected category groups
GET  /api/onboarding/seed-search    → search for seed papers (HTMX partial)
POST /api/onboarding/complete       → mark done, redirect to /
POST /api/onboarding/skip           → mark done (no categories), redirect to /
"""
import uuid
import json
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from app import db
from app.config import COOKIE_NAME, CATEGORY_GROUPS
from app.templates_env import templates

# Reuse the hybrid search backend for seed paper discovery
from app import hybrid_search_svc, arxiv_svc, turso_svc

router = APIRouter()


@router.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())

    # If already completed, go home
    state = await db.get_onboarding_state(user_id)
    if state and state["onboarding_completed"]:
        resp = RedirectResponse("/", status_code=302)
        resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
        return resp

    # Load any previously selected categories (if they started but didn't finish)
    selected = state["selected_categories"] if state else []

    resp = templates.TemplateResponse(
        request,
        "onboarding.html",
        {
            "categories": CATEGORY_GROUPS,
            "selected": selected,
        },
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.post("/api/onboarding/categories", response_class=HTMLResponse)
async def save_categories(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """Save selected categories and return the seed search step (HTMX partial)."""
    user_id = user_id or str(uuid.uuid4())

    # Parse JSON body from the HTMX request
    body = await request.json()
    categories = body.get("categories", [])

    # Validate: must be valid group keys
    valid = [c for c in categories if c in CATEGORY_GROUPS]
    await db.save_onboarding_categories(user_id, valid)

    # Return the seed search step partial
    from app import user_state as us
    state = await us.ensure_loaded(user_id)
    seed_count = len(state.positives)

    resp = templates.TemplateResponse(
        request,
        "partials/seed_search.html",
        {
            "seed_count": seed_count,
            "seed_target": 5,
        },
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.get("/api/onboarding/seed-search", response_class=HTMLResponse)
async def seed_search(
    request: Request,
    q: str = "",
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """Search for papers to save as seed interests during onboarding."""
    user_id = user_id or str(uuid.uuid4())

    papers = []
    if q.strip():
        try:
            results = await hybrid_search_svc.search(q.strip(), limit=6)
            arxiv_ids = results  # search() returns list[str] directly
            if arxiv_ids:
                meta = await turso_svc.fetch_metadata_batch(arxiv_ids)
                missing = [aid for aid in arxiv_ids if aid not in meta]
                if missing:
                    try:
                        arxiv_meta = await arxiv_svc.fetch_metadata_batch(missing)
                        meta.update(arxiv_meta)
                    except Exception:
                        pass
                papers = [meta[aid] for aid in arxiv_ids if aid in meta]
        except Exception as e:
            print(f"[onboarding] seed search failed: {e}")
            # Fallback to arXiv API keyword search
            try:
                from app import arxiv_svc
                papers = await arxiv_svc.search(q.strip(), max_results=6)
            except Exception:
                pass

    # HTMX request: return ONLY the results partial (swap target = #seed-results).
    # The full seed_search.html panel is rendered by save_categories() during the
    # step 1 → step 2 transition; subsequent searches must not re-render the whole
    # panel or it nests inside #seed-results and duplicates the wizard.
    resp = templates.TemplateResponse(
        request,
        "partials/seed_results.html",
        {"papers": papers, "query": q},
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.post("/api/onboarding/complete")
async def complete_onboarding(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """Mark onboarding as complete and redirect to home."""
    user_id = user_id or str(uuid.uuid4())
    await db.complete_onboarding(user_id)
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.post("/api/onboarding/skip")
async def skip_onboarding(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """Skip onboarding entirely — mark as complete with no categories."""
    user_id = user_id or str(uuid.uuid4())
    await db.complete_onboarding(user_id)
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp



