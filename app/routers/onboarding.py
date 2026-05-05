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
from fastapi import APIRouter, Request, Cookie, Form
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

    # Check current save count
    from app import user_state as us
    state = await us.ensure_loaded(user_id)
    seed_count = len(state.positives)

    resp = templates.TemplateResponse(
        request,
        "partials/seed_search.html",
        {
            "papers": papers,
            "query": q,
            "seed_count": seed_count,
            "seed_target": 5,
        },
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


@router.post("/api/onboarding/import-author", response_class=HTMLResponse)
async def import_author(
    request: Request,
    author_url: str = Form(default=""),
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """Phase 5.1: Import papers from a Semantic Scholar author profile.

    Accepts S2 URL, raw S2 author ID, or ORCID.
    Auto-saves the author's arXiv papers as seed interests.
    """
    user_id = user_id or str(uuid.uuid4())

    if not author_url.strip():
        return HTMLResponse(
            '<div class="alert alert-warning text-sm py-2">'
            '⚠️ Please paste a Semantic Scholar author URL, ID, or ORCID.</div>'
        )

    from app import s2_svc, user_state as us

    # 1. Parse input
    parsed_id, input_type = s2_svc.parse_author_input(author_url)
    if parsed_id is None:
        return HTMLResponse(
            '<div class="alert alert-error text-sm py-2">'
            '❌ Could not recognise input. Paste a Semantic Scholar author URL, '
            'a numeric author ID, or an ORCID (e.g. 0000-0003-3394-6622).</div>'
        )

    # 2. Resolve ORCID → S2 author ID if needed
    try:
        if input_type == "orcid":
            s2_id = await s2_svc.resolve_orcid(parsed_id)
            if not s2_id:
                return HTMLResponse(
                    '<div class="alert alert-warning text-sm py-2">'
                    f'⚠️ No Semantic Scholar author found for ORCID {parsed_id}.</div>'
                )
        else:
            s2_id = parsed_id
    except Exception as e:
        print(f"[onboarding] ORCID resolve failed: {e}")
        return HTMLResponse(
            '<div class="alert alert-error text-sm py-2">'
            '❌ Failed to look up ORCID. Please try pasting the S2 URL directly.</div>'
        )

    # 3. Fetch arXiv papers
    try:
        arxiv_ids = await s2_svc.fetch_author_arxiv_papers(s2_id, limit=20)
    except Exception as e:
        print(f"[onboarding] S2 author paper fetch failed: {e}")
        return HTMLResponse(
            '<div class="alert alert-error text-sm py-2">'
            '❌ Failed to fetch papers from Semantic Scholar. '
            'The author ID may be invalid, or the API may be down.</div>'
        )

    if not arxiv_ids:
        return HTMLResponse(
            '<div class="alert alert-warning text-sm py-2">'
            '⚠️ No arXiv papers found for this author. '
            'They may publish in venues not indexed on arXiv.</div>'
        )

    # 4. Auto-save each paper as a positive interaction
    for aid in arxiv_ids:
        us.record_positive(user_id, aid)
        await db.log_interaction(
            user_id=user_id,
            paper_id=aid,
            event_type="save",
            source="s2_import",
        )

    state = await us.ensure_loaded(user_id)
    seed_count = len(state.positives)

    resp = HTMLResponse(
        f'<div class="alert alert-success text-sm py-2">'
        f'✅ Imported {len(arxiv_ids)} papers! '
        f'You now have {seed_count} saved papers. '
        f'Click <strong>"Done — start exploring →"</strong> to see your recommendations.</div>'
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp
