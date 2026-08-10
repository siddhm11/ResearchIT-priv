"""
FastAPI application entry point.

Routes:
  GET  /          → home (recs + search bar) — redirects to /onboarding for new users
  GET  /onboarding → onboarding wizard (Phase 5)
  GET  /search    → search router
  POST /api/papers/{id}/save           → events router
  POST /api/papers/{id}/not-interested → events router
  GET  /api/recommendations            → recommendations router
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from app import db
from app.config import APP_TITLE, COOKIE_NAME
from app.templates_env import templates
from app.routers import search, events, recommendations, saved, onboarding, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()

    # Restore user data from Turso and start replicating.
    # DB_PATH is /tmp on HF Spaces, so without this every save, EWMA profile,
    # cluster and onboarding record is destroyed on each rebuild.
    try:
        from app import turso_sync
        await turso_sync.start()
    except Exception as e:
        print(f"[main] Turso sync unavailable ({e}) -- user data is ephemeral")
    # Phase 3: Warm up BGE-M3 at startup (graceful — app works without it)
    try:
        import asyncio
        from app import embed_svc
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, embed_svc.get_model)
        print("[main] BGE-M3 model loaded -- hybrid search ready")
    except Exception as e:
        print(f"[main] BGE-M3 not loaded ({e}) -- search will fall back to arXiv API")
    
    # Eagerly warm up Cross-Encoder Reranker at startup (graceful fallback)
    try:
        import asyncio
        from app import reranker_bge_svc
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, reranker_bge_svc.get_reranker)
        print("[main] Cross-Encoder Reranker loaded -- high relevance search ready")
    except Exception as e:
        print(f"[main] Reranker not loaded ({e}) -- search will fall back to RRF rankings")

    # Phase 6.5 B3: Prune old cluster snapshots (>30 days)
    try:
        pruned = await db.prune_old_snapshots(retention_days=30)
        if pruned:
            print(f"[main] Pruned {pruned} old cluster snapshot rows")
    except Exception as e:
        print(f"[main] Snapshot pruning skipped: {e}")
    yield

    # Final flush so the last sync interval is not lost on shutdown.
    try:
        from app import turso_sync
        await turso_sync.stop()
    except Exception as e:
        print(f"[main] Turso final flush skipped: {e}")


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(search.router)
app.include_router(events.router)
app.include_router(recommendations.router)
app.include_router(saved.router)
app.include_router(onboarding.router)
app.include_router(health.router)
# researchit-space (3D map client) JSON API. Guarded the same way as the Turso
# sync and BGE-M3 warmup above: this router is additive, and a fault inside it
# must never stop the main app from serving.
try:
    from app.routers import space as _space
    app.include_router(_space.router)
except Exception as _e:
    print(f"[main] space JSON API unavailable ({_e}) -- map client features disabled")


@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())

    # Phase 5: Redirect new users to onboarding.
    # Existing users (any interaction history) are auto-marked as onboarded.
    onboarding_state = await db.get_onboarding_state(user_id)
    if onboarding_state is None:
        # Check if they're an existing user with interactions
        interactions = await db.get_user_interactions(user_id, limit=1)
        if interactions:
            # Auto-mark as onboarded — don't interrupt returning users
            await db.complete_onboarding(user_id)
        else:
            # Brand new user → onboarding
            resp = RedirectResponse("/onboarding", status_code=302)
            resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
            return resp
    elif not onboarding_state["onboarding_completed"]:
        resp = RedirectResponse("/onboarding", status_code=302)
        resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
        return resp

    from app import user_state as us
    state = await us.ensure_loaded(user_id)

    resp = templates.TemplateResponse(
        request,
        "index.html",
        {
            "has_recs": state.has_enough_for_recs(),
            "save_count": len(state.positives),
        },
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp
