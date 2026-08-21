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
from datetime import datetime

from fastapi import FastAPI, Request, Cookie
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               PlainTextResponse)
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from app import db
from app.config import APP_TITLE, COOKIE_NAME
from app.templates_env import templates
from app.routers import (search, events, recommendations, saved, onboarding,
                         health, collections)


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

    # Feed impressions grow with every page served and are only a "do not show
    # this again yet" set, so anything this old has no influence on the feed.
    try:
        pruned = await db.prune_impressions(retention_days=90)
        if pruned:
            print(f"[main] Pruned {pruned} old feed impression rows")
    except Exception as e:
        print(f"[main] Impression pruning skipped: {e}")
    yield

    # Final flush so the last sync interval is not lost on shutdown.
    try:
        from app import turso_sync
        await turso_sync.stop()
    except Exception as e:
        print(f"[main] Turso final flush skipped: {e}")

    # Close the shared connection pool after the final flush, since that flush
    # goes through it.
    try:
        from app import http_client
        await http_client.aclose()
    except Exception as e:
        print(f"[main] HTTP pool close skipped: {e}")


app = FastAPI(title=APP_TITLE, lifespan=lifespan)


# ── Compression ──────────────────────────────────────────────────────────────
#
# 117KB of CSS+JS was served uncompressed on every load. Measured: styles.css
# 57.6KB -> 15.6KB (72%), htmx.min.js 48.1KB -> 15.7KB (67%), app.js 11.3KB ->
# 3.8KB (66%); 117KB -> 35KB overall, a 70% saving. Text/HTML responses benefit
# too, and the feed fragment is the largest thing this app returns.
#
# minimum_size skips the tiny htmx fragments, where the gzip header plus the
# CPU cost is not worth it — this box has 2 vCPUs.
app.add_middleware(GZipMiddleware, minimum_size=800)


# ── Response headers ─────────────────────────────────────────────────────────

# Filenames under /static are stable across deploys, so `immutable` would pin a
# stale stylesheet in every returning visitor's cache until they hard-reloaded.
# A one-hour max-age plus revalidation gets most of the benefit — the browser
# stops re-fetching 117KB on every navigation — without that trap. Going
# immutable needs a content hash in the URL first.
_STATIC_CACHE_CONTROL = "public, max-age=3600, must-revalidate"


@app.middleware("http")
async def _security_and_cache_headers(request: Request, call_next):
    """Baseline response headers.

    No CSP here. The templates use inline `onclick` handlers and an inline
    theme-restore script that has to run before first paint, so a meaningful
    policy needs either per-response hashes or a refactor of both — more than a
    header, and worth doing deliberately rather than as a side effect.
    """
    response = await call_next(request)

    # Stops a browser from second-guessing our Content-Type, which is how a
    # user-supplied string ends up executed as script.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")

    # Every card links out to arxiv.org. Without this the full referring URL
    # goes with it — including the user's search query.
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

    if request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", _STATIC_CACHE_CONTROL)

    return response


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    """Throttle the quota-hungry endpoints; see app/rate_limit.py.

    Wrapped in try/except on purpose. This sits in front of every request, so a
    fault in the limiter would take down the whole app -- for a feature whose
    only job is to shed load. Any error here allows the request through.
    """
    try:
        from app import rate_limit
        allowed, retry_after = rate_limit.check(
            request.url.path, rate_limit.client_key(request))
        if not allowed:
            return PlainTextResponse(
                "Too many requests. Please slow down.",
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
    except Exception as e:  # pragma: no cover - defensive
        print(f"[main] rate limiter skipped ({e})")
    return await call_next(request)


# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(search.router)
app.include_router(events.router)
app.include_router(recommendations.router)
app.include_router(saved.router)
app.include_router(onboarding.router)
app.include_router(health.router)
app.include_router(collections.router)
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

    # The masthead is rendered HERE rather than inside the feed fragment.
    #
    # It began life in the fragment because only the fragment knew the issue
    # number and date. That cost the homepage its only <h1>: the fragment is
    # fetched by htmx after first paint, so the delivered HTML had no heading
    # at all — bad for screen readers, for crawlers, and for anyone whose feed
    # request fails or is slow. It also meant the page title did not appear
    # until the whole tier cascade had run, which can take seconds.
    #
    # Both facts the masthead needs are cheap and available here: the date is
    # local, and count_feed_issues() is one indexed SQLite count.
    resp = templates.TemplateResponse(
        request,
        "index.html",
        {
            "has_recs": state.has_enough_for_recs(),
            "save_count": len(state.positives),
            "issue_date": datetime.now().strftime("%A %-d %B %Y"),
            "issue_number": await db.count_feed_issues(user_id),
        },
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


# ── HTML 404 ─────────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def _not_found(request: Request, exc):
    """Render a real page for a wrong URL.

    Content-negotiated by path rather than by Accept header: htmx sends
    `Accept: */*`, so keying on Accept would hand HTML to the JSON callers and
    defeat the point. Everything under /api/ and /healthz keeps the JSON shape
    its callers parse; every human-facing route gets the normal layout.
    """
    path = request.url.path
    if path.startswith("/api/") or path.startswith("/healthz"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
