"""
FastAPI application entry point.

Routes:
  GET  /          → home (recs + search bar)
  GET  /search    → search router
  POST /api/papers/{id}/save           → events router
  POST /api/papers/{id}/not-interested → events router
  GET  /api/recommendations            → recommendations router
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Cookie
from fastapi.responses import HTMLResponse
from app import db
from app.config import APP_TITLE, COOKIE_NAME
from app.templates_env import templates
from app.routers import search, events, recommendations, saved


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    # Phase 3: Warm up BGE-M3 at startup (graceful — app works without it)
    try:
        import asyncio
        from app import embed_svc
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, embed_svc.get_model)
        print("[main] BGE-M3 model loaded — hybrid search ready")
    except Exception as e:
        print(f"[main] BGE-M3 not loaded ({e}) — search will fall back to arXiv API")
    yield


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.include_router(search.router)
app.include_router(events.router)
app.include_router(recommendations.router)
app.include_router(saved.router)


@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())
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
