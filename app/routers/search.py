"""
Search router — Phase 3 hybrid semantic search.

GET /search?q=<query>
  – Returns full page on normal request
  – Returns partial <div id="search-results"> on HTMX request (hx-target swap)

Phase 3 replaces the arXiv keyword API with:
  LLM rewrite → BGE-M3 encode → Qdrant dense + Zilliz sparse → RRF → rerank

Phase 3.5: Metadata now fetched from Turso cloud DB (fast, includes citations)
  with arXiv API as fallback for papers not in Turso.
"""
import uuid
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse
from app import arxiv_svc, db, turso_svc, user_state as us, hybrid_search_svc
from app.config import COOKIE_NAME, ARXIV_MAX_RESULTS
from app.templates_env import templates

router = APIRouter()


@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = "",
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    papers = []
    if q.strip():
        # Phase 3: Hybrid semantic search (BGE-M3 + Qdrant + Zilliz + RRF)
        try:
            arxiv_ids = await hybrid_search_svc.search(q.strip(), limit=ARXIV_MAX_RESULTS)
        except Exception as e:
            print(f"[search] Hybrid search error: {e}")
            arxiv_ids = []

        if arxiv_ids:
            # Phase 3.5: Fetch metadata from Turso DB first (fast, ~50ms)
            try:
                meta = await turso_svc.fetch_metadata_batch(arxiv_ids)
            except Exception as e:
                print(f"[search] Turso metadata fetch failed: {e}")
                meta = {}

            # Fallback: fetch any missing IDs from arXiv API
            missing = [aid for aid in arxiv_ids if aid not in meta]
            if missing:
                try:
                    arxiv_meta = await arxiv_svc.fetch_metadata_batch(missing)
                    meta.update(arxiv_meta)
                except Exception as e:
                    print(f"[search] arXiv fallback for {len(missing)} IDs failed: {e}")

            # Phase 4.3: Cache to SQLite so dismissal category JOINs work
            await db.cache_turso_metadata_batch(list(meta.values()))

            # Preserve ranking order from hybrid search
            papers = [meta[aid] for aid in arxiv_ids if aid in meta]

        if not papers and q.strip():
            # Fallback: arXiv keyword API if hybrid returns nothing
            try:
                papers = await arxiv_svc.search(q.strip())
            except Exception as e:
                print(f"[search] arXiv fallback also failed: {e}")
                papers = []

    user_id = user_id or str(uuid.uuid4())
    state = await us.ensure_loaded(user_id)
    saved_ids = set(state.positive_list)
    dismissed_ids = set(state.negative_list)

    for p in papers:
        p["saved"] = p["arxiv_id"] in saved_ids
        p["dismissed"] = p["arxiv_id"] in dismissed_ids

    if request.headers.get("HX-Request"):
        resp = templates.TemplateResponse(
            request,
            "partials/search_results.html",
            {"papers": papers, "query": q},
        )
    else:
        resp = templates.TemplateResponse(
            request,
            "search.html",
            {
                "papers": papers,
                "query": q,
                "has_recs": state.has_enough_for_recs(),
            },
        )

    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp
