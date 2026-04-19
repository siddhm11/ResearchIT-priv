"""
Search router — Phase 3 hybrid semantic search.

GET /search?q=<query>
  – Returns full page on normal request
  – Returns partial <div id="search-results"> on HTMX request (hx-target swap)

Phase 3 replaces the arXiv keyword API with:
  LLM rewrite → BGE-M3 encode → Qdrant dense + Zilliz sparse → RRF → rerank
"""
import uuid
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse
from app import arxiv_svc, user_state as us, hybrid_search_svc
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
            # Fetch metadata for the ranked results
            try:
                meta = await arxiv_svc.fetch_metadata_batch(arxiv_ids)
                # Preserve ranking order from hybrid search
                papers = [meta[aid] for aid in arxiv_ids if aid in meta]
            except Exception as e:
                # arXiv API timeout — fall back to keyword search
                print(f"[search] Metadata fetch failed ({e}), falling back to arXiv API")
                papers = []

        if not papers and q.strip():
            # Fallback: arXiv keyword API if hybrid returns nothing or metadata failed
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
