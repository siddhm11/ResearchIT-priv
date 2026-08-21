"""
Paper page — the site's owned destination for a single paper.

WHY THIS EXISTS
---------------
Until now no URL on this site was about a paper. Every card sent its title and
its arXiv id straight to arxiv.org, so the product had no page of its own to
share, to link to, or to return to. Three consequences, all of them structural
rather than cosmetic:

  * Nothing is shareable. The Open Graph tags added to base.html describe the
    feed, because the feed was the only thing there was to describe. A pasted
    link could never carry a paper's title.
  * There is nowhere to put comprehension. Plain-language explanation,
    difficulty, "read this first" — every one of them needs a surface, and a
    card in an infinite feed is not it.
  * There is no reason to come back. A link out to arxiv.org is the end of the
    session.

GET /p/{arxiv_id}
  Full metadata, the untruncated-where-possible abstract, save/dismiss, related
  work drawn from the paper's own embedding, and per-paper card tags.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse

from app import arxiv_svc, qdrant_svc, turso_svc, user_state as us
from app.config import COOKIE_NAME
from app.templates_env import templates

router = APIRouter()

# How many neighbours to show. Small on purpose: this is "what sits next to
# this paper", not a second feed, and each one costs a metadata row.
_RELATED_LIMIT = 6


async def _fetch_one(arxiv_id: str) -> dict | None:
    """Metadata for a single paper: sidecar/Turso first, arXiv as fallback."""
    meta = await turso_svc.fetch_metadata_batch([arxiv_id])
    paper = meta.get(arxiv_id)
    if paper:
        return paper
    try:
        fallback = await arxiv_svc.fetch_metadata_batch([arxiv_id])
    except Exception as e:
        print(f"[paper] arXiv fallback failed for {arxiv_id}: {e}")
        return None
    return fallback.get(arxiv_id)


async def _related(arxiv_id: str, seen: set[str]) -> list[dict]:
    """Nearest neighbours of this paper, by its own embedding.

    Best-effort throughout: the page is worth serving without them, so a vector
    store that is down costs the section and nothing else.
    """
    try:
        vectors = await qdrant_svc.get_paper_vectors([arxiv_id])
        vec = vectors.get(arxiv_id)
        if vec is None:          # numpy array — `not vec` would raise
            return []
        hits = await qdrant_svc.search_by_vector(
            query_vector=vec.tolist(),
            limit=_RELATED_LIMIT + 1,
            exclude_ids=seen | {arxiv_id},
        )
    except Exception as e:
        print(f"[paper] related lookup failed for {arxiv_id}: {e}")
        return []

    ids = [h for h in hits if h != arxiv_id][:_RELATED_LIMIT]
    if not ids:
        return []
    meta = await turso_svc.fetch_metadata_batch(ids)
    return [meta[i] for i in ids if i in meta]


@router.get("/p/{arxiv_id}", response_class=HTMLResponse)
async def paper_page(
    arxiv_id: str,
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())
    state = await us.ensure_loaded(user_id)

    paper = await _fetch_one(arxiv_id)
    if paper is None:
        # Rendered through the normal layout by the 404 handler in main.py,
        # rather than as a bare JSON detail.
        return templates.TemplateResponse(
            request, "not_found.html", {}, status_code=404)

    paper = {
        **paper,
        "saved": arxiv_id in state.positives,
        "dismissed": arxiv_id in state.negatives,
        # This page is not a ranked surface, so it carries no ranking
        # instrumentation. Leaving the §3.11 fields empty is deliberate: a
        # save from here genuinely has no query_id, propensity or policy, and
        # inventing one would corrupt the very analysis those fields exist for.
        "candidate_source": "paper_page",
    }

    related = await _related(arxiv_id, us.all_seen(user_id))

    resp = templates.TemplateResponse(
        request,
        "paper.html",
        {
            "paper": paper,
            "related": related,
            "og_title": paper.get("title") or f"arXiv:{arxiv_id}",
            "og_description": _summary_for_card(paper),
            "og_type": "article",
        },
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


def _summary_for_card(paper: dict) -> str:
    """A one-line description for the link preview.

    Kept short and cut on a word boundary — a social card truncates anyway, and
    cutting mid-word is exactly the defect the abstract work just fixed on the
    card itself.
    """
    text = (paper.get("abstract") or "").strip().replace("\n", " ")
    if not text:
        authors = (paper.get("authors") or "").strip()
        return f"{authors} — on arXiv." if authors else "A paper on arXiv."
    if len(text) <= 200:
        return text
    return text[:200].rsplit(" ", 1)[0] + "…"
