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

from fastapi import APIRouter, Cookie, Query, Request
from fastapi.responses import HTMLResponse

from app import (arxiv_svc, db, groq_svc, qdrant_svc, readability,
                 turso_svc, user_state as us)
from app.config import COOKIE_NAME
from app.routers.events import _NO_POSITION, _position
from app.templates_env import templates

router = APIRouter()

# How many neighbours to show. Small on purpose: this is "what sits next to
# this paper", not a second feed, and each one costs a metadata row.
_RELATED_LIMIT = 6


def _cluster_of(candidate_source: str) -> int | None:
    """The interest cluster a candidate_source names, or None.

    Tier 1 tags every candidate with "cluster_<idx>"; every other source
    ("exploration", "ewma_longterm", "qdrant_recommend",
    "trending_category_fallback", "paper_page") genuinely has no cluster and
    must stay NULL rather than be coerced to 0 -- 0 is a real cluster index.
    """
    if not candidate_source or not candidate_source.startswith("cluster_"):
        return None
    try:
        return int(candidate_source[len("cluster_"):])
    except ValueError:
        return None


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
    qid: str = Query(default=""),
    pos: int = Query(default=_NO_POSITION),
    src: str = Query(default=""),
    sf: str = Query(default=""),
    prop: float = Query(default=0.0),
    pol: str = Query(default=""),
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

    # The click-through. app/db.py has declared `click` as an event_type since
    # the schema was written and nothing ever wrote one, so the system knew
    # which papers were saved and which were dismissed but not which were
    # OPENED — the difference between "scrolled past" and "read", and the
    # densest engagement signal a feed produces.
    #
    # Only logged when the visit carries a query_id, i.e. it came from a ranked
    # surface. A bare /p/{id} visit — a shared link, a bookmark, a crawler —
    # has no propensity and no policy, and recording it as though it did would
    # corrupt the §3.11 contract rather than honour it.
    if qid:
        try:
            await db.log_interaction(
                user_id=user_id,
                paper_id=arxiv_id,
                event_type="click",
                # The SURFACE, not the candidate source. This used to pass
                # `src`, which is the retrieval origin ("cluster_1"), so the
                # column that is supposed to hold search|recommendation|saved
                # filled up with cluster names instead: 2 rows said
                # "recommendation" where 21 belonged, and a click from a search
                # result -- which carries a query_id but no src -- was recorded
                # as a recommendation. Both made grouping by source wrong.
                source=sf or "recommendation",
                position=_position(pos),
                query_id=qid,
                ranker_version=pol or None,
                candidate_source=src or None,
                # Was hard-coded None while `src` literally spelled the cluster
                # out. Per-cluster CTR is the one measurement that can show
                # whether a minority interest earns its slots, and it was
                # unanswerable because this column was empty on every row.
                cluster_id=_cluster_of(src),
                propensity=prop if prop > 0 else None,
                policy_id=pol or None,
            )
        except Exception as e:   # a lost click must never cost the page
            print(f"[paper] click log failed for {arxiv_id}: {e}")

    related = await _related(arxiv_id, us.all_seen(user_id))

    resp = templates.TemplateResponse(
        request,
        "paper.html",
        {
            "paper": paper,
            "related": related,
            "reading": readability.assess(paper.get("abstract") or ""),
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


@router.get("/api/papers/{arxiv_id}/explain", response_class=HTMLResponse)
async def explain(arxiv_id: str, request: Request):
    """Plain-language explanation of one paper.

    Fetched by htmx AFTER the page paints rather than rendered inline. The
    generation is a network call to Groq with an 8s ceiling, and the paper page
    must not wait on it — a reader who wants the abstract should never pay for
    a summary they did not ask to wait for. On a cache hit this is one indexed
    SQLite read and returns immediately.

    Shared across users and content-addressed, so the corpus warms itself: the
    second reader of a paper pays nothing.
    """
    paper = await _fetch_one(arxiv_id)
    if paper is None:
        return HTMLResponse(content="")

    abstract = paper.get("abstract") or ""
    key = groq_svc.explain_cache_key(arxiv_id, abstract)

    cached = None
    try:
        cached = await db.get_explanation(key)
    except Exception as e:                      # cache is an optimisation
        print(f"[paper] explanation cache read failed: {e}")

    text = cached
    if text is None:
        text = await groq_svc.explain_paper(paper.get("title") or "", abstract)
        if text:
            try:
                await db.save_explanation(
                    key, arxiv_id, text, groq_svc._EXPLAIN_MODEL)
            except Exception as e:
                print(f"[paper] explanation cache write failed: {e}")

    # Nothing rather than an apology. The section simply does not appear when
    # there is no honest summary to give — a truncated abstract, no API key, a
    # timeout. An empty slot is quieter than an error the reader cannot act on.
    if not text:
        return HTMLResponse(content="")

    return templates.TemplateResponse(
        request, "partials/explanation.html",
        {"explanation": text, "cached": cached is not None},
    )
