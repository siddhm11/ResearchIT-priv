"""
The front page a new visitor sees at / (doc 06, 2026-11-17).

Until now `/` sent everyone without a finished onboarding straight to the
two-step form: no product name, no search box, no paper. 8 of the first 18
outside readers never reached a feed. This page says what ResearchIT is,
offers search right away, and shows the week's most upvoted papers as real
cards a visitor can open or save. "Personalise your feed" leads to the same
onboarding as before, which is no longer the only way in.

A save from here counts like any other: the next visit to / finds the save,
marks onboarding done and serves a personal feed (one save is enough, §3.3).
The cards are a ranked surface, so they carry query_id, position, propensity
and policy_id and are written to the exposure log (§3.11).
"""
from __future__ import annotations

import uuid

from fastapi import Request

from app import card_data, db, errors, trending_shelf
from app.templates_env import corpus_label, templates

SOURCE = "welcome"
POLICY = "welcome_top_v1"
N_PAPERS = 7            # one wide lead card and two rows of three
EXAMPLES = (
    "how do diffusion models generate images",
    "LoRA fine-tuning",
    "dark matter detection",
    "protein structure prediction",
)


async def render(request: Request, user_id: str, started: bool):
    """The page itself. `started`: the visitor began onboarding and left it."""
    papers = trending_shelf.week_top(N_PAPERS)
    query_id = str(uuid.uuid4())
    exposures = []
    for i, p in enumerate(papers):
        p.update(query_id=query_id, position=i, propensity=1.0, policy_id=POLICY,
                 ranker_version=POLICY, candidate_source=trending_shelf.CANDIDATE_SOURCE, cluster_id=None)
        exposures.append({"user_id": user_id, "paper_id": p["arxiv_id"], "query_id": query_id,
                          "position": i, "tier": None, "candidate_source": p["candidate_source"],
                          "cluster_id": None, "propensity": 1.0, "policy_id": POLICY})
    if exposures:
        try:
            await db.record_exposures(exposures)
        except Exception as e:  # pragma: no cover - analytics must not cost the page
            errors.report("welcome", "exposure write failed", e)
    await card_data.add_figures(papers, request)
    return templates.TemplateResponse(request, "welcome.html", {
        "papers": papers, "source": SOURCE, "query_id": query_id, "started": started,
        "examples": EXAMPLES, "window": trending_shelf.window(),
        "og_title": "ResearchIT",
        "og_description": (f"Search {corpus_label()} arXiv papers in plain words, or get a reading feed that "
                           "learns your separate interests from the papers you save. Free, no sign-up."),
    })
