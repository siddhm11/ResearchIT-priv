"""
What a redesigned feed card shows beyond today's card, from local reads only.

    papers = await card_data.enrich(papers, request)

Called where the feed renders a page, after the page is built and logged, never
where the ranked pool is cached. It returns new dicts in the same order with
the same ids and instrumentation fields, adding:

    card_figure  the first saved figure that may be shown, with its thumbnail
                 URL, size, caption and credit (switch FIGURES)
    card_code    the saved GitHub repository and stars, if any
    card_plain   "In plain terms" if one is already cached for this abstract;
                 the feed never asks a model, which would cost a call per card
    abstract     the full abstract when abstract_svc has one saved

No outside request is made: everything comes from paper_sources,
paper_explanations and paper_abstracts.
"""
from __future__ import annotations

import aiosqlite

from app import abstract_svc, config, errors, features, groq_svc, paper_source_svc
from app.routers.figures import figure_credit, figure_url


async def _cached_explanations(keys: list[str]) -> dict[str, str]:
    if not keys:
        return {}
    marks = ",".join("?" * len(keys))
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute(
            f"SELECT cache_key, explanation FROM paper_explanations WHERE cache_key IN ({marks})", keys)
        return dict(await cur.fetchall())


async def enrich(papers: list[dict], request=None) -> list[dict]:
    """Copies of `papers`, in order, with the card fields added. Never raises."""
    out = [dict(p) for p in papers]
    try:
        out = [dict(p) for p in await abstract_svc.overlay(out)]
        ids = [p["arxiv_id"] for p in out if p.get("arxiv_id")]
        sources = await paper_source_svc.get_many(ids)
        keys = {p["arxiv_id"]: groq_svc.explain_cache_key(p["arxiv_id"], p.get("abstract") or "")
                for p in out if p.get("arxiv_id")}
        plain = await _cached_explanations(list(keys.values()))
        show_figures = features.visible("FIGURES", request)
    except Exception as e:                      # a plain card beats a broken feed
        errors.report("card_data", "enriching feed cards failed", e)
        return out
    for p in out:
        aid = p.get("arxiv_id")
        source = sources.get(aid)
        p["card_plain"] = plain.get(keys.get(aid, ""))
        p["card_code"] = ({"repo": source.code_repo, "stars": source.code_stars}
                          if source is not None and source.code_repo else None)
        p["card_figure"] = first_figure(p, source) if show_figures else None
    return out


def first_figure(paper: dict, source) -> dict | None:
    """The first saved figure of `source` that may be shown, as a card shows it."""
    if source is None:
        return None
    for n, f in enumerate(source.figures):
        url = figure_url(source, n)
        if url:
            return {"url": url, "w": f.get("w"), "h": f.get("h"),
                    "label": f.get("label") or "", "caption": f.get("caption") or "",
                    "credit": figure_credit(paper, source)}
    return None


async def add_figures(papers: list[dict], request=None) -> None:
    """Set card_figure on each paper in place, from saved sources only (the
    trending cards, which build their own dicts). Never raises: on any failure
    every card goes without a figure."""
    for p in papers:
        p["card_figure"] = None
    if not papers or not features.visible("FIGURES", request):
        return
    try:
        sources = await paper_source_svc.get_many([p["arxiv_id"] for p in papers if p.get("arxiv_id")])
    except Exception as e:
        errors.report("card_data", "trending card figures failed", e)
        return
    for p in papers:
        p["card_figure"] = first_figure(p, sources.get(p.get("arxiv_id")))
