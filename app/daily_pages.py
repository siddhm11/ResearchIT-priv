"""Dated daily pages: /day/<yyyy-mm-dd>, one day's Hugging Face Daily Papers
list ranked by upvotes, with a permanent address to link to (HF Daily, Cool
Papers and alphaXiv all lead with a dated daily list; UI pass 2026-11-27).

Built from lists already in memory: the Trending shelf's last days (refreshed
every few hours, freshest upvotes) over the Trending topics' history (about
six months, refreshed daily). No network call, no model.
"""
from __future__ import annotations

from app import trending_shelf, trending_topics


def _days() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for day, papers in (trending_topics._state.get("days") or {}).items():
        out[day] = {p["arxiv_id"]: dict(p) for p in papers}
    for p in trending_shelf._state.get("papers", {}).values():
        day = (p.get("listed_on") or "")[:10]
        if not day:
            continue
        repo = trending_shelf._state.get("code", {}).get(p["arxiv_id"]) or ("", 0)
        out.setdefault(day, {})[p["arxiv_id"]] = {
            "arxiv_id": p["arxiv_id"], "title": p.get("title") or p["arxiv_id"], "summary": p.get("summary") or "",
            "upvotes": int(p.get("upvotes") or 0), "github": repo[0], "stars": int(repo[1] or 0)}
    return out


def available() -> list[str]:
    """Days with a list, newest first."""
    return sorted(_days(), reverse=True)


def papers(day: str) -> list[dict]:
    """That day's list, most upvoted first."""
    return sorted(_days().get(day, {}).values(), key=lambda p: (-p["upvotes"], p["arxiv_id"]))


def neighbours(day: str) -> tuple[str | None, str | None]:
    """(the previous day with a list, the next one)."""
    days = sorted(_days())
    older = [d for d in days if d < day]
    newer = [d for d in days if d > day]
    return (older[-1] if older else None, newer[0] if newer else None)
