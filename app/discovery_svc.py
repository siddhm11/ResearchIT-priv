"""Balanced starter candidates from the reader's selected research areas."""
from __future__ import annotations

import asyncio
from itertools import zip_longest

from app import local_meta, turso_svc


async def starter_papers(categories: set[str], limit: int = 30) -> list[dict]:
    """Give thin categories room without eight remote full-table scans.

    With the indexed sidecar, query each category and round-robin its ranked
    results. Without it, retain the existing single cached remote fallback.
    These are citation/popularity candidates, not measured live trends.
    """
    if not categories or limit <= 0:
        return []
    if len(categories) == 1 or not local_meta.is_available():
        return await turso_svc.fetch_trending_by_categories(categories, limit=limit)
    semaphore = asyncio.Semaphore(4)

    async def one(code: str) -> list[dict]:
        async with semaphore:
            return await turso_svc.fetch_trending_by_categories({code}, limit=limit)

    results = await asyncio.gather(*(one(code) for code in sorted(categories)),
                                   return_exceptions=True)
    pools = [r for r in results if isinstance(r, list)]
    seen: set[str] = set()
    papers: list[dict] = []
    for row in zip_longest(*pools):
        for paper in row:
            if not paper or not paper.get("arxiv_id") or paper["arxiv_id"] in seen:
                continue
            seen.add(paper["arxiv_id"])
            papers.append(paper)
            if len(papers) == limit:
                return papers
    return papers
