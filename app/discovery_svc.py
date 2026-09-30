"""Balanced starter candidates from the reader's selected research areas."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from itertools import zip_longest

from app import local_meta, turso_svc

# Every third starter slot within an interest goes to a paper from the corpus's
# last three months. Ranked by citations alone, those can never compete: a
# paper a few weeks old has almost none yet.
_FRESH_EVERY = 3


def _interleave(lists: list[list[dict]]) -> list[dict]:
    return [p for row in zip_longest(*lists) for p in row if p]


def _with_fresh(established: list[dict], fresh: list[dict]) -> list[dict]:
    out, e, f = [], iter(established), iter(fresh)
    while True:
        batch = [p for p in (next(e, None) for _ in range(_FRESH_EVERY - 1)) if p]
        batch += [p for p in [next(f, None)] if p]
        if not batch:
            return out + list(e) + list(f)
        out += batch


async def starter_papers(
    categories: set[str] | Mapping[str, set[str]], limit: int = 30,
) -> list[dict]:
    """Deal starter candidates out evenly across the reader's interests.

    `categories` is either a mapping of interest -> arXiv codes (what onboarding
    stores) or a flat set of codes, which is treated as one interest per code.
    Balancing per interest rather than per code matters: "Machine Learning"
    spans cs.LG and stat.ML while "Robotics" is only cs.RO, and dealing per code
    gave the two-code interests twice the share. Measured 2026-09-30 for six
    interests, Computer Vision got 22 of 200 starter papers and language-model
    papers 86.

    Within an interest, codes are round-robined and every third slot is a
    recent paper (see _FRESH_EVERY). These are citation/popularity candidates,
    not measured live trends. Without the sidecar, one cached remote query
    serves everything and there is no fresh lane.
    """
    groups = ({k: set(v) for k, v in categories.items() if v}
              if isinstance(categories, Mapping)
              else {code: {code} for code in categories})
    if not groups or limit <= 0:
        return []
    all_codes = set().union(*groups.values())
    if not local_meta.is_available():
        return await turso_svc.fetch_trending_by_categories(all_codes, limit=limit)
    semaphore = asyncio.Semaphore(4)

    async def guarded(coro):
        async with semaphore:
            try:
                return await coro
            except Exception as e:  # one bad lane must not empty the feed
                print(f"[discovery] starter lane failed: {e}")
                return []

    codes = sorted(all_codes)
    names = sorted(groups)
    results = await asyncio.gather(
        *(guarded(turso_svc.fetch_trending_by_categories({c}, limit=limit)) for c in codes),
        *(guarded(turso_svc.fetch_fresh_by_categories(groups[g], limit=limit)) for g in names),
    )
    trending = dict(zip(codes, results[:len(codes)]))
    fresh = dict(zip(names, results[len(codes):]))

    per_group = [
        _with_fresh(_interleave([trending[c] for c in sorted(groups[g])]), fresh[g])
        for g in names
    ]
    seen: set[str] = set()
    papers: list[dict] = []
    for paper in _interleave(per_group):
        aid = paper.get("arxiv_id")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        papers.append(paper)
        if len(papers) == limit:
            break
    return papers
