"""The weekly digest: new papers in a reader's interests, by link, RSS or email.

Products people come back to send a list on a schedule (Scholar Inbox's digest
was its users' favourite feature; HF Daily Papers mails 12k readers). This is
that list for a reader with no account:

    build(user_id, days)   the papers, grouped by interest
    GET /digest            the reader's own digest, with its private link,
                           RSS feed and (when email is configured) a sign-up
    GET /digest/<token>    the same digest from anywhere, by its private link

How papers are picked (doc 06 rules, as the feed applies them):
- Interests are the reader's stored clusters (user_clusters: Ward medoids,
  replicated), so the digest names the same interests the feed does.
- Each interest searches the recent shard by its medoid and keeps papers
  first published in the window (default the last 7 days), leaving out what
  the reader saved, passed on or opened.
- Slots are split by importance-weighted quota with a floor
  (recommend.fusion.allocate_quotas), never RRF, so a side interest keeps
  its share. Within an interest, order is nearness to the medoid.
- A reader with no interests yet gets the week's most upvoted Hugging Face
  papers in the areas they picked, else across all areas, labelled as such.

No model is called. Results are kept for three hours per reader and window.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from app import db, qdrant_svc, trending_shelf, turso_svc
from app import user_state as us
from app.recommend.fusion import allocate_quotas
from app.recommend.labels import label_clusters

DIGEST_SIZE = 10
POLICY = "digest_v1"
# Nearest recent papers looked at per interest. The recent shard holds every
# paper since June 2025, so a week's papers are ~2% of it: at depth 80 a real
# reader's three interests found 1, 0 and 0 papers from the last 7 days; at
# 1000, 11, 12 and 18 (2026-11-27). Hits are cut to the window's arXiv months
# (2611.xxxxx) before any metadata is read, so depth costs ~1.3 s of search.
SEARCH_DEPTH = 1000
CACHE_S = 3 * 3600
_cache: dict[tuple, tuple[float, dict]] = {}


def _first_sentence(text: str, limit: int = 200) -> str:
    from app.routers.share import first_sentence
    return first_sentence(text, limit)


def _authors(paper: dict) -> str:
    from app.routers.share import authors_of
    from app.share_cards import authors_line
    return authors_line(authors_of(paper))


def _card(p: dict) -> dict:
    return {"arxiv_id": p["arxiv_id"], "title": p.get("title") or p["arxiv_id"],
            "authors": _authors(p), "line": _first_sentence(p.get("abstract") or p.get("summary") or ""),
            "published": (p.get("published") or p.get("listed_on") or "")[:10],
            "category": p.get("category") or ""}


def window_months(now: datetime, days: int) -> set[str]:
    """arXiv id prefixes (YYMM) a paper first published in the window can have.
    The id is given at submission, which can come a few days before
    publication, so the month before the window's start is included too."""
    out, d = set(), now - timedelta(days=days + 7)
    while d <= now:
        out.add(d.strftime("%y%m"))
        d += timedelta(days=1)
    return out


def floor_for(n_interests: int, size: int = DIGEST_SIZE) -> int:
    """The per-interest floor: doc 06's F_min = 3, lowered when that many
    interests at 3 each would overrun the digest (7 interests, 10 slots)."""
    return max(1, min(3, size // max(1, n_interests)))


async def _interests(user_id: str) -> list[dict]:
    """The reader's stored interests, most important first, with medoid vectors."""
    out = []
    for row in await db.get_user_clusters(user_id):
        blob = row.get("medoid_embedding_blob")
        if blob is None:
            continue
        try:
            ids = json.loads(row.get("paper_ids") or "[]")
        except (TypeError, ValueError):
            ids = []
        out.append({"idx": row["cluster_idx"], "importance": float(row["importance"] or 0),
                    "vector": np.frombuffer(blob, dtype=np.float32).copy(), "paper_ids": ids})
    return out


async def _labels(interests: list[dict]) -> dict[int, str]:
    ids = [pid for c in interests for pid in c["paper_ids"]]
    try:
        meta = await db.get_cached_metadata_batch(ids) if ids else {}
        missing = [i for i in ids if i not in meta]
        if missing:            # the local cache misses papers saved before a deploy
            meta.update(await turso_svc.fetch_metadata_batch(missing))
        corpus = {c["idx"]: [meta[p] for p in c["paper_ids"] if p in meta] for c in interests}
        return label_clusters({k: v for k, v in corpus.items() if v})
    except Exception as e:     # labels are decoration
        print(f"[digest] labelling failed: {type(e).__name__}: {e}")
        return {}


async def build(user_id: str, days: int = 7, size: int = DIGEST_SIZE, now: datetime | None = None) -> dict:
    """{"kind", "sections": [{"label", "papers": [...]}], "count", "days", "since", "query_id"}.

    kind: "interests" (from the reader's clusters), "areas" (the week's top
    papers in their picked areas) or "week" (the week's top papers overall)."""
    now = now or datetime.now(timezone.utc)
    key = (user_id, days, size, now.strftime("%Y-%m-%d"))
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_S:
        return hit[1]
    since = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    await us.ensure_loaded(user_id)
    seen = us.all_seen(user_id)
    out = {"kind": "interests", "sections": [], "count": 0, "days": days, "since": since,
           "query_id": str(uuid.uuid4())}

    interests = await _interests(user_id)
    if interests:
        results = await asyncio.gather(
            *[qdrant_svc.search_recent(c["vector"].tolist(), limit=SEARCH_DEPTH, exclude_ids=seen)
              for c in interests], return_exceptions=True)
        months = window_months(now, days)
        ids = list(dict.fromkeys(h["arxiv_id"] for r in results if isinstance(r, list) for h in r
                                 if h["arxiv_id"][:4] in months))
        meta = await turso_svc.fetch_metadata_batch(ids) if ids else {}
        fresh = {a for a in ids if a in meta and (meta[a].get("published") or "")[:10] >= since}
        quotas = allocate_quotas([c["importance"] for c in interests], total_slots=size,
                                 min_slots=floor_for(len(interests), size))
        labels = await _labels(interests)
        taken: set[str] = set()
        for c, r, slots in zip(interests, results, quotas):
            if not isinstance(r, list):
                print(f"[digest] interest {c['idx']} search failed: {r!r}")
                continue
            picks = [h["arxiv_id"] for h in r if h["arxiv_id"] in fresh and h["arxiv_id"] not in taken][:slots]
            taken.update(picks)
            if picks:
                out["sections"].append({"label": labels.get(c["idx"]) or "One of your interests",
                                        "source": f"digest_cluster_{c['idx']}", "cluster_id": c["idx"],
                                        "papers": [_card(meta[a]) for a in picks]})
    else:
        state = await db.get_onboarding_state(user_id) or {}
        areas = state.get("selected_categories") or []
        if areas:
            for group in trending_shelf.by_area(areas, exclude=seen, n=max(2, size // max(1, len(areas)))):
                if group["papers"]:
                    out["sections"].append({"label": f"{group['label']}: most upvoted this week",
                                            "source": "digest_area", "cluster_id": None,
                                            "papers": [_card(p) for p in group["papers"]]})
            out["kind"] = "areas"
        if not out["sections"]:
            week = trending_shelf.week_top(size, exclude=seen)
            if week:
                out["sections"].append({"label": "Most upvoted on Hugging Face this week",
                                        "source": "digest_week", "cluster_id": None,
                                        "papers": [_card(p) for p in week]})
            out["kind"] = "week"

    position = 0
    for s in out["sections"]:
        for p in s["papers"]:
            p["position"] = position
            position += 1
    out["count"] = position
    _cache[key] = (time.monotonic(), out)
    if len(_cache) > 2000:
        for k in sorted(_cache, key=lambda k: _cache[k][0])[:500]:
            _cache.pop(k, None)
    return out


def exposures(user_id: str, digest: dict) -> list[dict]:
    """feed_exposures rows for a digest a reader was shown (§3.11)."""
    return [{"user_id": user_id, "paper_id": p["arxiv_id"], "query_id": digest["query_id"],
             "position": p["position"], "tier": None, "candidate_source": s["source"],
             "cluster_id": s["cluster_id"], "propensity": 1.0, "policy_id": POLICY}
            for s in digest["sections"] for p in s["papers"]]
