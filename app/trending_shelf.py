"""
Trending in your interests: Hugging Face Daily Papers on a shelf in the feed.

A separate surface, decided 2026-10-23 (doc 06 changelog). It never takes a
slot from the feed, so the quota rules of CLAUDE.md §3.1 are untouched: the
shelf is one block placed after the fourth card of page 1, holding papers the
feed did not pick.

Why a shelf and not swaps inside the feed: on the owner's profile the shadow
swap policy (discovery_shadow.compare_feed) removed papers at closeness
0.757-0.763 to add ones at 0.628-0.690 with 123-150 upvotes. That trades
closeness for popularity inside slots the reader's interests are entitled to.
The shelf adds the same papers without touching those slots.

Evidence for building it (HF comparison, 23 Oct 2026): 147 of the 383 Daily
Papers from 16-23 Oct were on-topic for the owner, and the feed's 60-paper pool
held 1 of them. 379 of the 383 were already in the corpus with vectors, so
nothing is embedded here; vectors come from Qdrant through the shared cache.

Switch TRENDING_SHELF (app/features.py): off does nothing, shadow refreshes
the papers but renders nothing, preview shows the owner, on shows everyone.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from collections import Counter
from datetime import date as Date, datetime, timedelta, timezone

import httpx
import numpy as np

from app import db, features, hf_papers_svc, http_client, qdrant_svc
from app.recommend.labels import cluster_tone, label_clusters

SWITCH = "TRENDING_SHELF"
POLICY_ID = "trending_shelf_v1"
SOURCE = "trending"
CANDIDATE_SOURCE = "hf_daily"

DAYS = 7                    # Daily Papers lists kept, today included
SHELF_SIZE = 6
SHELF_AFTER_CARD = 4      # page 1 shows this many feed cards before the shelf
PER_INTEREST = 2            # floor on each interest's share of the shelf
MIN_SHOWN = 3               # fewer than this and the shelf stays hidden

# Each tenfold increase in upvotes is worth this much closeness, so popularity
# breaks ties between similar papers without lifting a far one over a near one.
UPVOTE_WEIGHT = 0.07

# On-topic bar: the 25th percentile of how close a reader's own saves sit to
# their interest's medoid (0.586 for the owner on 23 Oct), clamped so a very
# tight or very loose profile cannot shut the shelf or open it to anything.
# Below MIN_FITS_FOR_BAR members (every save its own interest, from 2 to 4
# saves) there is no spread to measure, so the default applies.
DEFAULT_BAR = 0.60
BAR_MIN, BAR_MAX = 0.55, 0.70
MIN_FITS_FOR_BAR = 4

_state: dict = {"papers": {}, "vectors": {}, "code": {}, "refreshed_at": None,
                "days": 0, "failed_days": [], "error": None, "scheduler": "off"}
_task: asyncio.Task | None = None


def _refresh_seconds() -> int:
    return max(600, int(os.getenv("TRENDING_SHELF_REFRESH_S", "10800")))


# ── Source ───────────────────────────────────────────────────────────────────

async def refresh(client: httpx.AsyncClient | None = None, today: Date | None = None) -> dict:
    """Fetch the last DAYS Daily Papers lists and their vectors.

    A day that fails is skipped; if every day fails the previous papers stay
    in place, because an outage is not an empty trend.
    """
    today = today or datetime.now(timezone.utc).date()
    client = client or http_client.get_client()
    papers: dict[str, dict] = {}
    code: dict[str, tuple[str, int]] = {}
    failed: list[str] = []
    for i in range(DAYS):
        day = (today - timedelta(days=i)).isoformat()
        try:
            snap = await hf_papers_svc.fetch_snapshot(client, date=day)
        except (httpx.HTTPError, ValueError) as e:
            print(f"[trending_shelf] Daily Papers {day} failed: {type(e).__name__}")
            failed.append(day)
            continue
        for p in snap["papers"]:
            votes = p["upvotes"] or 0
            prev = papers.get(p["arxiv_id"])
            if prev is None or votes > prev["upvotes"]:
                papers[p["arxiv_id"]] = {"arxiv_id": p["arxiv_id"], "title": p["title"],
                                         "upvotes": votes, "published_at": p["published_at"],
                                         "listed_on": day}
        code.update(snap.get("code") or {})
    if len(failed) == DAYS:
        raise RuntimeError("no Daily Papers list could be fetched")
    vectors = await qdrant_svc.get_paper_vectors(list(papers))
    _state.update(papers=papers, vectors=vectors, code=code, days=DAYS - len(failed),
                  failed_days=failed, error=None,
                  refreshed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return stats()


async def _loop() -> None:
    while True:
        try:
            await refresh()
            _state["scheduler"] = "running"
        except Exception as e:  # keep the last good papers; try again next round
            _state["error"] = f"{type(e).__name__}: {e}"[:200]
            print(f"[trending_shelf] refresh failed: {_state['error']}")
        await asyncio.sleep(_refresh_seconds())


async def start() -> None:
    global _task
    if not features.active(SWITCH) or _task is not None:
        return
    _state["scheduler"] = "starting"
    _task = asyncio.create_task(_loop(), name="trending-shelf")


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
    _task = None


def stats() -> dict:
    with_vec = sum(1 for a in _state["papers"] if _state["vectors"].get(a) is not None)
    return {"switch": features.mode(SWITCH), "scheduler": _state["scheduler"],
            "papers": len(_state["papers"]), "with_vectors": with_vec,
            "days": _state["days"], "failed_days": _state["failed_days"],
            "refreshed_at": _state["refreshed_at"], "error": _state["error"]}


# ── Ranking (pure) ───────────────────────────────────────────────────────────

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def interest_bar(member_fits: list[float]) -> float:
    if len(member_fits) < MIN_FITS_FOR_BAR:
        return DEFAULT_BAR
    return float(min(max(np.percentile(member_fits, 25), BAR_MIN), BAR_MAX))


def rank(candidates: list[dict], vectors: dict, interests: list[dict], bar: float,
         exclude: set[str]) -> list[dict]:
    """The shelf for one reader, best first, or [] when too little is on-topic.

    `interests` carry a unit `medoid` vector, `cluster_idx`, `label`, `tone`.
    Each paper belongs to its closest interest. Every interest gets up to
    max(PER_INTEREST, ceil(SHELF_SIZE / K)) places first, so a dominant interest
    cannot fill the shelf, then any places left go to the next best overall.
    """
    if not interests:
        return []
    medoids = np.stack([_unit(i["medoid"]) for i in interests])
    scored = []
    for p in candidates:
        aid = p["arxiv_id"]
        vec = vectors.get(aid)
        if aid in exclude or vec is None:
            continue
        sims = medoids @ _unit(vec)
        k = int(np.argmax(sims))
        fit = float(sims[k])
        if fit < bar:
            continue
        score = fit + UPVOTE_WEIGHT * math.log10(max(p.get("upvotes") or 0, 1))
        scored.append({**p, "fit": round(fit, 4), "score": score, "interest": interests[k]})
    scored.sort(key=lambda s: -s["score"])

    cap = max(PER_INTEREST, math.ceil(SHELF_SIZE / len(interests)))
    taken, per = [], Counter()
    for s in scored:
        if per[s["interest"]["cluster_idx"]] < cap:
            taken.append(s)
            per[s["interest"]["cluster_idx"]] += 1
        if len(taken) == SHELF_SIZE:
            break
    if len(taken) < SHELF_SIZE:
        chosen = {s["arxiv_id"] for s in taken}
        taken += [s for s in scored if s["arxiv_id"] not in chosen][:SHELF_SIZE - len(taken)]
        taken.sort(key=lambda s: -s["score"])
    return taken if len(taken) >= MIN_SHOWN else []


# ── One reader ───────────────────────────────────────────────────────────────

async def reader_interests(user_id: str) -> tuple[list[dict], float]:
    """The reader's interests as the last feed build saved them, and their bar."""
    rows = await db.get_user_clusters(user_id)
    rows = [r for r in rows if r.get("medoid_embedding_blob") is not None]
    if not rows:
        return [], DEFAULT_BAR
    members = {r["cluster_idx"]: json.loads(r["paper_ids"] or "[]") for r in rows}
    all_ids = sorted({pid for ids in members.values() for pid in ids})
    vecs = await qdrant_svc.get_paper_vectors(all_ids)
    meta = await db.get_cached_metadata_batch(all_ids)
    labels = label_clusters({idx: [meta[p] for p in ids if p in meta] for idx, ids in members.items()})
    interests, fits = [], []
    for r in rows:
        medoid = _unit(np.frombuffer(r["medoid_embedding_blob"], dtype=np.float32))
        idx = r["cluster_idx"]
        for pid in members[idx]:
            if pid != r["medoid_paper_id"] and vecs.get(pid) is not None:
                fits.append(float(_unit(vecs[pid]) @ medoid))
        interests.append({"cluster_idx": idx, "medoid": medoid,
                          "label": labels.get(idx, ""), "tone": cluster_tone(idx)})
    return interests, interest_bar(fits)


async def shelf_for(user_id: str, exclude: set[str],
                    labels: dict[int, tuple[str, int]] | None = None) -> list[dict]:
    """`labels` ({cluster_idx: (name, tone)}) are the names on the feed page
    the shelf sits on; they win over names worked out here."""
    if not _state["papers"]:
        return []
    interests, bar = await reader_interests(user_id)
    for i in interests:
        if labels and i["cluster_idx"] in labels:
            i["label"], i["tone"] = labels[i["cluster_idx"]]
    picked = rank(list(_state["papers"].values()), _state["vectors"], interests, bar, exclude)
    for p in picked:
        repo = _state["code"].get(p["arxiv_id"])
        p["github"], p["stars"] = (repo if repo else ("", 0))
        listed = Date.fromisoformat(p["listed_on"])
        p["listed_label"] = f"{listed.day} {listed:%b}"
    return picked
