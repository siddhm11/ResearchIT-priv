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

Since 2026-10-25 (doc 06) it is there whenever the week has papers: a paper
also qualifies by being listed in an area the reader picked, papers shown
before fill places after new ones, and a reader nothing fits gets the week's
most upvoted. It shows on every feed tier, not only the clustered one.

Switch TRENDING_SHELF (app/features.py): off does nothing, shadow refreshes
the papers but renders nothing, preview shows the owner, on shows everyone.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from bisect import bisect_left
from collections import Counter
from datetime import date as Date, datetime, timedelta, timezone

import httpx
import numpy as np

from app import config, db, features, hf_papers_svc, http_client, qdrant_svc, turso_svc
from app.recommend.labels import cluster_tone, label_clusters

SWITCH = "TRENDING_SHELF"
SEARCH_SWITCH = "SEARCH_TRENDING"   # hybrid_search_svc reads the same lists
POLICY_ID = "trending_shelf_v1"
SOURCE = "trending"
CANDIDATE_SOURCE = "hf_daily"

DAYS = 7                    # Daily Papers lists kept, today included
FRESH_DAYS = 3              # days a paper counts fully as trending in search
SHELF_SIZE = 6
SHELF_AFTER_CARD = 4      # page 1 shows this many feed cards before the shelf
PER_INTEREST = 2            # floor on each interest's share of the shelf
MIN_SHOWN = 3               # fewer fit than this and the shelf shows the week's top instead

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

# A paper can also qualify by its arXiv categories alone, when it is listed in
# one of the areas the reader picked (onboarding, /interests), so a reader with
# few or no saves still gets a shelf (doc 06, 2026-10-25). Such a paper scores
# this much before upvotes: below any interest fit that clears the bar (0.55 at
# the lowest), so saves lead and the areas fill the places they leave.
AREA_SCORE = 0.50

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
        extra = snap.get("extra") or {}
        listed = [p for p in snap["papers"] if p["arxiv_id"] not in papers]
        listed.sort(key=lambda p: -(p["upvotes"] or 0))
        for rank, p in enumerate(listed, start=1):
            papers[p["arxiv_id"]] = {"arxiv_id": p["arxiv_id"], "title": p["title"],
                                     "upvotes": p["upvotes"] or 0, "published_at": p["published_at"],
                                     "listed_on": day, "day_rank": rank, "day_size": len(listed),
                                     **extra.get(p["arxiv_id"], {"keywords": [], "summary": "", "comments": 0})}
        code.update(snap.get("code") or {})
    if len(failed) == DAYS:
        raise RuntimeError("no Daily Papers list could be fetched")
    _place_in_week(papers)
    vectors = await qdrant_svc.get_paper_vectors(list(papers))
    # Categories for matching the reader's areas: sidecar first, Turso for the
    # newest papers, a few hundred ids by key. Without them only saves match.
    try:
        meta = await turso_svc.fetch_metadata_batch(list(papers))
    except Exception as e:  # pragma: no cover - fetch_metadata_batch reports and returns {}
        print(f"[trending_shelf] categories unavailable: {type(e).__name__}")
        meta = {}
    for aid, p in papers.items():
        p["categories"] = ((meta.get(aid) or {}).get("arxiv_categories") or "").split()
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
    if not (features.active(SWITCH) or features.active(SEARCH_SWITCH)) or _task is not None:
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


# ── Why a paper is trending (pure) ───────────────────────────────────────────
#
# Every claim is a count read off Hugging Face's own lists, never a model's
# opinion: where the paper placed on the day it was listed, and how its upvotes
# compare with every other paper of the week. Upvotes arrive in a paper's first
# day or two on the list (measured on 26 weeks, 24 Oct 2026), so "this week" is
# the window in which a trend still means something.

def _place_in_week(papers: dict[str, dict]) -> None:
    """Each paper's share of the week's papers it has more upvotes than."""
    votes = sorted(p["upvotes"] for p in papers.values())
    n = len(votes)
    for p in papers.values():
        below = bisect_left(votes, p["upvotes"])
        p["week_size"] = n
        p["week_percentile"] = below / max(n - 1, 1)


def trend_facts(p: dict) -> list[str]:
    """Short, checkable reasons this paper is trending, strongest first."""
    day = Date.fromisoformat(p["listed_on"])
    when = f"{day.day} {day:%b}"
    facts = []
    rank, size = p.get("day_rank") or 0, p.get("day_size") or 0
    if rank == 1 and size > 1:
        facts.append(f"Most upvoted of {size} papers on {when}")
    elif rank and size:
        facts.append(f"#{rank} of {size} papers on {when}")
    top = 1 - p.get("week_percentile", 0)
    if p.get("week_size", 0) >= 20 and top <= 0.25:
        pct = max(1, math.ceil(top * 100))
        facts.append(f"top {pct}% of this week's {p['week_size']}")
    return facts


def freshness(listed_on: str, today: Date) -> float:
    """1 for a paper's first FRESH_DAYS days on the list, then down to 0 at DAYS."""
    age = (today - Date.fromisoformat(listed_on)).days
    if age < FRESH_DAYS:
        return 1.0
    return max(0.0, 1 - (age - FRESH_DAYS + 1) / (DAYS - FRESH_DAYS + 1))


def trending_now(today: Date | None = None) -> dict[str, dict]:
    """The week's papers for search's temporary boost: each with its place in the
    week (`strength`, 0 to 1), how fresh it still is (`fresh`) and its facts."""
    today = today or datetime.now(timezone.utc).date()
    out = {}
    for aid, p in _state["papers"].items():
        fresh = freshness(p["listed_on"], today)
        if fresh > 0:
            out[aid] = {"strength": p.get("week_percentile", 0.0), "fresh": fresh,
                        "upvotes": p["upvotes"], "facts": trend_facts(p)}
    return out


# ── Ranking (pure) ───────────────────────────────────────────────────────────

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def interest_bar(member_fits: list[float]) -> float:
    if len(member_fits) < MIN_FITS_FOR_BAR:
        return DEFAULT_BAR
    return float(min(max(np.percentile(member_fits, 25), BAR_MIN), BAR_MAX))


def area_groups(keys: list[str], first_tone: int = 0) -> list[dict]:
    """The reader's chosen areas (CATEGORY_GROUPS keys) as shelf groups."""
    out = []
    for key in keys:
        grp = config.CATEGORY_GROUPS.get(key)
        if grp:
            out.append({"cluster_idx": None, "area": key, "codes": frozenset(grp["arxiv"]),
                        "label": grp["name"], "tone": cluster_tone(first_tone + len(out))})
    return out


def _area_of(categories: list[str], areas: list[dict]) -> dict | None:
    """The area a paper belongs to: by its primary category, else any it lists."""
    if not categories:
        return None
    for a in areas:
        if categories[0] in a["codes"]:
            return a
    listed = set(categories)
    return next((a for a in areas if listed & a["codes"]), None)


def _round_robin(pool: list[dict], n: int) -> list[dict]:
    """Up to n of `pool` (best first), one per group in turn, best group first,
    so no interest or area takes the shelf while another has papers."""
    groups: dict[int, list[dict]] = {}
    for s in pool:
        groups.setdefault(id(s["interest"]), []).append(s)
    queues = sorted(groups.values(), key=lambda g: -g[0]["score"])
    taken: list[dict] = []
    while len(taken) < n and any(queues):
        for q in queues:
            if q and len(taken) < n:
                taken.append(q.pop(0))
    return taken


def rank(candidates: list[dict], vectors: dict, interests: list[dict], bar: float,
         exclude: set[str], size: int = SHELF_SIZE, areas: list[dict] = (),
         seen: set[str] = frozenset()) -> list[dict]:
    """The shelf for one reader, best first, or [] when fewer than MIN_SHOWN qualify.

    A paper qualifies by closeness to one of the reader's saved interests (its
    medoid, at `bar` or above) or, failing that, by being listed in one of the
    reader's chosen `areas`. `interests` carry a unit `medoid`, `cluster_idx`,
    `label`, `tone`. Places go round the interests and areas, so a dominant
    interest or a big area cannot fill the shelf. `exclude` is never shown;
    papers in `seen` (shown before) come after every unseen one, filling places
    rather than leaving the shelf empty for a reader who has seen the week's
    best already.
    """
    if not interests and not areas:
        return []
    medoids = np.stack([_unit(i["medoid"]) for i in interests]) if interests else None
    scored = []
    for p in candidates:
        aid = p["arxiv_id"]
        if aid in exclude:
            continue
        votes = UPVOTE_WEIGHT * math.log10(max(p.get("upvotes") or 0, 1))
        vec = vectors.get(aid)
        if medoids is not None and vec is not None:
            sims = medoids @ _unit(vec)
            k = int(np.argmax(sims))
            fit = float(sims[k])
            if fit >= bar:
                scored.append({**p, "fit": round(fit, 4), "score": fit + votes, "interest": interests[k]})
                continue
        area = _area_of(p.get("categories") or [], list(areas))
        if area is not None:
            scored.append({**p, "fit": None, "score": AREA_SCORE + votes, "interest": area})
    if len(scored) < MIN_SHOWN:
        return []
    scored.sort(key=lambda s: -s["score"])
    fresh = _round_robin([s for s in scored if s["arxiv_id"] not in seen], size)
    again = _round_robin([s for s in scored if s["arxiv_id"] in seen], size - len(fresh))
    return sorted(fresh, key=lambda s: -s["score"]) + sorted(again, key=lambda s: -s["score"])


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


async def reader_areas(user_id: str) -> list[str]:
    """The areas the reader picked (CATEGORY_GROUPS keys), in their order."""
    state = await db.get_onboarding_state(user_id)
    return [k for k in (state or {}).get("selected_categories") or [] if isinstance(k, str)]


async def shelf_for(user_id: str, exclude: set[str],
                    labels: dict[int, tuple[str, int]] | None = None,
                    size: int = SHELF_SIZE, seen: set[str] = frozenset(),
                    fallback: bool = False) -> list[dict]:
    """`labels` ({cluster_idx: (name, tone)}) are the names on the feed page
    the shelf sits on; they win over names worked out here.

    With `fallback`, a reader nothing qualifies for gets the week's most
    upvoted instead (cards without an `interest`), so the shelf is there
    whenever the week has papers: weekends and readers without saves included.
    """
    if not _state["papers"]:
        return []
    interests, bar = await reader_interests(user_id)
    for i in interests:
        if labels and i["cluster_idx"] in labels:
            i["label"], i["tone"] = labels[i["cluster_idx"]]
    first_tone = 1 + max((i["cluster_idx"] for i in interests), default=-1)
    areas = area_groups(await reader_areas(user_id), first_tone)
    picked = rank(list(_state["papers"].values()), _state["vectors"], interests, bar, exclude,
                  size, areas=areas, seen=seen)
    if not picked and fallback:
        return week_top(size, exclude=exclude, seen=seen)
    return [_for_card(p) for p in picked]


def window(today: Date | None = None) -> dict | None:
    """Which lists the week's papers come from, in words, for the page header:
    {"label": "19–23 Oct", "next": "Mon 26 Oct" or None}. "next" is set at
    weekends, when no list arrives until Monday, so the page can say so instead
    of looking stale."""
    days = sorted({p["listed_on"] for p in _state["papers"].values()})
    if not days:
        return None
    first, last = Date.fromisoformat(days[0]), Date.fromisoformat(days[-1])
    if first.month == last.month:
        label = f"{first.day}–{last.day} {last:%b}" if first != last else f"{last.day} {last:%b}"
    else:
        label = f"{first.day} {first:%b} – {last.day} {last:%b}"
    today = today or datetime.now(timezone.utc).date()
    nxt = None
    if today.weekday() >= 5:
        monday = today + timedelta(days=7 - today.weekday())
        nxt = f"{monday:%a} {monday.day} {monday:%b}"
    return {"label": label, "next": nxt}


def week_top(n: int = 10, exclude: set[str] = frozenset(), seen: set[str] = frozenset()) -> list[dict]:
    """The week's most upvoted papers, for everyone (the /trending page). Papers
    in `seen` come after the rest."""
    papers = sorted((p for a, p in _state["papers"].items() if a not in exclude),
                    key=lambda p: (p["arxiv_id"] in seen, -p["upvotes"], p["arxiv_id"]))
    return [_for_card(dict(p)) for p in papers[:n]]


def by_area(keys: list[str], exclude: set[str] = frozenset(), n: int = 3) -> list[dict]:
    """For each chosen area, how many of the week's papers are listed in it and
    its n most upvoted (the /trending page). Areas with none are left out."""
    out = []
    for g in area_groups(keys):
        mine = [p for p in _state["papers"].values() if set(p.get("categories") or ()) & g["codes"]]
        if not mine:
            continue
        best = sorted((p for p in mine if p["arxiv_id"] not in exclude),
                      key=lambda p: (-p["upvotes"], p["arxiv_id"]))[:n]
        out.append({"area": g["area"], "label": g["label"], "count": len(mine),
                    "papers": [_for_card(dict(p)) for p in best]})
    return out


def _for_card(p: dict) -> dict:
    repo = _state["code"].get(p["arxiv_id"])
    p["github"], p["stars"] = (repo if repo else ("", 0))
    listed = Date.fromisoformat(p["listed_on"])
    p["listed_label"] = f"{listed.day} {listed:%b}"
    p["facts"] = trend_facts(p)
    return p
