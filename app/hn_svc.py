"""
Hacker News on ResearchIT: which papers HN brings up, by how many people and
over how long, from the tables scripts/hn_harvest.py fills each day (doc 06
changelog, 2026-11-03).

The idea is Karpathy's HackerTube (31 Oct 2026): a link counts as a
recommendation once several people have posted it in several different months,
which filters out one-day spikes. DAIR.AI's "Hacker Papers" applied it to
arXiv. Here the same counts serve three surfaces, each behind its own switch
(app/features.py):

  HN_PAPER    the "On Hacker News" block on a paper's page: counts, a strip of
              people per year, the threads where it was discussed
  HN_LASTING  /trending/lasting: the papers HN keeps coming back to, the ones
              closest to the reader first (trending_shelf.rank)
  HN_FRESH    papers HN discussed in the last FRESH_DAYS, on /trending

The counting rule
-----------------
Karpathy's rule is at least 3 people in at least 3 different months. Measured
on every HN link to arXiv (3 Nov 2026), one account wrote half or more of a
paper's mentions for 214 of the 2,078 papers that pass it; one paper passed
because one user linked it 14 times. So a month counts here only when someone
who had not linked the paper before links it ("new-voice months"): one person
linking a paper every month adds one month, not twelve. With that rule 2,022
papers pass, and 86 of the top 100 are the same.

These are counts of links, not of approval. Most papers on HN are cited in an
argument rather than recommended (8 of 10 sampled comments on "markets are
efficient iff P=NP" used it as a rebuttal), so nothing here calls a paper
"recommended"; the words are "brought up" and "linked".

Reads: the mentions and threads, paged by rowid, when the daily job has
written new ones (hn_state.loaded_at), polled every REFRESH_S. About 50,000 +
30,000 rows a load, once a day and at each start.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone

from app import features, qdrant_svc, turso_svc

SWITCH_PAPER = "HN_PAPER"
SWITCH_LASTING = "HN_LASTING"
SWITCH_FRESH = "HN_FRESH"
SWITCHES = (SWITCH_PAPER, SWITCH_LASTING, SWITCH_FRESH)

MIN_PEOPLE = 3              # Karpathy's rule, with months counted by new voices
MIN_MONTHS = 3
FRESH_DAYS = 30
FRESH_PEOPLE = 2            # a fresh paper: this many people in FRESH_DAYS ...
FRESH_POINTS = 20           # ... or submitted as a story that reached this many points
SUBMISSION_FIRST = 10       # a submission with fewer points is listed after the discussions
PAGE_ROWS = 5000
VECTOR_BATCH = 500
THREADS_SHOWN = 3

_state: dict = {"papers": {}, "threads": {}, "canon": [], "canon_rank": {}, "fresh": [], "cards": [],
                "meta": {}, "vectors": {}, "loaded_at": None, "refreshed_at": None,
                "rows": 0, "error": None, "scheduler": "off"}
_task: asyncio.Task | None = None


def _refresh_seconds() -> int:
    return max(300, int(os.getenv("HN_REFRESH_S", "3600")))


# ── Counting (pure) ──────────────────────────────────────────────────────────

def aggregate(rows: list[tuple], threads: dict[str, dict], now: int) -> dict[str, dict]:
    """Per-paper counts from mention rows (arxiv_id, item_id, kind, author,
    created_at, story_id). Mentions by a deleted account (no author) count as
    mentions but never as a person."""
    by_paper: dict[str, list[tuple]] = defaultdict(list)
    for r in rows:
        by_paper[r[0]].append(r)
    cutoff = now - FRESH_DAYS * 86400
    out = {}
    for aid, ms in by_paper.items():
        ms.sort(key=lambda m: (m[4], m[1]))
        people: set[str] = set()
        voice: set[tuple[int, int]] = set()
        months: set[tuple[int, int]] = set()
        years: dict[int, set[str]] = defaultdict(set)
        per_thread: Counter = Counter()
        submitted: set[str] = set()
        recent_people: set[str] = set()
        recent_points, recent_last = 0, 0
        for _aid, _item, kind, author, ts, story in ms:
            d = datetime.fromtimestamp(ts, timezone.utc)
            month = (d.year, d.month)
            months.add(month)
            if author:
                if author not in people:
                    people.add(author)
                    voice.add(month)
                years[d.year].add(author)
            per_thread[story] += 1
            if kind == "story":
                submitted.add(story)
            if ts >= cutoff:
                recent_last = max(recent_last, ts)
                if author:
                    recent_people.add(author)
                if kind == "story":
                    recent_points = max(recent_points, (threads.get(story) or {}).get("points") or 0)
        out[aid] = {
            "arxiv_id": aid, "people": len(people), "months": len(voice), "any_months": len(months),
            "mentions": len(ms), "threads": len(per_thread), "first": ms[0][4], "last": ms[-1][4],
            "years": {y: len(s) for y, s in years.items()}, "per_thread": per_thread,
            "submitted": submitted, "recent_people": len(recent_people), "recent_points": recent_points,
            "recent_last": recent_last,
            "lasting": len(people) >= MIN_PEOPLE and len(voice) >= MIN_MONTHS,
        }
    return out


def canon_order(papers: dict[str, dict]) -> list[str]:
    """Lasting papers, most new-voice months first, then most people, then
    most threads."""
    keep = [p for p in papers.values() if p["lasting"]]
    keep.sort(key=lambda p: (-p["months"], -p["people"], -p["threads"], p["arxiv_id"]))
    return [p["arxiv_id"] for p in keep]


def is_fresh(p: dict) -> bool:
    return p["recent_last"] > 0 and (p["recent_people"] >= FRESH_PEOPLE or p["recent_points"] >= FRESH_POINTS)


def top_threads(p: dict, threads: dict[str, dict], n: int = THREADS_SHOWN) -> list[dict]:
    """The threads to show for a paper: submissions that drew attention
    (SUBMISSION_FIRST points or more) by points, then the discussions that
    linked it most, then submissions nobody noticed."""
    out = []
    for story, k in p["per_thread"].items():
        t = threads.get(story) or {}
        sub = story in p["submitted"]
        out.append({"story_id": story, "title": t.get("title") or "", "points": t.get("points"),
                    "comments": t.get("comments"), "created_at": t.get("created_at"),
                    "submitted": sub, "mentions": k})

    def key(x):
        pts = x["points"] or 0
        if x["submitted"] and pts >= SUBMISSION_FIRST:
            return (0, -pts, 0, x["story_id"])
        if not x["submitted"]:
            return (1, -x["mentions"], -pts, x["story_id"])
        return (2, -pts, 0, x["story_id"])

    out.sort(key=key)
    for x in out[:n]:
        ts = x["created_at"]
        x["when"] = datetime.fromtimestamp(ts, timezone.utc).strftime("%b %Y") if ts else ""
        x["url"] = f"https://news.ycombinator.com/item?id={x['story_id']}"
    return out[:n]


def year_strip(p: dict) -> list[dict]:
    """People who linked the paper in each year from its first to its last,
    empty years included, with each bar's height as a share of the busiest."""
    if not p["years"]:
        return []
    first, last = min(p["years"]), max(p["years"])
    top = max(p["years"].values())
    return [{"year": y, "people": p["years"].get(y, 0),
             "h": round(100 * p["years"].get(y, 0) / top) if top else 0} for y in range(first, last + 1)]


def facts(p: dict) -> dict:
    """The words a card or a row uses, from the counts."""
    first = datetime.fromtimestamp(p["first"], timezone.utc).year
    last = datetime.fromtimestamp(p["last"], timezone.utc).year
    since = datetime.fromtimestamp(p["first"], timezone.utc).strftime("%b %Y")
    return {"people": p["people"], "months": p["months"], "threads": p["threads"],
            "span": f"{first}" if first == last else f"{first}–{last}", "first_year": first, "last_year": last,
            "since": since}


# ── Loading ──────────────────────────────────────────────────────────────────

async def _query(sql: str, args: tuple = ()) -> list[dict]:
    """Turso, or the SQLite file named by HN_SQLITE: a copy of the same tables
    (scripts/hn_harvest.py can fill one) for local runs and browser tests, so
    neither reads production."""
    path = os.getenv("HN_SQLITE", "").strip()
    if not path:
        return await turso_svc.query(sql, args, timeout=60)

    def read() -> list[dict]:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute(sql, args).fetchall()]
        finally:
            con.close()
    try:
        return await asyncio.to_thread(read)
    except sqlite3.Error as e:
        raise RuntimeError(f"HN_SQLITE: {e}") from e


async def _paged(sql: str) -> list[dict]:
    """Every row of `sql` (which must take `rowid > ?` and LIMIT ?), PAGE_ROWS at a time."""
    out, after = [], 0
    while True:
        page = await _query(sql, (after, PAGE_ROWS))
        out += page
        if len(page) < PAGE_ROWS:
            return out
        after = page[-1]["rid"]


async def refresh(force: bool = False) -> dict:
    """Load the mentions when the daily job has written new ones. Raises on a
    failed read, keeping the previous counts."""
    got = await _query("SELECT value FROM hn_state WHERE key = 'loaded_at'")
    loaded_at = got[0]["value"] if got else None
    if loaded_at is None:
        raise RuntimeError("no Hacker News harvest recorded yet")
    if loaded_at == _state["loaded_at"] and not force:
        return stats()
    mention_rows = await _paged("SELECT rowid AS rid, arxiv_id, item_id, kind, author, created_at, story_id "
                                "FROM hn_mentions WHERE rowid > ? ORDER BY rowid LIMIT ?")
    thread_rows = await _paged("SELECT rowid AS rid, story_id, title, points, comments, created_at "
                               "FROM hn_threads WHERE rowid > ? ORDER BY rowid LIMIT ?")
    threads = {r["story_id"]: {"title": r["title"] or "", "points": r["points"], "comments": r["comments"],
                               "created_at": r["created_at"]} for r in thread_rows}
    rows = [(r["arxiv_id"], r["item_id"], r["kind"], r["author"] or "", int(r["created_at"]), r["story_id"])
            for r in mention_rows]
    now = int(datetime.now(timezone.utc).timestamp())
    papers = aggregate(rows, threads, now)
    canon = canon_order(papers)
    fresh = sorted((a for a, p in papers.items() if is_fresh(p)),
                   key=lambda a: (-papers[a]["recent_people"], -papers[a]["recent_points"], a))
    meta, vectors = await _describe(canon + fresh)
    listed = [a for a in canon if a in meta]
    _state.update(papers=papers, threads=threads, canon=listed,
                  canon_rank={a: i + 1 for i, a in enumerate(listed)},
                  canon_all=len(canon), fresh=[a for a in fresh if a in meta], meta=meta, vectors=vectors,
                  loaded_at=loaded_at, rows=len(rows), error=None,
                  refreshed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    # Built once a load, not per request: the whole list is ~2,000 entries.
    _state["cards"] = [c for c in (card(a, "lasting") for a in listed) if c is not None]
    return stats()


def id_year(arxiv_id: str) -> int | None:
    """The year a paper first appeared, from its id: 1706.03762 -> 2017,
    hep-th/9711200 -> 1997. The stored year follows the latest version
    (Attention Is All You Need reads 2023), which is wrong on a list of
    papers people kept returning to."""
    digits = arxiv_id.split("/")[-1][:2]
    if not digits.isdigit():
        return None
    yy = int(digits)
    return 1900 + yy if "/" in arxiv_id and yy >= 91 else 2000 + yy


async def _describe(ids: list[str]) -> tuple[dict, dict]:
    """Titles, authors, years and categories (sidecar first, then Turso by key)
    and vectors (Qdrant, not kept in the shared cache) for the papers the lists
    can show. A paper the corpus lacks has no metadata and is left off the lists."""
    ids = list(dict.fromkeys(ids))
    meta: dict[str, dict] = {}
    try:
        got = await turso_svc.fetch_metadata_batch(ids)
    except Exception as e:  # pragma: no cover - fetch_metadata_batch reports and returns {}
        print(f"[hn] metadata unavailable: {type(e).__name__}")
        got = {}
    for aid, m in got.items():
        if not m.get("title"):
            continue
        meta[aid] = {"title": m["title"], "authors": m.get("authors") or "", "year": id_year(aid) or m.get("year") or "",
                     "categories": (m.get("arxiv_categories") or "").split(),
                     "citation_count": m.get("citation_count") or 0}
    vectors: dict = {}
    have = [a for a in ids if a in meta]
    for i in range(0, len(have), VECTOR_BATCH):
        try:
            vectors.update(await qdrant_svc.get_paper_vectors(have[i:i + VECTOR_BATCH], remember=False))
        except Exception as e:  # lists still work for everyone; only "for you" needs vectors
            print(f"[hn] vectors unavailable: {type(e).__name__}")
            break
    return meta, vectors


async def _loop() -> None:
    while True:
        try:
            await refresh()
            _state["scheduler"] = "running"
        except Exception as e:  # keep the last good counts; try again next round
            _state["error"] = f"{type(e).__name__}: {e}"[:200]
            print(f"[hn] refresh failed: {_state['error']}")
        await asyncio.sleep(_refresh_seconds())


async def start() -> None:
    global _task
    if not any(features.active(s) for s in SWITCHES) or _task is not None:
        return
    _state["scheduler"] = "starting"
    _task = asyncio.create_task(_loop(), name="hn-mentions")


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
    return {"switches": {s: features.mode(s) for s in SWITCHES}, "scheduler": _state["scheduler"],
            "papers": len(_state["papers"]), "mentions": _state["rows"], "threads": len(_state["threads"]),
            "lasting": len(_state["canon"]), "lasting_all": _state.get("canon_all", 0),
            "fresh": len(_state["fresh"]), "with_vectors": len(_state["vectors"]),
            "loaded_at": _state["loaded_at"], "refreshed_at": _state["refreshed_at"], "error": _state["error"]}


# ── What the surfaces read ───────────────────────────────────────────────────

def paper_block(arxiv_id: str) -> dict | None:
    """Everything the paper page's block shows, or None when HN never linked it."""
    p = _state["papers"].get(arxiv_id)
    if p is None:
        return None
    return {**facts(p), "mentions": p["mentions"], "strip": year_strip(p),
            "threads_shown": top_threads(p, _state["threads"]),
            "lasting": p["lasting"], "rank": _state["canon_rank"].get(arxiv_id),
            "of": len(_state["canon"]), "submitted": bool(p["submitted"])}


def card(arxiv_id: str, kind: str) -> dict | None:
    """A list entry: the paper's metadata and its HN facts, plus its best thread.
    `kind` is "lasting" or "fresh"; a fresh entry counts only recent activity."""
    p, m = _state["papers"].get(arxiv_id), _state["meta"].get(arxiv_id)
    if p is None or m is None:
        return None
    best = top_threads(p, _state["threads"], 1)
    out = {"arxiv_id": arxiv_id, **m, "hn": facts(p), "thread": best[0] if best else None,
           "rank": _state["canon_rank"].get(arxiv_id), "kind": kind,
           # trending_shelf.rank's popularity term: tenfold more people is
           # worth UPVOTE_WEIGHT of closeness.
           "people": p["recent_people"] if kind == "fresh" else p["people"]}
    if kind == "fresh":
        out["recent_people"], out["recent_points"] = p["recent_people"], p["recent_points"]
    return out


def lasting_cards(area_codes: frozenset[str] | None = None) -> list[dict]:
    """Every listed lasting paper in canon order, optionally only those with a
    category in `area_codes`. Copies: callers add per-request fields."""
    return [dict(c) for c in _state.get("cards") or []
            if area_codes is None or set(c["categories"]) & area_codes]


def fresh_cards() -> list[dict]:
    return [c for c in (card(a, "fresh") for a in _state["fresh"]) if c is not None]


def vectors() -> dict:
    return _state["vectors"]


def area_counts(groups: dict[str, dict]) -> list[tuple[str, str, int]]:
    """(key, name, lasting papers in it) for every area with at least one, biggest first."""
    out = []
    for key, g in groups.items():
        codes = set(g["arxiv"])
        n = sum(1 for a in _state["canon"] if set(_state["meta"][a]["categories"]) & codes)
        if n:
            out.append((key, g["name"], n))
    return sorted(out, key=lambda t: (-t[2], t[1]))
