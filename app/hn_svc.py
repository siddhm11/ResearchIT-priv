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
39,000 rows a load, once a day and at each start, plus the ~23,000 comments
that link at most QUOTE_MAX_PAPERS papers, read to pick quotes and then
dropped (doc 06, 2026-11-17).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone

from app import arxiv_svc, features, qdrant_svc, turso_svc

SWITCH_PAPER = "HN_PAPER"
SWITCH_LASTING = "HN_LASTING"
SWITCH_FRESH = "HN_FRESH"
SWITCHES = (SWITCH_PAPER, SWITCH_LASTING, SWITCH_FRESH)

MIN_PEOPLE = 3              # Karpathy's rule, with months counted by new voices
MIN_MONTHS = 3
FRESH_DAYS = 30
FRESH_PEOPLE = 2            # a fresh paper: this many people in FRESH_DAYS ...
FRESH_POINTS = 20           # ... or submitted as a story that reached this many points
SUBMISSION_FIRST = 50       # a submission with fewer points is listed after the discussions
PAGE_ROWS = 5000
VECTOR_BATCH = 500
ARXIV_FILL_MAX = 400         # lasting papers the corpus lacks, described from the arXiv API instead
ARXIV_FILL_SECONDS = 90
THREADS_SHOWN = 3
WEEK_DAYS = 7               # "Back on HN this week"
WEEK_SHOWN = 8
COMEBACK_DAYS = 365         # a comeback: linked lately after at least this long without a link
JOINED_DAYS = 92            # "Just joined": became lasting in about the last three months

_state: dict = {"papers": {}, "threads": {}, "canon": [], "canon_rank": {}, "fresh": [], "cards": [],
                "quotes": {}, "strata": {"lines": [], "months": 0},
                "meta": {}, "vectors": {}, "loaded_at": None, "refreshed_at": None,
                "rows": 0, "error": None, "scheduler": "off"}
_task: asyncio.Task | None = None


def _refresh_seconds() -> int:
    return max(300, int(os.getenv("HN_REFRESH_S", "3600")))


# ── Counting (pure) ──────────────────────────────────────────────────────────

def month_index(year: int, month: int) -> int:
    """Months since January 2007, when HN began: the shared x axis of every lifeline and the chart."""
    return (year - HN_FIRST_YEAR) * 12 + month - 1


def aggregate(rows: list[tuple], threads: dict[str, dict], now: int) -> dict[str, dict]:
    """Per-paper counts from mention rows (arxiv_id, item_id, kind, author,
    created_at, story_id). Mentions by a deleted account (no author) count as
    mentions but never as a person."""
    by_paper: dict[str, list[tuple]] = defaultdict(list)
    for r in rows:
        by_paper[r[0]].append(r)
    cutoff = now - FRESH_DAYS * 86400
    week_cutoff = now - WEEK_DAYS * 86400
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
        voices: Counter = Counter()            # month index -> people linking it for the first time
        joined = 0                             # when its third new-voice month began
        week_people: set[str] = set()
        week_story, week_last = "", 0
        before_window = 0                      # the latest link before the last FRESH_DAYS ...
        first_recent = 0                       # ... and the first one inside them
        for _aid, _item, kind, author, ts, story in ms:
            d = datetime.fromtimestamp(ts, timezone.utc)
            month = (d.year, d.month)
            months.add(month)
            if author:
                if author not in people:
                    people.add(author)
                    if month not in voice and len(voice) == MIN_MONTHS - 1:
                        joined = ts
                    voice.add(month)
                    voices[month_index(d.year, d.month)] += 1
                years[d.year].add(author)
            if ts < cutoff:
                before_window = ts
            elif not first_recent:
                first_recent = ts
            if ts >= week_cutoff:
                if author:
                    week_people.add(author)
                if ts >= week_last:
                    week_last, week_story = ts, story
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
            "first_story": ms[0][5], "first_kind": ms[0][2], "last_story": ms[-1][5], "last_kind": ms[-1][2],
            "mentions": len(ms), "threads": len(per_thread), "first": ms[0][4], "last": ms[-1][4],
            "years": {y: len(s) for y, s in years.items()}, "per_thread": per_thread,
            "submitted": submitted, "recent_people": len(recent_people), "recent_points": recent_points,
            "recent_last": recent_last, "voices": dict(voices), "joined": joined,
            "week_people": len(week_people), "week_story": week_story,
            # Days of silence before HN linked it again in the last FRESH_DAYS
            # (0 when it was not linked lately, or never before).
            "quiet_days": int((first_recent - before_window) / 86400) if first_recent and before_window else 0,
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


def _relevance(x: dict) -> tuple:
    """Threads about the paper before threads that mention it: submissions that
    drew attention by points, then discussions by how many comments link it,
    then submissions nobody noticed. A big thread with one passing link sorts
    by its points only among those."""
    pts = x["points"] or 0
    if x["submitted"] and pts >= SUBMISSION_FIRST:
        return (0, -pts, 0, x["story_id"])
    if not x["submitted"]:
        return (1, -x["mentions"], -pts, x["story_id"])
    return (2, -pts, 0, x["story_id"])


def top_threads(p: dict, threads: dict[str, dict], n: int = THREADS_SHOWN) -> list[dict]:
    """The threads to show for a paper: submissions that drew attention
    (SUBMISSION_FIRST points or more) by points, then the discussions that
    linked it most, then submissions nobody noticed."""
    out = []
    for story, k in p["per_thread"].items():
        t = threads.get(story) or {}
        if (t.get("title") or "").strip() in _GONE - {""}:      # removed by HN: nothing to read there
            continue
        sub = story in p["submitted"]
        out.append({"story_id": story, "title": t.get("title") or "", "points": t.get("points"),
                    "comments": t.get("comments"), "created_at": t.get("created_at"),
                    "submitted": sub, "mentions": k})

    out.sort(key=_relevance)
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
    quotes = await _quotes(listed, rows, meta, threads)
    _state.update(papers=papers, threads=threads, canon=listed,
                  canon_rank={a: i + 1 for i, a in enumerate(listed)},
                  canon_all=len(canon), fresh=[a for a in fresh if a in meta], meta=meta, vectors=vectors,
                  quotes=quotes,
                  loaded_at=loaded_at, rows=len(rows), error=None,
                  refreshed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    # Built once a load, not per request: the whole list is ~2,000 entries.
    _state["cards"] = [c for c in (card(a, "lasting") for a in listed) if c is not None]
    _state["strata"] = strata(_state["cards"], now)
    _state["strata_json"] = json.dumps(_state["strata"], separators=(",", ":"))
    return stats()


async def _quotes(listed: list[str], rows: list[tuple], meta: dict, threads: dict) -> dict[str, dict]:
    """One quote per lasting paper from the comments that link it (pick_quote).
    The comment texts are read, used and dropped; only the picks are kept. A
    failed read costs the quotes, never the lists."""
    want = set(listed)
    by_item: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for aid, item, kind, _author, ts, story in rows:
        if kind == "comment" and aid in want:
            by_item[item].append((aid, story, ts))
    if not by_item:
        return {}
    try:
        comment_rows = await _paged("SELECT rowid AS rid, item_id, text FROM hn_comments "
                                    f"WHERE rowid > ? AND papers <= {QUOTE_MAX_PAPERS} ORDER BY rowid LIMIT ?")
    except Exception as e:  # quotes are optional: any failed read (HTTP, timeout, missing table) costs only them
        print(f"[hn] comments unavailable, no quotes: {type(e).__name__}: {str(e)[:120]}")
        return {}
    candidates: dict[str, list[dict]] = defaultdict(list)
    for r in comment_rows:
        for aid, story, ts in by_item.get(r["item_id"], ()):
            candidates[aid].append({"text": r["text"] or "", "story_id": story, "created_at": ts,
                                    "item_id": r["item_id"]})
    out = {}
    for aid, cands in candidates.items():
        q = pick_quote(aid, meta[aid]["title"], cands)
        if q:
            t = threads.get(q["story_id"]) or {}
            q["thread"] = t.get("title") if (t.get("title") or "").strip() not in _GONE else ""
            out[aid] = q
    return out


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
    # A lasting paper outside the corpus (mostly physics and maths) is still
    # lasting: describe it from the arXiv API (cached) so every count on the
    # site is the same number. No vector, so it never shows under "for you".
    missing = [a for a in ids if a not in meta][:ARXIV_FILL_MAX]
    if missing:
        try:
            filled = await asyncio.wait_for(arxiv_svc.fetch_metadata_batch(missing), ARXIV_FILL_SECONDS)
        except Exception as e:  # lists still work; these papers stay off them until the next load
            print(f"[hn] arXiv fill unavailable: {type(e).__name__}")
            filled = {}
        for aid, m in filled.items():
            if aid in meta or not m.get("title"):
                continue
            meta[aid] = {"title": " ".join(m["title"].split()), "authors": _authors_text(m.get("authors")),
                         "year": id_year(aid) or m.get("year") or "",
                         "categories": [m["category"]] if m.get("category") else [], "citation_count": 0}
    vectors: dict = {}
    have = [a for a in ids if a in meta]
    for i in range(0, len(have), VECTOR_BATCH):
        try:
            vectors.update(await qdrant_svc.get_paper_vectors(have[i:i + VECTOR_BATCH], remember=False))
        except Exception as e:  # lists still work for everyone; only "for you" needs vectors
            print(f"[hn] vectors unavailable: {type(e).__name__}")
            break
    return meta, vectors


def _authors_text(raw) -> str:
    """arxiv_svc stores authors as a JSON list; the lists show them as text."""
    if isinstance(raw, list):
        return ", ".join(raw)
    try:
        got = json.loads(raw or "")
    except (TypeError, ValueError):
        return raw or ""
    return ", ".join(got) if isinstance(got, list) else str(got)


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
    """A list entry: the paper's metadata and its HN facts, its best thread,
    its threads in date order and a bar per year since HN began (the year-by-year
    record DAIR.AI's Hacker Papers shows on each card). `kind` is "lasting" or
    "fresh"; a fresh entry counts only recent activity."""
    p, m = _state["papers"].get(arxiv_id), _state["meta"].get(arxiv_id)
    if p is None or m is None:
        return None
    best = top_threads(p, _state["threads"], 1)
    now_dt = datetime.now(timezone.utc)
    now_ts, now_month = int(now_dt.timestamp()), month_index(now_dt.year, now_dt.month)
    first = datetime.fromtimestamp(p["first"], timezone.utc)
    last = datetime.fromtimestamp(p["last"], timezone.utc)
    out = {"arxiv_id": arxiv_id, **m, "hn": facts(p), "thread": best[0] if best else None,
           "rank": _state["canon_rank"].get(arxiv_id), "kind": kind,
           "field": field_of(m.get("categories") or [], arxiv_id),
           "first": p["first"], "last": p["last"], "last_label": last.strftime("%-d %b %Y"),
           "span_years": round((p["last"] - p["first"]) / (365.25 * 86400), 1),
           "best_post": max([(_state["threads"].get(t) or {}).get("points") or 0 for t in p["submitted"]] or [0]),
           "spark": spark(p, last_year=max(last.year, datetime.now(timezone.utc).year)),
           "dated": dated_threads(p, _state["threads"]), "n_threads": p["threads"],
           "life": lifeline(p, now_month), "beats": beats(p, _state["threads"]),
           "quote": (_state.get("quotes") or {}).get(arxiv_id),
           "joined": p.get("joined") or 0, "just_joined": bool(p.get("joined")) and p["joined"] >= now_ts - JOINED_DAYS * 86400,
           "joined_label": datetime.fromtimestamp(p["joined"], timezone.utc).strftime("%b %Y") if p.get("joined") else "",
           "comeback_days": p.get("quiet_days") if (p.get("quiet_days") or 0) >= COMEBACK_DAYS else 0,
           "week_people": p.get("week_people", 0),
           "week_thread": _week_thread(p),
           # trending_shelf.rank's popularity term: tenfold more people is
           # worth UPVOTE_WEIGHT of closeness.
           "people": p["recent_people"] if kind == "fresh" else p["people"]}
    out["first_label"] = first.strftime("%b %Y")
    if kind == "fresh":
        out["recent_people"], out["recent_points"] = p["recent_people"], p["recent_points"]
    return out


HN_FIRST_YEAR = 2007
THREADS_DATED = 8


def spark(p: dict, last_year: int) -> list[dict]:
    """People per year from HN's first year to `last_year`, every list on one
    axis so rows compare; heights are shares of the paper's own busiest year."""
    top = max(p["years"].values() or [0]) or 1
    return [{"year": y, "people": p["years"].get(y, 0), "h": round(100 * p["years"].get(y, 0) / top)}
            for y in range(HN_FIRST_YEAR, last_year + 1)]


_GONE = {"", "[dead]", "[flagged]", "[deleted]"}


def dated_threads(p: dict, threads: dict[str, dict], n: int = THREADS_DATED) -> list[dict]:
    """The paper's n most relevant threads (_relevance), oldest first: how its
    HN history unfolded, without the dozen two-point reposts of launch week.
    Threads HN removed, or whose date is unknown, are left out."""
    out = []
    for story, k in p["per_thread"].items():
        t = threads.get(story) or {}
        if (t.get("title") or "").strip() in _GONE or not t.get("created_at"):
            continue
        out.append({"story_id": story, "title": t["title"], "points": t.get("points"),
                    "created_at": t["created_at"], "submitted": story in p["submitted"], "mentions": k})
    out.sort(key=_relevance)
    out = sorted(out[:n], key=lambda x: (x["created_at"], x["story_id"]))
    for x in out:
        x["when"] = datetime.fromtimestamp(x["created_at"], timezone.utc).strftime("%b %Y")
        x["url"] = f"https://news.ycombinator.com/item?id={x['story_id']}"
    return out


# ── The row's record: lifeline, story, quote; the chart ──────────────────────
#
# Every lifeline shares one x axis (January 2007 to this month) and one height
# scale, so rows can be compared at a glance: small multiples only compare
# when their scales agree. The height is the square root of the people who
# linked it for the first time that month, full at LIFE_FULL, so a month with
# one new person still shows and a launch spike does not flatten the rest.

LIFE_FULL = 6               # new people in a month for a full-height tick
LIFE_MIN_H = 22             # % of full height for a month with one new person


def lifeline(p: dict, now_month: int) -> dict:
    """A tick per month someone new linked the paper, on the shared axis."""
    ticks = [{"i": i, "n": n, "h": max(LIFE_MIN_H, round(100 * min(1.0, math.sqrt(n / LIFE_FULL))))}
             for i, n in sorted((p.get("voices") or {}).items())]
    first, last = datetime.fromtimestamp(p["first"], timezone.utc), datetime.fromtimestamp(p["last"], timezone.utc)
    return {"months": now_month + 1, "ticks": ticks,
            "first": month_index(first.year, first.month), "last": month_index(last.year, last.month)}


BEAT_BUSY = 5               # a "busiest year" needs at least this many people
BEAT_BIG = 100              # a large thread needs at least this many points


def _link_beat(kind: str, story: str, threads: dict, label: str, ts: int) -> dict:
    t = threads.get(story) or {}
    title = (t.get("title") or "").strip()
    posted = kind == "story"
    return {"ts": ts, "when": datetime.fromtimestamp(ts, timezone.utc).strftime("%b %Y"), "k": label,
            "posted": posted, "title": "" if posted or title in _GONE else title,
            "points": t.get("points") if posted else None,
            "url": f"https://news.ycombinator.com/item?id={story}"}


def beats(p: dict, threads: dict[str, dict]) -> list[dict]:
    """The paper's HN story in up to four moments, oldest first: its first
    link, its busiest year, the largest thread it came up in (that thread's
    points, never the paper's) and its latest link. Written from the counts."""
    out = [_link_beat(p["first_kind"], p["first_story"], threads, "First link", p["first"])]
    if p["years"]:
        year, n = max(p["years"].items(), key=lambda kv: (kv[1], -kv[0]))
        if n >= BEAT_BUSY and len(p["years"]) > 1:
            out.append({"ts": int(datetime(year, 7, 1, tzinfo=timezone.utc).timestamp()), "when": str(year),
                        "k": "Busiest year", "people": n, "year": year})
    big = None
    for story in p["per_thread"]:
        t = threads.get(story) or {}
        if (t.get("title") or "").strip() in _GONE or not t.get("created_at"):
            continue
        if (t.get("points") or 0) >= BEAT_BIG and story not in (p["first_story"], p["last_story"]):
            if big is None or t["points"] > big[1]["points"]:
                big = (story, t)
    if big:
        story, t = big
        out.append({"ts": t["created_at"], "when": datetime.fromtimestamp(t["created_at"], timezone.utc).strftime("%b %Y"),
                    "k": "Largest thread it came up in", "posted": story in p["submitted"], "title": t["title"],
                    "points": t["points"], "url": f"https://news.ycombinator.com/item?id={story}"})
    if p["last_story"] != p["first_story"]:
        out.append(_link_beat(p["last_kind"], p["last_story"], threads, "Latest link", p["last"]))
    out.sort(key=lambda b: b["ts"])
    return out


def _week_thread(p: dict) -> dict | None:
    """The thread of the paper's latest link this week."""
    story = p.get("week_story")
    if not story or not p.get("week_people"):
        return None
    t = _state["threads"].get(story) or {}
    title = (t.get("title") or "").strip()
    return {"title": "" if title in _GONE else title, "points": t.get("points"),
            "submitted": story in p["submitted"], "url": f"https://news.ycombinator.com/item?id={story}"}


# Quotes. One sentence (or two) from a comment that links the paper, so a
# reader sees why people reach for it. Nothing is rewritten: a quote is cut
# from the comment, with only links removed, and links back to the comment.
# Measured on the 30 most lasting papers (17 Nov 2026), a plain "sentence
# around the link" rule gave weak quotes for about a third (a bibliography
# line, half an argument, the title alone); the rules below drop those, and
# no quote is better than a bad one.

QUOTE_MAX_PAPERS = 2        # comments linking more papers than this are reading lists, not a voice
QUOTE_MIN, QUOTE_MAX = 60, 240
QUOTE_MIN_SCORE = 2.0
_URL = re.compile(r"https?://\S+|www\.\S+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[A-Z0-9])")
_PRAISE = re.compile(r"\b(recommend\w*|classic|seminal|landmark|must[- ]read|favou?rite|excellent|brilliant|"
                     r"fascinating|beautiful|elegant|influential|well worth|worth (?:a )?read\w*|great (?:paper|read)|"
                     r"changed how|eye[- ]opening|one of the best)\b", re.I)
_EXPLAIN = re.compile(r"\b(shows? that|showed that|argues? that|proves? that|proved that|found that|demonstrat\w+|"
                      r"the (?:key )?idea|tl;?dr|in short|the gist|the point is|they show|it shows)\b", re.I)
_JUNK = re.compile(r"arxiv preprint|arxiv:\s*\d|et al\.,|\bpp\.\s*\d|\bvol\.\s*\d|\bdoi\b|\[non|"
                   r"\bpdf\b|abs/\d|^\s*\W*\s*$|\b(?:19|20)\d\d\)\.", re.I)
_RUDE = re.compile(r"\b(fuck\w*|shit\w*|idiot\w*|stupid|crap\w*|dumb|moron\w*|bullshit|retard\w*)\b", re.I)
_WORDS = re.compile(r"[A-Za-z]{4,}")
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”')
_ASKS = re.compile(r"^(?:wasn't|isn't|aren't|doesn't|didn't|don't|couldn't|wouldn't|shouldn't|hasn't|haven't|"
                   r"is there|are there|does anyone|has anyone|did anyone)\b", re.I)
_LINK_AS_WORD = re.compile(r"[:,]\s*(?:https?://|www\.)\S+\s+[a-z]")


def _balanced(text: str) -> str:
    """Drops a bracket left open or closed alone, as a removed link leaves them."""
    out, stack = list(text), []
    for i, ch in enumerate(text):
        if ch in "([":
            stack.append(i)
        elif ch in ")]":
            if stack and text[stack[-1]] == "([" [")]".index(ch)]:
                stack.pop()
            else:
                out[i] = ""
    for i in stack:
        out[i] = ""
    return "".join(out)


def _clean(text: str) -> str:
    text = _URL.sub("", text)
    text = re.sub(r"\(\s*[,;:.]?\s*\)|\[\s*\d*\s*\]", "", text)
    text = _balanced(text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" :;-–—")


def _trim(text: str) -> str:
    if len(text) <= QUOTE_MAX:
        return text
    cut = max((m.end() for m in re.finditer(r"[.!?](?=\s)", text[:QUOTE_MAX + 1])), default=0)
    if cut >= QUOTE_MIN:
        return text[:cut].strip()
    return text[:QUOTE_MAX - 1].rsplit(" ", 1)[0].rstrip(",;:-–— ") + "…"


def quote_score(text: str, title: str) -> float | None:
    """How well a cut sentence speaks for the paper; None when it must not be shown."""
    if (len(text) < QUOTE_MIN or text[0] in ">)]" or text[0].islower()
            or not text.endswith((".", "!", '"', "”", ")", "…", "'"))
            or _JUNK.search(text) or _RUDE.search(text) or "?" in text
            # an odd quotation mark: the comment was quoting someone else (often the abstract)
            or text.count('"') % 2 or text.count("“") != text.count("”")
            or sum(len(m) for m in _QUOTED.findall(text)) > len(text) / 2     # mostly someone else's words
            or _ASKS.search(text)):
        return None
    words = {w.lower() for w in _WORDS.findall(text)}
    own = words - {w.lower() for w in _WORDS.findall(title)}
    if len(own) < 6:                          # the title and little else
        return None
    score = 3.0 * bool(_PRAISE.search(text)) + 2.0 * bool(_EXPLAIN.search(text))
    score += 1.0 if len(words & {w.lower() for w in _WORDS.findall(title)}) >= 2 else 0.0
    score += 0.5 if 90 <= len(text) <= 210 else 0.0
    score -= 1.0 if len(re.findall(r"\bI\b", text)) > 2 else 0.0
    return score


def pick_quote(arxiv_id: str, title: str, candidates: list[dict]) -> dict | None:
    """The best passage around a link to the paper among `candidates`
    ({"text", "story_id", "item_id", "created_at"}), or None. The passage is
    the sentence with the link, with the next one when it explains it."""
    best: tuple | None = None
    bare = arxiv_id.split("v")[0] if "/" not in arxiv_id else arxiv_id
    for c in candidates:
        for para in (c.get("text") or "").split("\n"):
            if bare not in para:
                continue
            sents = _SENT.split(para)
            for i, sent in enumerate(sents):
                if bare not in sent:
                    continue
                options = [sent]
                if i + 1 < len(sents):
                    options.append(sent + " " + sents[i + 1])
                if i > 0:
                    options.append(sents[i - 1] + " " + sent)
                for raw in options:
                    # More than one link is a reading list; a link that was a
                    # word of the sentence ("I recommend: <link> and ...")
                    # leaves it broken once removed.
                    if len(_URL.findall(raw)) > 1 or _LINK_AS_WORD.search(raw):
                        continue
                    opt = _trim(_clean(raw))
                    sc = quote_score(opt, title)
                    if sc is None or sc < QUOTE_MIN_SCORE:
                        continue
                    key = (sc, -abs(len(opt) - 150), c["created_at"])
                    if best is None or key > best[0]:
                        best = (key, opt, c)
    if best is None:
        return None
    _key, text, c = best
    return {"text": text, "story_id": c["story_id"], "item_id": c["item_id"],
            "url": f"https://news.ycombinator.com/item?id={c['item_id']}",
            "when": datetime.fromtimestamp(c["created_at"], timezone.utc).strftime("%b %Y")}


def strata(cards: list[dict], now: int) -> dict:
    """The chart on /trending/lasting: every listed lasting paper as a line
    from its first link to its latest, with its new-voice months, stacked by
    when HN first linked it. [first, last, [months], rank, title, id]."""
    lines = sorted(([c["life"]["first"], c["life"]["last"], [t["i"] for t in c["life"]["ticks"]],
                     c["rank"], (c["title"] if len(c["title"]) <= 90 else c["title"][:88] + "…"), c["arxiv_id"]]
                    for c in cards), key=lambda l: (l[0], l[1], l[3] or 0))
    d = datetime.fromtimestamp(now, timezone.utc)
    return {"lines": lines, "months": month_index(d.year, d.month) + 1}


# ── Browsing the list: fields, years, views, sorts ───────────────────────────
#
# The controls DAIR.AI's Hacker Papers offers, filled from our counts. Fields
# are five broad groups by a paper's primary arXiv category, coarser than the
# onboarding areas on purpose: a list of 1,843 papers is browsed, not matched.

FIELDS = (("ai", "AI & ML"), ("cs", "Computer science"), ("math", "Mathematics"),
          ("physics", "Physics"), ("bio", "Biology, economics & statistics"))
_AI = {"cs.LG", "cs.CL", "cs.AI", "cs.CV", "cs.NE", "cs.IR", "cs.RO", "cs.MA", "stat.ML"}
_PHYSICS = ("quant-ph", "physics", "hep-", "gr-qc", "astro-ph", "cond-mat", "nlin", "nucl-", "math-ph")


def field_of(categories: list[str], arxiv_id: str = "") -> str:
    """The broad field of a paper's primary category (or, without one, its old-style archive)."""
    prim = categories[0] if categories else arxiv_id.split("/")[0] if "/" in arxiv_id else ""
    if prim in _AI:
        return "ai"
    if prim.startswith(("cs", "eess")):
        return "cs"
    if prim.startswith(_PHYSICS):
        return "physics"
    if prim.startswith("math"):
        return "math"
    if prim.startswith(("q-bio", "q-fin", "econ", "stat")):
        return "bio"
    return "cs" if not prim else "bio"


def year_buckets(this_year: int) -> list[tuple[str, str, int, int]]:
    """(key, label, from, to) by the year a paper first appeared: the last four
    years one by one, then 2020–2022, 2015–2019 and before 2015 (for 2026)."""
    y = this_year
    single = [(str(y - i), str(y - i), y - i, y - i) for i in range(4)]
    return single + [(f"{y - 6}-{y - 4}", f"{y - 6}–{y - 4}", y - 6, y - 4),
                     (f"{y - 11}-{y - 7}", f"{y - 11}–{y - 7}", y - 11, y - 7),
                     (f"before-{y - 11}", f"Before {y - 11}", 0, y - 12)]


GEM_MAX_CITATIONS = 50      # hidden gem: lasting on HN, cited fewer times than this ...
GEM_MIN_AGE = 2             # ... and old enough (years) for citations to have arrived

VIEWS = (("lasting", "Most lasting"), ("comebacks", "Comebacks"), ("joined", "Just joined"),
         ("running", "Longest running"), ("gems", "Hidden gems"), ("recent", "Recently mentioned"),
         ("saved", "Saved"))
SORTS = (("default", "Default"), ("newest", "Newest papers"), ("oldest", "Oldest papers"),
         ("people", "Most people"), ("threads", "Most threads"))


def is_gem(c: dict, this_year: int) -> bool:
    """Lasting on HN and rarely cited: 1 to GEM_MAX_CITATIONS - 1 citations. A
    count of 0 is left out, since it usually means Semantic Scholar has no
    record (NeRF, DeepSeek-V3 and Gemini all read 0), and so are papers too new
    to have been cited."""
    cites = c.get("citation_count") or 0
    year = c.get("year") or this_year
    return 0 < cites < GEM_MAX_CITATIONS and int(year) <= this_year - GEM_MIN_AGE


def browse(cards: list[dict], view: str = "lasting", sort: str = "default", year: str = "",
           field: str = "", saved: frozenset[str] = frozenset(), this_year: int | None = None) -> list[dict]:
    """The list as the controls ask: a view picks and orders papers, a sort
    reorders them, and the year and field narrow them. Pure."""
    this_year = this_year or datetime.now(timezone.utc).year
    span = next(((lo, hi) for k, _l, lo, hi in year_buckets(this_year) if k == year), None)
    out = [c for c in cards
           if (not field or c["field"] == field)
           and (span is None or span[0] <= int(c.get("year") or 0) <= span[1])]
    if view == "gems":
        out = [c for c in out if is_gem(c, this_year)]
    elif view == "saved":
        out = [c for c in out if c["arxiv_id"] in saved]
    elif view == "comebacks":
        out = [c for c in out if c.get("comeback_days")]
        out.sort(key=lambda c: (-c["comeback_days"], c["rank"] or 0))
    elif view == "joined":
        out = [c for c in out if c.get("just_joined")]
        out.sort(key=lambda c: (-c["joined"], c["rank"] or 0))
    if view == "running":
        out.sort(key=lambda c: (-c["span_years"], c["rank"] or 0))
    elif view == "recent":
        out.sort(key=lambda c: (-c["last"], c["rank"] or 0))
    if sort == "newest":
        out.sort(key=lambda c: (-int(c.get("year") or 0), c["rank"] or 0))
    elif sort == "oldest":
        out.sort(key=lambda c: (int(c.get("year") or 0), c["rank"] or 0))
    elif sort == "people":
        out.sort(key=lambda c: (-c["hn"]["people"], c["rank"] or 0))
    elif sort == "threads":
        out.sort(key=lambda c: (-c["hn"]["threads"], c["rank"] or 0))
    return out


def week_cards(exclude: frozenset[str] | set[str] = frozenset(), year: str = "", field: str = "",
               this_year: int | None = None) -> list[dict]:
    """Lasting papers HN linked in the last WEEK_DAYS, narrowed like the list
    below them: comebacks first (longest silence), then papers that just
    joined the list, then by people this week. Copies, like lasting_cards."""
    week = [dict(c) for c in _state.get("cards") or [] if c.get("week_people") and c["arxiv_id"] not in exclude]
    week = browse(week, "lasting", "default", year, field, this_year=this_year)
    week.sort(key=lambda c: (-(c["comeback_days"] or 0), -(c["joined"] if c["just_joined"] else 0),
                             -c["week_people"], c["rank"] or 0))
    return week[:WEEK_SHOWN]


def strata_json() -> str:
    """The chart's data, serialised once a load (about 200 KB before gzip)."""
    return _state.get("strata_json") or '{"lines":[],"months":0}'


def lasting_cards() -> list[dict]:
    """Every listed lasting paper in canon order. Copies: callers add
    per-request fields."""
    return [dict(c) for c in _state.get("cards") or []]


def fresh_cards() -> list[dict]:
    return [c for c in (card(a, "fresh") for a in _state["fresh"]) if c is not None]


def vectors() -> dict:
    return _state["vectors"]
