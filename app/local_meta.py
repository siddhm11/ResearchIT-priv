"""
Local metadata sidecar — a read-only SQLite mirror of the Turso `papers` table
that ships inside the image.

Why
---
Turso has exactly one index (`arxiv_id`), so it serves point lookups and
nothing else.  Measured against the live deployment:

  * a 50-id metadata fetch on the search hot path costs ~1.2 s from the Space
    (the DB is in aws-ap-south-1; the same query is ~190 ms from a machine
    closer to it, so most of that is network distance, not query cost)
  * `fetch_trending_by_categories` is a `LIKE '%code%'` scan plus a temp
    B-tree sort over 1.6 M rows — the code comments put it at ~15 s cold, and
    it lands on brand-new users during onboarding
  * `SELECT MIN(update_date), MAX(update_date)` does not complete at all

The sidecar carries the indexes Turso lacks, so those become local reads.
It is strictly an accelerator: every entry point degrades to Turso when the
file is absent, so the app runs unchanged without it.

Build it with `scripts/build_metadata_sidecar.py`.  The file is large (~1.5 GB
with abstracts) and must NOT be committed — it is fetched at image build time.
"""
from __future__ import annotations

import os
import sqlite3
import threading

# Columns are named to match the Turso `papers` table exactly, so rows can be
# handed to turso_svc._to_paper_dict without a second mapping layer.
_COLUMNS = (
    "arxiv_id", "title", "authors", "abstract_preview", "categories",
    "primary_topic", "update_date", "citation_count", "influential_citations",
)

SIDECAR_PATH = os.getenv("METADATA_SIDECAR_PATH", "data/metadata.sqlite")

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()
_probed = False
_available = False


def _probe() -> bool:
    """Open the sidecar once and confirm it is usable."""
    global _conn, _probed, _available
    if _probed:
        return _available
    with _lock:
        if _probed:
            return _available
        _probed = True
        path = SIDECAR_PATH
        if not path or not os.path.isfile(path):
            print(f"[local_meta] no sidecar at {path!r} — falling back to Turso")
            _available = False
            return False
        try:
            # Read-only URI so a corrupt or partially-written file can never be
            # mutated, and so several threads can share the handle safely.
            conn = sqlite3.connect(
                f"file:{os.path.abspath(path)}?mode=ro", uri=True,
                check_same_thread=False,
            )
            n = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            if not n:
                print("[local_meta] sidecar present but empty — using Turso")
                _available = False
                return False
            _conn = conn
            _available = True
            size_mb = os.path.getsize(path) / 1e6
            print(f"[local_meta] sidecar loaded: {n:,} papers, {size_mb:,.0f} MB")
        except Exception as e:
            print(f"[local_meta] sidecar unusable ({e}) — falling back to Turso")
            _available = False
    return _available


def is_available() -> bool:
    return _probe()


def stats() -> dict:
    """Diagnostics for the health endpoint."""
    if not _probe() or _conn is None:
        return {"available": False, "path": SIDECAR_PATH}
    try:
        n = _conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        newest = _conn.execute(
            "SELECT MAX(update_date) FROM papers").fetchone()[0]
        return {
            "available": True,
            "path": SIDECAR_PATH,
            "papers": n,
            "newest_update_date": newest,
            "size_mb": round(os.path.getsize(SIDECAR_PATH) / 1e6),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def fetch_rows(arxiv_ids: list[str]) -> list[dict]:
    """
    Look up papers by arxiv_id.  Returns rows keyed by Turso column names.

    Missing ids are simply absent from the result — the caller falls back to
    Turso for those, so a partially-built sidecar is still useful.
    """
    if not arxiv_ids or not _probe() or _conn is None:
        return []
    out: list[dict] = []
    # Chunked to stay well under SQLITE_MAX_VARIABLE_NUMBER.
    for i in range(0, len(arxiv_ids), 500):
        chunk = arxiv_ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        try:
            cur = _conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM papers "
                f"WHERE arxiv_id IN ({ph})", chunk)
            out += [dict(zip(_COLUMNS, r)) for r in cur.fetchall()]
        except Exception as e:
            print(f"[local_meta] lookup failed ({e}) — deferring to Turso")
            return out
    return out


def pub_year_month(arxiv_id: str) -> tuple[int, int] | None:
    """Publication (year, month) decoded from the arXiv identifier.

    `update_date` is the *last revision* date, not publication, so filtering on
    it surfaces old landmarks that happen to have a recent vN — "Attention Is
    All You Need" (2017) carries update_date 2023-08-03 and outranks everything
    actually new.  The identifier itself is a reliable publication stamp:

        2407.21783        -> 2024-07   (post-2007 scheme, YYMM.NNNNN)
        0704.0002         -> 2007-04
        math/0309136      -> 2003-09   (pre-2007 scheme, archive/YYMMNNN)

    Returns None for anything unparseable, and callers treat that as "unknown"
    rather than excluding the paper.
    """
    if not arxiv_id:
        return None
    tail = arxiv_id.split("/")[-1]
    if len(tail) < 4 or not tail[:4].isdigit():
        return None
    yy, mm = int(tail[:2]), int(tail[2:4])
    if not 1 <= mm <= 12:
        return None
    # arXiv began in 1991; two-digit years below ~50 are 20xx.
    return (2000 + yy if yy < 50 else 1900 + yy), mm


def _months_before(anchor: str, months: int) -> tuple[int, int] | None:
    """(year, month) that is `months` before the YYYY-MM-DD `anchor`."""
    if not anchor or len(anchor) < 7:
        return None
    try:
        y, m = int(anchor[:4]), int(anchor[5:7])
    except ValueError:
        return None
    total = y * 12 + (m - 1) - months
    return total // 12, total % 12 + 1


_max_date: str | None = None


def newest_update_date() -> str | None:
    """Most recent update_date in the corpus, computed once and memoised.

    Cheap here (~50 ms against the indexed column) and impossible upstream —
    the same aggregate does not complete on Turso.
    """
    global _max_date
    if _max_date is not None:
        return _max_date
    if not _probe() or _conn is None:
        return None
    try:
        _max_date = _conn.execute(
            "SELECT MAX(update_date) FROM papers").fetchone()[0]
    except Exception:
        _max_date = None
    return _max_date


def fetch_trending(
    codes: set[str],
    limit: int = 10,
    recency_months: int = 24,
) -> list[dict]:
    """
    Well-cited *recent* papers in any of `codes`.

    Ordering by all-time citations returns the same canonical papers to every
    user forever — for cs.LG that is Adam (2014), scikit-learn (2011) and
    BatchNorm (2015).  Those are famous, not trending, and this feeds Tier 0,
    which is the very first thing a new user sees.  Restricting to a recent
    window over the same data returns Llama 3, DPO and DeepSeek-R1 instead.

    Two details that matter:

      * The window is measured back from the newest paper in the corpus, not
        from today.  The corpus is a static snapshot ending 2025-05-30, so an
        absolute cutoff would silently empty out; anchoring to the data means
        this keeps working if ingestion is added later.
      * Categories are wildly uneven (~302k papers in cs.LG vs ~7.9k in
        q-bio.NC), so a fixed window starves the thin ones.  The window widens
        and finally drops away entirely rather than returning a short list.

    Returns [] when unavailable so the caller can fall back to Turso.
    """
    if not codes or not _probe() or _conn is None:
        return []

    ph = ",".join("?" * len(codes))
    # Read the most-cited papers in these categories in one index-ordered pass,
    # then apply the publication-date window in Python.  The date cannot be
    # filtered in SQL because it is derived from the identifier rather than
    # stored; the pool is sized so the window still has plenty to choose from.
    pool = max(2000, limit * 200)
    try:
        cur = _conn.execute(
            f"""SELECT {', '.join('p.' + c for c in _COLUMNS)}, t.cit
                FROM papers p
                JOIN (
                    SELECT arxiv_id, MAX(citation_count) AS cit
                    FROM paper_categories
                    WHERE code IN ({ph})
                    GROUP BY arxiv_id
                    ORDER BY cit DESC
                    LIMIT ?
                ) t ON t.arxiv_id = p.arxiv_id
                ORDER BY t.cit DESC""",
            (*codes, pool),
        )
        candidates = [dict(zip(_COLUMNS, r[:len(_COLUMNS)])) for r in cur.fetchall()]
    except Exception as e:
        print(f"[local_meta] trending failed ({e}) — deferring to Turso")
        return []

    if not candidates:
        return []

    anchor = newest_update_date()
    if not anchor:
        return candidates[:limit]

    # Widen rather than return a short list: categories range from ~302k papers
    # (cs.LG) to ~7.9k (q-bio.NC), so one fixed window cannot serve both.
    # `candidates` is already ordered by citations, so each filtered subset
    # keeps that ordering.
    best: list[dict] = []
    for months in (recency_months, recency_months * 2, recency_months * 4):
        cutoff = _months_before(anchor, months)
        if cutoff is None:
            break
        picked = [
            row for row in candidates
            if (ym := pub_year_month(row["arxiv_id"])) is not None and ym >= cutoff
        ]
        if len(picked) >= limit:
            return picked[:limit]
        if len(picked) > len(best):
            best = picked

    # No window could fill the slots — fall back to plain citation order.
    return (best or candidates)[:limit]
