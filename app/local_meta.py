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


def fetch_trending(codes: set[str], limit: int = 10) -> list[dict]:
    """
    Most-cited papers in any of `codes`.

    Uses the paper_categories side table, which stores one row per
    (paper, arXiv code) and is indexed on (code, citation_count DESC).  That
    turns Turso's full-scan-plus-sort into an index range read.

    Returns [] when unavailable so the caller can fall back.
    """
    if not codes or not _probe() or _conn is None:
        return []
    ph = ",".join("?" * len(codes))
    try:
        cur = _conn.execute(
            f"""SELECT {', '.join('p.' + c for c in _COLUMNS)}
                FROM papers p
                JOIN (
                    SELECT DISTINCT arxiv_id FROM paper_categories
                    WHERE code IN ({ph})
                    ORDER BY citation_count DESC
                    LIMIT ?
                ) t ON t.arxiv_id = p.arxiv_id
                ORDER BY p.citation_count DESC""",
            (*codes, limit * 4),
        )
        rows = [dict(zip(_COLUMNS, r)) for r in cur.fetchall()]
        return rows[:limit]
    except Exception as e:
        print(f"[local_meta] trending failed ({e}) — deferring to Turso")
        return []
