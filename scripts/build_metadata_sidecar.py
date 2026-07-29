#!/usr/bin/env python3
"""
Build the local metadata sidecar from Turso.

Why this exists
---------------
Turso's `papers` table has exactly one index (`arxiv_id`), so it supports point
lookups and nothing else.  Measured against the live DB:

    SELECT 1                                  ->   187 ms
    SELECT COUNT(*)                           ->   834 ms
    SELECT MIN(update_date), MAX(update_date) ->   TIMES OUT at 100 s
    trending (LIKE + ORDER BY)                ->   SCAN papers + TEMP B-TREE

and a 50-id metadata fetch on the search hot path costs ~4.7 s round trip.

This script mirrors the table into a local SQLite file that ships inside the
Docker image, with the indexes Turso lacks.  That buys two things:

  1. Hot-path metadata lookups become local reads (microseconds, no network).
  2. Query patterns that are currently impossible become cheap — trending by
     category, recency filters, "most cited in cs.CL since 2024".

Usage
-----
    export TURSO_URL=... TURSO_DB_TOKEN=...
    python scripts/build_metadata_sidecar.py --out data/metadata.sqlite

    # smaller/faster build without abstracts (~250 MB instead of ~1.1 GB)
    python scripts/build_metadata_sidecar.py --out data/meta.sqlite --no-abstracts

Resumable: re-running continues from the highest rowid already imported.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request

TOTAL_ESTIMATE = 1_600_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id              TEXT PRIMARY KEY,
    title                 TEXT,
    authors               TEXT,
    abstract_preview      TEXT,
    categories            TEXT,
    primary_topic         TEXT,
    update_date           TEXT,
    citation_count        INTEGER,
    influential_citations INTEGER
);

-- One row per (paper, arXiv code).  Turso can only do LIKE '%cs.CL%' against a
-- space-separated string, which forces a full scan; this makes category
-- filtering index-driven.
CREATE TABLE IF NOT EXISTS paper_categories (
    code           TEXT NOT NULL,
    arxiv_id       TEXT NOT NULL,
    citation_count INTEGER,
    update_date    TEXT
);

CREATE TABLE IF NOT EXISTS build_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_papers_update_date ON papers(update_date);
CREATE INDEX IF NOT EXISTS idx_papers_citations   ON papers(citation_count DESC);
-- Covers the trending query: filter by code, already ordered by citations.
CREATE INDEX IF NOT EXISTS idx_pc_code_cit ON paper_categories(code, citation_count DESC);
CREATE INDEX IF NOT EXISTS idx_pc_code_date ON paper_categories(code, update_date DESC);
CREATE INDEX IF NOT EXISTS idx_pc_arxiv ON paper_categories(arxiv_id);
"""


def turso_query(url: str, token: str, sql: str, timeout: int = 120):
    payload = json.dumps({"requests": [
        {"type": "execute", "stmt": {"sql": sql}},
        {"type": "close"},
    ]}).encode()
    req = urllib.request.Request(
        f"{url}/v2/pipeline", data=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    res = data["results"][0]
    if res.get("type") == "error":
        raise RuntimeError(str(res.get("error"))[:300])
    rd = res["response"]["result"]
    return [[c.get("value") for c in row] for row in rd["rows"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output sqlite path")
    ap.add_argument("--chunk", type=int, default=2000, help="rows per request")
    ap.add_argument("--no-abstracts", action="store_true",
                    help="omit abstract_preview (much smaller file)")
    ap.add_argument("--max-rowid", type=int, default=1_700_000)
    args = ap.parse_args()

    url = os.environ.get("TURSO_URL", "").rstrip("/")
    token = os.environ.get("TURSO_DB_TOKEN", "")
    if not url or not token:
        print("TURSO_URL and TURSO_DB_TOKEN must be set", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    con = sqlite3.connect(args.out)
    con.executescript(SCHEMA)
    # Bulk-load pragmas.  Safe: if the build dies we just re-run it.
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.commit()

    cur = con.execute("SELECT value FROM build_state WHERE key='max_rowid_done'")
    row = cur.fetchone()
    start = int(row[0]) if row else 0
    if start:
        print(f"resuming from rowid {start:,}")

    abstract_col = "''" if args.no_abstracts else "abstract_preview"
    chunk = args.chunk
    lo = start + 1
    imported = 0
    t_start = time.time()

    while lo <= args.max_rowid:
        hi = lo + chunk - 1
        sql = (
            "SELECT rowid, arxiv_id, title, authors, "
            f"{abstract_col}, categories, primary_topic, update_date, "
            "citation_count, influential_citations "
            f"FROM papers WHERE rowid BETWEEN {lo} AND {hi}"
        )
        try:
            rows = turso_query(url, token, sql)
        except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
            if chunk > 250:
                chunk = max(250, chunk // 2)
                print(f"\n  chunk too big/slow ({str(e)[:60]}); retrying at {chunk}")
                continue
            print(f"\n  chunk {lo}-{hi} failed permanently: {str(e)[:120]}")
            lo = hi + 1
            continue

        if rows:
            papers = []
            cats = []
            for r in rows:
                (_rid, aid, title, authors, abstract, categories,
                 topic, udate, cit, infl) = r
                if not aid:
                    continue
                try:
                    cit_i = int(cit) if cit is not None else 0
                except (TypeError, ValueError):
                    cit_i = 0
                try:
                    infl_i = int(infl) if infl is not None else 0
                except (TypeError, ValueError):
                    infl_i = 0
                papers.append((aid, title, authors, abstract, categories,
                               topic, udate, cit_i, infl_i))
                for code in str(categories or "").split():
                    cats.append((code, aid, cit_i, udate))

            con.executemany(
                "INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?)", papers)
            con.executemany(
                "INSERT INTO paper_categories VALUES (?,?,?,?)", cats)
            imported += len(papers)

        con.execute(
            "INSERT OR REPLACE INTO build_state VALUES ('max_rowid_done', ?)",
            (str(hi),))
        con.commit()

        elapsed = time.time() - t_start
        rate = imported / elapsed if elapsed > 0 else 0
        pct = 100 * min(hi, TOTAL_ESTIMATE) / TOTAL_ESTIMATE
        eta = (TOTAL_ESTIMATE - imported) / rate / 60 if rate > 0 else 0
        print(f"\r  rowid {hi:,}  imported {imported:,}  "
              f"({pct:5.1f}%)  {rate:,.0f} rows/s  ETA {eta:5.1f} min",
              end="", flush=True)

        if not rows and hi > TOTAL_ESTIMATE:
            break
        lo = hi + 1

    print("\nbuilding indexes...")
    con.executescript(INDEXES)
    con.commit()
    print("running ANALYZE...")
    con.execute("ANALYZE")
    con.commit()
    con.execute("PRAGMA journal_mode=DELETE")
    con.close()

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"done: {imported:,} papers -> {args.out} ({size_mb:,.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
