#!/usr/bin/env python3
"""
Append papers that Turso has and the metadata sidecar does not.

Why this exists
---------------
The sidecar ships inside the Docker image, and the starter feed and keyword
search only see what it holds. build_metadata_sidecar.py rebuilds it from
scratch, which re-reads 1.8M rows; after an ingest only the new rows are
needed. New Turso rows have rowids above the last ingest's, so they are
fetched by `rowid > --after-rowid`.

Three things must stay consistent, and this does all of them in one
transaction:
  * papers            one row per paper (plain INSERT; an existing arxiv_id is
                      skipped, never replaced: INSERT OR REPLACE would give it
                      a new rowid and desynchronise the search index)
  * paper_categories  one row per (paper, arXiv code), as the builder writes
  * papers_fts        external-content FTS5 over papers(rowid); new rows must
                      be indexed explicitly, under the rowid they were given

Run it on a COPY of the sidecar, then check with --verify.

Usage
-----
    export TURSO_URL=... TURSO_DB_TOKEN=...
    python scripts/append_to_sidecar.py --sidecar publish/metadata.sqlite --after-rowid 1799348
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import httpx

COLUMNS = ("arxiv_id", "title", "authors", "abstract_preview", "categories",
           "primary_topic", "update_date", "citation_count", "influential_citations")


def turso_rows_after(url: str, token: str, after: int, page: int = 2000):
    """Yield Turso paper rows with rowid > after, in rowid order."""
    url = url.replace("libsql://", "https://").rstrip("/")
    last = after
    with httpx.Client(timeout=180) as client:
        while True:
            sql = (f"SELECT rowid, {', '.join(COLUMNS)} FROM papers "
                   f"WHERE rowid > {int(last)} ORDER BY rowid LIMIT {int(page)}")
            r = client.post(f"{url}/v2/pipeline", headers={"Authorization": f"Bearer {token}"},
                            json={"requests": [{"type": "execute", "stmt": {"sql": sql}},
                                               {"type": "close"}]})
            r.raise_for_status()
            res = r.json()["results"][0]
            if res.get("type") != "ok":
                raise RuntimeError(str(res)[:200])
            rows = [[c.get("value") for c in row] for row in res["response"]["result"]["rows"]]
            if not rows:
                return
            for row in rows:
                yield dict(zip(COLUMNS, row[1:]))
            last = int(rows[-1][0])


def append_rows(conn: sqlite3.Connection, rows) -> dict:
    """Insert papers, their category rows and their FTS entries. Returns counts."""
    stats = {"inserted": 0, "skipped_existing": 0, "category_rows": 0}
    with conn:
        for r in rows:
            cit = int(r.get("citation_count") or 0)
            inf = int(r.get("influential_citations") or 0)
            cur = conn.execute(
                "INSERT OR IGNORE INTO papers (arxiv_id, title, authors, abstract_preview, "
                "categories, primary_topic, update_date, citation_count, influential_citations) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (r["arxiv_id"], r["title"], r["authors"], r["abstract_preview"], r["categories"],
                 r["primary_topic"], r["update_date"], cit, inf))
            if cur.rowcount != 1:
                stats["skipped_existing"] += 1
                continue
            rowid = cur.lastrowid
            stats["inserted"] += 1
            codes = str(r.get("categories") or "").split()
            conn.executemany("INSERT INTO paper_categories VALUES (?,?,?,?)",
                             [(code, r["arxiv_id"], cit, r["update_date"]) for code in codes])
            stats["category_rows"] += len(codes)
            conn.execute("INSERT INTO papers_fts(rowid, title, abstract_preview) VALUES (?,?,?)",
                         (rowid, r["title"], r["abstract_preview"]))
    return stats


def verify(conn: sqlite3.Connection) -> dict:
    """Consistency checks the published file must pass."""
    out = {
        "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
        "papers": conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
        "fts_rows": conn.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0],
        "papers_without_categories": conn.execute(
            "SELECT COUNT(*) FROM papers p WHERE categories IS NOT NULL AND categories != '' "
            "AND NOT EXISTS (SELECT 1 FROM paper_categories c WHERE c.arxiv_id = p.arxiv_id)"
        ).fetchone()[0],
        "newest_update_date": conn.execute("SELECT MAX(update_date) FROM papers").fetchone()[0],
    }
    try:
        conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('integrity-check')")
        out["fts_integrity"] = "ok"
    except sqlite3.DatabaseError as e:
        out["fts_integrity"] = f"FAILED: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True, help="sidecar COPY to modify")
    ap.add_argument("--after-rowid", type=int, help="append Turso rows with rowid above this")
    ap.add_argument("--verify", action="store_true", help="only run the consistency checks")
    args = ap.parse_args()
    conn = sqlite3.connect(args.sidecar)
    if not args.verify:
        if args.after_rowid is None:
            print("--after-rowid is required to append", file=sys.stderr)
            return 2
        url, token = os.environ.get("TURSO_URL", ""), os.environ.get("TURSO_DB_TOKEN", "")
        if not url or not token:
            print("TURSO_URL and TURSO_DB_TOKEN must be set", file=sys.stderr)
            return 2
        stats = append_rows(conn, turso_rows_after(url, token, args.after_rowid))
        print(f"[sidecar] appended {stats['inserted']:,} papers, {stats['category_rows']:,} "
              f"category rows; skipped {stats['skipped_existing']:,} already present")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    report = verify(conn)
    print("[sidecar] verify:", report)
    ok = (report["quick_check"] == "ok" and report["fts_integrity"] == "ok"
          and report["fts_rows"] == report["papers"] and report["papers_without_categories"] == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
