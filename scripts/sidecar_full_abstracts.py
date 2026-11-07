#!/usr/bin/env python3
"""
Give the metadata sidecar full abstracts and an author-searchable keyword index.

Why this exists
---------------
1,440,395 of 1,972,571 sidecar papers (73%, all published before June 2025)
carry an abstract cut at 500 characters. The search cross-encoder and the
plain-language summaries read that stump: on 847 known-item queries the
reranker scored MRR@10 0.563 on it and 0.618 on full text (doc 06,
2026-10-15). The local Kaggle CSV holds the full text of those papers.

The keyword index (papers_fts) also covered only title and abstract, so an
author's name found nothing.

What it does, on a COPY of the sidecar
--------------------------------------
1. Adds papers.abstract_full and fills it from the CSV wherever the sidecar's
   abstract is a stump AND the CSV text starts the same way (whitespace
   normalised), so a mismatched id can never attach another paper's abstract.
   abstract_preview is left as it is: the keyword index keeps indexing the
   opening 500 characters, so its size and speed do not change.
2. Rebuilds papers_fts over (title, abstract_preview, authors).
3. Records what it did in build_state, then ANALYZE and VACUUM.

app/local_meta.py serves abstract_full in place of the stump when the column
exists, and older sidecars keep working unchanged.

Usage
-----
    .venv/bin/python scripts/sidecar_full_abstracts.py --sidecar publish/metadata.sqlite \\
        --csv ~/.cache/researchit-embedding-eval/csv/arxiv_comprehensive_papers.csv [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time

STUMP_LENGTHS = (496, 503)   # the cap is 500; a few stumps end a character or so either side
PREFIX = 120                 # characters that must agree between stump and CSV text


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def stump_prefixes(conn: sqlite3.Connection) -> dict[str, str]:
    lo, hi = STUMP_LENGTHS
    rows = conn.execute(
        "SELECT arxiv_id, abstract_preview FROM papers "
        "WHERE length(abstract_preview) BETWEEN ? AND ?", (lo, hi))
    return {aid: _norm(text)[:PREFIX] for aid, text in rows}


def csv_matches(path: str, wanted: dict[str, str]):
    """Yield (arxiv_id, full abstract) for stumps whose CSV text starts the same way."""
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = (row.get("id") or "").strip()
            prefix = wanted.get(aid)
            if prefix is None:
                continue
            full = _norm(row.get("abstract"))
            if len(full) > len(prefix) and full.startswith(prefix):
                yield aid, full


def add_full_abstracts(conn: sqlite3.Connection, csv_path: str, dry_run: bool) -> dict:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
    if "abstract_full" not in cols and not dry_run:
        conn.execute("ALTER TABLE papers ADD COLUMN abstract_full TEXT")
    wanted = stump_prefixes(conn)
    print(f"[sidecar] {len(wanted):,} cut abstracts")
    filled, batch, started = 0, [], time.time()
    for aid, full in csv_matches(csv_path, wanted):
        filled += 1
        if dry_run:
            continue
        batch.append((full, aid))
        if len(batch) >= 20_000:
            with conn:
                conn.executemany("UPDATE papers SET abstract_full = ? WHERE arxiv_id = ?", batch)
            batch.clear()
            print(f"[sidecar]   {filled:,} filled ({time.time() - started:.0f} s)")
    if batch:
        with conn:
            conn.executemany("UPDATE papers SET abstract_full = ? WHERE arxiv_id = ?", batch)
    return {"cut": len(wanted), "filled": filled}


def rebuild_fts(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("DROP TABLE IF EXISTS papers_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE papers_fts USING fts5("
            "title, abstract_preview, authors, content='papers', content_rowid='rowid', "
            "tokenize='porter unicode61')")
        conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True, help="sidecar COPY to modify")
    ap.add_argument("--csv", required=True, help="Kaggle arXiv CSV with full abstracts")
    ap.add_argument("--dry-run", action="store_true", help="count matches, change nothing")
    args = ap.parse_args()
    conn = sqlite3.connect(args.sidecar)
    stats = add_full_abstracts(conn, args.csv, args.dry_run)
    print(f"[sidecar] full abstracts: {stats['filled']:,} of {stats['cut']:,} cut")
    if args.dry_run:
        return 0
    t = time.time()
    rebuild_fts(conn)
    print(f"[sidecar] keyword index rebuilt with authors ({time.time() - t:.0f} s)")
    with conn:
        conn.executemany("INSERT OR REPLACE INTO build_state VALUES (?, ?)", [
            ("abstract_full_filled", str(stats["filled"])),
            ("fts_columns", "title,abstract_preview,authors"),
        ])
    conn.execute("ANALYZE")
    conn.execute("VACUUM")
    print("[sidecar] analysed and vacuumed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
