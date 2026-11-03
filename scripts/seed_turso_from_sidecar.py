#!/usr/bin/env python3
"""
Rebuild Turso's paper data from the metadata sidecar (and arXiv's listing).

Why this exists
---------------
On 2026-10-24 Turso's free plan blocked every read once 500M rows had been read
that month. The sidecar published to Hugging Face (siddhm11/researchit-metadata)
is an exact copy of Turso's `papers` table: same rows, same rowids, same
citation counts. So a new or emptied database can be seeded from it without
reading Turso at all.

What it writes, to the database named by --url / --token:
  papers          every sidecar row, with its sidecar rowid (the daily job and
                  append_to_sidecar.py rely on new papers having the highest
                  rowids), in multi-row INSERTs; resumable from MAX(rowid)
  paper_licences  with --listing: the licence arXiv's metadata listing (Kaggle
                  Cornell-University/arxiv) gives each paper from --licences-from
                  on (default 2506, the papers the daily job harvests)

Everything else rebuilds itself: the app creates the user tables
(turso_sync.ensure_remote_schema), the daily job creates citation_history and
the Hugging Face tables, and `daily_refresh.py --hf-only --hf-days 182` reloads
the lists.

Reads on the target: MAX(rowid) once, then point lookups to verify. Writes:
one per paper (~2M of the free plan's 10M a month) plus the licences.

Usage
-----
    python scripts/seed_turso_from_sidecar.py --sidecar ~/.cache/researchit/metadata.sqlite \\
        --url "$TURSO2_URL" --token "$TURSO2_DB_TOKEN" [--listing arxiv.zip] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
import zipfile

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLUMNS = ("arxiv_id", "title", "authors", "abstract_preview", "categories", "primary_topic",
           "update_date", "citation_count", "influential_citations")
PAPERS_DDL = ("CREATE TABLE IF NOT EXISTS papers (arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT, "
              "abstract_preview TEXT, categories TEXT, primary_topic TEXT, update_date TEXT, "
              "citation_count INTEGER, influential_citations INTEGER)")
ROWS_PER_INSERT = 200
INSERTS_PER_REQUEST = 10


def cell(v) -> dict:
    if v is None:
        return {"type": "null"}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    return {"type": "text", "value": str(v)}


def insert_stmt(table: str, cols: tuple[str, ...], rows: list[tuple]) -> dict:
    """One multi-row INSERT OR IGNORE: re-running a batch writes nothing twice."""
    one = "(" + ", ".join("?" * len(cols)) + ")"
    return {"sql": f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES " + ", ".join([one] * len(rows)),
            "args": [cell(v) for row in rows for v in row]}


class Target:
    def __init__(self, url: str, token: str):
        self.url = url.replace("libsql://", "https://", 1).rstrip("/") + "/v2/pipeline"
        self.client = httpx.Client(timeout=180, headers={"Authorization": f"Bearer {token}"})

    def run(self, stmts: list[dict]) -> list[list[list]]:
        for attempt in range(5):
            try:
                r = self.client.post(self.url, json={"requests": [{"type": "execute", "stmt": s} for s in stmts]
                                                     + [{"type": "close"}]})
                r.raise_for_status()
                break
            except httpx.HTTPError as e:
                if attempt == 4:
                    raise
                print(f"  retry {attempt + 1}: {str(e)[:100]}", flush=True)
                time.sleep(10 * (attempt + 1))
        out = []
        for res in r.json()["results"][:len(stmts)]:
            if res.get("type") == "error":
                raise RuntimeError(str(res.get("error"))[:300])
            out.append([[c.get("value") for c in row] for row in res["response"]["result"]["rows"]])
        return out


def seed_papers(side: sqlite3.Connection, target: Target, dry_run: bool) -> int:
    target.run([{"sql": PAPERS_DDL}]) if not dry_run else None
    start = 0 if dry_run else int(target.run([{"sql": "SELECT MAX(rowid) FROM papers"}])[0][0][0] or 0)
    total = side.execute("SELECT MAX(rowid) FROM papers").fetchone()[0]
    print(f"papers: sidecar has rowids 1..{total:,}; target already holds up to {start:,}", flush=True)
    cur = side.execute(f"SELECT rowid, {', '.join(COLUMNS)} FROM papers WHERE rowid > ? ORDER BY rowid", (start,))
    done, t0 = 0, time.time()
    while True:
        chunk = cur.fetchmany(ROWS_PER_INSERT * INSERTS_PER_REQUEST)
        if not chunk:
            break
        stmts = [insert_stmt("papers", ("rowid",) + COLUMNS, chunk[i:i + ROWS_PER_INSERT])
                 for i in range(0, len(chunk), ROWS_PER_INSERT)]
        if not dry_run:
            target.run(stmts)
        done += len(chunk)
        if done % 50_000 < len(chunk) or len(chunk) < ROWS_PER_INSERT * INSERTS_PER_REQUEST:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  papers {start + done:,}/{total:,}  {rate:,.0f}/s  eta {(total - start - done) / rate / 60:.0f} min",
                  flush=True)
    return done


def listing_licences(path: str, wanted: set[str]):
    """(arxiv_id, licence code, licence url) for `wanted` papers in arXiv's listing."""
    from app.licences import code_from_url
    opener = zipfile.ZipFile(path).open(zipfile.ZipFile(path).namelist()[0]) if path.endswith(".zip") else open(path, "rb")
    with opener as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] in wanted:
                url = (r.get("license") or "").strip()
                if url:
                    yield r["id"], code_from_url(url), url


def seed_licences(side: sqlite3.Connection, target: Target, listing: str, since: str, dry_run: bool) -> int:
    from ingest_backends import LICENCE_DDL
    wanted = {a for (a,) in side.execute("SELECT arxiv_id FROM papers WHERE arxiv_id >= ? AND arxiv_id NOT LIKE '%/%'",
                                         (since,))}
    print(f"licences: {len(wanted):,} papers from {since} on", flush=True)
    if not dry_run:
        target.run([{"sql": LICENCE_DDL}])
    rows, done = [], 0
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for aid, code, url in listing_licences(listing, wanted):
        rows.append((aid, code or "", url, "listing", stamp))
        if len(rows) == ROWS_PER_INSERT * INSERTS_PER_REQUEST:
            done += _flush_licences(target, rows, dry_run)
            rows = []
    done += _flush_licences(target, rows, dry_run)
    print(f"  licences written: {done:,}", flush=True)
    return done


def _flush_licences(target: Target, rows: list[tuple], dry_run: bool) -> int:
    if rows and not dry_run:
        cols = ("arxiv_id", "licence", "licence_url", "source", "updated_at")
        target.run([insert_stmt("paper_licences", cols, rows[i:i + ROWS_PER_INSERT])
                    for i in range(0, len(rows), ROWS_PER_INSERT)])
    return len(rows)


def verify(side: sqlite3.Connection, target: Target, n: int = 300) -> list[str]:
    """MAX(rowid) matches, and `n` random papers match column for column (point lookups only)."""
    problems = []
    total = side.execute("SELECT MAX(rowid) FROM papers").fetchone()[0]
    got = int(target.run([{"sql": "SELECT MAX(rowid) FROM papers"}])[0][0][0] or 0)
    if got != total:
        problems.append(f"MAX(rowid) {got:,} != sidecar {total:,}")
    random.seed(25)
    picks = random.sample(range(1, total + 1), n)
    want = {r[0]: r for r in side.execute(
        f"SELECT rowid, {', '.join(COLUMNS)} FROM papers WHERE rowid IN ({','.join('?' * n)})", picks)}
    rows = target.run([{"sql": f"SELECT rowid, {', '.join(COLUMNS)} FROM papers WHERE rowid IN ({','.join('?' * n)})",
                        "args": [cell(p) for p in picks]}])[0]
    for row in rows:
        w = want.get(int(row[0]))
        if w is None or [str(x) if x is not None else None for x in w] != [x for x in row]:
            problems.append(f"rowid {row[0]} differs")
    if len(rows) != n:
        problems.append(f"{n - len(rows)} sampled rowids missing")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--url", default=os.getenv("TURSO2_URL", ""))
    ap.add_argument("--token", default=os.getenv("TURSO2_DB_TOKEN", ""))
    ap.add_argument("--listing", default="", help="arXiv metadata listing (.json or .zip) for licences")
    ap.add_argument("--licences-from", default="2506")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not (args.url and args.token) and not args.dry_run:
        print("--url/--token (or TURSO2_URL/TURSO2_DB_TOKEN) required", file=sys.stderr)
        return 2
    side = sqlite3.connect(f"file:{os.path.expanduser(args.sidecar)}?mode=ro&immutable=1", uri=True)
    target = Target(args.url, args.token) if args.url else None
    if not args.verify_only:
        seed_papers(side, target, args.dry_run)
        if args.listing:
            seed_licences(side, target, args.listing, args.licences_from, args.dry_run)
    if args.dry_run:
        return 0
    problems = verify(side, target)
    print("verify: " + ("ok" if not problems else "; ".join(problems[:10])), flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
