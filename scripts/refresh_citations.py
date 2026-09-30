#!/usr/bin/env python3
"""
Fetch current citation counts from Semantic Scholar into a staging SQLite file.

Why this exists
---------------
Citation counts were collected once, with the Kaggle snapshot. Every paper
ingested since (all 202,251 from 2025-06 onward) arrived with 0, and the
existing ones have not moved since. Tier 0 ranks new readers' feeds by these
counts, so measured on 2026-09-30 it could not surface a single paper from the
last 14 months: 2506.05176 had 1,488 citations on Semantic Scholar and 0 here.

This script only READS Semantic Scholar and writes a local staging file.
Applying the counts to Turso and the sidecar is a separate, explicit step
(`--apply-sidecar`, `--apply-turso`), so a fetch can never touch production.

Usage
-----
    # fetch, newest papers first; resumable, safe to interrupt
    python scripts/refresh_citations.py fetch --sidecar data/metadata.sqlite --staging data/citations.sqlite

    # report what would change
    python scripts/refresh_citations.py diff --sidecar data/metadata.sqlite --staging data/citations.sqlite

    # write the counts into a sidecar file (edit a COPY; the image ships it)
    python scripts/refresh_citations.py apply-sidecar --sidecar data/metadata.sqlite --staging data/citations.sqlite

    # write changed counts to Turso (production metadata; run deliberately)
    python scripts/refresh_citations.py apply-turso --sidecar data/metadata.sqlite --staging data/citations.sqlite

S2_API_KEY is used when set and valid, throttled to its 1 request/second limit
(500 papers per request). Without it the shared public pool is used, with
backoff on 429; that pool gave up mid-run twice on 2026-09-30.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

import httpx

API = "https://api.semanticscholar.org/graph/v1/paper/batch"
BATCH = 500   # the batch endpoint's maximum


def _staging(path: str) -> sqlite3.Connection:
    # URI mode, so the read-only `file:...?mode=ro` ATTACH in diff/apply-turso
    # is honoured; a plain connection rejects it as "unable to open database".
    db = sqlite3.connect(f"file:{os.path.abspath(path)}", uri=True)
    db.execute("""CREATE TABLE IF NOT EXISTS citations (
        arxiv_id TEXT PRIMARY KEY, citation_count INTEGER, influential_citations INTEGER,
        found INTEGER NOT NULL, fetched_at TEXT NOT NULL)""")
    return db


_KEY = {"value": os.getenv("S2_API_KEY") or None}
# A key is limited to 1 request/second across all endpoints; stay under it.
_KEYED_INTERVAL_S = 1.1
_last = {"t": 0.0}


def _post(client: httpx.Client, ids: list[str]) -> list | None:
    headers = {"x-api-key": _KEY["value"]} if _KEY["value"] else {}
    for attempt in range(8):
        if headers:
            wait = _last["t"] + _KEYED_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last["t"] = time.monotonic()
        try:
            r = client.post(API, params={"fields": "citationCount,influentialCitationCount"},
                            headers=headers, json={"ids": [f"ARXIV:{i}" for i in ids]})
        except httpx.HTTPError as e:
            print(f"[citations] network error ({e}); retrying", file=sys.stderr)
            r = None
        if r is not None and r.status_code == 200:
            return r.json()
        if r is not None and r.status_code in (401, 403) and headers:
            print("[citations] S2_API_KEY rejected; using the public pool from now on", file=sys.stderr)
            _KEY["value"], headers = None, {}
            continue
        time.sleep(min(60, 2 ** attempt))
    return None


def fetch(args) -> int:
    src = sqlite3.connect(f"file:{os.path.abspath(args.sidecar)}?mode=ro", uri=True)
    db = _staging(args.staging)
    done = {r[0] for r in db.execute("SELECT arxiv_id FROM citations")}
    # Newest first. New-style ids (YYMM.NNNNN) sort by date; old-style ids
    # (hep-th/9901001) predate 2007 and would sort ahead of them on letters alone.
    ids = [r[0] for r in src.execute(
        "SELECT arxiv_id FROM papers ORDER BY arxiv_id GLOB '[0-9]*' DESC, arxiv_id DESC")
           if r[0] not in done]
    if args.since:
        ids = [i for i in ids if i[:4].isdigit() and i >= args.since]
    print(f"[citations] {len(ids):,} to fetch ({len(done):,} already staged)")
    t0 = time.time()
    with httpx.Client(timeout=120) as client:
        for n, i in enumerate(range(0, len(ids), BATCH)):
            chunk = ids[i:i + BATCH]
            res = _post(client, chunk)
            if res is None:
                print(f"[citations] giving up on batch at {chunk[0]}; re-run to resume", file=sys.stderr)
                return 1
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            db.executemany(
                "INSERT OR REPLACE INTO citations VALUES (?,?,?,?,?)",
                [(a, (p or {}).get("citationCount"), (p or {}).get("influentialCitationCount"),
                  int(p is not None), now) for a, p in zip(chunk, res)])
            db.commit()
            if n % 20 == 0:
                rate = (i + len(chunk)) / max(1e-9, time.time() - t0)
                print(f"[citations] {i + len(chunk):,}/{len(ids):,} at {chunk[-1]} "
                      f"({rate:.0f}/s, ~{(len(ids) - i) / max(rate, 1e-9) / 60:.0f} min left)", flush=True)
    print(f"[citations] done in {(time.time() - t0) / 60:.1f} min")
    return 0


def diff(args) -> int:
    db = _staging(args.staging)
    db.execute(f"ATTACH DATABASE 'file:{os.path.abspath(args.sidecar)}?mode=ro' AS s")
    rows = db.execute("""SELECT substr(c.arxiv_id,1,2) yy, count(*), sum(c.found),
            sum(c.citation_count > coalesce(p.citation_count,0)),
            sum(coalesce(p.citation_count,0) = 0 AND c.citation_count > 0)
        FROM citations c JOIN s.papers p USING (arxiv_id)
        GROUP BY yy ORDER BY yy DESC""").fetchall()
    print("year  staged   found   increased   was-0-now->0")
    for yy, n, found, inc, zero in rows:
        print(f"20{yy}  {n:>7,} {found:>7,} {inc:>10,} {zero:>13,}")
    return 0


def _changed(db: sqlite3.Connection) -> str:
    """Staged rows whose count differs from the sidecar's. Papers S2 does not
    know keep their stored counts: absent is not zero."""
    return """SELECT c.arxiv_id, c.citation_count, coalesce(c.influential_citations, 0)
              FROM citations c JOIN s.papers p USING (arxiv_id)
              WHERE c.found = 1 AND c.citation_count IS NOT NULL
                AND (c.citation_count IS NOT coalesce(p.citation_count, 0)
                     OR coalesce(c.influential_citations, 0) IS NOT coalesce(p.influential_citations, 0))"""


def apply_sidecar(args) -> int:
    db = sqlite3.connect(os.path.abspath(args.sidecar))
    db.execute(f"ATTACH DATABASE '{os.path.abspath(args.staging)}' AS st")
    with db:
        db.execute("""CREATE TEMP TABLE upd AS
            SELECT c.arxiv_id, c.citation_count AS cit, coalesce(c.influential_citations, 0) AS inf
            FROM st.citations c WHERE c.found = 1 AND c.citation_count IS NOT NULL""")
        db.execute("CREATE INDEX temp.upd_id ON upd(arxiv_id)")
        n = db.execute("""UPDATE papers SET citation_count = upd.cit, influential_citations = upd.inf
                          FROM upd WHERE papers.arxiv_id = upd.arxiv_id""").rowcount
        m = db.execute("""UPDATE paper_categories SET citation_count = upd.cit
                          FROM upd WHERE paper_categories.arxiv_id = upd.arxiv_id""").rowcount
    print(f"[citations] sidecar: {n:,} papers and {m:,} category rows updated")
    return 0


def apply_turso(args) -> int:
    url = os.environ.get("TURSO_URL", "").replace("libsql://", "https://").rstrip("/")
    token = os.environ.get("TURSO_DB_TOKEN", "")
    if not url or not token:
        print("TURSO_URL and TURSO_DB_TOKEN must be set", file=sys.stderr)
        return 2
    db = _staging(args.staging)
    db.execute(f"ATTACH DATABASE 'file:{os.path.abspath(args.sidecar)}?mode=ro' AS s")
    rows = db.execute(_changed(db)).fetchall()
    print(f"[citations] {len(rows):,} changed rows to write to Turso")
    sql = "UPDATE papers SET citation_count = ?, influential_citations = ? WHERE arxiv_id = ?"
    with httpx.Client(timeout=180) as client:
        for i in range(0, len(rows), 500):
            stmts = [{"type": "execute", "stmt": {"sql": sql, "args": [
                {"type": "integer", "value": str(c)}, {"type": "integer", "value": str(f)},
                {"type": "text", "value": a}]}} for a, c, f in rows[i:i + 500]]
            r = client.post(f"{url}/v2/pipeline", headers={"Authorization": f"Bearer {token}"},
                            json={"requests": stmts + [{"type": "close"}]})
            r.raise_for_status()
            errs = [x for x in r.json().get("results", []) if x.get("type") == "error"]
            if errs:
                raise RuntimeError(str(errs[0])[:200])
            if (i // 500) % 50 == 0:
                print(f"[citations] turso {i + len(stmts):,}/{len(rows):,}", flush=True)
    print("[citations] turso done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "diff", "apply-sidecar", "apply-turso"):
        p = sub.add_parser(name)
        p.add_argument("--sidecar", required=True, help="metadata sqlite to read ids/current counts from")
        p.add_argument("--staging", required=True, help="where fetched counts are written")
        if name == "fetch":
            p.add_argument("--since", default="", help="only ids >= this prefix, e.g. 2506")
    args = ap.parse_args()
    return {"fetch": fetch, "diff": diff, "apply-sidecar": apply_sidecar,
            "apply-turso": apply_turso}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
