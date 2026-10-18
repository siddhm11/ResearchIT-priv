#!/usr/bin/env python3
"""
Generate briefings for a fixed set of papers and write a page to rate them.

    python scripts/eval_briefings.py --papers scripts/eval_papers.txt \\
        --db /tmp/eval.db --out reports/briefings/2026-10-19

Reads the real arXiv HTML pages (one request each, paced) and calls the
configured model (LLM_PROVIDER, default the Groq chain) once per paper, into a
scratch database, never the app's. Writes:

    index.html    each paper: title, the briefing as readers would see it, every
                  point the checks dropped and why, time taken
    ratings.csv   one row per paper for the owner to mark good or bad, with a note
    summary.json  counts by status, median seconds, model calls and spend

The gate for switching BRIEFING on (docs/redesign/06-pr-6-7-paper-page-briefing.md):
at least 45 of 50 rated good, no displayed number missing from the paper (true
by construction; the dropped list proves the checks ran), median under 8 s.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def render(rows: list[dict], summary: dict) -> str:
    esc = html.escape
    parts = ["<!doctype html><meta charset='utf-8'><title>Briefing evaluation</title>",
             "<style>body{font:15px/1.5 system-ui;max-width:52rem;margin:2rem auto;padding:0 1rem;color:#16140f}"
             "h2{font-size:1.1rem;margin:2.2rem 0 .3rem}.m{color:#6e6860;font-size:.85rem}"
             ".s-ok{color:#2c5638}.s-insufficient,.s-failed{color:#8e3033}"
             "li{margin:.3rem 0}.drop{color:#8e3033;font-size:.85rem}.chip{font:600 .7rem monospace;"
             "background:#ddeafa;color:#13528c;padding:0 .3rem;border-radius:3px}</style>",
             f"<h1>Briefing evaluation</h1><p class='m'>{esc(json.dumps(summary))}</p>"]
    for r in rows:
        b = r.get("briefing") or {}
        parts.append(f"<h2><a href='https://arxiv.org/abs/{esc(r['id'])}'>{esc(r['id'])}</a> "
                     f"{esc(r.get('title', ''))}</h2>"
                     f"<p class='m'><span class='s-{esc(r['status'])}'>{esc(r['status'])}</span> · "
                     f"{r['seconds']:.1f} s</p>")
        if b.get("things"):
            parts.append("<ul>" + "".join(
                f"<li><b>{esc(t['lead'])}</b> {esc(t['text'])} <span class='chip'>{esc(t['label'])}</span></li>"
                for t in b["things"]) + "</ul>")
        if b.get("how"):
            parts.append("<ol>" + "".join(f"<li><b>{esc(s['name'])}</b>: {esc(s['text'])}</li>" for s in b["how"]) + "</ol>")
        if b.get("use"):
            parts.append(f"<p>{esc(b['use'])}</p>")
        for d in b.get("dropped", []):
            parts.append(f"<p class='drop'>Dropped: {esc(json.dumps(d))}</p>")
        if b.get("error"):
            parts.append(f"<p class='drop'>{esc(b['error'])}</p>")
    return "\n".join(parts)


async def run(ids: list[str], out: Path) -> dict:
    from app import briefing_svc, db, llm, paper_source_svc, turso_svc
    await db.init_db()
    rows = []
    for i, aid in enumerate(ids, 1):
        t0 = time.perf_counter()
        meta = (await turso_svc.fetch_metadata_batch([aid])).get(aid) or {}
        for attempt in range(4):
            try:
                status, b = await briefing_svc.get_or_generate(aid, meta.get("title") or "", meta.get("abstract") or "")
            except Exception as e:                  # keep going; the row records it
                status, b = "failed", {"error": f"{type(e).__name__}: {e}"}
            if b.get("error") != "model unavailable":
                break
            print(f"    every model is rate-limited; waiting 30 s (attempt {attempt + 1})", flush=True)
            await asyncio.sleep(30)
        secs = time.perf_counter() - t0
        rows.append({"id": aid, "title": meta.get("title", ""), "status": status, "briefing": b, "seconds": secs})
        print(f"[{i}/{len(ids)}] {aid} {status} {secs:.1f}s", flush=True)
    await paper_source_svc.stop()
    seconds = [r["seconds"] for r in rows if r["status"] == "ok"]
    summary = {
        "papers": len(rows),
        "by_status": {s: sum(r["status"] == s for r in rows) for s in ("ok", "insufficient", "failed")},
        "dropped_points": sum(len((r["briefing"] or {}).get("dropped", [])) for r in rows),
        "median_seconds_ok": round(statistics.median(seconds), 2) if seconds else None,
        "model": llm.configured_model(),
        "today": await llm.today(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(rows, summary))
    with open(out / "ratings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "status", "rating (good/bad)", "note"])
        for r in rows:
            w.writerow([r["id"], r["status"], "", ""])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=Path, default=ROOT / "scripts" / "eval_papers.txt")
    ap.add_argument("--db", required=True, help="a scratch database; never the app's")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    os.environ["DB_PATH"] = args.db
    os.environ.setdefault("TURSO_SYNC_DISABLED", "1")
    sys.path.insert(0, str(ROOT))
    ids = read_ids(args.papers)[: args.limit or None]
    summary = asyncio.run(run(ids, args.out))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
