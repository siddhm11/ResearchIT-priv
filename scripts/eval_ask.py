#!/usr/bin/env python3
"""
Ask questions of a fixed set of papers and write a page to rate the answers.

    python scripts/eval_ask.py --papers scripts/eval_papers.txt --limit 20 \\
        --db /tmp/eval_ask.db --out reports/ask/2026-10-19 [--extra questions.csv]

Per paper it asks the three suggested questions plus one about something the
paper almost certainly doesn't cover (a correct answer is "the paper doesn't
say"), and any owner-written questions from --extra (a CSV of arxiv_id,question).
Real arXiv pages, the configured model, a scratch database. Writes index.html
(question, answer, quotes, cites, kind, time), ratings.csv and summary.json.

The gate for switching ASK on (docs/redesign/07-pr-8-ask-this-paper.md): every
shown quote verbatim (by construction; the summary counts dropped ones), at
least 85 of 100 answers rated correct and useful, at least 18 of 20 off-topic
questions answered "the paper doesn't say", median under 5 s, 95th percentile
under 12 s.
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
OFF_TOPIC = "What does this paper say about protein folding in baker's yeast?"


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def render(rows: list[dict], summary: dict) -> str:
    esc = html.escape
    out = ["<!doctype html><meta charset='utf-8'><title>Ask evaluation</title>",
           "<style>body{font:15px/1.5 system-ui;max-width:52rem;margin:2rem auto;padding:0 1rem}"
           ".q{font-weight:600;margin-top:1.4rem}.m{color:#6e6860;font-size:.85rem}"
           "blockquote{border-left:2px solid #cadfcd;margin:.4rem 0;padding-left:.6rem;font-style:italic}"
           ".chip{font:600 .7rem monospace;background:#ddeafa;color:#13528c;padding:0 .3rem;border-radius:3px}</style>",
           f"<h1>Ask evaluation</h1><p class='m'>{esc(json.dumps(summary))}</p>"]
    current = None
    for r in rows:
        if r["id"] != current:
            current = r["id"]
            out.append(f"<h2><a href='https://arxiv.org/abs/{esc(r['id'])}'>{esc(r['id'])}</a> {esc(r['title'])}</h2>")
        a = r["answer"]
        out.append(f"<p class='q'>{esc(r['question'])}</p><p>{esc(a.get('answer', ''))}</p>")
        for q in a.get("quotes", []):
            out.append(f"<blockquote>{esc(q['text'])}</blockquote>")
        chips = " ".join(f"<span class='chip'>{esc(c['label'])}</span>" for c in a.get("cites", []))
        out.append(f"<p class='m'>{esc(a.get('kind', ''))} · {r['seconds']:.1f} s {chips}</p>")
    return "\n".join(out)


async def run(ids: list[str], extra: dict[str, list[str]], out: Path) -> dict:
    from app import ask_svc, db, llm, paper_source_svc, turso_svc
    await db.init_db()
    rows = []
    for i, aid in enumerate(ids, 1):
        meta = (await turso_svc.fetch_metadata_batch([aid])).get(aid) or {"arxiv_id": aid}
        await paper_source_svc.ensure(aid)
        questions = [(q, k) for k, q in ask_svc.SUGGESTED.items()] + [(OFF_TOPIC, None)] + \
                    [(q, None) for q in extra.get(aid, [])]
        for question, key in questions:
            for attempt in range(4):
                t0 = time.perf_counter()
                a = await ask_svc.answer(aid, meta, question, user_id=None, suggested=key)
                if a.get("kind") != "busy":
                    break
                print("    every model is rate-limited; waiting 30 s", flush=True)
                await asyncio.sleep(30)
            secs = time.perf_counter() - t0
            rows.append({"id": aid, "title": meta.get("title", ""), "question": question, "key": key,
                         "answer": a, "seconds": secs})
            print(f"[{i}/{len(ids)}] {aid} {key or 'free'} {a.get('kind')} {secs:.1f}s", flush=True)
            await asyncio.sleep(3)                    # stay inside the free tier's per-minute budget
    await paper_source_svc.stop()
    times = sorted(r["seconds"] for r in rows if not r["answer"].get("cached"))
    off = [r for r in rows if r["question"] == OFF_TOPIC]
    summary = {
        "questions": len(rows),
        "by_kind": {k: sum(r["answer"].get("kind") == k for r in rows)
                    for k in ("answer", "unanswerable", "nosupport", "busy", "error")},
        "off_topic_unanswerable": f"{sum(r['answer'].get('kind') in ('unanswerable', 'nosupport') for r in off)}/{len(off)}",
        "quotes_shown": sum(len(r["answer"].get("quotes", [])) for r in rows),
        "quotes_dropped": sum((r["answer"].get("dropped") or {}).get("quotes", 0) for r in rows),
        "median_s": round(statistics.median(times), 2) if times else None,
        "p95_s": round(times[int(0.95 * (len(times) - 1))], 2) if times else None,
        "model": llm.configured_model(),
        "today": await llm.today(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(rows, summary))
    with open(out / "ratings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "question", "kind", "rating (good/bad)", "note"])
        for r in rows:
            w.writerow([r["id"], r["question"], r["answer"].get("kind"), "", ""])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=Path, default=ROOT / "scripts" / "eval_papers.txt")
    ap.add_argument("--extra", type=Path)
    ap.add_argument("--db", required=True, help="a scratch database; never the app's")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)
    os.environ["DB_PATH"] = args.db
    os.environ.setdefault("TURSO_SYNC_DISABLED", "1")
    sys.path.insert(0, str(ROOT))
    extra: dict[str, list[str]] = {}
    if args.extra:
        with open(args.extra, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and not row[0].startswith("#"):
                    extra.setdefault(row[0].strip(), []).append(row[1].strip())
    summary = asyncio.run(run(read_ids(args.papers)[: args.limit or None], extra, args.out))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
