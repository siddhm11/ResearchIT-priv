#!/usr/bin/env python3
"""Nightly search-quality check against the live Space (audit F24).

Runs 100 fixed known-answer searches (scripts/search_check_queries.json: 50
keyword and 50 question queries, each written from one paper's abstract; that
paper is the answer) through the real /search page and reports where the
answer landed: hit@10, MRR@10, and time per search. Exits non-zero when the
numbers fall below the floors, so a scheduled GitHub Action fails loudly.

    python scripts/search_check.py [--base URL] [--out results.json]

With PREVIEW_TOKEN set (the Space's owner token) the requests count as the
owner's, so the readers and searches dashboards leave them out. Searches are
spaced to stay inside /search's rate limit (30 a minute). Standard library only.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

QUERIES = Path(__file__).with_name("search_check_queries.json")
BASE = "https://siddhm11-researchit.hf.space"
# First live run sets the real baseline; these floors only catch a collapse
# (a dead retrieval arm or reranker), not a 2-point drift.
FLOOR_HIT10 = float(os.getenv("SEARCH_CHECK_FLOOR_HIT10", "0.50"))
FLOOR_MRR = float(os.getenv("SEARCH_CHECK_FLOOR_MRR", "0.35"))
SPACING_S = 3.0
_PAPER = re.compile(r'id="paper-([^"]+)"')


def ranked_ids(html: str) -> list[str]:
    """Paper ids in the order the results page shows them, first occurrence only."""
    return list(dict.fromkeys(_PAPER.findall(html)))


def score(ids: list[str], gold: list[str]) -> tuple[bool, float]:
    for rank, aid in enumerate(ids[:10], start=1):
        if aid in gold:
            return True, 1.0 / rank
    return False, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.getenv("SEARCH_CHECK_BASE", BASE))
    ap.add_argument("--out", default="search_check.json")
    ap.add_argument("--limit", type=int, default=0, help="first N queries only")
    args = ap.parse_args()
    queries = json.loads(QUERIES.read_text())
    if args.limit:
        queries = queries[:args.limit]

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "researchit-search-check/1"), ("HX-Request", "true")]
    token = os.getenv("PREVIEW_TOKEN", "").strip()
    if token:      # sets the owner cookie once; every later request carries it
        opener.open(f"{args.base}/?preview={urllib.parse.quote(token)}", timeout=60).read()

    rows, errors = [], 0
    for i, q in enumerate(queries):
        t0 = time.time()
        try:
            url = f"{args.base}/search?q={urllib.parse.quote(q['query'])}"
            html = opener.open(url, timeout=90).read().decode("utf-8", "replace")
            ids = ranked_ids(html)
        except Exception as e:                     # one failed request is a data point, not a crash
            print(f"[search_check] {q['qid']}: {e}", file=sys.stderr)
            ids, errors = [], errors + 1
        hit, rr = score(ids, q["gold"])
        rows.append({"qid": q["qid"], "kind": q["kind"], "hit10": hit, "rr10": rr,
                     "seconds": round(time.time() - t0, 2), "n": len(ids)})
        print(f"[search_check] {i + 1}/{len(queries)} {q['kind']:8s} {'hit' if hit else '   '} "
              f"rr={rr:.2f} {rows[-1]['seconds']:.1f}s  {q['query'][:60]}", flush=True)
        time.sleep(max(0.0, SPACING_S - (time.time() - t0)))

    def summary(rs):
        n = len(rs) or 1
        secs = sorted(r["seconds"] for r in rs) or [0]
        return {"n": len(rs), "hit10": sum(r["hit10"] for r in rs) / n,
                "mrr10": sum(r["rr10"] for r in rs) / n, "median_s": secs[len(secs) // 2],
                "p90_s": secs[int(0.9 * (len(secs) - 1))]}

    out = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base": args.base,
           "all": summary(rows), "errors": errors,
           "by_kind": {k: summary([r for r in rows if r["kind"] == k]) for k in ("keyword", "question")},
           "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    a = out["all"]
    print(f"\nhit@10 {a['hit10']:.1%}  MRR@10 {a['mrr10']:.3f}  median {a['median_s']}s  p90 {a['p90_s']}s"
          f"  errors {errors}  (keyword MRR {out['by_kind']['keyword']['mrr10']:.3f},"
          f" question MRR {out['by_kind']['question']['mrr10']:.3f})")
    failed = [m for m, ok in (("hit@10", a["hit10"] >= FLOOR_HIT10), ("MRR@10", a["mrr10"] >= FLOOR_MRR),
                              ("errors", errors <= len(rows) // 10)) if not ok]
    if failed:
        print(f"FAILED: {', '.join(failed)} below the floor", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
