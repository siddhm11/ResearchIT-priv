#!/usr/bin/env python3
"""
Ingest new arXiv papers into Qdrant + Zilliz + Turso.

Why this exists
---------------
The corpus is a one-time Kaggle snapshot. Measured 2026-07-29:

    newest publication in corpus : 2025-05
    today                        : 2026-07-29

Against the live arXiv API, papers published since 2025-06 in just eight of
the configured categories:

    cs.AI     62,438      cs.LG   58,747      cs.CV   43,269
    cs.CL     29,295      quant-ph 21,026     cs.RO   14,370
    stat.ML    7,841      cs.IR    5,441

roughly 242k paper-slots, against a corpus of 1.6M. arXiv has published
almost half again as many cs.AI papers in that window as the entire index
contains. A recommender that cannot surface them is an archive, not a feed.

Design notes
------------
* Embedding text matches the original ingest exactly -- title[:256] plus
  abstract[:1024], max_length=512. New vectors have to live in the same
  distribution as the existing 1.6M or ANN results become inconsistent.
* Stored metadata keeps the FULL abstract. The old pipeline truncated to 500
  chars, which is what 90% of rows still look like, and that truncated text is
  what the cross-encoder reranks on. New rows do not inherit the flaw.
* Idempotent: papers already present are skipped, so re-running is safe.
* Checkpointed after every batch, so an interrupted run resumes.

Usage
-----
    export QDRANT_RECENT_URL=... QDRANT_RECENT_API_KEY=... ZILLIZ_URI=... ZILLIZ_TOKEN=...
    export TURSO_URL=... TURSO_DB_TOKEN=...

    # see what would be fetched, no model load, no writes
    python scripts/ingest_arxiv.py --since 2025-06-01 --dry-run

    # real run
    python scripts/ingest_arxiv.py --since 2025-06-01 --batch 200
"""
from __future__ import annotations

import argparse
import json
from collections import deque
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net_deadline import deadline  # noqa: E402

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom",
      "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

# arXiv asks for roughly one request every three seconds.
ARXIV_DELAY = 3.5
PAGE = 200  # results per API call; arXiv caps this well below its documented max

# Must match app/config.py CATEGORY_GROUPS.
DEFAULT_CATEGORIES = [
    "cs.CL", "cs.IR", "cs.CV", "cs.LG", "stat.ML", "cs.AI", "cs.RO",
    "hep-ph", "hep-th", "hep-ex", "hep-lat",
    "astro-ph.GA", "astro-ph.CO", "astro-ph.SR", "astro-ph.HE",
    "quant-ph", "math.CO", "math.AG", "math.NT", "math.PR", "math.AP",
    "q-bio.BM", "q-bio.GN", "q-bio.QM", "q-bio.NC",
    "econ.TH", "cs.GT", "cs.CR", "cs.DC", "cs.NI", "cs.HC",
    "cs.SD", "eess.AS", "physics.flu-dyn", "physics.comp-ph", "physics.optics",
    "cond-mat.mes-hall", "cond-mat.mtrl-sci", "cond-mat.str-el",
    "cs.SE", "cs.PL", "eess.SP", "eess.IV",
]


# ── arXiv fetch ──────────────────────────────────────────────────────────────

def _text(el, path: str) -> str:
    node = el.find(path, NS)
    return (node.text or "").strip() if node is not None else ""


def parse_entry(entry) -> dict | None:
    raw_id = _text(entry, "atom:id")
    m = re.search(r"abs/([^v]+)", raw_id)
    if not m:
        return None
    arxiv_id = m.group(1)

    title = re.sub(r"\s+", " ", _text(entry, "atom:title"))
    abstract = re.sub(r"\s+", " ", _text(entry, "atom:summary"))
    if len(title) < 5 or len(abstract) < 20:
        return None

    authors = [
        (a.findtext("atom:name", namespaces=NS) or "").strip()
        for a in entry.findall("atom:author", NS)
    ]
    cats = [c.attrib.get("term", "") for c in entry.findall("atom:category", NS)]
    cats = [c for c in cats if c]
    prim = entry.find("arxiv:primary_category", NS)
    primary = prim.attrib.get("term", "") if prim is not None else (
        cats[0] if cats else "")

    updated = _text(entry, "atom:updated")[:10]
    published = _text(entry, "atom:published")[:10]

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ", ".join(authors[:20]),
        "abstract": abstract,            # FULL text, deliberately not truncated
        "categories": " ".join(cats),
        "primary_topic": primary,
        "update_date": updated or published,
    }


FETCH_ATTEMPTS = 6


def retry_wait(error: Exception, attempt: int) -> float:
    """Seconds to wait before retrying a failed arXiv request.

    A rate limit (429/503) gets arXiv's Retry-After when it sends one, else an
    exponential back-off from 30 s, capped at 5 min: on 2026-10-01 retries 5,
    10 and 15 s apart failed every time once arXiv had started throttling.
    Other errors (timeouts, resets) retry sooner.
    """
    code = getattr(error, "code", None)
    if code in (429, 503):
        after = (getattr(error, "headers", None) or {}).get("Retry-After", "")
        if str(after).strip().isdigit():
            return min(300.0, float(after))
        return min(300.0, 30.0 * 2 ** attempt)
    return 5.0 * (attempt + 1)


def fetch_page(category: str, since: str, until: str, start: int) -> tuple[list[dict], int | None]:
    """One page of results. Returns (papers, total_available); total is None
    when the request failed, so a failure is never mistaken for an empty
    category."""
    q = (f"cat:{category} AND submittedDate:"
         f"[{since.replace('-', '')}0000 TO {until.replace('-', '')}0000]")
    url = f"{ARXIV_API}?" + urllib.parse.urlencode({
        "search_query": q,
        "start": start,
        "max_results": PAGE,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    })
    for attempt in range(FETCH_ATTEMPTS):
        try:
            with deadline(150, f"arXiv {category} @{start}"), \
                    urllib.request.urlopen(url, timeout=120) as r:
                xml = r.read()
            break
        except Exception as e:
            if attempt == FETCH_ATTEMPTS - 1:
                print(f"    [{category}] fetch failed: {str(e)[:90]}")
                return [], None
            time.sleep(retry_wait(e, attempt))
    root = ET.fromstring(xml)
    total_el = root.find("opensearch:totalResults", NS)
    total = int(total_el.text) if total_el is not None and total_el.text else 0
    papers = [p for p in (parse_entry(e) for e in root.findall("atom:entry", NS)) if p]
    return papers, total


# arXiv's query API fails (HTTP 500, every time) once start + max_results passes
# 10,000 for one query: measured 2026-10-01, cs.AI over nine weeks had 11,092
# papers and stopped at 10,000. Larger windows are split by date.
MAX_RESULTS_PER_QUERY = 9_800


def _midpoint(since: str, until: str) -> str:
    from datetime import date
    a, b = date.fromisoformat(since), date.fromisoformat(until)
    return (a + (b - a) / 2).isoformat()


def walk(cats, since, until, state, handle_page, save_state, *, fetch=None,
         limit: int = 0, delay: float = ARXIV_DELAY, log=print) -> tuple[int, int]:
    """Fetch every category's window page by page; returns (examined, new).

    A window is split by date only when its own first page reports more than
    MAX_RESULTS_PER_QUERY results, so small windows (the daily case) cost no
    extra requests. Probing every category up front doubled the requests and
    drew 19 minutes of HTTP 429 back-off from arXiv on 2026-10-01. Splits are
    recorded in `state` so a resumed run keeps the same windows.
    """
    fetch = fetch or fetch_page
    queue: deque = deque()
    for cat in cats:
        windows = state.setdefault(f"{cat}#windows", [[since, until]])
        queue.extend((cat, w) for w in windows)
    seen = new = 0
    t0 = time.time()
    while queue:
        cat, w = queue.popleft()
        windows = state[f"{cat}#windows"]
        key = window_key(cat, windows, w)
        start = int(state.get(key, 0))
        while True:
            papers, total = fetch(cat, w[0], w[1], start)
            if delay:
                time.sleep(delay)
            if total is not None and start == 0 and total > MAX_RESULTS_PER_QUERY:
                mid = _midpoint(w[0], w[1])
                if mid not in (w[0], w[1]):
                    halves = [[w[0], mid], [mid, w[1]]]
                    i = windows.index(w)
                    windows[i:i + 1] = halves
                    save_state()
                    queue.extendleft(reversed([(cat, h) for h in halves]))
                    log(f"  {cat}: {total:,} results in {w[0]}..{w[1]} -> split at {mid}")
                    break
            if total is not None:
                state[f"{key}#total"] = total
            if not papers:
                break
            seen += len(papers)
            new += handle_page(papers)
            start += len(papers)
            state[key] = start
            save_state()
            log(f"  {key:<16} {start:>6}/{total:<7} seen={seen:,} new={new:,}  "
                f"{(time.time() - t0) / 60:.1f} min")
            if start >= (total or 0) or (limit and seen >= limit):
                break
        if limit and seen >= limit:
            break
    return seen, new


def window_key(category: str, windows: list[list[str]], w: list[str]) -> str:
    return category if len(windows) == 1 else f"{category}@{w[0]}..{w[1]}"


# ── Stores ───────────────────────────────────────────────────────────────────

def turso_execute(url: str, token: str, stmts: list[dict]) -> None:
    payload = json.dumps({
        "requests": [{"type": "execute", "stmt": s} for s in stmts]
                    + [{"type": "close"}]}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v2/pipeline", data=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    with deadline(210, "Turso request"), urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    for res in data.get("results", []):
        if res.get("type") == "error":
            raise RuntimeError(str(res.get("error"))[:200])


def turso_existing(url: str, token: str, ids: list[str]) -> set[str]:
    ph = ", ".join("?" * len(ids))
    stmt = {"sql": f"SELECT arxiv_id FROM papers WHERE arxiv_id IN ({ph})",
            "args": [{"type": "text", "value": i} for i in ids]}
    payload = json.dumps({"requests": [{"type": "execute", "stmt": stmt},
                                       {"type": "close"}]}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v2/pipeline", data=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    with deadline(210, "Turso request"), urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    res = data["results"][0]
    if res.get("type") == "error":
        raise RuntimeError(str(res.get("error"))[:200])
    rows = res["response"]["result"]["rows"]
    return {row[0].get("value") for row in rows if row[0].get("value")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-06-01")
    ap.add_argument("--until", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--categories", default="", help="comma-separated; default = all")
    ap.add_argument("--batch", type=int, default=200, help="papers per encode/upsert")
    ap.add_argument("--limit", type=int, default=0, help="stop after N papers (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and parse only; no model load, no writes")
    ap.add_argument("--encode-only", action="store_true",
                    help="fetch, parse and encode, but write nothing. Validates "
                         "the GPU path without needing database credentials.")
    ap.add_argument("--state", default="data/ingest_state.json")
    args = ap.parse_args()

    # If a run stalls, its log names the line: dump every thread's stack every
    # INGEST_STACK_DUMP_S seconds (default 10 min) for as long as it runs.
    import faulthandler
    faulthandler.dump_traceback_later(int(os.getenv("INGEST_STACK_DUMP_S", "600")), repeat=True)

    cats = ([c.strip() for c in args.categories.split(",") if c.strip()]
            or DEFAULT_CATEGORIES)

    turso_url = os.environ.get("TURSO_URL", "")
    turso_tok = os.environ.get("TURSO_DB_TOKEN", "")
    writing = not (args.dry_run or args.encode_only)
    if writing and not (turso_url and turso_tok):
        print("TURSO_URL / TURSO_DB_TOKEN required for a real run", file=sys.stderr)
        return 2

    state = {}
    if os.path.isfile(args.state):
        state = json.load(open(args.state))

    encoder = upserter = None
    if not args.dry_run:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ingest_backends import Encoder
        encoder = Encoder()
        if writing:
            from ingest_backends import Upserter
            upserter = Upserter()

    seen_total = new_total = 0
    t0 = time.time()

    def handle_page(papers: list[dict]) -> int:
        ids = [p["arxiv_id"] for p in papers]
        if args.dry_run:
            return len(ids)
        if args.encode_only:
            vecs = encoder.encode([f"{p['title'][:256]} {p['abstract'][:1024]}" for p in papers])
            d, _s = vecs[0]
            nz = sum(len(sp) for _dv, sp in vecs) / len(vecs)
            print(f"    encoded {len(vecs)}: dense dim={len(d)} norm={sum(x * x for x in d) ** 0.5:.4f}"
                  f" | sparse avg {nz:.0f} terms/doc", flush=True)
            return len(ids)
        have = turso_existing(turso_url, turso_tok, ids)
        todo = [p for p in papers if p["arxiv_id"] not in have]
        if todo:
            vecs = encoder.encode([f"{p['title'][:256]} {p['abstract'][:1024]}" for p in todo])
            upserter.upsert(todo, vecs)
        return len(todo)

    def save_state() -> None:
        os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
        json.dump(state, open(args.state, "w"))

    seen_total, new_total = walk(cats, args.since, args.until, state, handle_page, save_state,
                                 limit=args.limit, log=lambda m: print(m, flush=True))

    print(f"\ndone: examined {seen_total:,}, new {new_total:,} "
          f"in {(time.time()-t0)/60:.1f} min")
    if args.limit:
        return 0
    return report_coverage(cats, state)


def report_coverage(cats: list[str], state: dict) -> int:
    """Compare each category's offset with arXiv's own total.

    An empty page mid-listing ends a category's loop, so "done" alone does not
    mean complete. Re-running with the same --state resumes from the offset.
    """
    gaps = []
    for cat in cats:
        windows = state.get(f"{cat}#windows") or [None]
        for w in windows:
            key = cat if w is None else window_key(cat, windows, w)
            got, total = int(state.get(key, 0)), state.get(f"{key}#total")
            if total is None or got < int(total):
                gaps.append(f"{key} {got}/{total}")
    if gaps:
        print(f"coverage: {len(gaps)} categories short of arXiv's total -> re-run to resume: "
              + ", ".join(gaps))
        return 3
    print(f"coverage: all {len(cats)} categories reached arXiv's total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
