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
    export QDRANT_URL=... QDRANT_API_KEY=... ZILLIZ_URI=... ZILLIZ_TOKEN=...
    export TURSO_URL=... TURSO_DB_TOKEN=...

    # see what would be fetched, no model load, no writes
    python scripts/ingest_arxiv.py --since 2025-06-01 --dry-run

    # real run
    python scripts/ingest_arxiv.py --since 2025-06-01 --batch 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

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


def fetch_page(category: str, since: str, until: str, start: int) -> tuple[list[dict], int]:
    """One page of results. Returns (papers, total_available)."""
    q = (f"cat:{category} AND submittedDate:"
         f"[{since.replace('-', '')}0000 TO {until.replace('-', '')}0000]")
    url = f"{ARXIV_API}?" + urllib.parse.urlencode({
        "search_query": q,
        "start": start,
        "max_results": PAGE,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                xml = r.read()
            break
        except Exception as e:
            if attempt == 3:
                print(f"    [{category}] fetch failed: {str(e)[:90]}")
                return [], 0
            time.sleep(5 * (attempt + 1))
    root = ET.fromstring(xml)
    total_el = root.find("opensearch:totalResults", NS)
    total = int(total_el.text) if total_el is not None and total_el.text else 0
    papers = [p for p in (parse_entry(e) for e in root.findall("atom:entry", NS)) if p]
    return papers, total


# ── Stores ───────────────────────────────────────────────────────────────────

def turso_execute(url: str, token: str, stmts: list[dict]) -> None:
    payload = json.dumps({
        "requests": [{"type": "execute", "stmt": s} for s in stmts]
                    + [{"type": "close"}]}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v2/pipeline", data=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
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
    with urllib.request.urlopen(req, timeout=180) as r:
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

    for cat in cats:
        start = int(state.get(cat, 0))
        while True:
            papers, total = fetch_page(cat, args.since, args.until, start)
            time.sleep(ARXIV_DELAY)
            if not papers:
                break

            seen_total += len(papers)
            ids = [p["arxiv_id"] for p in papers]

            if args.dry_run:
                fresh = ids
            elif args.encode_only:
                fresh = ids
                vecs = encoder.encode([
                    f"{p['title'][:256]} {p['abstract'][:1024]}" for p in papers])
                d, s = vecs[0]
                nz = sum(len(sp) for _dv, sp in vecs) / len(vecs)
                norm = sum(x * x for x in d) ** 0.5
                print(f"    encoded {len(vecs)}: dense dim={len(d)} "
                      f"norm={norm:.4f} | sparse avg {nz:.0f} terms/doc",
                      flush=True)
            else:
                have = turso_existing(turso_url, turso_tok, ids)
                fresh = [i for i in ids if i not in have]
                todo = [p for p in papers if p["arxiv_id"] in set(fresh)]
                if todo:
                    vecs = encoder.encode([
                        f"{p['title'][:256]} {p['abstract'][:1024]}" for p in todo])
                    upserter.upsert(todo, vecs)

            new_total += len(fresh)
            start += len(papers)
            state[cat] = start
            os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
            json.dump(state, open(args.state, "w"))

            el = time.time() - t0
            print(f"  {cat:<16} {start:>6}/{total:<7} seen={seen_total:,} "
                  f"new={new_total:,}  {el/60:.1f} min", flush=True)

            if start >= total or (args.limit and seen_total >= args.limit):
                break
        if args.limit and seen_total >= args.limit:
            break

    print(f"\ndone: examined {seen_total:,}, new {new_total:,} "
          f"in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
