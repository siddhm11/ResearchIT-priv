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
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net_deadline import deadline  # noqa: E402
from app.licences import code_from_url  # noqa: E402  (standard library only)

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom",
      "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

# arXiv asks clients to identify themselves. On 2026-10-01, after a heavy night,
# oaipmh.arxiv.org answered ListRecords with HTTP 406 for a while (curl got 200
# at the same moment), then served the identical request normally: a temporary
# block, treated like a rate limit below.
USER_AGENT = "ResearchIT-ingest/1.0 (+https://github.com/siddhm11/ResearchIT-priv)"


def _arxiv_get(url: str, timeout: int):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}),
        timeout=timeout)


# Status codes that mean "slow down / blocked for now": wait and retry.
SOFT_BLOCK = (403, 406, 429, 503)


def _permanent(error: Exception) -> bool:
    """A malformed or missing-resource request (400, 404, 422...) fails at once;
    a soft block (SOFT_BLOCK) is waited out."""
    code = getattr(error, "code", None)
    return isinstance(code, int) and 400 <= code < 500 and code not in SOFT_BLOCK


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

    title = clean_text(_text(entry, "atom:title"))
    abstract = clean_text(_text(entry, "atom:summary"))
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
    if code in SOFT_BLOCK:
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
            with deadline(150, f"arXiv {category} @{start}"), _arxiv_get(url, 120) as r:
                xml = r.read()
            break
        except Exception as e:
            if _permanent(e) or attempt == FETCH_ATTEMPTS - 1:
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


# ── OAI-PMH harvest (the daily path) ─────────────────────────────────────────
#
# arXiv's OAI-PMH interface is built for incremental harvesting: everything
# changed between two dates, a whole archive per request stream, ~1,300
# records a page. A day across our seven archives is ~10-15 requests, where
# the search API needs one query per category (~100). Measured parity against
# rows written from the search API (901 overlapping papers, 2026-09-29):
# identical embedding text once NFKC-normalised (the search API sends U+03C6
# where OAI sends U+03D5) and identical categories once arXiv's alias pairs
# are collapsed as the search API does.

OAI_API = "https://oaipmh.arxiv.org/oai"
OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
# OAI lists both halves of an alias pair; the search API keeps the canonical one.
CATEGORY_ALIASES = {"math.IT": "cs.IT", "cs.SY": "eess.SY", "cs.NA": "math.NA",
                    "stat.TH": "math.ST", "math.MP": "math-ph", "q-fin.EC": "econ.GN"}


def clean_text(text: str) -> str:
    return unicodedata.normalize("NFKC", re.sub(r"\s+", " ", text or "").strip())


def canonical_categories(categories: str) -> str:
    cats = categories.split()
    present, out = set(cats), []
    for c in cats:
        if c in CATEGORY_ALIASES and CATEGORY_ALIASES[c] in present:
            continue                  # canonical is listed too: keep it at its own position
        c = CATEGORY_ALIASES.get(c, c)
        if c not in out:
            out.append(c)
    return " ".join(out)


def oai_set(category: str) -> str:
    """The archive-level OAI set holding `category`."""
    archive = category.split(".")[0]
    if archive in ("cs", "math", "stat", "q-bio", "econ", "eess", "q-fin"):
        return archive
    return "physics"                  # hep-*, astro-ph, quant-ph, cond-mat, physics.*


def parse_oai_record(rec) -> dict | None:
    header = rec.find("oai:header", OAI_NS)
    if header is not None and header.get("status") == "deleted":
        return None
    m = rec.find(".//a:arXiv", OAI_NS)
    if m is None:
        return None
    title = clean_text(m.findtext("a:title", "", OAI_NS))
    abstract = clean_text(m.findtext("a:abstract", "", OAI_NS))
    if len(title) < 5 or len(abstract) < 20:
        return None
    authors = []
    for au in m.findall("a:authors/a:author", OAI_NS):
        parts = (au.findtext("a:forenames", "", OAI_NS), au.findtext("a:keyname", "", OAI_NS),
                 au.findtext("a:suffix", "", OAI_NS))
        authors.append(" ".join(p.strip() for p in parts if p and p.strip()))
    cats = canonical_categories(" ".join((m.findtext("a:categories", "", OAI_NS) or "").split()))
    licence_url = (m.findtext("a:license", "", OAI_NS) or "").strip()
    return {
        "arxiv_id": m.findtext("a:id", "", OAI_NS).strip(),
        "title": title,
        "authors": ", ".join(authors[:20]),
        "abstract": abstract,
        "categories": cats,
        "primary_topic": cats.split()[0] if cats else "",
        "update_date": (m.findtext("a:updated", "", OAI_NS) or m.findtext("a:created", "", OAI_NS) or "")[:10],
        "licence_url": licence_url,           # "" when the record names none
        "licence": code_from_url(licence_url),
    }


def fetch_oai_page(set_spec: str, since: str, until: str, token: str | None):
    """One ListRecords page: (papers, next_token, complete_list_size, records).

    `records` counts every record on the page (deleted ones too, as arXiv's
    completeListSize does); it is None when the request failed."""
    params = ({"verb": "ListRecords", "resumptionToken": token} if token else
              {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_spec,
               "from": since, "until": until})
    url = f"{OAI_API}?" + urllib.parse.urlencode(params)
    for attempt in range(FETCH_ATTEMPTS):
        try:
            with deadline(240, f"OAI {set_spec}"), _arxiv_get(url, 180) as r:
                xml = r.read()
            break
        except Exception as e:
            if _permanent(e) or attempt == FETCH_ATTEMPTS - 1:
                print(f"    [oai {set_spec}] fetch failed: {str(e)[:90]}")
                return [], None, None, None
            time.sleep(retry_wait(e, attempt))
    root = ET.fromstring(xml)
    err = root.find("oai:error", OAI_NS)
    if err is not None:
        if err.get("code") == "noRecordsMatch":
            return [], None, 0, 0
        print(f"    [oai {set_spec}] error {err.get('code')}: {(err.text or '')[:90]}")
        return [], None, None, None
    papers = [p for p in (parse_oai_record(r) for r in root.iter(f"{{{OAI_NS['oai']}}}record")) if p]
    tok = root.find(".//oai:resumptionToken", OAI_NS)
    size = tok.get("completeListSize") if tok is not None else None
    records = sum(1 for _ in root.iter(f"{{{OAI_NS['oai']}}}record"))
    next_token = ((tok.text or "").strip() or None) if tok is not None else None
    return papers, next_token, (int(size) if size and size.isdigit() else None), records


def harvest(cats, since: str, until: str, min_month: str, state: dict, handle_page, save_state,
            *, fetch=None, delay: float = ARXIV_DELAY, log=print) -> tuple[int, int]:
    """Harvest every archive set holding `cats`; returns (examined, new).

    Keeps records with at least one of `cats` and a new-style id from
    `min_month` (YYMM) on. OAI's <created> is not the first-version date for
    revised papers (1810.04449 showed 2026-09-10), so the id decides "new".
    A set is complete only when its last page carries no resumption token;
    `state["oai:<set>"]` records that for the coverage report.
    """
    fetch = fetch or fetch_oai_page
    wanted = set(cats)
    seen = new = 0
    for set_spec in sorted({oai_set(c) for c in cats}):
        state[f"oai:{set_spec}"] = "started"
        token, records, expected = None, 0, None
        while True:
            papers, token, size, got = fetch(set_spec, since, until, token)
            if delay:
                time.sleep(delay)
            if got is None:
                state[f"oai:{set_spec}"] = "failed"
                break
            records += got
            expected = size if size is not None else expected
            keep = [p for p in papers
                    if wanted & set(p["categories"].split())
                    and re.match(r"\d{4}\.\d{4,5}$", p["arxiv_id"]) and p["arxiv_id"][:4] >= min_month]
            seen += len(keep)
            if keep:
                new += handle_page(keep)
            log(f"  oai {set_spec:<8} records={records:,}{'/' + format(expected, ',') if expected else ''} "
                f"kept={seen:,} new={new:,}")
            if not token:
                ok = expected is None or records == expected
                state[f"oai:{set_spec}"] = "done" if ok else f"short {records}/{expected}"
                break
        save_state()
    return seen, new


def report_oai_coverage(cats, state: dict) -> int:
    sets = sorted({oai_set(c) for c in cats})
    bad = [f"{s}={state.get(f'oai:{s}')}" for s in sets if state.get(f"oai:{s}") != "done"]
    if bad:
        print(f"coverage: {len(bad)} OAI sets incomplete -> re-run: " + ", ".join(bad))
        return 3
    print(f"coverage: all {len(sets)} OAI sets harvested to the end")
    return 0


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
    ap.add_argument("--source", choices=("search", "oai"), default="search",
                    help="search: per-category query API (backfills); oai: OAI-PMH (daily)")
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
    licence_tally: dict[str, int] = {}

    def handle_page(papers: list[dict]) -> int:
        ids = [p["arxiv_id"] for p in papers]
        if args.source == "oai":
            for p in papers:
                code = p["licence"] or ("unrecognised" if p["licence_url"] else "none")
                licence_tally[code] = licence_tally.get(code, 0) + 1
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
            upserter.upsert(todo, vecs)               # their licences ride in the same Turso batch
        # Papers stored before licences were kept get theirs when OAI lists them again.
        upserter.write_licences([p for p in papers if p["arxiv_id"] in have])
        return len(todo)

    def save_state() -> None:
        os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
        json.dump(state, open(args.state, "w"))

    if args.source == "oai":
        # OAI dates are inclusive datestamps; --until is exclusive like the search
        # path. Papers are "new" from the month before --since (weekend and
        # announcement lag), judged by id; re-runs skip stored papers anyway.
        from datetime import date, timedelta
        oai_until = (date.fromisoformat(args.until) - timedelta(days=1)).isoformat()
        first = date.fromisoformat(args.since).replace(day=1) - timedelta(days=1)
        seen_total, new_total = harvest(cats, args.since, oai_until, first.strftime("%y%m"), state,
                                        handle_page, save_state, log=lambda m: print(m, flush=True))
    else:
        seen_total, new_total = walk(cats, args.since, args.until, state, handle_page, save_state,
                                     limit=args.limit, log=lambda m: print(m, flush=True))

    print(f"\ndone: examined {seen_total:,}, new {new_total:,} "
          f"in {(time.time()-t0)/60:.1f} min")
    if licence_tally:
        print("licences: " + ", ".join(f"{k}={v:,}" for k, v in
                                        sorted(licence_tally.items(), key=lambda kv: -kv[1])))
    if args.limit:
        return 0
    if args.source == "oai":
        return report_oai_coverage(cats, state)
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
