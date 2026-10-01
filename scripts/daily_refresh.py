#!/usr/bin/env python3
"""
Daily refresh: yesterday's arXiv papers and recent citation counts, into the
live stores. Small by design: a few hundred to ~1,500 papers a day.

What it updates, and what that makes current on the site without a deploy
--------------------------------------------------------------------------
  * Qdrant `arxiv_recent` (vectors) and Turso `papers` (metadata): new papers
    reach personalised feeds, paper pages, similarity/neighbours and the map's
    on-demand placement the moment they are written.
  * Turso citation counts for the last three months of papers.
The metadata sidecar is NOT touched: it ships inside the Docker image, so the
starter feed and the keyword half of search only see new papers after the next
sidecar publish and deploy (docs/phases/PHASE7 §8).

Stages, each with a check that fails the run
--------------------------------------------
  1. preflight  credentials, both stores reachable, recent-shard size under
                --max-recent-points, baseline counts
  2. ingest     scripts/ingest_arxiv.py over [today - lookback, today + 1);
                re-run once if its coverage check reports a gap (exit 3)
  3. citations  Semantic Scholar counts for papers from the last 3 months;
                only changed rows are written
  4. verify     new Qdrant points == new Turso rows == papers the ingest
                reported; sampled new vectors are 1024-dim unit vectors and
                their Turso rows are complete
  5. summary    printed as JSON, and to $GITHUB_STEP_SUMMARY when set

Re-running is always safe: the ingest skips papers already stored and point
ids are derived from the arXiv id, so overlapping runs cannot collide.

Usage
-----
    python scripts/daily_refresh.py                 # the daily run
    python scripts/daily_refresh.py --dry-run       # stages 1 and 5 only
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REQUIRED_ENV = ("QDRANT_RECENT_URL", "QDRANT_RECENT_API_KEY", "TURSO_URL", "TURSO_DB_TOKEN")
COLLECTION = os.getenv("QDRANT_RECENT_COLLECTION", "arxiv_recent")


# ── small clients ────────────────────────────────────────────────────────────

class Stores:
    def __init__(self):
        self.qurl = os.environ["QDRANT_RECENT_URL"].rstrip("/")
        self.qkey = os.environ["QDRANT_RECENT_API_KEY"]
        self.turl = os.environ["TURSO_URL"].replace("libsql://", "https://").rstrip("/")
        self.ttok = os.environ["TURSO_DB_TOKEN"]
        self.http = httpx.Client(timeout=180)

    def q(self, path: str, body: dict | None = None, method: str = "POST"):
        r = self.http.request(method, f"{self.qurl}{path}", json=body,
                              headers={"api-key": self.qkey})
        r.raise_for_status()
        return r.json()["result"]

    def t(self, sql: str, args: tuple = ()) -> list[list]:
        stmt = {"sql": sql, "args": [{"type": "integer" if isinstance(a, int) else "text",
                                      "value": str(a)} for a in args]}
        r = self.http.post(f"{self.turl}/v2/pipeline", headers={"Authorization": f"Bearer {self.ttok}"},
                           json={"requests": [{"type": "execute", "stmt": stmt}, {"type": "close"}]})
        r.raise_for_status()
        res = r.json()["results"][0]
        if res.get("type") != "ok":
            raise RuntimeError(str(res)[:300])
        return [[c.get("value") for c in row] for row in res["response"]["result"]["rows"]]

    def t_many(self, stmts: list[dict]) -> None:
        r = self.http.post(f"{self.turl}/v2/pipeline", headers={"Authorization": f"Bearer {self.ttok}"},
                           json={"requests": [{"type": "execute", "stmt": s} for s in stmts] + [{"type": "close"}]})
        r.raise_for_status()
        errs = [x for x in r.json().get("results", []) if x.get("type") == "error"]
        if errs:
            raise RuntimeError(str(errs[0])[:300])

    def counts(self) -> dict:
        tmax, tcnt = self.t("SELECT MAX(rowid), COUNT(*) FROM papers")[0]
        return {"recent_points": self.q(f"/collections/{COLLECTION}/points/count", {"exact": True})["count"],
                "turso_rows": int(tcnt), "turso_max_rowid": int(tmax)}


# ── pure helpers (unit-tested) ───────────────────────────────────────────────

def window(today: dt.date, lookback_days: int) -> tuple[str, str]:
    """[today - lookback, today + 1): arXiv's submittedDate upper bound is
    midnight, so +1 day includes today. Overlap is free, since re-runs skip
    stored papers, and it absorbs weekend and announcement delays."""
    return ((today - dt.timedelta(days=lookback_days)).isoformat(),
            (today + dt.timedelta(days=1)).isoformat())


def recent_prefix(today: dt.date, months: int = 3) -> str:
    """YYMM id prefix of the first month in the citation window."""
    total = today.year * 12 + (today.month - 1) - (months - 1)
    return f"{(total // 12) % 100:02d}{total % 12 + 1:02d}"


def parse_ingest_new(output: str) -> int | None:
    m = re.findall(r"done: examined [\d,]+, new ([\d,]+)", output)
    return int(m[-1].replace(",", "")) if m else None


def changed_counts(current: dict[str, tuple[int, int]], fetched: dict[str, dict | None]) -> list[tuple[str, int, int]]:
    """Rows whose (citations, influential) differ. Papers Semantic Scholar
    does not know keep their stored counts: absent is not zero."""
    out = []
    for aid, paper in fetched.items():
        if not paper or paper.get("citationCount") is None:
            continue
        new = (int(paper["citationCount"]), int(paper.get("influentialCitationCount") or 0))
        if current.get(aid) != new:
            out.append((aid, *new))
    return out


def verify_counts(before: dict, after: dict, ingest_new: int | None) -> list[str]:
    problems = []
    dq = after["recent_points"] - before["recent_points"]
    dt_rows = after["turso_rows"] - before["turso_rows"]
    if dq != dt_rows:
        problems.append(f"new Qdrant points ({dq}) != new Turso rows ({dt_rows})")
    if ingest_new is not None and dt_rows != ingest_new:
        problems.append(f"new Turso rows ({dt_rows}) != papers the ingest reported ({ingest_new})")
    if after["turso_max_rowid"] != after["turso_rows"]:
        problems.append("Turso rowids are no longer contiguous (a row was deleted or replaced)")
    return problems


# ── stages ───────────────────────────────────────────────────────────────────

def stage_ingest(since: str, until: str, timeout_s: int = 3600) -> tuple[int | None, str]:
    state = Path(tempfile.mkdtemp()) / "state.json"
    cmd = [sys.executable, str(HERE / "ingest_arxiv.py"), "--since", since, "--until", until,
           "--state", str(state)]
    log = ""
    for attempt in (1, 2):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            print(out[-4000:], err[-6000:], sep="\n", flush=True)    # includes faulthandler stacks
            raise SystemExit(f"ingest exceeded {timeout_s // 60} min; stacks above") from None
        log += p.stdout + p.stderr
        print(p.stdout[-4000:], p.stderr[-2000:], sep="\n", flush=True)
        if p.returncode == 0:
            break
        if p.returncode != 3 or attempt == 2:
            raise SystemExit(f"ingest failed with exit {p.returncode}")
        print("[daily] coverage gap; resuming once", flush=True)
    # The resumed run only reports what IT added, so sum both.
    news = [int(n.replace(",", "")) for n in re.findall(r"done: examined [\d,]+, new ([\d,]+)", log)]
    return (sum(news) if news else None), log


def stage_citations(st: Stores, prefix: str) -> dict:
    spec = importlib.util.spec_from_file_location("refresh_citations", HERE / "refresh_citations.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    rows = st.t("SELECT arxiv_id, citation_count, influential_citations FROM papers "
                "WHERE arxiv_id >= ? AND arxiv_id < ?", (prefix, "9999"))
    current = {r[0]: (int(r[1] or 0), int(r[2] or 0)) for r in rows}
    ids = sorted(current)
    fetched: dict[str, dict | None] = {}
    chunks = [ids[i:i + rc.BATCH] for i in range(0, len(ids), rc.BATCH)]
    workers = 4 if rc._KEY["value"] else 1       # _post keeps keyed starts 1.1 s apart
    with httpx.Client(timeout=120) as client, ThreadPoolExecutor(workers) as pool:
        pending: deque = deque()
        nxt = 0
        for chunk in chunks:           # bounded window, consumed in order
            while nxt < len(chunks) and len(pending) < workers:
                pending.append(pool.submit(rc._post, client, chunks[nxt]))
                nxt += 1
            res = pending.popleft().result()
            if res is None:
                for f in pending:
                    f.cancel()
                raise SystemExit(f"Semantic Scholar batch failed at {chunk[0]}")
            fetched.update(zip(chunk, res))
    changes = changed_counts(current, fetched)
    sql = "UPDATE papers SET citation_count = ?, influential_citations = ? WHERE arxiv_id = ?"
    for i in range(0, len(changes), 500):
        st.t_many([{"sql": sql, "args": [{"type": "integer", "value": str(c)},
                                         {"type": "integer", "value": str(f)},
                                         {"type": "text", "value": a}]} for a, c, f in changes[i:i + 500]])
    found = sum(1 for v in fetched.values() if v)
    return {"papers": len(ids), "found": found, "changed": len(changes)}


def stage_sample_new(st: Stores, after_rowid: int, n: int = 5) -> list[str]:
    """Spot-check up to n papers this run added."""
    rows = st.t("SELECT arxiv_id, title, authors, abstract_preview, categories, primary_topic "
                "FROM papers WHERE rowid > ? ORDER BY rowid DESC LIMIT ?", (after_rowid, n))
    problems = []
    if not rows:
        return problems
    got = st.q(f"/collections/{COLLECTION}/points/scroll", {
        "filter": {"must": [{"key": "arxiv_id", "match": {"any": [r[0] for r in rows]}}]},
        "limit": 3 * len(rows), "with_payload": True, "with_vector": True})["points"]
    by_id: dict[str, list] = {}
    for p in got:
        by_id.setdefault(p["payload"]["arxiv_id"], []).append(p["vector"])
    for aid, title, authors, abstract, cats, label in rows:
        vecs = by_id.get(aid, [])
        if len(vecs) != 1:
            problems.append(f"{aid}: {len(vecs)} vectors in {COLLECTION} (expected 1)")
            continue
        v = vecs[0]
        norm = sum(x * x for x in v) ** 0.5
        if len(v) != 1024 or not 0.99 < norm < 1.01:
            problems.append(f"{aid}: vector dim {len(v)} norm {norm:.3f}")
        if not (title and authors and len(abstract or "") >= 20 and cats and label):
            problems.append(f"{aid}: incomplete Turso row")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=3)
    ap.add_argument("--citation-months", type=int, default=3)
    ap.add_argument("--max-recent-points", type=int, default=600_000,
                    help="refuse to write past this many points (free-tier headroom)")
    ap.add_argument("--today", default="", help="YYYY-MM-DD, for re-running a past day")
    ap.add_argument("--ingest-timeout-min", type=int, default=60,
                    help="fail the run if one ingest attempt takes longer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-citations", action="store_true")
    args = ap.parse_args()

    summary: dict = {"started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        print(f"[daily] missing env: {', '.join(missing)}", file=sys.stderr)
        return 2
    today = dt.date.fromisoformat(args.today) if args.today else dt.datetime.now(dt.timezone.utc).date()
    since, until = window(today, args.lookback_days)
    st = Stores()

    before = st.counts()
    summary.update({"window": [since, until], "before": before})
    if before["recent_points"] >= args.max_recent_points:
        summary["error"] = f"recent shard at {before['recent_points']:,} points, limit {args.max_recent_points:,}"
        return _finish(summary, 4)
    if args.dry_run:
        summary["dry_run"] = True
        return _finish(summary, 0)

    t0 = time.time()
    ingest_new, _log = stage_ingest(since, until, args.ingest_timeout_min * 60)
    summary["ingest"] = {"new_papers": ingest_new, "minutes": round((time.time() - t0) / 60, 1)}

    if not args.skip_citations:
        t0 = time.time()
        summary["citations"] = stage_citations(st, recent_prefix(today, args.citation_months))
        summary["citations"]["minutes"] = round((time.time() - t0) / 60, 1)

    after = st.counts()
    summary["after"] = after
    problems = verify_counts(before, after, ingest_new) + stage_sample_new(st, before["turso_max_rowid"])
    summary["problems"] = problems
    return _finish(summary, 1 if problems else 0)


def _finish(summary: dict, code: int) -> int:
    summary["exit"] = code
    text = json.dumps(summary, indent=1)
    print(text)
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as fh:
            fh.write(f"### Daily refresh {'OK' if code == 0 else 'FAILED'}\n\n```json\n{text}\n```\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
