#!/usr/bin/env python3
"""
Baseline reading metrics: the "before" numbers the redesign is judged against.

Two modes, both read-only against production:

    # Owner only: copy user_interactions from Turso into a local file.
    python scripts/baseline_metrics.py --export data/baseline.sqlite

    # Anyone: weekly metrics from that file, written to reports/baseline/.
    python scripts/baseline_metrics.py --db data/baseline.sqlite \\
        --since 2026-08-01 --fix-date 2026-10-20 \\
        --runtime-url https://siddhm11-researchit.hf.space/healthz/runtime

The export mode reads real reader activity, which is why only the project
owner runs it. The report holds counts and rates only, never a user id.

Definitions (also printed into every report, so nobody has to guess later)
---------------------------------------------------------------------------
  reader            a user_id with any event that week
  session           one reader's events with no gap over 30 minutes
  pass share        not_interested / (save + not_interested); unsave is not a pass
  opens             click + view events (feed click-throughs and direct visits)
  open-to-save      sessions where a paper is opened and later saved in the same
                    session, divided by sessions with an open
  seven-day return  reader-days followed by another session 1-7 days later,
                    divided by reader-days; days in the last week of data are
                    left out because their week is not over yet
  clicks by surface click events grouped by `source`, only from --fix-date on:
                    before the card fix every click was stored as
                    "recommendation" whatever page it came from
  excluded          user_ids with more than 500 events in one UTC day
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFINITIONS = __doc__[__doc__.index("  reader"):].rstrip()

SESSION_GAP = dt.timedelta(minutes=30)
RETURN_WINDOW_DAYS = 7
BOT_EVENTS_PER_DAY = 500

COLUMNS = ["id", "user_id", "paper_id", "event_type", "source", "position", "query_id",
           "ranker_version", "candidate_source", "cluster_id", "propensity", "policy_id",
           "timestamp"]


# ── loading ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Event:
    user_id: str
    paper_id: str
    event_type: str
    source: str | None
    ts: dt.datetime


def parse_ts(raw: str) -> dt.datetime:
    """SQLite datetime('now') strings carry no zone and are UTC."""
    text = raw.strip().replace("T", " ").rstrip("Z")
    ts = dt.datetime.fromisoformat(text)
    return ts.replace(tzinfo=dt.timezone.utc) if ts.tzinfo is None else ts.astimezone(dt.timezone.utc)


def load_events(db_path: str, since: dt.date | None = None) -> list[Event]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT user_id, paper_id, event_type, source, timestamp FROM interactions "
            "WHERE timestamp IS NOT NULL ORDER BY timestamp").fetchall()
    finally:
        conn.close()
    events = [Event(str(u), str(p), str(e), s, parse_ts(t)) for u, p, e, s, t in rows]
    if since:
        events = [e for e in events if e.ts.date() >= since]
    return events


# ── metrics ──────────────────────────────────────────────────────────────────

def heavy_users(events: list[Event], limit: int = BOT_EVENTS_PER_DAY) -> set[str]:
    per_day = Counter((e.user_id, e.ts.date()) for e in events)
    return {u for (u, _day), n in per_day.items() if n > limit}


def sessions(events: list[Event], gap: dt.timedelta = SESSION_GAP) -> list[list[Event]]:
    """Every reader's events split wherever two events are more than `gap` apart."""
    by_user: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    out = []
    for evs in by_user.values():
        evs.sort(key=lambda e: e.ts)
        cur = [evs[0]]
        for prev, e in zip(evs, evs[1:]):
            if e.ts - prev.ts > gap:
                out.append(cur)
                cur = []
            cur.append(e)
        out.append(cur)
    return out


def opened_then_saved(session: list[Event]) -> bool:
    opened: set[str] = set()
    for e in session:
        if e.event_type in ("click", "view"):
            opened.add(e.paper_id)
        elif e.event_type == "save" and e.paper_id in opened:
            return True
    return False


def iso_week(day: dt.date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


@dataclass
class Week:
    readers: set = field(default_factory=set)
    sessions: int = 0
    sessions_with_open: int = 0
    open_then_save: int = 0
    events: Counter = field(default_factory=Counter)
    reader_days: int = 0
    returning_reader_days: int = 0
    clicks_by_surface: Counter = field(default_factory=Counter)


def compute(events: list[Event], *, fix_date: dt.date | None = None,
            today: dt.date | None = None) -> dict:
    excluded = heavy_users(events)
    events = [e for e in events if e.user_id not in excluded]
    weeks: dict[str, Week] = defaultdict(Week)
    if not events:
        return {"weeks": {}, "excluded_users": len(excluded)}
    last_day = today or max(e.ts for e in events).date()

    for e in events:
        w = weeks[iso_week(e.ts.date())]
        w.readers.add(e.user_id)
        w.events[e.event_type] += 1
        if e.event_type == "click" and fix_date and e.ts.date() >= fix_date:
            w.clicks_by_surface[e.source or "unknown"] += 1

    session_days: dict[str, set[dt.date]] = defaultdict(set)
    for s in sessions(events):
        start = s[0].ts.date()
        w = weeks[iso_week(start)]
        w.sessions += 1
        if any(e.event_type in ("click", "view") for e in s):
            w.sessions_with_open += 1
            w.open_then_save += opened_then_saved(s)
        for e in s:
            session_days[e.user_id].add(e.ts.date())

    for days in session_days.values():
        ordered = sorted(days)
        for i, d in enumerate(ordered):
            if (last_day - d).days < RETURN_WINDOW_DAYS:
                continue                     # its return window isn't over yet
            w = weeks[iso_week(d)]
            w.reader_days += 1
            nxt = ordered[i + 1] if i + 1 < len(ordered) else None
            if nxt and 1 <= (nxt - d).days <= RETURN_WINDOW_DAYS:
                w.returning_reader_days += 1

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    out = {}
    for key in sorted(weeks):
        w = weeks[key]
        saves, passes = w.events["save"], w.events["not_interested"]
        out[key] = {
            "readers": len(w.readers),
            "sessions": w.sessions,
            "saves": saves,
            "passes": passes,
            "unsaves": w.events["unsave"],
            "opens": w.events["click"] + w.events["view"],
            "saves_per_session": ratio(saves, w.sessions),
            "pass_share": ratio(passes, saves + passes),
            "open_to_save": ratio(w.open_then_save, w.sessions_with_open),
            "seven_day_return": ratio(w.returning_reader_days, w.reader_days),
            "clicks_by_surface": dict(w.clicks_by_surface),
        }
    return {"weeks": out, "excluded_users": len(excluded)}


# ── report ───────────────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def render_markdown(result: dict, *, generated: str, fix_date: dt.date | None,
                    runtime: dict | None) -> str:
    lines = [f"# Baseline reading metrics, {generated}", ""]
    lines += ["| Week | Readers | Sessions | Saves per session | Pass share | Opens | "
              "Open-to-save | Seven-day return |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for week, m in result["weeks"].items():
        sps = "–" if m["saves_per_session"] is None else f"{m['saves_per_session']:.2f}"
        lines.append(f"| {week} | {m['readers']} | {m['sessions']} | {sps} | {_pct(m['pass_share'])} | "
                     f"{m['opens']} | {_pct(m['open_to_save'])} | {_pct(m['seven_day_return'])} |")
    lines.append("")
    lines.append(f"Excluded as automated: {result['excluded_users']} user ids.")
    if fix_date:
        lines.append(f"Clicks by surface count from {fix_date} only.")
        for week, m in result["weeks"].items():
            if m["clicks_by_surface"]:
                parts = ", ".join(f"{k} {v}" for k, v in sorted(m["clicks_by_surface"].items()))
                lines.append(f"- {week}: {parts}")
    else:
        lines.append("Clicks are not split by surface: before the card fix every click was "
                     "stored as \"recommendation\". Pass --fix-date once it is deployed.")
    if runtime:
        feed = (runtime.get("timings") or {}).get("feed", {}).get("build_ms")
        page = (runtime.get("timings") or {}).get("paper_page", {}).get("total_ms")
        lines.append("")
        lines.append("Live timings (last 200 requests per kind, since the last restart):")
        if feed:
            lines.append(f"- Feed build: p50 {feed['p50']} ms, p95 {feed['p95']} ms")
        if page:
            lines.append(f"- Paper page: p50 {page['p50']} ms, p95 {page['p95']} ms")
    lines += ["", "## Definitions", "", "```", DEFINITIONS, "```", ""]
    return "\n".join(lines)


# ── export (owner only) ──────────────────────────────────────────────────────

def export_from_turso(out_path: str, page: int = 5000) -> int:
    import httpx
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app import config
    from app.turso_sync import _uncell, _url

    if not (config.TURSO_URL and config.TURSO_DB_TOKEN):
        raise SystemExit("TURSO_URL and TURSO_DB_TOKEN are required for --export")
    conn = sqlite3.connect(out_path)
    conn.execute(f"CREATE TABLE IF NOT EXISTS interactions ({', '.join(COLUMNS)})")
    conn.execute("DELETE FROM interactions")
    last, total = 0, 0
    with httpx.Client(timeout=120) as http:
        while True:
            stmt = {"sql": f"SELECT {', '.join(COLUMNS)} FROM user_interactions WHERE id > ? ORDER BY id LIMIT {page}",
                    "args": [{"type": "integer", "value": str(last)}]}
            r = http.post(f"{_url()}/v2/pipeline",
                          headers={"Authorization": f"Bearer {config.TURSO_DB_TOKEN}"},
                          json={"requests": [{"type": "execute", "stmt": stmt}, {"type": "close"}]})
            r.raise_for_status()
            rows = r.json()["results"][0]["response"]["result"]["rows"]
            if not rows:
                break
            values = [[_uncell(c) for c in row] for row in rows]
            conn.executemany(f"INSERT INTO interactions VALUES ({', '.join('?' * len(COLUMNS))})", values)
            total += len(values)
            last = values[-1][0]
            print(f"[baseline] exported {total} rows", flush=True)
    conn.commit()
    conn.close()
    return total


def fetch_runtime(url: str) -> dict | None:
    import httpx
    try:
        r = httpx.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError) as e:
        print(f"[baseline] runtime snapshot failed: {e}")
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", metavar="OUT", help="copy user_interactions from Turso (owner only)")
    ap.add_argument("--db", help="local SQLite file made by --export")
    ap.add_argument("--since", type=dt.date.fromisoformat)
    ap.add_argument("--fix-date", type=dt.date.fromisoformat,
                    help="deploy date of the card-link fix; clicks are split by surface from then on")
    ap.add_argument("--runtime-url", help="a /healthz/runtime URL to snapshot (read-only GET)")
    ap.add_argument("--out", default="reports/baseline")
    args = ap.parse_args(argv)

    if args.export:
        n = export_from_turso(args.export)
        print(f"[baseline] wrote {n} interactions to {args.export}")
        return 0
    if not args.db:
        ap.error("--db or --export is required")

    events = load_events(args.db, args.since)
    result = compute(events, fix_date=args.fix_date)
    runtime = fetch_runtime(args.runtime_url) if args.runtime_url else None
    today = dt.date.today().isoformat()
    os.makedirs(args.out, exist_ok=True)
    md = render_markdown(result, generated=today, fix_date=args.fix_date, runtime=runtime)
    Path(args.out, f"{today}.md").write_text(md)
    Path(args.out, f"{today}.json").write_text(json.dumps(
        {"generated": today, "since": str(args.since or ""), "fix_date": str(args.fix_date or ""),
         **result, "runtime": runtime}, indent=1))
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
