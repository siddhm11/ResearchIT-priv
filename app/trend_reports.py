"""
Monthly trend reports: /trending/<yyyy-mm> and its CSV (docs/AEO-PLAN.md phase 3).

/trending changes every day, so nothing can cite "what it said in November".
Each daily computation of the topics (app/trending_topics.py) is also written
here as that month's report: the named topics, whether each passed the trend
test, its share of the lists at the start and end of the 26 weeks, the slope,
the q value and its most upvoted papers. Once the month is over its row is
never written again, so a past month's address shows what was computed on the
month's last run, for good. The current month's report says it is still
updated daily.

Numbers only: every sentence on the page is written by code from these rows
(app/routers/trending.py trend_sentence), never by a model.

Stored in `trend_snapshots` (one row per month, JSON), replicated to Turso so a
deploy keeps past months.
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import date as Date, datetime, timezone

import aiosqlite

from app import config, errors

MONTH = re.compile(r"^([0-9]{4})-(0[1-9]|1[0-2])$")
TOP_PAPERS = 3
VERSION = 1


def is_month(s: str) -> bool:
    return bool(MONTH.match(s or ""))


def month_of(iso: str) -> str:
    return iso[:7]


def month_label(month: str) -> str:
    """"2026-11" -> "November 2026"."""
    return f"{Date.fromisoformat(month + '-01'):%B %Y}"


def snapshot(topics: list[dict], coverage: dict, q_max: float, label) -> dict:
    """The report row from one computation (pure). `label(t)` says rising,
    cooling or steady; unnamed topics are left out, as everywhere a reader
    looks."""
    rows = []
    for t in topics:
        if not t.get("name"):
            continue
        rows.append({
            "slug": t.get("slug"), "name": t["name"], "label": label(t),
            "from_share": round(float(t["from_share"]), 5), "to_share": round(float(t["to_share"]), 5),
            "slope": round(float(t["slope"]), 6), "q": round(float(t["q"]), 4), "papers": int(t["papers"]),
            "also": list(t.get("also") or []),
            "rolling": [round(float(v), 5) for v in t.get("rolling") or []],
            "top": [{"arxiv_id": p["arxiv_id"], "title": " ".join((p["title"] or "").split()), "upvotes": p["upvotes"]}
                    for p in (t.get("top") or [])[:TOP_PAPERS]],
        })
    order = {"rising": 0, "cooling": 1, "steady": 2}
    rows.sort(key=lambda r: (order[r["label"]], -abs(r["slope"]) if r["label"] != "steady" else -r["to_share"]))
    return {"version": VERSION, "computed_at": coverage.get("computed_at"), "first_week": coverage.get("first_week"),
            "weeks": coverage.get("weeks"), "papers": coverage.get("papers"), "q_max": q_max, "topics": rows}


async def save(snap: dict, now: datetime | None = None) -> str | None:
    """Write `snap` as the current month's report. Past months are never
    touched: a computation stamped in an earlier month than now is dropped."""
    now = now or datetime.now(timezone.utc)
    computed = snap.get("computed_at") or ""
    month = month_of(computed)
    if not is_month(month) or month != now.strftime("%Y-%m"):
        return None
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO trend_snapshots (month, computed_at, payload, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(month) DO UPDATE SET computed_at = excluded.computed_at, payload = excluded.payload, "
            "updated_at = excluded.updated_at",
            (month, computed, json.dumps(snap, separators=(",", ":")), now.strftime("%Y-%m-%d %H:%M:%S")))
        await conn.commit()
    return month


async def record() -> None:
    """Called after each topic computation; never raises."""
    from app import trending_topics
    try:
        await save(snapshot(trending_topics._state["topics"], trending_topics.coverage(),
                            trending_topics.Q_MAX, trending_topics.label))
    except Exception as e:
        errors.report("trend_reports", "snapshot write failed", e)


async def get(month: str) -> dict | None:
    if not is_month(month):
        return None
    async with aiosqlite.connect(config.DB_PATH) as conn:
        row = await (await conn.execute("SELECT payload FROM trend_snapshots WHERE month = ?",
                                        (month,))).fetchone()
    return json.loads(row[0]) if row else None


async def months() -> list[str]:
    """Every month with a report, newest first."""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        rows = await (await conn.execute("SELECT month FROM trend_snapshots ORDER BY month DESC")).fetchall()
    return [r[0] for r in rows]


async def dates() -> dict[str, str]:
    """{month: day it was last computed}, for the sitemap's lastmod."""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        rows = await (await conn.execute("SELECT month, computed_at FROM trend_snapshots")).fetchall()
    return {m: (c or "")[:10] for m, c in rows}


def is_final(month: str, today: Date | None = None) -> bool:
    today = today or datetime.now(timezone.utc).date()
    return month < today.strftime("%Y-%m")


CSV_FIELDS = ("month", "topic", "slug", "trend", "share_start", "share_end", "change_points",
              "slope_points_per_4_weeks", "q_value", "papers", "other_phrases", "top_paper_arxiv_id",
              "top_paper_title")


def to_csv(month: str, snap: dict) -> str:
    """One row per topic. Shares are fractions of each week's list (0-1);
    change and slope in percentage points."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_FIELDS)
    for t in snap.get("topics") or []:
        top = (t.get("top") or [{}])[0]
        w.writerow([month, t["name"], t.get("slug") or "", t["label"], f"{t['from_share']:.5f}",
                    f"{t['to_share']:.5f}", f"{(t['to_share'] - t['from_share']) * 100:.2f}",
                    f"{t['slope'] * 100:.3f}", f"{t['q']:.4f}", t["papers"], "; ".join(t.get("also") or []),
                    top.get("arxiv_id", ""), top.get("title", "")])
    return buf.getvalue()
