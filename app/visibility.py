"""
What search engines and AI assistants do with the site, for /admin/visibility
(docs/AEO-PLAN.md phase 6).

Reads the counts app/analytics.py writes to ops_daily:

    bot:<class>:<name>:<page>     a crawler or assistant fetched a page
    bot_status:<class>:<code>     ... and got an error (status >= 400)
    ai_ref:<engine>:<page>        a person arrived from an AI answer
    indexnow:<status>             the daily IndexNow ping (app/indexnow.py)

Classes (app/bots.py): search crawlers build indexes; user_fetch is an
assistant reading a page because someone asked it something right now, the
closest thing to being used in an answer; training crawlers collect text for
models; preview fetches build link cards.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta

import time

from app import ops_digest

CLASSES = (("search", "Search crawlers"), ("user_fetch", "Assistants reading for a person"),
           ("training", "Training crawlers"), ("preview", "Link previews"), ("other", "Other bots"))
DAYS = 28
WEEK = 7


def summarise(days: dict[str, dict[str, int]], today: str) -> dict:
    """The dashboard's numbers from {day: {metric: n}} (pure). `today` is the
    newest day (YYYY-MM-DD); "this week" is it and the six days before."""
    t = date.fromisoformat(today)
    week = {(t - timedelta(days=i)).isoformat() for i in range(WEEK)}
    ordered = [(t - timedelta(days=i)).isoformat() for i in reversed(range(DAYS))]

    crawlers: dict[tuple[str, str], dict] = {}
    pages: Counter = Counter()
    refs: dict[tuple[str, str], dict] = {}
    errors: Counter = Counter()
    indexnow: Counter = Counter()
    per_day: dict[str, Counter] = defaultdict(Counter)
    for day, metrics in days.items():
        in_week = day in week
        for metric, n in metrics.items():
            parts = metric.split(":")
            if parts[0] == "bot" and len(parts) == 4:
                _b, cls, name, page = parts
                row = crawlers.setdefault((cls, name), {"cls": cls, "name": name, "week": 0, "month": 0,
                                                        "pages": Counter()})
                row["month"] += n
                row["week"] += n if in_week else 0
                row["pages"][page] += n
                pages[(cls, page)] += n
                per_day[day][cls] += n
            elif parts[0] == "bot_status" and len(parts) == 3:
                errors[(parts[1], parts[2])] += n
            elif parts[0] == "ai_ref" and len(parts) == 3:
                row = refs.setdefault((parts[1], parts[2]), {"engine": parts[1], "page": parts[2],
                                                             "week": 0, "month": 0})
                row["month"] += n
                row["week"] += n if in_week else 0
            elif parts[0] == "indexnow" and len(parts) == 2:
                indexnow[parts[1]] += n
    rows = sorted(crawlers.values(), key=lambda r: (-r["month"], r["name"]))
    for r in rows:
        r["top_pages"] = ", ".join(f"{p} {c}" for p, c in r["pages"].most_common(3))
        del r["pages"]

    def total(cls: str, window: str) -> int:
        return sum(r[window] for r in rows if r["cls"] == cls)

    return {
        "kpis": {"search_week": total("search", "week"), "user_fetch_week": total("user_fetch", "week"),
                 "training_week": total("training", "week"),
                 "referrals_week": sum(r["week"] for r in refs.values()),
                 "referrals_month": sum(r["month"] for r in refs.values())},
        "crawlers": rows,
        "referrals": sorted(refs.values(), key=lambda r: (-r["month"], r["engine"], r["page"])),
        "errors": [{"cls": c, "status": s, "n": n} for (c, s), n in sorted(errors.items(), key=lambda x: -x[1])],
        "indexnow": dict(indexnow),
        "days": [{"day": d, "values": dict(per_day.get(d, {}))} for d in ordered],
    }


async def report(today: str | None = None) -> dict:
    today = today or time.strftime("%Y-%m-%d", time.gmtime())
    return summarise(await ops_digest.last_days(DAYS), today)
