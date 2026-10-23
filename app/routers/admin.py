"""
Owner-only dashboards. Visible only to a browser holding the preview cookie, set
by visiting any page once with ?preview=<PREVIEW_TOKEN> (app/features.py), or
to the request carrying that token itself; everyone else gets a 404.

    GET /admin/ai          where the AI goes (app/ai_usage.py)
    GET /admin/readers     readers and what they do (app/analytics.py)
    GET /admin/searches    what people search for and what comes of it
    GET /admin/ai/history  every AI call, by the paper or search it was for

The owner's own use is left out of readers and searches unless ?me=1.
Counts and text only: no reader is shown against a query.
"""
from __future__ import annotations

import math
import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import ai_usage, analytics, features
from app.templates_env import templates

router = APIRouter()


def _owner(request: Request) -> bool:
    return features.has_preview(request) or \
        features.preview_cookie_action(request.query_params.get("preview")) == "set"


def _nice(top: float) -> float:
    """A round scale top: a multiple of 4 for small counts (whole-number ticks),
    else the smallest 1, 2, 2.5 or 5 × 10^k at or above `top`."""
    if top <= 0:
        return 4
    if top <= 20:
        return max(4, 4 * math.ceil(top / 4))
    k = 10 ** (len(str(int(top))) - 1)
    for step in (1, 2, 2.5, 5, 10):
        if step * k >= top:
            return step * k
    return 10 * k


def _compact(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1000:
        return f"{n / 1000:.1f}K".replace(".0K", "K")
    return f"{n:g}"


def _day_label(day: str) -> str:
    return time.strftime("%d %b", time.strptime(day, "%Y-%m-%d")).lstrip("0")


def bars(points: list[dict], series: list[tuple[str, str]], unit: str = "") -> dict:
    """Stacked columns as percentages.

    points: [{"label": str, "values": {key: n}}], oldest first.
    series: [(key, label)], stacked bottom-up, coloured by slot in this order.
    About seven columns are labelled (every other one for two weeks, one in five
    for a month), counting back from the newest, so labels never collide.
    """
    totals = [sum(p["values"].get(k, 0) or 0 for k, _ in series) for p in points]
    top = _nice(max(totals, default=0))
    stride = max(1, math.ceil(len(points) / 7))
    columns = []
    for i, (p, total) in enumerate(zip(points, totals)):
        values = {k: p["values"].get(k, 0) or 0 for k, _ in series}
        columns.append({
            "label": p["label"], "show_label": (len(points) - 1 - i) % stride == 0,
            "height": round(100 * total / top, 2),
            "segments": [{"slot": j + 1, "n": values[k]} for j, (k, _) in enumerate(series) if values[k] > 0],
            "tip": {"title": p["label"], "total": total, "unit": unit,
                    "rows": [{"slot": j + 1, "label": lab, "n": values[k]}
                             for j, (k, lab) in reversed(list(enumerate(series))) if values[k]]},
        })
    legend = [{"slot": j + 1, "label": lab, "total": sum(p["values"].get(k, 0) or 0 for p in points)}
              for j, (k, lab) in enumerate(series)]
    ticks = [{"pct": 25 * k, "label": _compact(top * k / 4)} for k in range(5)]
    return {"columns": columns, "ticks": ticks, "legend": legend, "empty": not any(totals)}


def _me(request: Request) -> bool:
    return request.query_params.get("me") == "1"


@router.get("/admin/ai", response_class=HTMLResponse)
async def ai_dashboard(request: Request):
    if not _owner(request):
        return HTMLResponse("Not found", status_code=404)
    data = await ai_usage.dashboard()
    chart = bars([{"label": _day_label(d["day"]), "values": d["tokens"]} for d in data["history"]],
                 [(f, ai_usage.LABELS[f]) for f in ai_usage.FEATURES], unit="tokens")
    return templates.TemplateResponse(request, "admin_ai.html", {
        **data, "chart": chart, "labels": ai_usage.LABELS, "features_order": ai_usage.FEATURES, "active": "ai"})


@router.get("/admin/readers", response_class=HTMLResponse)
async def readers_dashboard(request: Request):
    if not _owner(request):
        return HTMLResponse("Not found", status_code=404)
    data = await analytics.readers_report(include_owner=_me(request))
    daily = bars([{"label": _day_label(d["day"]), "values": {"new": d["new"], "returning": d["returning"]}}
                  for d in data["days"]], [("new", "New readers"), ("returning", "Returning readers")],
                 unit="readers")
    monthly = bars([{"label": time.strftime("%b %Y", time.strptime(m["month"], "%Y-%m")),
                     "values": {"r": m["readers"]}} for m in data["months"]],
                   [("r", "Readers who saved, passed or opened a paper")], unit="readers")
    return templates.TemplateResponse(request, "admin_readers.html", {
        **data, "daily": daily, "monthly": monthly, "active": "readers", "owner_toggle": data["include_owner"]})


@router.get("/admin/searches", response_class=HTMLResponse)
async def searches_dashboard(request: Request):
    if not _owner(request):
        return HTMLResponse("Not found", status_code=404)
    data = await analytics.searches_report(include_owner=_me(request))
    chart = bars([{"label": _day_label(d["day"]),
                   "values": {"found": d["searches"] - d["no_results"], "none": d["no_results"]}}
                  for d in data["days"]], [("found", "Found results"), ("none", "Found nothing")], unit="searches")
    return templates.TemplateResponse(request, "admin_searches.html", {
        **data, "chart": chart, "active": "searches", "owner_toggle": data["include_owner"]})


@router.get("/admin/ai/history", response_class=HTMLResponse)
async def ai_history_dashboard(request: Request):
    if not _owner(request):
        return HTMLResponse("Not found", status_code=404)
    data = await analytics.ai_history()
    return templates.TemplateResponse(request, "admin_history.html", {**data, "active": "history"})
