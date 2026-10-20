"""
Owner-only pages. Visible only to a browser holding the preview cookie, set by
visiting any page once with ?preview=<PREVIEW_TOKEN> (app/features.py), or to
the request carrying that token itself; everyone else gets a 404.

    GET /admin/ai   where the AI goes: today's calls, tokens and outcomes per
                    feature, each model's limits, 14 days of token history and
                    what stands out (app/ai_usage.py). Counts only.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import ai_usage, features
from app.templates_env import templates

router = APIRouter()

def _owner(request: Request) -> bool:
    return features.has_preview(request) or \
        features.preview_cookie_action(request.query_params.get("preview")) == "set"


def _nice(top: float) -> float:
    """The smallest 1, 2, 2.5 or 5 × 10^k at or above `top`."""
    if top <= 0:
        return 1000
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
    return str(int(n))


def chart(history: list[dict]) -> dict:
    """The stacked token columns, one per day, oldest first, as percentages.

    Drawn in HTML, not SVG, so its text stays the same size on a phone and a
    laptop; segments stack bottom-up in FEATURES order.
    """
    top = _nice(max((d["total"] for d in history), default=0))
    columns = []
    for i, d in enumerate(history):
        day = time.strptime(d["day"], "%Y-%m-%d")
        columns.append({"day": d["day"], "label": time.strftime("%d %b", day).lstrip("0"),
                        "show_label": (len(history) - 1 - i) % 2 == 0,
                        "height": round(100 * d["total"] / top, 2), "total": d["total"], "tokens": d["tokens"],
                        "segments": [{"feature": f, "tokens": d["tokens"][f]}
                                     for f in ai_usage.FEATURES if d["tokens"][f] > 0]})
    ticks = [{"pct": 25 * k, "label": _compact(top * k / 4)} for k in range(5)]
    return {"columns": columns, "ticks": ticks, "empty": all(d["total"] == 0 for d in history)}


@router.get("/admin/ai", response_class=HTMLResponse)
async def ai_dashboard(request: Request):
    if not _owner(request):
        return HTMLResponse("Not found", status_code=404)
    data = await ai_usage.dashboard()
    return templates.TemplateResponse(request, "admin_ai.html",
                                      {**data, "chart": chart(data["history"]), "labels": ai_usage.LABELS,
                                       "features_order": ai_usage.FEATURES})
