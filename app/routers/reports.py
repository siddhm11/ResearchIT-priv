"""
Pages and files built to be cited (docs/AEO-PLAN.md phase 3):

  GET /trending/reports          every monthly trend report
  GET /trending/<yyyy-mm>        one month's rising, cooling and steady topics
  GET /trending/<yyyy-mm>.csv    the same as a table
  GET /trending/lasting.csv      every lasting paper on Hacker News
  GET /trending/lasting.rss      papers that most recently became lasting

Included after the trending router, so /trending/lasting and
/trending/topics/... are matched there first.

None of these is a ranked surface for a reader: they list topics and papers in
a fixed order, set no cookie and write no exposures. Each sentence is written
from the stored numbers. Behind the same switches as the pages they summarise.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from email.utils import format_datetime
from xml.sax.saxutils import escape

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from app import config, features, hn_svc, trend_reports, trending_topics
from app.templates_env import one_in, templates

router = APIRouter()

RSS_ITEMS = 50
LASTING_CSV_FIELDS = ("rank", "arxiv_id", "title", "year", "field", "people", "new_voice_months", "threads",
                      "first_linked", "last_linked", "became_lasting", "url")


def _not_found(request: Request):
    return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)


def _pct(share: float) -> str:
    return f"{share * 100:.1f}%"


def report_lead(snap: dict, month: str, final: bool) -> str:
    """The report's answer in a sentence or two, from its numbers (pure)."""
    rising = [t for t in snap["topics"] if t["label"] == "rising"]
    cooling = [t for t in snap["topics"] if t["label"] == "cooling"]
    since = trend_reports.month_label(snap["first_week"][:7]) if snap.get("first_week") else ""
    basis = (f"{snap.get('papers') or 0:,} papers on Hugging Face's Daily Papers over the "
             f"{snap.get('weeks') or 0} weeks from {since}")
    head = month_label(month) + ("" if final else " so far")
    if not rising and not cooling:
        lead = f"{head}: no topic's share of {basis} moved more than chance would move it."
        steady = [t for t in snap["topics"] if t["label"] == "steady"]
        if steady:
            t = max(steady, key=lambda t: (abs(t["to_share"] - t["from_share"]), t["name"]))
            verb = "rose" if t["to_share"] > t["from_share"] else "fell"
            lead += (f" The largest change, {t['name']} ({verb} from {one_in(t['from_share'])} to "
                     f"{one_in(t['to_share'])} papers), did not pass the test.")
        return lead
    parts = []
    if rising:
        t = rising[0]
        parts.append(f"{t['name']} rose fastest, from {one_in(t['from_share'])} to {one_in(t['to_share'])} papers")
    if cooling:
        t = cooling[0]
        parts.append(f"{t['name']} cooled most, from {one_in(t['from_share'])} to {one_in(t['to_share'])}")
    count = f"{len(rising)} topic{'s' if len(rising) != 1 else ''} rising and {len(cooling)} cooling"
    return f"{head}: {count}. " + "; ".join(parts) + f". Measured on {basis}."


def month_label(month: str) -> str:
    return trend_reports.month_label(month)


@router.get("/trending/reports", response_class=HTMLResponse)
async def reports_index(request: Request):
    if not features.visible(trending_topics.SWITCH, request):
        return _not_found(request)
    months = await trend_reports.months()
    return templates.TemplateResponse(request, "trend_reports.html", {
        "months": [{"month": m, "label": month_label(m), "final": trend_reports.is_final(m)} for m in months],
        "og_title": "Monthly AI research trend reports from Hugging Face Daily Papers",
        "og_description": "Each month: which research topics rose or cooled in Hugging Face's Daily Papers, "
                          "tested against chance, with the data as CSV."})


@router.get("/trending/lasting.csv")
async def lasting_csv(request: Request) -> Response:
    if not features.visible(hn_svc.SWITCH_LASTING, request):
        return Response(status_code=404)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(LASTING_CSV_FIELDS)
    for c in hn_svc.lasting_cards():
        joined = c.get("joined") or 0
        w.writerow([c.get("rank") or "", c["arxiv_id"], _title(c), c.get("year") or "",
                    c.get("field") or "", c["hn"]["people"], c["hn"]["months"], c["hn"]["threads"],
                    _iso(c.get("first")), _iso(c.get("last")), _iso(joined),
                    f"{config.PUBLIC_URL}/p/{c['arxiv_id']}"])
    return Response(buf.getvalue(), media_type="text/csv; charset=utf-8", headers={
        "Content-Disposition": 'inline; filename="researchit-hn-lasting.csv"',
        "Cache-Control": "public, max-age=3600"})


def _title(c: dict) -> str:
    """arXiv titles keep the line breaks of the submitted LaTeX; one space each."""
    return " ".join((c.get("title") or c["arxiv_id"]).split())


def _iso(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") if ts else ""


def lasting_rss(cards: list[dict], base: str) -> str:
    """RSS 2.0 of the papers that most recently became lasting (pure): the
    day a paper's third new-voice month began is its date."""
    items = sorted((c for c in cards if c.get("joined")), key=lambda c: (-c["joined"], c["arxiv_id"]))[:RSS_ITEMS]
    out = []
    for c in items:
        url = f"{base}/p/{c['arxiv_id']}"
        desc = (f"Linked on Hacker News by {c['hn']['people']} people in {c['hn']['months']} separate months "
                f"({c['hn']['span']}). arXiv {c['arxiv_id']}.")
        out.append(f"<item><title>{escape(_title(c))}</title><link>{escape(url)}</link>"
                   f"<guid isPermaLink=\"true\">{escape(url)}</guid>"
                   f"<pubDate>{format_datetime(datetime.fromtimestamp(c['joined'], timezone.utc))}</pubDate>"
                   f"<description>{escape(desc)}</description></item>")
    built = format_datetime(datetime.now(timezone.utc))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel>'
            "<title>Lasting on Hacker News — ResearchIT</title>"
            f"<link>{escape(base)}/trending/lasting</link>"
            "<description>arXiv papers that have just been linked on Hacker News by at least 3 people in at "
            "least 3 different months.</description>"
            f"<lastBuildDate>{built}</lastBuildDate>" + "".join(out) + "</channel></rss>\n")


@router.get("/trending/lasting.rss")
async def lasting_feed(request: Request) -> Response:
    if not features.visible(hn_svc.SWITCH_LASTING, request):
        return Response(status_code=404)
    return Response(lasting_rss(hn_svc.lasting_cards(), config.PUBLIC_URL),
                    media_type="application/rss+xml; charset=utf-8",
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/trending/{name}")
async def monthly_report(name: str, request: Request):
    month, as_csv = (name[:-4], True) if name.endswith(".csv") else (name, False)
    if not trend_reports.is_month(month) or not features.visible(trending_topics.SWITCH, request):
        return _not_found(request)
    snap = await trend_reports.get(month)
    if snap is None:
        return _not_found(request)
    if as_csv:
        return Response(trend_reports.to_csv(month, snap), media_type="text/csv; charset=utf-8", headers={
            "Content-Disposition": f'inline; filename="researchit-trends-{month}.csv"',
            "Cache-Control": "public, max-age=3600"})
    final = trend_reports.is_final(month)
    months = await trend_reports.months()
    older = next((m for m in months if m < month), None)
    newer = next((m for m in reversed(months) if m > month), None)
    computed = (snap.get("computed_at") or "")[:10]
    lead = report_lead(snap, month, final)
    topics = snap["topics"]
    for t in topics:
        live = trending_topics.by_slug(t["slug"]) if t.get("slug") else None
        t["href"] = trending_topics.path(live) if live else None
        t["change"] = (t["to_share"] - t["from_share"]) * 100
    label = month_label(month)
    return templates.TemplateResponse(request, "trend_report.html", {
        "month": month, "label": label, "final": final, "lead": lead, "computed": computed,
        "computed_label": _long(computed), "snap": snap,
        "rising": [t for t in topics if t["label"] == "rising"],
        "cooling": [t for t in topics if t["label"] == "cooling"],
        # Largest change first: what moved most, shown as not passing the test.
        "steady": sorted((t for t in topics if t["label"] == "steady"), key=lambda t: -abs(t["change"])),
        "since": month_label(snap["first_week"][:7]) if snap.get("first_week") else "",
        "older": older, "newer": newer, "older_label": month_label(older) if older else "",
        "newer_label": month_label(newer) if newer else "", "pct": _pct, "one_in": one_in,
        "og_title": f"Rising AI research topics, {label}: Hugging Face Daily Papers trend report",
        "og_description": lead,
        "jsonld": {
            "@context": "https://schema.org", "@type": "Dataset",
            "name": f"AI research topic trends in Hugging Face Daily Papers, {label}",
            "description": lead,
            "url": f"{config.PUBLIC_URL}/trending/{month}",
            "creator": {"@type": "Organization", "name": "ResearchIT", "url": config.PUBLIC_URL + "/"},
            "isAccessibleForFree": True,
            "temporalCoverage": f"{snap.get('first_week') or ''}/{computed}",
            **({"dateModified": computed} if computed else {}),
            "variableMeasured": "Weekly share of Hugging Face Daily Papers per research topic",
            "measurementTechnique": "Ward clustering of BGE-M3 embeddings; permutation test of the weekly-share "
                                    "slope (2,000 shuffles), Benjamini-Hochberg q < 0.05",
            "distribution": [{"@type": "DataDownload", "encodingFormat": "text/csv",
                              "contentUrl": f"{config.PUBLIC_URL}/trending/{month}.csv"}],
        }})


def _long(iso: str) -> str:
    if not iso:
        return ""
    d = datetime.fromisoformat(iso[:10])
    return f"{d.day} {d:%B %Y}"
