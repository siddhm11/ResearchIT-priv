"""The weekly digest's pages (app/digest_svc.py builds the list).

    GET  /digest                        your digest this week, its private link and feed,
                                        and the email sign-up when email is configured
    GET  /digest/{token}                the same digest from any browser, by its link
    GET  /digest/{token}/feed.xml       RSS: the digest in a feed reader
    POST /digest/email                  ask for it by email (sends a confirmation link)
    GET  /digest/confirm/{code}         the confirmation page (a button: mail scanners
    POST /digest/confirm/{code}         open links, so a GET never subscribes anyone)
    GET  /digest/unsubscribe/{token}    unsubscribe page
    POST /digest/unsubscribe/{token}    unsubscribe; also RFC 8058 one-click from the inbox
    POST /digest/send                   send the digests due (a weekly cron, Bearer DIGEST_CRON_KEY)

The token is 128 random bits and is not the user id. None of these pages is
for search engines (robots.txt, noindex). A crawler gets no row, no cookie,
no exposure (CLAUDE.md §3.11); nothing here calls a model.
"""
from __future__ import annotations

import hmac
import os
import re
import secrets
import uuid
from datetime import datetime, timezone
from email.utils import format_datetime
from xml.sax.saxutils import escape

from fastapi import APIRouter, Cookie, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from app import bots, config, db, digest_svc, email_svc
from app.config import COOKIE_MAX_AGE, COOKIE_NAME
from app.routers.library import _same_site
from app.templates_env import templates

router = APIRouter()

_EMAIL = re.compile(r"^[^@\s<>\"']{1,64}@[^@\s<>\"']{1,255}\.[A-Za-z]{2,24}$")
RESEND_AFTER_S = 600        # a second confirmation email waits ten minutes


def digest_url(token: str) -> str:
    return f"{config.PUBLIC_URL}/digest/{token}"


def _cookie(resp, user_id: str):
    resp.set_cookie(COOKIE_NAME, user_id, max_age=COOKIE_MAX_AGE, httponly=True)
    return resp


async def _row_for(user_id: str) -> dict:
    return await db.get_digest_by_user(user_id) or await db.create_digest(user_id, secrets.token_urlsafe(16))


async def _record(user_id: str, digest: dict) -> None:
    rows = digest_svc.exposures(user_id, digest)
    if rows:
        try:
            await db.record_exposures(rows)
        except Exception as e:      # analytics must not cost the page
            print(f"[digest] exposure write failed: {type(e).__name__}: {e}")


def _context(row: dict | None, digest: dict | None, **extra) -> dict:
    return {"row": row, "digest": digest, "email_on": email_svc.configured(),
            "link": digest_url(row["token"]) if row else None,
            "feed": f"{digest_url(row['token'])}/feed.xml" if row else None,
            "page_noindex": True, "og_title": "Your weekly digest — ResearchIT", **extra}


@router.get("/digest", response_class=HTMLResponse)
async def my_digest(request: Request, notice: str = "", days: int = 7,
                    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    if bots.is_bot(request):
        return templates.TemplateResponse(request, "digest.html", _context(None, None))
    user_id = user_id or str(uuid.uuid4())
    row = await _row_for(user_id)
    digest = await digest_svc.build(user_id, days=min(max(days, 1), 14))
    await _record(user_id, digest)
    return _cookie(templates.TemplateResponse(request, "digest.html", _context(
        row, digest, own=True, notice=notice)), user_id)


@router.get("/digest/since", response_class=HTMLResponse)
async def since_last_visit(request: Request, user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    """The feed page's "since your last visit" box, fetched after the page
    loads: the digest built over the days since the reader was last here
    (Scholar Inbox's catch-up digest). Empty for a first visit, a visit
    earlier today, a crawler, or when nothing new appeared."""
    if bots.is_bot(request) or not user_id:
        return HTMLResponse("")
    today = datetime.now(timezone.utc).date()
    last = await db.last_visit_day(user_id, today.isoformat())
    if not last:
        return HTMLResponse("")
    try:
        gap = (today - datetime.fromisoformat(last).date()).days
    except ValueError:
        return HTMLResponse("")
    if gap < 1:
        return HTMLResponse("")
    digest = await digest_svc.build(user_id, days=min(max(gap, 1), 14))
    if digest["kind"] != "interests" or not digest["count"]:
        return HTMLResponse("")
    await _record(user_id, digest)
    return templates.TemplateResponse(request, "partials/since_last_visit.html",
                                      {"digest": digest, "gap": gap})


@router.get("/digest/{token}/feed.xml")
async def digest_feed(token: str, request: Request):
    row = await db.get_digest_by_token(token)
    if row is None:
        return Response("Not found", status_code=404, media_type="text/plain")
    digest = await digest_svc.build(row["user_id"])
    now = format_datetime(datetime.now(timezone.utc))
    items = []
    for s in digest["sections"]:
        for p in s["papers"]:
            # A plain address: a feed reader's visit is a direct visit, not a
            # ranked click (§3.11), since no exposure was recorded for it.
            url = f"{config.PUBLIC_URL}/p/{p['arxiv_id']}"
            items.append(
                f"<item><title>{escape(p['title'])}</title><link>{escape(url)}</link>"
                f"<guid isPermaLink=\"true\">{escape(url)}</guid><category>{escape(s['label'])}</category>"
                f"<description>{escape(' — '.join(x for x in (p['authors'], p['line']) if x))}</description></item>")
    xml = ('<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
           "<title>ResearchIT: new papers in your interests</title>"
           f"<link>{escape(digest_url(token))}</link>"
           "<description>New arXiv papers in your interests from the last 7 days, picked by ResearchIT.</description>"
           f"<lastBuildDate>{now}</lastBuildDate>{''.join(items)}</channel></rss>")
    return Response(xml, media_type="application/rss+xml; charset=utf-8",
                    headers={"Cache-Control": "private, max-age=3600", "X-Robots-Tag": "noindex"})


@router.get("/digest/confirm/{code}", response_class=HTMLResponse)
async def confirm_page(code: str, request: Request):
    row = await db.get_digest_by_confirm(code)
    return templates.TemplateResponse(request, "digest_status.html", {
        "state": "confirm" if row and row["status"] == "pending" else "gone", "code": code,
        "page_noindex": True}, status_code=200 if row else 404)


@router.post("/digest/confirm/{code}", response_class=HTMLResponse)
async def confirm(code: str, request: Request):
    row = await db.get_digest_by_confirm(code)
    if row is None or row["status"] != "pending":
        return templates.TemplateResponse(request, "digest_status.html",
                                          {"state": "gone", "page_noindex": True}, status_code=404)
    await db.update_digest(row["token"], status="active", confirm_token=None,
                           confirmed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return templates.TemplateResponse(request, "digest_status.html", {
        "state": "subscribed", "link": digest_url(row["token"]), "page_noindex": True})


@router.get("/digest/unsubscribe/{token}", response_class=HTMLResponse)
async def unsubscribe_page(token: str, request: Request):
    row = await db.get_digest_by_token(token)
    return templates.TemplateResponse(request, "digest_status.html", {
        "state": "unsubscribe" if row and row["email"] else "unsubscribed", "token": token,
        "page_noindex": True}, status_code=200 if row else 404)


@router.post("/digest/unsubscribe/{token}", response_class=HTMLResponse)
async def unsubscribe(token: str, request: Request):
    """From the page's button, or a mail provider's one-click POST (RFC 8058),
    which comes from no site at all: the secret token is the authority."""
    row = await db.get_digest_by_token(token)
    if row is not None and row["status"] != "unsubscribed":
        await db.update_digest(token, status="unsubscribed", email=None, confirm_token=None)
    return templates.TemplateResponse(request, "digest_status.html",
                                      {"state": "unsubscribed", "page_noindex": True},
                                      status_code=200 if row else 404)


@router.get("/digest/{token}", response_class=HTMLResponse)
async def shared_digest(token: str, request: Request):
    row = await db.get_digest_by_token(token)
    if row is None:
        return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
    digest = await digest_svc.build(row["user_id"])
    if not bots.is_bot(request):
        await _record(row["user_id"], digest)
    return templates.TemplateResponse(request, "digest.html", _context(row, digest, own=False))


@router.post("/digest/email")
async def ask_by_email(request: Request, email: str = Form(default=""),
                       user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    if not _same_site(request) or bots.is_bot(request) or not user_id:
        return RedirectResponse("/digest", status_code=303)
    if not email_svc.configured():
        return RedirectResponse("/digest?notice=email-off", status_code=303)
    email = email.strip()
    if not _EMAIL.match(email):
        return RedirectResponse("/digest?notice=bad-email", status_code=303)
    row = await _row_for(user_id)
    if row["status"] == "active" and row["email"] == email:
        return RedirectResponse("/digest?notice=already", status_code=303)
    if row["status"] == "pending" and _age_s(row["updated_at"]) < RESEND_AFTER_S:
        return RedirectResponse("/digest?notice=wait", status_code=303)
    code = secrets.token_urlsafe(16)
    await db.update_digest(row["token"], email=email, status="pending", confirm_token=code)
    confirm_url = f"{config.PUBLIC_URL}/digest/confirm/{code}"
    try:
        await email_svc.send(
            email, "Confirm your ResearchIT weekly digest",
            templates.get_template("emails/digest_confirm.html").render(confirm_url=confirm_url),
            f"Confirm your weekly digest of new papers in your interests:\n{confirm_url}\n\n"
            "If you didn't ask for this, ignore this email; nothing will be sent.")
    except (email_svc.NotConfigured, email_svc.SendFailed) as e:
        print(f"[digest] confirmation email failed: {e}")
        await db.update_digest(row["token"], status="link", email=None, confirm_token=None)
        return RedirectResponse("/digest?notice=send-failed", status_code=303)
    return RedirectResponse("/digest?notice=check-inbox", status_code=303)


def _age_s(ts: str | None) -> float:
    if not ts:
        return 1e9
    try:
        t = datetime.fromisoformat(ts.replace(" ", "T"))
    except ValueError:
        return 1e9
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds()


def render_email(row: dict, digest: dict) -> tuple[str, str, str, dict]:
    """(subject, html, text, headers) for one reader's digest."""
    unsub = f"{config.PUBLIC_URL}/digest/unsubscribe/{row['token']}"
    n = digest["count"]
    first = next((s["papers"][0]["title"] for s in digest["sections"] if s["papers"]), "")
    subject = f"{n} new paper{'s' if n != 1 else ''} for you this week" + (f": {first[:60]}" if first else "")
    html = templates.get_template("emails/digest.html").render(
        digest=digest, link=digest_url(row["token"]), unsub=unsub, base=config.PUBLIC_URL)
    lines = [f"New papers in your interests, last {digest['days']} days.", ""]
    for s in digest["sections"]:
        lines += [s["label"].upper(), ""]
        for p in s["papers"]:
            lines += [p["title"], f"  {config.PUBLIC_URL}/p/{p['arxiv_id']}?sf=digest_email&qid={digest['query_id']}"
                      f"&pos={p['position']}&prop=1.0&pol={digest_svc.POLICY}", ""]
    lines += [f"Open it on the web: {digest_url(row['token'])}", f"Unsubscribe: {unsub}"]
    headers = {"List-Unsubscribe": f"<{unsub}>", "List-Unsubscribe-Post": "List-Unsubscribe=One-Click"}
    return subject, html, "\n".join(lines), headers


@router.post("/digest/send")
async def send_due(request: Request):
    """Send every digest that is due. Called weekly by a scheduler outside the
    Space (it sleeps when idle), with Authorization: Bearer DIGEST_CRON_KEY."""
    key = os.getenv("DIGEST_CRON_KEY", "").strip()
    given = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
    if not key or not hmac.compare_digest(given, key):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not email_svc.configured():
        return JSONResponse({"configured": False, "sent": 0}, status_code=503)
    sent = skipped = failed = 0
    for row in await db.digests_due():
        digest = await digest_svc.build(row["user_id"])
        if not digest["count"]:
            skipped += 1                    # an empty email teaches people to ignore the next one
            continue
        subject, html, text, headers = render_email(row, digest)
        try:
            await email_svc.send(row["email"], subject, html, text, headers)
        except (email_svc.NotConfigured, email_svc.SendFailed) as e:
            print(f"[digest] send failed for {row['token'][:6]}…: {e}")
            failed += 1
            continue
        await db.update_digest(row["token"], last_sent_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        await _record(row["user_id"], digest)
        sent += 1
    print(f"[digest] sent {sent}, skipped {skipped} empty, {failed} failed")
    return {"configured": True, "sent": sent, "skipped_empty": skipped, "failed": failed}
