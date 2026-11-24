"""
IndexNow: tell search engines which pages changed today (docs/AEO-PLAN.md, phase 2).

Bing (which feeds Copilot and is one of ChatGPT's sources), Yandex, Seznam and
Naver share pings sent to api.indexnow.org. Once a day, about START_DELAY_S
after the app starts (so the trend topics and Hacker News lists have loaded)
and then every DAY_S, this sends the sitemap's URLs whose lastmod is today:
the trend pages, Lasting, topic pages and papers that got a summary.

Off unless INDEXNOW_KEY is set; the key file is served at /{key}.txt
(app/routers/seo.py). Never raises into the app.
"""
from __future__ import annotations

import asyncio
import time
from urllib.parse import urlsplit

from app import config, errors, ops_digest

ENDPOINT = "https://api.indexnow.org/indexnow"
START_DELAY_S = 30 * 60
DAY_S = 24 * 3600
MAX_URLS = 10_000           # the protocol's limit per request
_task: asyncio.Task | None = None


def enabled() -> bool:
    return bool(config.INDEXNOW_KEY)


async def changed_urls() -> list[str]:
    """Absolute URLs of sitemap entries dated today."""
    from app.routers import seo
    today = time.strftime("%Y-%m-%d", time.gmtime())
    return [config.PUBLIC_URL + path for path, day in await seo.entries() if day == today][:MAX_URLS]


def payload(urls: list[str]) -> dict:
    return {"host": urlsplit(config.PUBLIC_URL).hostname, "key": config.INDEXNOW_KEY,
            "keyLocation": f"{config.PUBLIC_URL}/{config.INDEXNOW_KEY}.txt", "urlList": urls}


async def ping(urls: list[str]) -> int | None:
    """POST the URLs; the HTTP status, or None when nothing was sent."""
    if not enabled() or not urls:
        return None
    from app import http_client
    try:
        resp = await http_client.get_client().post(ENDPOINT, json=payload(urls), timeout=30,
                                                   headers={"User-Agent": config.OUTBOUND_USER_AGENT})
    except Exception as e:
        errors.report("indexnow", "ping failed", e)
        return None
    ops_digest.count(f"indexnow:{resp.status_code}")
    ops_digest.count("indexnow_urls", len(urls))
    if resp.status_code not in (200, 202):
        errors.report("indexnow", f"ping answered {resp.status_code}: {resp.text[:200]}",
                      RuntimeError(str(resp.status_code)))
    return resp.status_code


async def _loop() -> None:
    await asyncio.sleep(START_DELAY_S)
    while True:
        try:
            await ping(await changed_urls())
        except Exception as e:
            errors.report("indexnow", "daily ping failed", e)
        await asyncio.sleep(DAY_S)


async def start() -> None:
    global _task
    if enabled() and _task is None:
        _task = asyncio.create_task(_loop(), name="indexnow")


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None
