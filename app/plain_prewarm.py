"""
"In plain terms" written ahead for the papers new readers see first (audit F21).

Cards show a plain-language line only when one is already stored (card_data
reads the cache, never generates), and one was written only when somebody
opened the paper. So a new reader's first feed, made of papers nobody had
opened yet, fell back to abstracts. Every INTERVAL_S this writes lines for:

    the first page of each area's starter list (what onboarding serves first)
    the week's trending papers, strongest first

Skips papers already explained under any configured model, uses the full
abstract when it can be had (as the paper page does), and stops at DAILY_MAX
lines a day (PLAIN_PREWARM_DAILY; 0 turns it off) and after three failed calls
in a row, so Groq's free daily quota stays for readers. Lines are stored with
db.save_explanation, content-addressed and replicated like the paper page's.
"""
from __future__ import annotations

import asyncio
import os
import time

from app import abstract_svc, ai_usage, db, groq_svc, turso_svc

INTERVAL_S = int(os.getenv("PLAIN_PREWARM_INTERVAL_S", "21600"))
DAILY_MAX = int(os.getenv("PLAIN_PREWARM_DAILY", "150"))
FIRST_DELAY_S = int(os.getenv("PLAIN_PREWARM_FIRST_DELAY_S", "300"))
STARTER_FIRST_PAGE = 6
TRENDING_TOP = 40
SPACING_S = 2.0

_task: asyncio.Task | None = None
_state: dict = {"day": "", "written": 0, "last_run": None, "last_written": 0, "status": "not started"}


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _left() -> int:
    if _state["day"] != _today():
        _state.update(day=_today(), written=0)
    return max(0, DAILY_MAX - _state["written"])


async def candidates() -> list[str]:
    """Starter first pages, area by area, then trending papers by strength."""
    from app import discovery_svc, trending_shelf
    from app.config import CATEGORY_GROUPS, expand_category_groups
    out: list[str] = []
    for key in CATEGORY_GROUPS:
        try:
            papers = await discovery_svc.starter_papers({key: expand_category_groups([key])}, limit=12)
        except Exception as e:          # one area's list costs only its own lines
            print(f"[plain_prewarm] starter list {key} unavailable: {e}")
            continue
        out += [p["arxiv_id"] for p in papers[:STARTER_FIRST_PAGE] if p.get("arxiv_id")]
    trending = sorted(trending_shelf.trending_now().items(), key=lambda kv: -kv[1]["strength"])
    out += [aid for aid, _ in trending[:TRENDING_TOP]]
    return list(dict.fromkeys(out))


async def _explained(arxiv_id: str, abstract: str) -> bool:
    for model in groq_svc.explanation_models():
        if await db.get_explanation(groq_svc.explain_cache_key(arxiv_id, abstract, model=model)) is not None:
            return True
    return False


async def run_once() -> int:
    """One pass. Returns lines written."""
    if _left() <= 0:
        _state["status"] = "daily cap reached"
        return 0
    ids = await candidates()
    meta = await turso_svc.fetch_metadata_batch(ids) if ids else {}
    written, failures = 0, 0
    for aid in ids:
        if _left() <= 0 or failures >= 3:
            break
        paper = meta.get(aid)
        if not paper or not paper.get("abstract"):
            continue
        abstract = paper["abstract"]
        if abstract_svc.is_truncated(abstract):
            full = await abstract_svc.ensure(aid, timeout=4.0)
            if full and len(full) > len(abstract):
                abstract = full
        if await _explained(aid, abstract):
            continue
        generated = await groq_svc.explain_paper_result(paper.get("title") or "", abstract, aid)
        if generated.text and generated.model:
            key = groq_svc.explain_cache_key(aid, abstract, model=generated.model)
            await db.save_explanation(key, aid, generated.text, generated.model)
            written += 1
            failures = 0
            _state["written"] += 1
            ai_usage.count("explain", "prewarmed")
        else:
            failures += 1
        await asyncio.sleep(SPACING_S)
    _state.update(last_run=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), last_written=written,
                  status="stopped after failures" if failures >= 3 else "done")
    print(f"[plain_prewarm] wrote {written} plain-language lines ({_state['written']} today)")
    return written


async def _loop() -> None:
    await asyncio.sleep(FIRST_DELAY_S)
    while True:
        try:
            await run_once()
        except Exception as e:
            from app import errors
            errors.report("plain_prewarm", "pass failed", e)
        await asyncio.sleep(INTERVAL_S)


async def start() -> None:
    global _task
    if DAILY_MAX <= 0:
        _state["status"] = "off"
        return
    if _task is None:
        _state["status"] = "scheduled"
        _task = asyncio.create_task(_loop(), name="plain-prewarm")


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None


def stats() -> dict:
    return {**_state, "daily_max": DAILY_MAX, "left_today": _left()}
