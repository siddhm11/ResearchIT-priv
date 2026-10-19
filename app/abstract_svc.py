"""
Full abstracts for papers whose stored copy stops at 500 characters.

WHY THIS EXISTS
---------------
The metadata sidecar caps abstracts at 500 characters, and about 90% of
pre-2025 papers hit the cap. The paper page then shows a stump, and "In plain
terms" is written from it. The sidecar ships inside the Docker image, so it
can't be repaired in place; the repair lives in the `paper_abstracts` table,
which turso_sync replicates so it survives deploys.

HOW
---
The first time anyone opens such a paper, `request()` queues it. One worker
drains the queue in batches of up to 20 ids per arXiv API call, at least 3.1 s
apart (arXiv asks for one request every three seconds), honours Retry-After,
and stops for the day at FULL_ABSTRACTS_DAILY_CAP ids. `overlay()` swaps stored
full abstracts into paper dicts on every read.

It does not use arxiv_svc.fetch_metadata_batch, whose 0.35 s spacing is ten
times faster than arXiv's published rate.

Switch: FULL_ABSTRACTS (off | on). Reading stored abstracts happens either way.
"""
from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET

import aiosqlite
import httpx

from app import config, errors, features, http_client, runtime_stats

SWITCH = "FULL_ABSTRACTS"
BATCH = 20
MIN_INTERVAL_S = 3.1
FAIL_SKIP_S = 24 * 3600
SOFT_BLOCK = (403, 429, 503)
MAX_RETRY_AFTER_S = 300.0
DEFAULT_RETRY_AFTER_S = 30.0

# New-style ids (2609.31507) and old-style ones (hep-th/9901001, math.GT/0309136).
_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z][a-z\-]*(\.[A-Z]{2})?/\d{7})$")


def _now() -> float:               # indirection so tests can drive the clock
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def is_truncated(abstract: str | None) -> bool:
    from app.templates_env import _is_truncated
    return _is_truncated(abstract)


# ── reads ────────────────────────────────────────────────────────────────────

async def stored(arxiv_ids: list[str]) -> dict[str, str]:
    """Full abstracts already saved, for whichever of these ids have one."""
    ids = [i for i in dict.fromkeys(arxiv_ids) if i]
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute(
            f"SELECT arxiv_id, abstract FROM paper_abstracts WHERE arxiv_id IN ({marks})", ids)
        return {a: t for a, t in await cur.fetchall()}


async def overlay(papers: list[dict]) -> list[dict]:
    """Copies of `papers` with a saved full abstract in place of a cut one.

    Never edits the dicts it is given: turso_svc's in-process cache hands the
    same dict to every request, so an in-place edit would leak between them.
    """
    cut = [p["arxiv_id"] for p in papers if p and is_truncated(p.get("abstract"))]
    try:
        full = await stored(cut) if cut else {}
    except Exception as e:                       # an optimisation, never a failure
        errors.report("abstract_svc", "reading stored abstracts failed", e)
        full = {}
    out = []
    for p in papers:
        text = full.get(p["arxiv_id"]) if p else None
        if text and len(text) > len(p.get("abstract") or ""):
            p = {**p, "abstract": text, "abstract_source": "full"}
        out.append(p)
    return out


async def save(arxiv_id: str, abstract: str, source: str = "arxiv_api") -> None:
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO paper_abstracts (arxiv_id, abstract, source) VALUES (?, ?, ?)",
            (arxiv_id, abstract, source))
        await conn.commit()


# ── the fetch queue ──────────────────────────────────────────────────────────

class _State:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.pending: dict[str, asyncio.Future] = {}
        self.queue: list[str] = []
        self.wake = asyncio.Event()
        self.worker: asyncio.Task | None = None
        self.last_call = -MIN_INTERVAL_S
        self.failed_until: dict[str, float] = {}
        self.day = ""
        self.used_today = 0


_state: _State | None = None


def _get_state() -> _State:
    """Queue state bound to the running loop (tests start a new loop per client)."""
    global _state
    loop = asyncio.get_running_loop()
    if _state is None or _state.loop is not loop:
        _state = _State(loop)
    return _state


def valid_id(arxiv_id: str) -> bool:
    return bool(_ID.match(arxiv_id or ""))


def request(arxiv_id: str) -> asyncio.Future | None:
    """Queue a fetch of this paper's full abstract; None when it won't be fetched.

    Callers pass only papers whose stored abstract is cut, which means the paper
    exists in our metadata: an id that came from nowhere never reaches arXiv.
    """
    if not features.active(SWITCH) or not valid_id(arxiv_id):
        return None
    st = _get_state()
    if st.failed_until.get(arxiv_id, 0.0) > _now():
        return None
    fut = st.pending.get(arxiv_id)
    if fut is not None:
        return fut
    fut = st.loop.create_future()
    st.pending[arxiv_id] = fut
    st.queue.append(arxiv_id)
    if st.worker is None or st.worker.done():
        st.worker = st.loop.create_task(_run(st), name="abstract-fetcher")
    st.wake.set()
    return fut


async def ensure(arxiv_id: str, timeout: float = 4.0) -> str | None:
    """The full abstract, waiting up to `timeout` for a fetch; None otherwise.

    A timeout leaves the fetch running, so the row still lands for next time.
    """
    have = (await stored([arxiv_id])).get(arxiv_id)
    if have:
        return have
    fut = request(arxiv_id)
    if fut is None:
        return None
    try:
        return await asyncio.wait_for(asyncio.shield(fut), timeout)
    except asyncio.TimeoutError:
        return None


def _resolve(st: _State, arxiv_id: str, value: str | None) -> None:
    fut = st.pending.pop(arxiv_id, None)
    if fut is not None and not fut.done():
        fut.set_result(value)


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


async def _run(st: _State) -> None:
    while True:
        if not st.queue:
            st.wake.clear()
            await st.wake.wait()
            continue
        day = _utc_day()
        if day != st.day:
            st.day, st.used_today = day, 0
        if st.used_today >= config.FULL_ABSTRACTS_DAILY_CAP:
            for aid in st.queue:
                _resolve(st, aid, None)
            st.queue.clear()
            runtime_stats.count("abstracts_capped")
            continue
        batch = st.queue[:BATCH]
        del st.queue[:BATCH]
        wait = st.last_call + MIN_INTERVAL_S - _now()
        if wait > 0:
            await _sleep(wait)
        st.last_call = _now()
        st.used_today += len(batch)
        try:
            found = await _fetch(batch)
        except _SoftBlock as e:
            st.queue[:0] = batch                  # try the same ids again, later
            st.used_today -= len(batch)
            runtime_stats.count("abstracts_soft_blocked")
            print(f"[abstract_svc] arXiv asked us to wait {e.retry_after:.0f}s")
            await _sleep(e.retry_after)
            continue
        except Exception as e:
            errors.report("abstract_svc", f"fetching {len(batch)} abstracts failed", e)
            found = {}
        for aid in batch:
            text = found.get(aid)
            if text:
                try:
                    await save(aid, text)
                except Exception as e:
                    errors.report("abstract_svc", f"saving the abstract of {aid} failed", e)
                runtime_stats.count("abstracts_fetched")
            else:
                st.failed_until[aid] = _now() + FAIL_SKIP_S
                runtime_stats.count("abstracts_failed")
            _resolve(st, aid, text)


class _SoftBlock(Exception):
    def __init__(self, retry_after: float):
        super().__init__(f"retry after {retry_after}s")
        self.retry_after = retry_after


def _retry_after(resp: httpx.Response) -> float:
    try:
        value = float(resp.headers.get("retry-after", ""))
    except ValueError:
        value = DEFAULT_RETRY_AFTER_S
    return min(max(value, 1.0), MAX_RETRY_AFTER_S)


async def _fetch(ids: list[str]) -> dict[str, str]:
    """{id: full abstract} for the ids arXiv returned; raises _SoftBlock on 403/429/503."""
    from app.arxiv_svc import _NS, _parse_entry
    resp = await http_client.get_client().get(
        config.ARXIV_API_URL,
        params={"id_list": ",".join(ids), "max_results": len(ids)},
        headers={"User-Agent": config.OUTBOUND_USER_AGENT},
        timeout=20)
    if resp.status_code in SOFT_BLOCK:
        raise _SoftBlock(_retry_after(resp))
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    wanted = set(ids)
    out = {}
    for entry in root.findall("atom:entry", _NS):
        paper = _parse_entry(entry)
        aid = paper.get("arxiv_id", "")
        text = re.sub(r"\s+", " ", paper.get("abstract") or "").strip()
        # No truncation check here: the API returns whole abstracts, and one
        # that legitimately ends in a URL would look "cut" to that heuristic.
        if aid in wanted and text:
            out[aid] = text
    return out


async def stop() -> None:
    """Cancel the worker (app shutdown, tests). Queued fetches resolve to None."""
    global _state
    st = _state
    if st is None:
        return
    for aid in list(st.pending):
        _resolve(st, aid, None)
    st.queue.clear()
    if st.worker is not None and not st.worker.done():
        st.worker.cancel()
        try:
            await st.worker
        except (asyncio.CancelledError, Exception):
            pass
    _state = None


def stats() -> dict:
    st = _state
    return {
        "switch": features.mode(SWITCH),
        "queued": len(st.queue) if st else 0,
        "in_flight": len(st.pending) if st else 0,
        "used_today": st.used_today if st else 0,
        "daily_cap": config.FULL_ABSTRACTS_DAILY_CAP,
    }
