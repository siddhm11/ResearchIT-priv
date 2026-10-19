"""
Read each paper's arXiv HTML page once and keep what the redesign needs.

WHY THIS EXISTS
---------------
Figures, the briefing and Ask this paper all need more than the abstract. arXiv
publishes most papers as HTML (54 of 60 new papers in the September 2026
sample; 28 of 30 from 2019-2024), and every page states its licence. Reading a
page costs a median 0.57 s, so it happens once per paper, in the background, and
every reader-facing request reads only what was saved here. The feed and the
paper page never wait on arXiv.

WHAT IT SAVES
-------------
    paper_sources      replicated; licence, version, up to 8 figures, counts,
                       and (from the code-link PR) the paper's GitHub repo
    paper_fulltext     local; sections, all figures, tables
    paper_paragraphs   local; paragraph text with anchors, for Ask and briefings

BEING POLITE TO ARXIV
---------------------
One shared pacer allows a request every 2 s with bursts of 3; figure images use
it too. Three soft blocks (403, 406, 429, 503) inside 10 minutes open a breaker
for 15 minutes, or for Retry-After if longer; while it is open, nothing is
fetched and callers get what is saved. One fetch is in flight per paper. Only
new-style ids are fetched, and only for papers already in our metadata: callers
pass ids they found there.

Switch: PAPER_SOURCES (off | shadow | on). Reads work whatever it says.
"""
from __future__ import annotations

import asyncio
import calendar
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field

import aiosqlite
import httpx

from app import arxiv_html, config, errors, features, http_client, licences, runtime_stats

SWITCH = "PAPER_SOURCES"
ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}$")

REPLICATED_FIGURES = 8
REPLICATED_CAPTION = 300
FETCH_TIMEOUT_S = 15.0
RETRY_NO_HTML_S = 7 * 24 * 3600
RETRY_ERROR_S = 24 * 3600
PRUNE_EVERY_S = 24 * 3600

PACE_INTERVAL_S = 2.0
PACE_BURST = 3
SOFT_BLOCK = (403, 406, 429, 503)
BREAKER_THRESHOLD = 3
BREAKER_WINDOW_S = 600.0
BREAKER_OPEN_S = 900.0
MAX_RETRY_AFTER_S = 3600.0


def _now() -> float:                 # indirection so tests can drive the clock
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass
class Source:
    arxiv_id: str
    status: str
    version: str = ""
    licence: str = ""
    licence_text: str = ""
    figures: list[dict] = field(default_factory=list)
    n_figures: int = 0
    n_sections: int = 0
    n_paragraphs: int = 0
    code_repo: str = ""
    code_stars: int | None = None
    code_checked_at: str = ""
    parser_version: int = 0
    fetched_at: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def figures_allowed(self) -> bool:
        return self.ok and licences.allows_figures(self.licence)

    @property
    def licence_label(self) -> str:
        return licences.label(self.licence) if self.licence else (self.licence_text or "")

    def html_url(self, anchor: str = "") -> str:
        base = f"https://arxiv.org/html/{self.arxiv_id}{self.version}"
        return f"{base}#{anchor}" if anchor and re.fullmatch(r"[A-Za-z0-9.]+", anchor) else base


_COLS = ["arxiv_id", "status", "version", "licence", "licence_text", "figures_json", "n_figures",
         "n_sections", "n_paragraphs", "code_repo", "code_stars", "code_checked_at",
         "parser_version", "fetched_at", "error"]


def _row_to_source(row) -> Source:
    d = dict(zip(_COLS, row))
    try:
        figures = json.loads(d["figures_json"] or "[]")
    except ValueError:
        figures = []
    return Source(arxiv_id=d["arxiv_id"], status=d["status"], version=d["version"] or "",
                  licence=d["licence"] or "", licence_text=d["licence_text"] or "", figures=figures,
                  n_figures=d["n_figures"] or 0, n_sections=d["n_sections"] or 0,
                  n_paragraphs=d["n_paragraphs"] or 0, code_repo=d["code_repo"] or "",
                  code_stars=d["code_stars"], code_checked_at=d["code_checked_at"] or "",
                  parser_version=d["parser_version"] or 0, fetched_at=d["fetched_at"] or "",
                  error=d["error"] or "")


# ── reads (local only; safe on every request) ────────────────────────────────

async def get_many(arxiv_ids: list[str]) -> dict[str, Source]:
    ids = [i for i in dict.fromkeys(arxiv_ids) if i]
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute(
            f"SELECT {', '.join(_COLS)} FROM paper_sources WHERE arxiv_id IN ({marks})", ids)
        return {s.arxiv_id: s for s in map(_row_to_source, await cur.fetchall())}


async def get(arxiv_id: str) -> Source | None:
    return (await get_many([arxiv_id])).get(arxiv_id)


async def fulltext(arxiv_id: str) -> dict | None:
    """Sections, every figure and tables of a fetched paper, or None."""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute(
            "SELECT version, title, abstract, sections_json, figures_json, tables_json "
            "FROM paper_fulltext WHERE arxiv_id = ?", (arxiv_id,))
        row = await cur.fetchone()
        if row is None:
            return None
        await conn.execute("UPDATE paper_fulltext SET last_used_at = datetime('now') WHERE arxiv_id = ?",
                           (arxiv_id,))
        await conn.commit()
    version, title, abstract, sections, figures, tables = row
    return {"version": version or "", "title": title or "", "abstract": abstract or "",
            "sections": json.loads(sections or "[]"), "figures": json.loads(figures or "[]"),
            "tables": json.loads(tables or "[]")}


async def paragraphs(arxiv_id: str) -> list[dict]:
    """Paragraphs in page order: anchor, section, text, refs."""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute(
            "SELECT anchor, section_anchor, text, refs_json FROM paper_paragraphs "
            "WHERE arxiv_id = ? ORDER BY ord", (arxiv_id,))
        rows = await cur.fetchall()
    return [{"anchor": a, "section": s or "", "text": t, "refs": json.loads(r or "[]")}
            for a, s, t, r in rows]


# ── fetch state (pacer, breaker, single flight), bound to the running loop ───

class _State:
    def __init__(self, loop):
        self.loop = loop
        self.in_flight: dict[str, asyncio.Future] = {}
        self.pace_lock = asyncio.Lock()
        self.tokens = float(PACE_BURST)
        self.token_time = _now()
        self.soft_blocks: deque[float] = deque()
        self.breaker_until = 0.0
        self.tasks: set[asyncio.Task] = set()
        self.recent: deque[tuple[float, str]] = deque(maxlen=500)
        self.last_prune = _now()


_state: _State | None = None


def _get_state() -> _State:
    global _state
    loop = asyncio.get_running_loop()
    if _state is None or _state.loop is not loop:
        _state = _State(loop)
    return _state


def breaker_open() -> bool:
    st = _state
    return bool(st and st.breaker_until > _now())


async def pace() -> bool:
    """Wait for a turn to call arXiv. False when the breaker is open."""
    st = _get_state()
    async with st.pace_lock:
        if st.breaker_until > _now():
            return False
        now = _now()
        st.tokens = min(float(PACE_BURST), st.tokens + (now - st.token_time) / PACE_INTERVAL_S)
        st.token_time = now
        if st.tokens < 1.0:
            await _sleep((1.0 - st.tokens) * PACE_INTERVAL_S)
            st.tokens, st.token_time = 1.0, _now()
        st.tokens -= 1.0
        return True


def soft_block(retry_after: float | None = None) -> None:
    """Record that arXiv asked us to slow down; open the breaker on the third."""
    st = _get_state()
    now = _now()
    st.soft_blocks.append(now)
    while st.soft_blocks and now - st.soft_blocks[0] > BREAKER_WINDOW_S:
        st.soft_blocks.popleft()
    runtime_stats.count("paper_source_soft_block")
    if len(st.soft_blocks) >= BREAKER_THRESHOLD or (retry_after and retry_after > PACE_INTERVAL_S * 10):
        open_for = max(BREAKER_OPEN_S, min(retry_after or 0.0, MAX_RETRY_AFTER_S))
        st.breaker_until = now + open_for
        st.soft_blocks.clear()
        print(f"[paper_source_svc] arXiv pushed back; pausing fetches for {open_for:.0f}s")


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None


# ── fetching ─────────────────────────────────────────────────────────────────

def _age_s(fetched_at: str, wall_now: float) -> float:
    """Seconds since a datetime('now') stamp, which is UTC."""
    try:
        return wall_now - calendar.timegm(time.strptime(fetched_at, "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return float("inf")


def _fresh(src: Source | None, wall_now: float) -> bool:
    """True when the saved row is good enough not to fetch again."""
    if src is None or src.status == "pending":      # pending: only a code link so far
        return False
    if src.status == "ok":
        return src.parser_version >= arxiv_html.PARSER_VERSION
    age = _age_s(src.fetched_at, wall_now)
    if src.status == "no_html":
        return age < RETRY_NO_HTML_S
    return age < RETRY_ERROR_S


async def _download(arxiv_id: str) -> tuple[int, bytes, str, float | None]:
    """(status, body capped at the parser's limit, final URL path, Retry-After)."""
    url = f"{config.ARXIV_HTML_BASE}/{arxiv_id}"
    client = http_client.get_client()
    body = bytearray()
    async with client.stream("GET", url, headers={"User-Agent": config.OUTBOUND_USER_AGENT},
                             timeout=FETCH_TIMEOUT_S, follow_redirects=True) as resp:
        if resp.status_code == 200:
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) > arxiv_html.MAX_HTML_BYTES:
                    break
        return resp.status_code, bytes(body), resp.url.path, _retry_after(resp)


def _slim_figures(parsed: arxiv_html.ParsedPaper) -> list[dict]:
    out = []
    for f in parsed.figures:
        img = f.first_image()
        if img is None:
            continue
        out.append({"anchor": f.anchor, "label": f.label, "caption": f.caption[:REPLICATED_CAPTION],
                    "src": img.src, "w": img.width, "h": img.height})
        if len(out) >= REPLICATED_FIGURES:
            break
    return out


async def _save_parsed(parsed: arxiv_html.ParsedPaper, keep: Source | None) -> None:
    figures = _slim_figures(parsed)
    code = (keep.code_repo, keep.code_stars, keep.code_checked_at) if keep else ("", None, "")
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute("BEGIN")
        # Whole-row replace, never UPDATE: a new id is what makes turso_sync push
        # the change, and every column, the code link included, is carried over.
        await conn.execute(
            f"INSERT OR REPLACE INTO paper_sources ({', '.join(_COLS)}) "
            f"VALUES ({', '.join('?' * len(_COLS))})",
            (parsed.arxiv_id, "ok", parsed.version, licences.code_from_text(parsed.licence_text),
             parsed.licence_text, json.dumps(figures), len(parsed.figures), len(parsed.sections),
             len(parsed.paragraphs), code[0] or None, code[1], code[2] or None,
             arxiv_html.PARSER_VERSION, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), None))
        await conn.execute(
            "INSERT OR REPLACE INTO paper_fulltext (arxiv_id, version, title, abstract, sections_json, "
            "figures_json, tables_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (parsed.arxiv_id, parsed.version, parsed.title, parsed.abstract,
             json.dumps([s.__dict__ for s in parsed.sections]),
             json.dumps([{"anchor": f.anchor, "label": f.label, "caption": f.caption,
                          "images": [i.__dict__ for i in f.images]} for f in parsed.figures]),
             json.dumps([t.__dict__ for t in parsed.tables])))
        await conn.execute("DELETE FROM paper_paragraphs WHERE arxiv_id = ?", (parsed.arxiv_id,))
        await conn.executemany(
            "INSERT INTO paper_paragraphs (arxiv_id, anchor, section_anchor, ord, text, refs_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(parsed.arxiv_id, p.anchor, p.section, i, p.text, json.dumps(p.refs))
             for i, p in enumerate(parsed.paragraphs)])
        await conn.commit()


async def _save_status(arxiv_id: str, status: str, error: str, keep: Source | None) -> None:
    row = {c: None for c in _COLS}
    if keep:
        row.update(version=keep.version or None, licence=keep.licence or None,
                   licence_text=keep.licence_text or None, figures_json=json.dumps(keep.figures),
                   n_figures=keep.n_figures, n_sections=keep.n_sections, n_paragraphs=keep.n_paragraphs,
                   code_repo=keep.code_repo or None, code_stars=keep.code_stars,
                   code_checked_at=keep.code_checked_at or None, parser_version=keep.parser_version)
    row.update(arxiv_id=arxiv_id, status=status, error=error[:300] or None,
               fetched_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    if status == "error" and keep and keep.ok:
        row["status"] = "ok"            # keep serving the last good copy
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(
            f"INSERT OR REPLACE INTO paper_sources ({', '.join(_COLS)}) VALUES ({', '.join('?' * len(_COLS))})",
            [row[c] for c in _COLS])
        await conn.commit()


async def _fetch_and_save(arxiv_id: str, existing: Source | None) -> Source | None:
    st = _get_state()
    if not await pace():
        runtime_stats.count("paper_source_breaker_skip")
        return existing
    t0 = time.perf_counter()
    try:
        status, body, path, retry_after = await _download(arxiv_id)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        # A blip, not a verdict on the paper: nothing is written, next open retries.
        runtime_stats.count("paper_source_network_error")
        print(f"[paper_source_svc] {arxiv_id}: {type(e).__name__}")
        return existing
    fetch_ms = (time.perf_counter() - t0) * 1000
    st.recent.append((_now(), str(status)))
    if status in SOFT_BLOCK:
        soft_block(retry_after)
        return existing
    if status == 404 or (status == 200 and "/html/" not in path):
        await _save_status(arxiv_id, "no_html", "", existing)
        runtime_stats.count("paper_source_no_html")
        return await get(arxiv_id)
    if status != 200:
        await _save_status(arxiv_id, "error", f"HTTP {status}", existing)
        runtime_stats.count("paper_source_error")
        return await get(arxiv_id)
    t1 = time.perf_counter()
    parsed = await asyncio.to_thread(arxiv_html.parse, body, arxiv_id)
    parse_ms = (time.perf_counter() - t1) * 1000
    if parsed.status != "ok":
        await _save_status(arxiv_id, "error", parsed.error, existing)
        runtime_stats.count("paper_source_error")
        return await get(arxiv_id)
    await _save_parsed(parsed, existing)
    runtime_stats.count("paper_source_ok")
    runtime_stats.record("paper_source", {"fetch_ms": round(fetch_ms, 1), "parse_ms": round(parse_ms, 1),
                                          "kb": len(body) // 1024})
    return await get(arxiv_id)


async def ensure(arxiv_id: str, *, force: bool = False) -> Source | None:
    """The saved source, fetching it first if needed. Never raises.

    Callers pass ids they found in our metadata; anything that isn't a
    new-style arXiv id is refused before any request.
    """
    if not ID_PATTERN.match(arxiv_id or ""):
        return None
    try:
        existing = await get(arxiv_id)
        if not force and _fresh(existing, time.time()):
            return existing
        st = _get_state()
        fut = st.in_flight.get(arxiv_id)
        if fut is not None:
            return await asyncio.shield(fut)
        fut = st.loop.create_future()
        st.in_flight[arxiv_id] = fut
        try:
            result = await _fetch_and_save(arxiv_id, existing)
        except Exception as e:
            errors.report("paper_source_svc", f"fetching {arxiv_id} failed", e)
            result = existing
        finally:
            st.in_flight.pop(arxiv_id, None)
        if not fut.done():
            fut.set_result(result)
        if _now() - st.last_prune > PRUNE_EVERY_S:
            st.last_prune = _now()
            schedule_task(prune())
        return result
    except Exception as e:
        errors.report("paper_source_svc", f"ensure({arxiv_id}) failed", e)
        return None


def schedule_task(coro) -> None:
    """Run a coroutine in the background, strongly referenced."""
    st = _get_state()
    task = st.loop.create_task(coro)
    st.tasks.add(task)
    task.add_done_callback(st.tasks.discard)


def schedule(arxiv_id: str) -> None:
    """Fetch this paper's page in the background when the switch allows it."""
    if features.active(SWITCH) and ID_PATTERN.match(arxiv_id or ""):
        schedule_task(ensure(arxiv_id))


async def save_code_link(arxiv_id: str, repo: str, stars: int | None) -> None:
    """Record a paper's code link by rewriting its whole row (see _save_parsed)."""
    existing = await get(arxiv_id)
    checked = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    if existing is None:
        row = {c: None for c in _COLS}
        row.update(arxiv_id=arxiv_id, status="pending", fetched_at=checked)
    else:
        row = {"arxiv_id": arxiv_id, "status": existing.status, "version": existing.version or None,
               "licence": existing.licence or None, "licence_text": existing.licence_text or None,
               "figures_json": json.dumps(existing.figures), "n_figures": existing.n_figures,
               "n_sections": existing.n_sections, "n_paragraphs": existing.n_paragraphs,
               "parser_version": existing.parser_version, "fetched_at": existing.fetched_at or checked,
               "error": existing.error or None}
    row.update(code_repo=repo or "", code_stars=stars, code_checked_at=checked)
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(
            f"INSERT OR REPLACE INTO paper_sources ({', '.join(_COLS)}) VALUES ({', '.join('?' * len(_COLS))})",
            [row[c] for c in _COLS])
        await conn.commit()


async def prune(max_papers: int | None = None) -> int:
    """Drop full text beyond the most recently used `max_papers` papers."""
    keep = config.FULLTEXT_MAX_PAPERS if max_papers is None else max_papers
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute(
            "SELECT arxiv_id FROM paper_fulltext ORDER BY last_used_at DESC, fetched_at DESC "
            "LIMIT -1 OFFSET ?", (keep,))
        old = [r[0] for r in await cur.fetchall()]
        for i in range(0, len(old), 500):
            chunk = old[i:i + 500]
            marks = ",".join("?" * len(chunk))
            await conn.execute(f"DELETE FROM paper_paragraphs WHERE arxiv_id IN ({marks})", chunk)
            await conn.execute(f"DELETE FROM paper_fulltext WHERE arxiv_id IN ({marks})", chunk)
        await conn.commit()
    if old:
        print(f"[paper_source_svc] pruned full text of {len(old)} papers")
    return len(old)


async def stats() -> dict:
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute("SELECT status, COUNT(*) FROM paper_sources GROUP BY status")
        by_status = dict(await cur.fetchall())
        cur = await conn.execute(
            "SELECT COALESCE(licence, ''), COUNT(*) FROM paper_sources WHERE status = 'ok' "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
        licence_mix = dict(await cur.fetchall())
        cur = await conn.execute("SELECT COUNT(*) FROM paper_fulltext")
        fulltext_rows = (await cur.fetchone())[0]
        cur = await conn.execute(
            "SELECT COUNT(*) FROM paper_sources WHERE status = 'ok' AND figures_json NOT IN ('[]', '')")
        with_figures = (await cur.fetchone())[0]
    st = _state
    hour_ago = _now() - 3600
    recent = [s for t, s in (st.recent if st else []) if t >= hour_ago]
    try:
        db_mb = round(os.path.getsize(config.DB_PATH) / 1e6, 1)
    except OSError:
        db_mb = None
    return {
        "switch": features.mode(SWITCH),
        "by_status": by_status,
        "with_figures": with_figures,
        "licence_mix": licence_mix,
        "fulltext_papers": fulltext_rows,
        "fulltext_max": config.FULLTEXT_MAX_PAPERS,
        "fetches_last_hour": len(recent),
        "soft_blocks_last_hour": sum(1 for s in recent if s in {str(c) for c in SOFT_BLOCK}),
        "breaker_open": breaker_open(),
        "in_flight": len(st.in_flight) if st else 0,
        "db_mb": db_mb,
    }


async def stop() -> None:
    """Cancel background fetches (app shutdown, tests)."""
    global _state
    st = _state
    if st is None:
        return
    for task in list(st.tasks):
        task.cancel()
    for task in list(st.tasks):
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    _state = None
