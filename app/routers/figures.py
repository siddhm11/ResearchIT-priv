"""
GET /fig/{arxiv_id}/{n}-{h8}.webp: one saved figure, as a small cached WebP.

WHY NOT LINK ARXIV'S IMAGES
---------------------------
base.html promises the page loads nothing from other origins, originals run to
2.1 MB, and a figure should cost arXiv one request ever, not one per reader.

HOW A REQUEST IS ANSWERED
-------------------------
The URL names a figure by its position in the saved list (`n`, 0-7) plus the
first 8 hex digits of the SHA-1 of its saved path (`h8`), so a refetched figure
gets a new URL and the old one can be cached forever. The URL never carries an
address: the image is fetched from https://arxiv.org/html/ + the saved path,
after that path passes a strict pattern, so this route can't be used as a proxy.

Checks, in order, each failing with 404: switch FIGURES visible to this browser;
a well-formed id; a saved source with status ok; licences.allows_figures();
figure n exists; h8 matches. Then the disk cache, a 24-hour failure marker, and
finally one generation in flight per figure, through paper_source_svc's pacer
and breaker, with Pillow in a thread behind a semaphore of 2 (the Space has two
vCPUs).
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response

from app import config, errors, features, http_client, licences, paper_source_svc, runtime_stats, thumbnails

router = APIRouter()

SWITCH = "FIGURES"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
FAIL_SKIP_S = 24 * 3600
SWEEP_EVERY_S = 3600
CACHE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}
_NAME = re.compile(r"^(\d{1,2})-([0-9a-f]{8})\.webp$")


def path_hash(src: str) -> str:
    return hashlib.sha1(src.encode()).hexdigest()[:8]


def safe_src(arxiv_id: str, src: str) -> bool:
    """A saved image path we are willing to fetch: under this paper, an image type."""
    return bool(src) and ".." not in src and bool(
        re.fullmatch(rf"{re.escape(arxiv_id)}v\d+/[A-Za-z0-9._\-/]+\.(?:png|jpe?g|gif)", src, re.I))


def figure_url(source, n: int) -> str | None:
    """The thumbnail URL for figure n of a saved source, or None if it can't be shown."""
    if source is None or not source.figures_allowed or not 0 <= n < len(source.figures):
        return None
    src = source.figures[n].get("src") or ""
    if not safe_src(source.arxiv_id, src):
        return None
    return f"/fig/{source.arxiv_id}/{n}-{path_hash(src)}.webp"


def figure_credit(paper: dict, source) -> str:
    """"Jiang et al. 2026, CC BY 4.0": first author's surname, year, licence."""
    from app.templates_env import _tojson_parse
    authors = _tojson_parse(paper.get("authors") or "[]") or []
    if not authors and paper.get("authors"):
        raw = paper["authors"]
        authors = [a.strip() for a in raw.split(";" if ";" in raw else ",") if a.strip()]
    first = authors[0].split()[-1] if authors else ""
    who = f"{first} et al." if len(authors) > 1 else first
    year = str(paper.get("year") or (paper.get("published") or "")[:4] or "")
    parts = [" ".join(p for p in (who, year) if p), source.licence_label if source else ""]
    return ", ".join(p for p in parts if p)


# ── per-loop state: single flight, CPU semaphore, sweep clock ────────────────

class _State:
    def __init__(self, loop):
        self.loop = loop
        self.in_flight: dict[str, asyncio.Future] = {}
        self.cpu = asyncio.Semaphore(2)
        self.last_sweep = 0.0


_state: _State | None = None


def _get_state() -> _State:
    global _state
    loop = asyncio.get_running_loop()
    if _state is None or _state.loop is not loop:
        _state = _State(loop)
    return _state


def _cache_path(arxiv_id: str, name: str) -> Path:
    return Path(config.FIG_CACHE_DIR) / arxiv_id / name


def _failed_recently(marker: Path) -> bool:
    try:
        return time.time() - marker.stat().st_mtime < FAIL_SKIP_S
    except OSError:
        return False


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def sweep(max_mb: int | None = None) -> int:
    """Delete the least recently used thumbnails until the cache fits its cap."""
    cap = (config.FIG_CACHE_MAX_MB if max_mb is None else max_mb) * 1024 * 1024
    root = Path(config.FIG_CACHE_DIR)
    if not root.exists():
        return 0
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            files.append((max(st.st_atime, st.st_mtime), st.st_size, p))
    total = sum(size for _, size, _ in files)
    removed = 0
    for _, size, p in sorted(files):
        if total <= cap:
            break
        try:
            p.unlink()
            total -= size
            removed += 1
        except OSError:
            pass
    return removed


async def _download(src: str) -> tuple[int, bytes, str]:
    url = f"{config.ARXIV_HTML_BASE}/{src}"
    body = bytearray()
    async with http_client.get_client().stream(
            "GET", url, headers={"User-Agent": config.OUTBOUND_USER_AGENT},
            timeout=20, follow_redirects=True) as resp:
        ctype = resp.headers.get("content-type", "")
        if resp.status_code == 200 and ctype.startswith("image/"):
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_IMAGE_BYTES:
                    return 413, b"", ctype
        retry = resp.headers.get("retry-after")
        if resp.status_code in paper_source_svc.SOFT_BLOCK:
            try:
                paper_source_svc.soft_block(float(retry) if retry else None)
            except ValueError:
                paper_source_svc.soft_block(None)
        return resp.status_code, bytes(body), ctype


async def _generate(arxiv_id: str, src: str, path: Path) -> bytes | None:
    st = _get_state()
    marker = path.with_suffix(".fail")
    if not await paper_source_svc.pace():
        return None                                   # breaker open: try again later
    try:
        status, raw, ctype = await _download(src)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        print(f"[figures] {arxiv_id} {src}: {type(e).__name__}")
        return None
    if status in paper_source_svc.SOFT_BLOCK:
        return None
    if status != 200 or not ctype.startswith("image/") or not raw:
        _write_atomic(marker, f"HTTP {status} {ctype}".encode())
        runtime_stats.count("figure_failed")
        return None
    t0 = time.perf_counter()
    async with st.cpu:
        try:
            data = await asyncio.to_thread(thumbnails.make, raw)
        except thumbnails.NotAnImage as e:
            _write_atomic(marker, str(e).encode()[:300])
            runtime_stats.count("figure_failed")
            return None
    _write_atomic(path, data)
    runtime_stats.count("figure_made")
    runtime_stats.record("figure", {"make_ms": round((time.perf_counter() - t0) * 1000, 1),
                                    "in_kb": len(raw) // 1024, "out_kb": len(data) // 1024})
    if time.monotonic() - st.last_sweep > SWEEP_EVERY_S:
        st.last_sweep = time.monotonic()
        paper_source_svc.schedule_task(asyncio.to_thread(sweep))
    return data


async def thumbnail(arxiv_id: str, n: int, h8: str) -> tuple[Path | None, bytes | None]:
    """(cached file, or freshly made bytes) for a figure, or (None, None)."""
    if not paper_source_svc.ID_PATTERN.match(arxiv_id):
        return None, None
    source = await paper_source_svc.get(arxiv_id)
    if source is None or not source.figures_allowed or not 0 <= n < len(source.figures):
        return None, None
    src = source.figures[n].get("src") or ""
    if path_hash(src) != h8 or not safe_src(arxiv_id, src):
        return None, None
    path = _cache_path(arxiv_id, f"{n}-{h8}.webp")
    if path.exists():
        return path, None
    if _failed_recently(path.with_suffix(".fail")):
        return None, None
    st = _get_state()
    key = f"{arxiv_id}/{n}-{h8}"
    fut = st.in_flight.get(key)
    if fut is not None:
        return None, await asyncio.shield(fut)
    fut = st.loop.create_future()
    st.in_flight[key] = fut
    data = None
    try:
        data = await _generate(arxiv_id, src, path)
    except Exception as e:
        errors.report("figures", f"thumbnail {key} failed", e)
    finally:
        st.in_flight.pop(key, None)
        if not fut.done():
            fut.set_result(data)
    return None, data


@router.get("/fig/{arxiv_id}/{name}")
async def figure(arxiv_id: str, name: str, request: Request):
    m = _NAME.match(name)
    if not features.visible(SWITCH, request) or m is None:
        return Response(status_code=404)
    path, data = await thumbnail(arxiv_id, int(m.group(1)), m.group(2))
    if path is not None:
        return FileResponse(path, media_type="image/webp", headers=CACHE_HEADERS)
    if data:
        return Response(content=data, media_type="image/webp", headers=CACHE_HEADERS)
    return Response(status_code=404)
