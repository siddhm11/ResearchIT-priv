"""
GET /fig/{arxiv_id}/{n}-{h8}.webp or .svg: one saved figure, cached.

Pictures (PNG, JPEG, GIF) become a small WebP. Vector figures are served as
SVG after app/svg_clean.py has rebuilt them from drawing elements only: an SVG
file is fetched from its path, and a drawing inline in the page is lifted out
of the page itself (app.arxiv_html.inline_svgs), one fetch for all of a
paper's drawings. SVG responses carry a Content-Security-Policy that blocks
scripts and sandboxes the file if it is opened directly; pages only show
figures through <img>, where browsers never run scripts.

WHY NOT LINK ARXIV'S IMAGES
---------------------------
base.html promises the page loads nothing from other origins, originals run to
2.1 MB, and a figure should cost arXiv one request ever, not one per reader.

HOW A REQUEST IS ANSWERED
-------------------------
The URL names a figure by its position in the saved list (`n`) plus the first
8 hex digits of the SHA-1 of its saved path (`h8`; for a drawing in the page,
the page path and the figure's anchor), so a refetched figure
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

from app import (arxiv_html, config, errors, features, http_client, licences, paper_source_svc, runtime_stats,
                 svg_clean, thumbnails)

router = APIRouter()

SWITCH = "FIGURES"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
FAIL_SKIP_S = 24 * 3600
SWEEP_EVERY_S = 3600
CACHE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}
SVG_HEADERS = {**CACHE_HEADERS, "Content-Disposition": "inline",
               "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox"}
SVG_KINDS = ("svg", "svg-inline")
_NAME = re.compile(r"^(\d{1,2})-([0-9a-f]{8})\.(webp|svg)$")


def path_hash(src: str) -> str:
    return hashlib.sha1(src.encode()).hexdigest()[:8]


def figure_kind(fig: dict) -> str:
    """img (a picture), svg (an SVG file) or svg-inline (drawn in the page)."""
    return fig.get("kind") or "img"


def figure_key(fig: dict) -> str:
    """What a figure's URL hash is taken from: its path, plus the anchor for a drawing."""
    src = fig.get("src") or ""
    return f"{src}#{fig.get('anchor') or ''}" if figure_kind(fig) == "svg-inline" else src


def _ext(kind: str) -> str:
    return "svg" if kind in SVG_KINDS else "webp"


def safe_src(arxiv_id: str, src: str, kind: str = "img") -> bool:
    """A saved path we are willing to fetch: under this paper, of the figure's type."""
    if not src or ".." in src:
        return False
    base = re.escape(arxiv_id)
    if kind == "svg-inline":                             # the paper's own page
        return bool(re.fullmatch(rf"{base}(?:v\d+)?", src))
    types = "svg" if kind == "svg" else "png|jpe?g|gif"
    return bool(re.fullmatch(rf"{base}v\d+/[A-Za-z0-9._\-/]+\.(?:{types})", src, re.I))


def figure_url(source, n: int) -> str | None:
    """The URL for figure n of a saved source, or None if it can't be shown."""
    if source is None or not source.figures_allowed or not 0 <= n < len(source.figures):
        return None
    fig = source.figures[n]
    kind = figure_kind(fig)
    if not safe_src(source.arxiv_id, fig.get("src") or "", kind):
        return None
    return f"/fig/{source.arxiv_id}/{n}-{path_hash(figure_key(fig))}.{_ext(kind)}"


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


async def _download(src: str, accept: str = "image/", cap: int = MAX_IMAGE_BYTES) -> tuple[int, bytes, str]:
    url = f"{config.ARXIV_HTML_BASE}/{src}"
    body = bytearray()
    async with http_client.get_client().stream(
            "GET", url, headers={"User-Agent": config.OUTBOUND_USER_AGENT},
            timeout=20, follow_redirects=True) as resp:
        ctype = resp.headers.get("content-type", "")
        if resp.status_code == 200 and ctype.startswith(accept):
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) > cap:
                    return 413, b"", ctype
        retry = resp.headers.get("retry-after")
        if resp.status_code in paper_source_svc.SOFT_BLOCK:
            try:
                paper_source_svc.soft_block(float(retry) if retry else None)
            except ValueError:
                paper_source_svc.soft_block(None)
        return resp.status_code, bytes(body), ctype


async def _generate(arxiv_id: str, src: str, path: Path, kind: str = "img") -> bytes | None:
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
    make = svg_clean.clean if kind == "svg" else thumbnails.make
    async with st.cpu:
        try:
            data = await asyncio.to_thread(make, raw)
        except (thumbnails.NotAnImage, svg_clean.NotSVG) as e:
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


def _drawings(html: bytes, anchors: list[str]) -> dict[str, bytes | str]:
    """Cleaned SVG, or why there is none, for each anchor. CPU-bound: run in a thread."""
    found = arxiv_html.inline_svgs(html)
    out: dict[str, bytes | str] = {}
    for anchor in anchors:
        if anchor not in found:
            out[anchor] = "no drawing for this figure in the page"
            continue
        try:
            out[anchor] = svg_clean.clean_tree(found[anchor])
        except svg_clean.NotSVG as e:
            out[anchor] = str(e)
    return out


async def _generate_inline(source, src: str) -> None:
    """Every drawing of a paper from one fetch of its page, written to the cache."""
    st = _get_state()
    wanted = {f["anchor"]: _cache_path(source.arxiv_id, f"{i}-{path_hash(figure_key(f))}.svg")
              for i, f in enumerate(source.figures)
              if figure_kind(f) == "svg-inline" and f.get("src") == src and f.get("anchor")}
    if not wanted or not await paper_source_svc.pace():
        return
    try:
        status, raw, ctype = await _download(src, accept="text/html", cap=arxiv_html.MAX_HTML_BYTES)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        print(f"[figures] {src}: {type(e).__name__}")
        return
    if status in paper_source_svc.SOFT_BLOCK:
        return
    if status != 200 or not raw:
        results: dict[str, bytes | str] = {a: f"HTTP {status} {ctype}" for a in wanted}
    else:
        async with st.cpu:
            results = await asyncio.to_thread(_drawings, raw, list(wanted))
    for anchor, path in wanted.items():
        result = results[anchor]
        if isinstance(result, bytes):
            _write_atomic(path, result)
            runtime_stats.count("figure_made")
        else:
            _write_atomic(path.with_suffix(".fail"), result.encode()[:300])
            runtime_stats.count("figure_failed")


async def thumbnail(arxiv_id: str, n: int, h8: str, ext: str = "webp") -> tuple[Path | None, bytes | None]:
    """(cached file, or freshly made bytes) for a figure, or (None, None)."""
    if not paper_source_svc.ID_PATTERN.match(arxiv_id):
        return None, None
    source = await paper_source_svc.get(arxiv_id)
    if source is None or not source.figures_allowed or not 0 <= n < len(source.figures):
        return None, None
    fig = source.figures[n]
    kind, src = figure_kind(fig), fig.get("src") or ""
    if ext != _ext(kind) or path_hash(figure_key(fig)) != h8 or not safe_src(arxiv_id, src, kind):
        return None, None
    path = _cache_path(arxiv_id, f"{n}-{h8}.{ext}")
    if path.exists():
        return path, None
    if _failed_recently(path.with_suffix(".fail")):
        return None, None
    st = _get_state()
    # A paper's drawings share one fetch of its page, so they share one flight.
    key = f"{arxiv_id}/page" if kind == "svg-inline" else f"{arxiv_id}/{n}-{h8}"
    fut = st.in_flight.get(key)
    if fut is not None:
        data = await asyncio.shield(fut)
        if kind == "svg-inline":
            return (path, None) if path.exists() else (None, None)
        return None, data
    fut = st.loop.create_future()
    st.in_flight[key] = fut
    data = None
    try:
        if kind == "svg-inline":
            await _generate_inline(source, src)
            data = path.read_bytes() if path.exists() else None
        else:
            data = await _generate(arxiv_id, src, path, kind)
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
    ext = m.group(3)
    path, data = await thumbnail(arxiv_id, int(m.group(1)), m.group(2), ext)
    media, headers = ("image/svg+xml", SVG_HEADERS) if ext == "svg" else ("image/webp", CACHE_HEADERS)
    if path is not None:
        return FileResponse(path, media_type=media, headers=headers)
    if data:
        return Response(content=data, media_type=media, headers=headers)
    return Response(status_code=404)


async def prepare(source, n: int) -> bool:
    """Make figure n ready in the cache, as a page request for it would."""
    url = figure_url(source, n)
    m = _NAME.match(url.rsplit("/", 1)[1]) if url else None
    if m is None:
        return False
    path, data = await thumbnail(source.arxiv_id, int(m.group(1)), m.group(2), m.group(3))
    return bool(path or data)
