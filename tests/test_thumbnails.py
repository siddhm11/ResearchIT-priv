"""
Figure thumbnails: app/thumbnails.py (pure) and GET /fig/... (app/routers/figures.py).

Images are generated inside the tests with Pillow and never committed. arXiv is
an httpx.MockTransport; saved sources are written straight into the test
database.
"""
import asyncio
import io
import json
import random

import aiosqlite
import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, http_client, paper_source_svc, thumbnails
from app.routers import figures

AID = "2609.31507"
SRC = f"{AID}v1/overview-final.png"


def png(w, h, mode="RGB", color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new(mode, (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def noisy(w, h):
    rng = random.Random(7)
    im = Image.frombytes("RGB", (w, h), bytes(rng.getrandbits(8) for _ in range(w * h * 3)))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def decode(webp):
    im = Image.open(io.BytesIO(webp))
    im.load()
    return im


# ── thumbnails.make ──────────────────────────────────────────────────────────

def test_a_wide_image_shrinks_to_800_px_and_keeps_its_shape():
    im = decode(thumbnails.make(png(1600, 400)))
    assert (im.width, im.height, im.format) == (800, 200, "WEBP")


def test_a_small_image_is_not_enlarged():
    assert decode(thumbnails.make(png(300, 120))).size == (300, 120)


def test_transparency_lands_on_white():
    im = decode(thumbnails.make(png(10, 10, "RGBA", (0, 0, 0, 0))))
    assert im.convert("RGB").getpixel((5, 5)) == (255, 255, 255)


def test_a_cmyk_jpeg_is_converted():
    buf = io.BytesIO()
    Image.new("CMYK", (40, 20), (0, 255, 255, 0)).save(buf, "JPEG")
    assert decode(thumbnails.make(buf.getvalue())).size == (40, 20)


def test_an_animated_gif_gives_its_first_frame():
    frames = [Image.new("P", (20, 20), i) for i in (10, 200)]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:])
    assert decode(thumbnails.make(buf.getvalue())).size == (20, 20)


def test_an_exif_rotated_photo_comes_out_upright():
    im = Image.new("RGB", (40, 20), (0, 0, 255))
    exif = im.getexif()
    exif[0x0112] = 6                                    # rotate 90 degrees clockwise
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    assert decode(thumbnails.make(buf.getvalue())).size == (20, 40)


def test_an_image_bomb_is_refused_before_decoding():
    buf = io.BytesIO()
    Image.new("1", (12_000, 12_000)).save(buf, "PNG")
    with pytest.raises(thumbnails.NotAnImage):
        thumbnails.make(buf.getvalue())


def test_bytes_that_are_not_an_image_are_refused():
    with pytest.raises(thumbnails.NotAnImage):
        thumbnails.make(b"<html>not an image</html>")


def test_a_noisy_image_that_cannot_reach_the_target_gets_the_smallest_attempt(monkeypatch):
    sizes = []
    real_save = Image.Image.save

    def spy(self, fp, fmt=None, **kw):
        real_save(self, fp, fmt, **kw)
        if fmt == "WEBP":
            sizes.append(len(fp.getvalue()))

    monkeypatch.setattr(Image.Image, "save", spy)
    out = thumbnails.make(noisy(1200, 900))
    assert len(sizes) == len(thumbnails.QUALITIES)          # every quality was tried
    assert len(out) == min(sizes) > thumbnails.TARGET_BYTES


# ── helpers ──────────────────────────────────────────────────────────────────

def source(licence="CC-BY-4.0", src=SRC, status="ok"):
    return paper_source_svc.Source(arxiv_id=AID, status=status, version="v1", licence=licence,
                                   figures=[{"anchor": "S1.F1", "label": "Figure 1", "caption": "Overview.",
                                             "src": src, "w": 476, "h": 235}])


def test_figure_url_names_the_figure_by_position_and_path_hash():
    assert figures.figure_url(source(), 0) == f"/fig/{AID}/0-{figures.path_hash(SRC)}.webp"
    assert figures.figure_url(source(), 1) is None
    assert figures.figure_url(source(status="no_html"), 0) is None
    assert figures.figure_url(None, 0) is None


def test_figure_url_follows_the_licence_policy(monkeypatch):
    monkeypatch.setenv("FIGURE_LICENCES", "CC-BY-4.0")
    assert figures.figure_url(source("CC-BY-4.0"), 0)
    assert figures.figure_url(source("ARXIV-NONEXCLUSIVE-1.0"), 0) is None


@pytest.mark.parametrize("bad", [f"{AID}v1/../../etc/passwd.png", "https://evil.example/a.png",
                                 "2609.99999v1/other-paper.png", f"{AID}v1/figure.svg", ""])
def test_unsafe_saved_paths_get_no_url(bad):
    assert figures.figure_url(source(src=bad), 0) is None


def test_the_credit_line_names_author_year_and_licence():
    paper = {"authors": '["Jiajun Jiang", "Chunliang Hua"]', "year": 2026}
    assert figures.figure_credit(paper, source()) == "Jiang et al. 2026, CC BY 4.0"
    assert figures.figure_credit({"authors": "Ada Lovelace", "published": "1843-01-01"},
                                 source("ARXIV-NONEXCLUSIVE-1.0")) == "Lovelace 1843, arXiv licence"


# ── the route ────────────────────────────────────────────────────────────────

async def save_source(licence="CC-BY-4.0", src=SRC):
    figs = [{"anchor": "S1.F1", "label": "Figure 1", "caption": "Overview.", "src": src, "w": 476, "h": 235}]
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO paper_sources (arxiv_id, status, version, licence, figures_json, "
            "parser_version) VALUES (?, 'ok', 'v1', ?, ?, 1)", (AID, licence, json.dumps(figs)))
        await conn.commit()


@pytest.fixture
def world(monkeypatch, tmp_path):
    """FIGURES on, a temp cache, and a fake arXiv that serves one PNG."""
    monkeypatch.setenv("FIGURES", "on")
    monkeypatch.setattr(config, "FIG_CACHE_DIR", str(tmp_path / "figcache"))
    state = {"requests": [], "status": 200, "body": png(1600, 790), "ctype": "image/png"}

    def handler(request):
        state["requests"].append(str(request.url))
        return httpx.Response(state["status"], content=state["body"], headers={"content-type": state["ctype"]})

    real_get_client = http_client.get_client
    clients = {}

    def get_client():
        loop = asyncio.get_running_loop()
        if loop not in clients:
            clients[loop] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return clients[loop]

    monkeypatch.setattr(http_client, "get_client", get_client)
    asyncio.run(save_source())
    yield state
    monkeypatch.setattr(http_client, "get_client", real_get_client)


def url(n=0, src=SRC):
    return f"/fig/{AID}/{n}-{figures.path_hash(src)}.webp"


def test_an_allowed_figure_is_served_as_webp_and_cached(world):
    from app.main import app
    with TestClient(app) as c:
        first = c.get(url())
        second = c.get(url())
    assert first.status_code == 200 and first.headers["content-type"] == "image/webp"
    assert "immutable" in first.headers["cache-control"]
    assert decode(first.content).size == (800, 395)
    assert second.status_code == 200 and second.content == first.content
    assert world["requests"] == [f"{config.ARXIV_HTML_BASE}/{SRC}"]       # arXiv asked once


@pytest.mark.parametrize("path", [
    f"/fig/{AID}/1-00000000.webp",          # no such figure
    f"/fig/{AID}/0-deadbeef.webp",          # wrong path hash
    f"/fig/{AID}/0-{figures.path_hash(SRC)}.png",
    f"/fig/2609.00000/0-{figures.path_hash(SRC)}.webp",   # no saved source
    f"/fig/..%2F..%2Fetc/0-{figures.path_hash(SRC)}.webp",
])
def test_anything_else_is_a_404_without_calling_arxiv(world, path):
    from app.main import app
    with TestClient(app) as c:
        assert c.get(path).status_code == 404
    assert world["requests"] == []


def test_a_licence_outside_the_policy_is_a_404(world, monkeypatch):
    from app.main import app
    monkeypatch.setenv("FIGURE_LICENCES", "CC0-1.0")
    with TestClient(app) as c:
        assert c.get(url()).status_code == 404
    assert world["requests"] == []


def test_with_the_switch_off_nothing_is_served(world, monkeypatch):
    from app.main import app
    monkeypatch.setenv("FIGURES", "off")
    with TestClient(app) as c:
        assert c.get(url()).status_code == 404


def test_an_html_response_is_not_an_image(world):
    from app.main import app
    world.update(body=b"<html>oops</html>", ctype="text/html")
    with TestClient(app) as c:
        assert c.get(url()).status_code == 404
        assert c.get(url()).status_code == 404
    assert len(world["requests"]) == 1          # the failure is remembered for a day


def test_a_soft_block_is_not_remembered_as_a_failure(world):
    from app.main import app
    world.update(status=429, body=b"", ctype="text/plain")
    with TestClient(app) as c:
        assert c.get(url()).status_code == 404
        world.update(status=200, body=png(100, 50), ctype="image/png")
        assert c.get(url()).status_code == 200


async def test_five_concurrent_requests_make_one_fetch(world):
    calls = []
    original = figures._download

    async def counting(src):
        calls.append(src)
        await asyncio.sleep(0.01)
        return await original(src)

    figures._download = counting
    try:
        results = await asyncio.gather(*[figures.thumbnail(AID, 0, figures.path_hash(SRC)) for _ in range(5)])
    finally:
        figures._download = original
        await paper_source_svc.stop()
    assert len(calls) == 1
    assert all(path is not None or data for path, data in results)


def test_the_sweep_removes_the_oldest_files_first(tmp_path, monkeypatch):
    import os
    import time
    monkeypatch.setattr(config, "FIG_CACHE_DIR", str(tmp_path))
    now = time.time()
    for i in range(4):
        p = tmp_path / AID / f"{i}-00000000.webp"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 600_000)
        os.utime(p, (now - 1000 + i, now - 1000 + i))
    assert figures.sweep(max_mb=1) == 3
    assert [p.name for p in (tmp_path / AID).iterdir()] == ["3-00000000.webp"]
