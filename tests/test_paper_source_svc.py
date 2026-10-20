"""
app/paper_source_svc.py: read each arXiv HTML page once, politely.

arXiv is never called: the shared HTTP client is swapped for one on an
httpx.MockTransport that serves the trimmed fixtures. A fake clock drives the
pacer and breaker where timing is the point.
"""
import asyncio
import json
import time
from pathlib import Path

import aiosqlite
import httpx
import pytest

from app import config, http_client, paper_source_svc as pss

FIXTURES = Path(__file__).parent / "fixtures" / "arxiv_html"
SATNAV = (FIXTURES / "2609.31507.html").read_bytes()


@pytest.fixture(autouse=True)
async def _clean(monkeypatch):
    monkeypatch.setenv("PAPER_SOURCES", "on")
    yield
    await pss.stop()


@pytest.fixture
def fake_arxiv(monkeypatch):
    """Serve pages by id; record every request."""
    state = {"pages": {"2609.31507": (200, SATNAV, {})}, "requests": []}

    def handler(request):
        state["requests"].append(str(request.url))
        aid = request.url.path.rsplit("/", 1)[-1]
        status, body, headers = state["pages"].get(aid, (404, b"", {}))
        if callable(body):
            return body(request)
        return httpx.Response(status, content=body, headers=headers)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(http_client, "_client", client)
    monkeypatch.setattr(http_client, "_loop", asyncio.get_event_loop())
    return state


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 10_000.0, "sleeps": []}
    monkeypatch.setattr(pss, "_now", lambda: t["now"])

    async def fake_sleep(seconds):
        t["sleeps"].append(seconds)
        t["now"] += seconds
        await asyncio.sleep(0)

    monkeypatch.setattr(pss, "_sleep", fake_sleep)
    return t


async def test_a_page_is_saved_to_every_table(fake_arxiv):
    src = await pss.ensure("2609.31507")
    assert src.ok and src.version == "v1"
    assert src.licence == "CC-BY-4.0" and src.licence_label == "CC BY 4.0"
    assert [f["anchor"] for f in src.figures] == ["S1.F1", "S3.F2", "S3.F3", "S4.F4"]
    assert src.figures[0]["src"] == "2609.31507v1/overview-final.png"
    assert (src.figures[0]["w"], src.figures[0]["h"]) == (476, 235)
    full = await pss.fulltext("2609.31507")
    assert any(s["anchor"] == "S3.SS2" for s in full["sections"])
    paras = await pss.paragraphs("2609.31507")
    assert paras and paras[0]["anchor"].startswith("S1.p")
    assert src.n_paragraphs == len(paras)
    assert fake_arxiv["requests"] == [f"{config.ARXIV_HTML_BASE}/2609.31507"]


async def test_a_saved_page_is_not_fetched_again(fake_arxiv):
    await pss.ensure("2609.31507")
    await pss.ensure("2609.31507")
    assert len(fake_arxiv["requests"]) == 1


async def test_requests_identify_themselves(fake_arxiv):
    seen = {}

    def capture(request):
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, content=SATNAV)

    fake_arxiv["pages"]["2609.31507"] = (200, capture, {})
    await pss.ensure("2609.31507")
    assert seen["ua"] == config.OUTBOUND_USER_AGENT


async def test_no_html_is_remembered_for_a_week(fake_arxiv):
    src = await pss.ensure("2609.00404")
    assert src.status == "no_html" and not src.figures_allowed
    await pss.ensure("2609.00404")
    assert len(fake_arxiv["requests"]) == 1


async def test_three_soft_blocks_open_the_breaker(fake_arxiv, clock):
    for i in range(3):
        fake_arxiv["pages"][f"2609.0000{i}"] = (429, b"", {})
    for i in range(3):
        assert await pss.ensure(f"2609.0000{i}") is None
    assert pss.breaker_open()
    before = len(fake_arxiv["requests"])
    assert await pss.ensure("2609.31507") is None          # nothing fetched while open
    assert len(fake_arxiv["requests"]) == before
    clock["now"] += pss.BREAKER_OPEN_S + 1
    assert (await pss.ensure("2609.31507")).ok


async def test_a_long_retry_after_opens_the_breaker_at_once(fake_arxiv, clock):
    fake_arxiv["pages"]["2609.00001"] = (503, b"", {"Retry-After": "1200"})
    await pss.ensure("2609.00001")
    assert pss.breaker_open()


async def test_a_redirect_to_the_versioned_page_is_followed(fake_arxiv):
    def redirect(request):
        if request.url.path.endswith("v1"):
            return httpx.Response(200, content=SATNAV)
        return httpx.Response(301, headers={"Location": "/html/2609.31507v1"})

    fake_arxiv["pages"]["2609.31507"] = (0, redirect, {})
    fake_arxiv["pages"]["2609.31507v1"] = (0, redirect, {})
    assert (await pss.ensure("2609.31507")).ok


async def test_a_redirect_away_from_html_means_no_html(fake_arxiv):
    def to_abs(request):
        if "/abs/" in request.url.path:
            return httpx.Response(200, content=b"<html>abstract page</html>")
        return httpx.Response(302, headers={"Location": "https://arxiv.org/abs/2609.00002"})

    fake_arxiv["pages"]["2609.00002"] = (0, to_abs, {})
    assert (await pss.ensure("2609.00002")).status == "no_html"


async def test_an_oversized_page_is_cut_and_recorded_as_an_error(fake_arxiv):
    big = b"<html><body>" + b"<p>x</p>" * (9 * 1024 * 1024 // 8) + b"</body></html>"
    fake_arxiv["pages"]["2609.00003"] = (200, big, {})
    src = await pss.ensure("2609.00003")
    assert src.status == "error" and "8 MB" in src.error


async def test_ten_concurrent_opens_make_one_request(fake_arxiv):
    results = await asyncio.gather(*[pss.ensure("2609.31507") for _ in range(10)])
    assert all(r and r.ok for r in results)
    assert len(fake_arxiv["requests"]) == 1


@pytest.mark.parametrize("bad", ["", "hep-th/9901001", "2609.31507v1", "../x", "2609.315"])
async def test_ids_that_are_not_new_style_are_never_fetched(fake_arxiv, bad):
    assert await pss.ensure(bad) is None
    assert fake_arxiv["requests"] == []


async def test_a_network_error_writes_nothing_so_the_next_open_retries(fake_arxiv):
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, content=SATNAV)

    fake_arxiv["pages"]["2609.31507"] = (0, flaky, {})
    assert await pss.ensure("2609.31507") is None
    assert await pss.get("2609.31507") is None
    assert (await pss.ensure("2609.31507")).ok


async def test_a_newer_parser_refetches_saved_pages(fake_arxiv):
    await pss.ensure("2609.31507")
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute("UPDATE paper_sources SET parser_version = 0")
        await conn.commit()
    await pss.ensure("2609.31507")
    assert len(fake_arxiv["requests"]) == 2


async def test_a_failed_refetch_keeps_serving_the_last_good_copy(fake_arxiv):
    await pss.ensure("2609.31507")
    fake_arxiv["pages"]["2609.31507"] = (500, b"", {})
    src = await pss.ensure("2609.31507", force=True)
    assert src.ok and src.figures and "HTTP 500" in src.error


async def test_a_refetch_keeps_the_code_link(fake_arxiv):
    await pss.ensure("2609.31507")
    await pss.save_code_link("2609.31507", "https://github.com/Eku127/SatNav", 15)
    src = await pss.ensure("2609.31507", force=True)
    assert (src.code_repo, src.code_stars) == ("https://github.com/Eku127/SatNav", 15)


async def test_a_code_link_saved_first_does_not_block_the_fetch(fake_arxiv):
    await pss.save_code_link("2609.31507", "", None)
    src = await pss.ensure("2609.31507")
    assert src.ok and src.code_checked_at


async def test_replicated_rows_stay_slim(fake_arxiv):
    figs = "".join(
        f'<figure id="S1.F{i}" class="ltx_figure"><img src="2609.00005v1/f{i}.png" class="ltx_graphics" '
        f'width="100" height="50"><figcaption>Figure {i}: {"long caption " * 60}</figcaption></figure>'
        for i in range(1, 13))
    page = (f'<div id="infobox"><a id="license-tr">License: CC BY 4.0</a>'
            f'<div id="watermark-tr">arXiv:2609.00005v1 [cs.LG] 1 Sep 2026</div></div>'
            f'<h1 class="ltx_title ltx_title_document">Twelve figures</h1>{figs}'
            f'<div class="ltx_para" id="S1.p1">Text.</div>').encode()
    fake_arxiv["pages"]["2609.00005"] = (200, page, {})
    src = await pss.ensure("2609.00005")
    assert len(src.figures) == pss.REPLICATED_FIGURES and src.n_figures == 12
    assert all(len(f["caption"]) <= pss.REPLICATED_CAPTION for f in src.figures)
    full = await pss.fulltext("2609.00005")
    assert len(full["figures"]) == 12


async def test_the_pacer_allows_a_burst_of_three_then_one_every_two_seconds(clock):
    for _ in range(5):
        assert await pss.pace()
    assert clock["sleeps"] == [pytest.approx(2.0), pytest.approx(2.0)]


async def test_parsing_a_large_page_does_not_stall_the_event_loop(fake_arxiv):
    body = SATNAV.replace(b"</body>", b"".join(
        b'<div class="ltx_para" id="S9.p%d"><p class="ltx_p">%s</p></div>' % (i, b"word " * 200)
        for i in range(1500)) + b"</body>")
    assert len(body) > 1_400_000
    fake_arxiv["pages"]["2609.31507"] = (200, body, {})
    gaps, stop = [], asyncio.Event()

    async def ticker():
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    tick = asyncio.create_task(ticker())
    await pss.ensure("2609.31507")
    stop.set()
    await tick
    assert max(gaps) < 0.1


async def test_prune_keeps_the_most_recently_used_full_text(fake_arxiv):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        for i in range(5):
            await conn.execute(
                "INSERT INTO paper_fulltext (arxiv_id, last_used_at) VALUES (?, ?)",
                (f"2609.0000{i}", f"2026-10-0{i + 1} 00:00:00"))
            await conn.execute(
                "INSERT INTO paper_paragraphs (arxiv_id, anchor, ord, text) VALUES (?, 'S1.p1', 0, 't')",
                (f"2609.0000{i}",))
        await conn.commit()
    assert await pss.prune(2) == 3
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute("SELECT arxiv_id FROM paper_fulltext ORDER BY arxiv_id")
        assert [r[0] for r in await cur.fetchall()] == ["2609.00003", "2609.00004"]
        cur = await conn.execute("SELECT COUNT(*) FROM paper_paragraphs")
        assert (await cur.fetchone())[0] == 2


async def test_schedule_does_nothing_with_the_switch_off(fake_arxiv, monkeypatch):
    monkeypatch.setenv("PAPER_SOURCES", "off")
    pss.schedule("2609.31507")
    await asyncio.sleep(0.05)
    assert fake_arxiv["requests"] == []


async def test_stats_report_status_and_breaker(fake_arxiv):
    await pss.ensure("2609.31507")
    await pss.ensure("2609.00404")
    s = await pss.stats()
    assert s["by_status"] == {"ok": 1, "no_html": 1}
    assert s["with_figures"] == 1 and s["licence_mix"] == {"CC-BY-4.0": 1}
    assert s["breaker_open"] is False and s["switch"] == "on"


def test_html_links_only_take_safe_anchors():
    src = pss.Source(arxiv_id="2609.31507", status="ok", version="v1")
    assert src.html_url("S3.SS2") == "https://arxiv.org/html/2609.31507v1#S3.SS2"
    assert src.html_url('"><script>') == "https://arxiv.org/html/2609.31507v1"
