"""
Browser tests for search thumbnails (SEARCH_FIGURES), against the seeded world.

Same world and server as tests/test_e2e_paper.py (see scripts/seed_e2e.py);
its sidecar has the word index, so "navigation" finds SatNav and GPT-6-Astra
without any outside service. Skips itself when no such server answers.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
import uuid

import pytest

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:7861").rstrip("/")


def _search_world() -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}/search?q=navigation", headers={"HX-Request": "true"})
        return "card-thumb" in urllib.request.urlopen(req, timeout=30).read().decode()
    except (urllib.error.URLError, OSError):
        return False


pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(not _search_world(),
                       reason=f"no seeded server with SEARCH_FIGURES at {BASE_URL} (see scripts/seed_e2e.py)"),
]


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def search(browser, width, height=900):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    ctx.add_cookies([{"name": "arxiv_user_id", "value": f"e2e-{uuid.uuid4()}", "url": BASE_URL}])
    page = ctx.new_page()
    page.errors = []
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    page.goto(f"{BASE_URL}/search?q=navigation", wait_until="networkidle")
    page.wait_for_function("""() => { const i = Array.from(document.querySelectorAll('.card-thumb img'));
        return i.length > 0 && i.every(x => x.complete && x.naturalWidth > 0); }""")
    return ctx, page


@pytest.mark.parametrize("width,max_thumb", [(1440, 125), (390, 80)])
def test_thumbnails_sit_beside_the_text_and_stay_small(browser, width, max_thumb):
    ctx, page = search(browser, width)
    try:
        boxes = page.evaluate("""() => Array.from(document.querySelectorAll('.card-split')).map(s => {
            const t = s.querySelector('.card-title').getBoundingClientRect();
            const m = s.querySelector('.card-thumb').getBoundingClientRect();
            return {titleRight: t.right, thumbLeft: m.left, thumbWidth: m.width, thumbTop: m.top, titleTop: t.top}; })""")
        assert boxes
        for b in boxes:
            assert b["thumbLeft"] >= b["titleRight"] - 1          # beside the text, not over it
            assert b["thumbWidth"] <= max_thumb
            assert abs(b["thumbTop"] - b["titleTop"]) < 20         # level with the title
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.errors == []
    finally:
        ctx.close()


def test_a_thumbnail_opens_the_paper_at_its_figure_with_the_search_link(browser):
    ctx, page = search(browser, 1440)
    try:
        thumb = page.locator(".card-thumb").first
        assert thumb.get_attribute("tabindex") == "-1" and thumb.get_attribute("aria-hidden") == "true"
        with page.expect_navigation():
            thumb.click()
        assert "/p/" in page.url and "sf=search" in page.url and page.url.endswith("#sec-fig")
    finally:
        ctx.close()
