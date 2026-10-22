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


def test_back_from_a_paper_returns_to_the_search(browser):
    """Search, open a result, go back (reported 2026-10-21). Back used to show the
    cached results fragment without the page around it, and the paper page's own
    link said "Your feed" and went to /. Going back must not run the search again."""
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    ctx.add_cookies([{"name": "arxiv_user_id", "value": f"e2e-{uuid.uuid4()}", "url": BASE_URL}])
    page = ctx.new_page()
    try:
        page.goto(f"{BASE_URL}/search", wait_until="networkidle")
        page.fill('form.searchbar input[name="q"]', "navigation")
        page.press('form.searchbar input[name="q"]', "Enter")
        page.wait_for_selector("#search-results .card .card-title")
        assert page.url.endswith("/search?q=navigation")

        page.click("#search-results .card .card-title >> nth=0")
        page.wait_for_load_state("networkidle")
        link = page.locator(".back-link")
        assert link.inner_text() == "← Search results" and link.get_attribute("href") == "/search?q=navigation"

        page.go_back(wait_until="networkidle")                       # the browser's Back button
        assert page.url.endswith("/search?q=navigation")
        assert page.locator("form.searchbar").count() == 1 and page.locator("#search-results .card").count() > 0
        assert "navigation" in page.title()
        assert page.locator("#search-results [data-cached]").count() == 1   # served from memory, not run again

        page.go_forward(wait_until="networkidle")                    # the page's own link
        page.click(".back-link")
        page.wait_for_load_state("networkidle")
        assert page.url.endswith("/search?q=navigation") and page.locator("#search-results .card").count() > 0
        assert page.locator("#search-results [data-cached]").count() == 1
    finally:
        ctx.close()


def test_a_search_from_the_box_renames_the_heading_and_the_tab(browser):
    """Open one search, run another from the box (reported 2026-10-23). Only the
    results were swapped, so the heading and the tab kept the first query above
    the second one's results. Back and Forward must show each its own."""
    ctx, page = search(browser, 1280)                                # opens /search?q=navigation
    heading = "() => document.getElementById('search-title').textContent"
    try:
        assert page.evaluate(heading) == "navigation" and page.title() == "navigation — Search — ResearchIT"
        page.fill('form.searchbar input[name="q"]', "astra")
        page.press('form.searchbar input[name="q"]', "Enter")
        page.wait_for_function(f"({heading})() === 'astra'")
        # htmx sets the title as the swap settles, a moment after the heading.
        page.wait_for_function("document.title === 'astra — Search — ResearchIT'")
        assert page.url.endswith("/search?q=astra")
        assert page.locator("#search-title").count() == 1 and page.locator("#search-title").is_visible()

        page.go_back()
        page.wait_for_function(f"({heading})() === 'navigation'")
        page.wait_for_function("document.title === 'navigation — Search — ResearchIT'")
        page.go_forward()
        page.wait_for_function(f"({heading})() === 'astra'")
        page.wait_for_function("document.title === 'astra — Search — ResearchIT'")
        assert not page.errors
    finally:
        ctx.close()
