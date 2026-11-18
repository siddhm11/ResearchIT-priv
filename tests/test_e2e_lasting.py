"""
Browser tests for the redesigned /trending/lasting (doc 06, 2026-11-17).

RUNNING
-------
Needs a server with Hacker News data and HN_LASTING on. The seeded e2e world
has none, so point it at a server reading a local copy of the HN tables
(scripts/hn_harvest.py can fill one; HN_SQLITE names it):

    HN_SQLITE=/path/hn.sqlite HN_LASTING=on TURSO_SYNC_DISABLED=1 DB_PATH=/tmp/e2e.db \\
        .venv/bin/python -m uvicorn app.main:app --port 7881
    E2E_LASTING_URL=http://127.0.0.1:7881 .venv/bin/python -m pytest tests/test_e2e_lasting.py -v

Skips itself when playwright is missing, no server answers, or the server's
chart has fewer than 50 papers.

What matters: the chart draws, names a paper on hover, tap and keyboard, opens
it as a plain visit (no ranking fields), follows the theme, and its failure
removes only the chart; the menus close each other, on Escape and outside;
the story opens; nothing scrolls sideways on a phone.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

BASE_URL = os.getenv("E2E_LASTING_URL", "http://127.0.0.1:7881").rstrip("/")


def _hn_world() -> bool:
    try:
        data = json.loads(urllib.request.urlopen(f"{BASE_URL}/trending/lasting/strata.json", timeout=15).read())
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return len(data.get("lines") or []) >= 50


pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(not _hn_world(), reason=f"no server with Hacker News data at {BASE_URL}"),
]


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def open_page(browser, width=1280, height=900, scheme="light", touch=False, block_chart=False):
    ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme,
                              is_mobile=touch, has_touch=touch)
    page = ctx.new_page()
    page.errors = []
    page.on("console", lambda m: page.errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    if block_chart:
        page.route("**/strata.json", lambda r: r.fulfill(status=500, body="no"))
    page.goto(BASE_URL + "/trending/lasting", wait_until="domcontentloaded")
    if not block_chart:
        page.wait_for_selector(".hn-strata.is-ready", timeout=15000)
    return page


def ink(page) -> int:
    return page.evaluate("""() => { const c = document.querySelector('.hn-strata-canvas');
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0;
      for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++; return n; }""")


def dot_colour(page):
    return page.evaluate("""() => { const c = document.querySelector('.hn-strata-canvas');
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      for (let i = 0; i < d.length; i += 4) if (d[i+3] > 120 && d[i+1] > d[i] + 15) return [d[i], d[i+1], d[i+2]];
      return null; }""")


def test_the_chart_draws_and_names_a_paper_on_hover(browser):
    page = open_page(browser)
    assert ink(page) > 10000
    box = page.locator(".hn-strata-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.7)
    tip = page.locator(".hn-strata-tip")
    assert tip.is_visible() and tip.inner_text().count("#") == 1
    page.mouse.move(2, 2)
    assert not tip.is_visible()
    assert not page.errors


def test_a_click_on_a_line_opens_the_paper_as_a_plain_visit(browser):
    page = open_page(browser)
    box = page.locator(".hn-strata-canvas").bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.9, box["y"] + box["height"] * 0.5)
    page.wait_for_url("**/p/**", timeout=10000)
    assert "qid=" not in page.url                   # §3.11: a visit from the chart is not a ranked click


def test_the_keyboard_steps_through_papers_and_opens_one(browser):
    page = open_page(browser)
    page.locator(".hn-strata-canvas").focus()
    page.keyboard.press("ArrowDown")
    first = page.locator(".hn-strata-tip").inner_text()
    page.keyboard.press("PageDown")
    assert page.locator(".hn-strata-tip").inner_text() != first
    page.keyboard.press("Enter")
    page.wait_for_url("**/p/**", timeout=10000)


def test_a_tap_names_the_paper_before_opening_it(browser):
    page = open_page(browser, 390, 844, touch=True)
    box = page.locator(".hn-strata-canvas").bounding_box()
    page.touchscreen.tap(box["x"] + box["width"] * 0.85, box["y"] + box["height"] * 0.6)
    assert page.locator(".hn-strata-tip").is_visible() and page.url.endswith("/trending/lasting")


def test_the_chart_follows_the_chosen_theme(browser):
    page = open_page(browser)
    light = dot_colour(page)
    page.evaluate("document.documentElement.setAttribute('data-theme', 'dark')")
    page.wait_for_timeout(300)
    dark = dot_colour(page)
    assert light and dark and dark[1] > light[1] + 40


def test_a_failed_chart_load_removes_only_the_chart(browser):
    page = open_page(browser, block_chart=True)
    page.wait_for_load_state("networkidle")
    assert page.locator(".hn-strata").count() == 0
    assert page.locator("li.hn-row--lasting").count() == 25


def test_the_menus_close_each_other_on_escape_and_outside(browser):
    page = open_page(browser)
    menus = page.locator("details.hn-menu")
    menus.nth(0).locator("summary").click()
    menus.nth(1).locator("summary").click()
    assert menus.nth(0).get_attribute("open") is None and menus.nth(1).get_attribute("open") is not None
    page.keyboard.press("Escape")
    assert menus.nth(1).get_attribute("open") is None
    menus.nth(0).locator("summary").click()
    page.mouse.click(4, 300)
    assert menus.nth(0).get_attribute("open") is None


def test_the_story_opens_with_its_moments(browser):
    page = open_page(browser)
    row = page.locator("li.hn-row--lasting").nth(0)
    row.locator(".hn-story > summary").click()
    assert row.locator(".hn-beats li").count() >= 2 and row.locator(".hn-dated li").count() >= 1


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_nothing_scrolls_sideways_on_a_phone(browser, scheme):
    page = open_page(browser, 390, 844, scheme=scheme, touch=True)
    page.locator("li.hn-row--lasting").nth(1).locator(".hn-story > summary").click()
    page.locator("details.hn-menu").nth(0).locator("summary").click()
    assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 0
    pop = page.locator(".hn-menu-pop").first.bounding_box()
    assert pop["x"] >= 0 and pop["x"] + pop["width"] <= 390
    assert not page.errors
