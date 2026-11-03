"""
Browser tests for onboarding's "In a sentence" step, against the seeded world.

Same world and server as tests/test_e2e_search.py (see scripts/seed_e2e.py):
its sidecar has the word index, so "navigation" finds SatNav and GPT-6-Astra
(both cs.CV, cross-listed cs.RO) without any outside service. Skips itself
when no such server answers.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
import uuid

import pytest

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:7861").rstrip("/")


def _seeded_world() -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}/search?q=navigation", headers={"HX-Request": "true"})
        return "SatNav" in urllib.request.urlopen(req, timeout=30).read().decode()
    except (urllib.error.URLError, OSError):
        return False


pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(not _seeded_world(), reason=f"no seeded server at {BASE_URL} (see scripts/seed_e2e.py)"),
]


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    ctx.add_cookies([{"name": "arxiv_user_id", "value": f"e2e-{uuid.uuid4()}", "url": BASE_URL}])
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.goto(f"{BASE_URL}/onboarding", wait_until="networkidle")
    yield pg
    ctx.close()


def test_a_described_interest_picks_its_areas_and_opens_on_its_papers(page):
    page.fill("#describe-q", "navigation")
    page.click("#describe-btn")
    page.wait_for_function("document.getElementById('describe-status').textContent.startsWith('Picked')")

    assert page.get_attribute('.cat-opt[data-key="cv"]', "aria-pressed") == "true"
    assert page.locator(".cat-opt[data-suggested]").count() == 1      # cs.RO is only cross-listed
    assert "Computer Vision" in page.inner_text("#describe-status")
    assert page.inner_text("#selected-count") == "1"
    assert page.is_enabled("#continue-btn")

    page.click("#continue-btn")
    page.wait_for_selector("#step-2 .seed-row")
    assert page.input_value('#step-2 input[name="q"]') == "navigation"
    assert "SatNav" in page.inner_text("#seed-results")
    assert not page.errors


def test_a_description_adds_to_what_is_picked_and_can_be_undone(page):
    page.click('.cat-opt[data-key="hep"]')
    page.fill("#describe-q", "navigation")
    page.press("#describe-q", "Enter")
    page.wait_for_function("document.getElementById('describe-status').textContent.startsWith('Picked')")
    assert page.inner_text("#selected-count") == "2"

    page.click('.cat-opt[data-key="cv"]')                              # the reader disagrees
    assert page.get_attribute('.cat-opt[data-key="cv"]', "aria-pressed") == "false"
    assert page.get_attribute('.cat-opt[data-key="hep"]', "aria-pressed") == "true"
    assert not page.errors


def test_a_description_nothing_matches_says_so_and_picks_nothing(page):
    page.fill("#describe-q", "zzqx qqzv")
    page.click("#describe-btn")
    page.wait_for_function("document.getElementById('describe-status').textContent.startsWith('Nothing placed')")
    assert page.locator('.cat-opt[aria-pressed="true"]').count() == 0
    assert not page.is_enabled("#continue-btn")
