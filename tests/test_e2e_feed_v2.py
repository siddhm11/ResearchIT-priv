"""
Browser tests for the redesigned feed card (FEED_CARD_V2) and the wide feed
(FEED_WIDE), against the seeded test world.

RUNNING
-------
Same world and server as tests/test_e2e_paper.py (the command in
scripts/seed_e2e.py already turns FEED_CARD_V2 and FEED_WIDE on):

    E2E_BASE_URL=http://127.0.0.1:7861 .venv/bin/python -m pytest tests/test_e2e_feed_v2.py -v

Each test onboards a new reader through the category step, so the feed is the
popularity-per-category tier over the six seeded papers; no vector store is
involved. Skips itself when playwright is missing, no server answers, or the
server's home page has no wide feed.

What matters most here is order: the two columns must show the feed in the
order it was served (quota decides that order, CLAUDE.md §3.1), so a card may
never sit higher on the page than the card served before it.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid

import pytest

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:7861").rstrip("/")


def _wide_world() -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}/", headers={"Cookie": "arxiv_user_id=e2e-probe"})
        html = urllib.request.urlopen(req, timeout=15).read().decode()
    except (urllib.error.URLError, OSError):
        return False
    # A new reader is sent to onboarding; a finished one sees the feed.
    return "data-wide-feed" in html or "/onboarding" in html


pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(not _wide_world(),
                       reason=f"no seeded redesign server at {BASE_URL} (see scripts/seed_e2e.py)"),
]


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def feed(browser, width=1440, height=900):
    """A new reader, onboarded on the four seeded areas, looking at the feed."""
    ctx = browser.new_context(viewport={"width": width, "height": height})
    ctx.add_cookies([{"name": "arxiv_user_id", "value": f"e2e-{uuid.uuid4()}", "url": BASE_URL}])
    page = ctx.new_page()
    page.errors = []
    page.on("console", lambda m: page.errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    api = ctx.request
    assert api.post(f"{BASE_URL}/api/onboarding/categories",
                    data=json.dumps({"categories": ["cv", "ml", "nlp", "ai"]}),
                    headers={"Content-Type": "application/json"}).ok
    assert api.post(f"{BASE_URL}/api/onboarding/complete", max_redirects=0).status == 303
    page.goto(BASE_URL + "/", wait_until="networkidle")
    page.wait_for_selector("[data-wide-feed] .card.pc")
    page.wait_for_function("document.querySelectorAll('[data-wide-feed] .card.pc')"
                           ".length > 0 && document.querySelector('[data-wide-feed] .card.pc').style.gridRowEnd !== ''"
                           " || window.innerWidth < 1180")
    return ctx, page


def boxes(page):
    """Each visible card's box, in DOM (served) order."""
    return page.evaluate("""() => Array.from(document.querySelectorAll('[data-wide-feed] .feed > .card'))
        .filter(c => !c.hidden)
        .map(c => { const r = c.getBoundingClientRect();
                    return {id: c.dataset.arxivId, x: r.left, y: r.top + scrollY, w: r.width, h: r.height}; })""")


def overlaps(a, b):
    return a["x"] < b["x"] + b["w"] - 1 and b["x"] < a["x"] + a["w"] - 1 and \
        a["y"] < b["y"] + b["h"] - 1 and b["y"] < a["y"] + a["h"] - 1


def no_sideways_scroll(page):
    return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_wide_screens_get_two_columns_in_served_order(browser):
    ctx, page = feed(browser, 1440)
    try:
        cards = boxes(page)
        assert len(cards) >= 4
        assert len({round(c["x"]) for c in cards}) == 2, cards
        for before, after in zip(cards, cards[1:]):
            assert after["y"] >= before["y"] - 1, (before, after)
        for i, a in enumerate(cards):
            for b in cards[i + 1:]:
                assert not overlaps(a, b), (a, b)
        assert no_sideways_scroll(page)
        assert page.errors == []
    finally:
        ctx.close()


def test_below_the_wide_breakpoint_it_is_one_column(browser):
    ctx, page = feed(browser, 1179)
    try:
        cards = boxes(page)
        assert len({round(c["x"]) for c in cards}) == 1, cards
        assert [c["y"] for c in cards] == sorted(c["y"] for c in cards)
        assert no_sideways_scroll(page)
        assert page.errors == []
    finally:
        ctx.close()


def test_a_phone_gets_one_column_without_sideways_scrolling(browser):
    ctx, page = feed(browser, 390, 844)
    try:
        assert len({round(c["x"]) for c in boxes(page)}) == 1
        assert no_sideways_scroll(page)
    finally:
        ctx.close()


def test_save_sends_the_layout_it_happened_in(browser):
    ctx, page = feed(browser, 1440)
    try:
        card = page.locator("[data-wide-feed] .card.pc").first
        with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/save")) as req:
            card.locator("button", has_text="Save").first.click()
        body = req.value.post_data or ""
        assert "layout=grid2" in body, body
        assert "query_id=" in body and "propensity=" in body       # §3.11 fields still go with it
    finally:
        ctx.close()


def test_clicking_a_title_carries_the_layout(browser):
    ctx, page = feed(browser, 1440)
    try:
        link = page.locator("[data-wide-feed] .card.pc .card-title").first
        with page.expect_navigation():
            link.click()
        assert "lay=grid2" in page.url and "qid=" in page.url
    finally:
        ctx.close()


def test_the_rail_counts_papers_and_filters_by_interest(browser):
    ctx, page = feed(browser, 1440)
    try:
        total = len(boxes(page))
        assert page.inner_text("#rail-ints .is-all [data-count]") == str(total)
        # Readers who only picked categories get no interest labels, so label
        # the cards the way the cluster tiers do and let the rail re-read them.
        page.evaluate("""() => {
            document.querySelectorAll('[data-wide-feed] .feed > .card').forEach((c, i) => {
                c.dataset.interest = i % 3 === 2 ? 'explore' : (i % 3 ? 'Speech <b>x</b>' : 'Drone navigation');
                c.dataset.tone = String(i % 3); });
            document.body.dispatchEvent(new Event('htmx:afterSettle')); }""")
        chips = page.locator("#rail-ints [data-filter]:not(.is-all)")
        assert chips.all_inner_texts()[-1].startswith("Something different")     # exploration goes last
        assert page.locator("#rail-ints b").count() == 0                         # labels are text, not markup
        chip = page.locator('#rail-ints [data-filter="Drone navigation"]')
        want = int(chip.locator(".n").inner_text())
        with page.expect_request(lambda r: r.url.endswith("/api/ui-event")):
            chip.click()
        shown = boxes(page)
        assert len(shown) == want and want < total
        assert chip.get_attribute("aria-pressed") == "true"
        assert page.inner_text("#filter-note").startswith(f"Showing {want} of {total}")
        assert len({round(c["x"]) for c in shown}) == min(2, want)                # repacked, no holes
        page.click("#filter-note [data-filter]")
        assert len(boxes(page)) == total and not page.is_visible("#filter-note")
        assert page.errors == []
    finally:
        ctx.close()


def test_card_text_tabs_and_the_rail_preference(browser):
    ctx, page = feed(browser, 1440)
    try:
        card = page.locator("[data-wide-feed] .card.pc:has([data-cardtab])").first
        assert card.locator('[data-panel="plain"]').is_visible()
        card.locator('[data-cardtab="abs"]').click()
        assert card.locator('[data-panel="abs"]').is_visible()
        assert not card.locator('[data-panel="plain"]').is_visible()
        page.click('[data-cardpref="abs"]')
        page.reload(wait_until="networkidle")
        page.wait_for_selector("[data-wide-feed] .card.pc")
        card = page.locator("[data-wide-feed] .card.pc:has([data-cardtab])").first
        assert card.locator('[data-panel="abs"]').is_visible()
        assert page.get_attribute('[data-cardpref="abs"]', "aria-pressed") == "true"
    finally:
        ctx.close()


def test_card_figures_load_and_the_menu_opens(browser):
    ctx, page = feed(browser, 1440)
    try:
        page.wait_for_function("""() => { const imgs = document.querySelectorAll('.pc-fig img');
            return imgs.length > 0 && Array.from(imgs).every(i => i.complete && i.naturalWidth > 0); }""")
        menu_btn = page.locator("[data-wide-feed] .card.pc [data-menu]").first
        menu_btn.click()
        assert menu_btn.get_attribute("aria-expanded") == "true"
        page.keyboard.press("Escape")
        assert menu_btn.get_attribute("aria-expanded") == "false"
        assert page.errors == []
    finally:
        ctx.close()


def test_a_card_figure_that_cannot_be_made_is_hidden(browser):
    ctx, page = feed(browser, 1440)
    try:
        card = page.locator("[data-wide-feed] .card.pc:has(.pc-fig)").first
        card.locator(".pc-fig img").evaluate("i => { i.loading = 'eager'; i.src = '/fig/2609.31507/99-00000000.webp'; }")
        page.wait_for_function("el => el.querySelector('.pc-fig').hidden", arg=card.element_handle())
        assert card.locator(".card-title").is_visible()
    finally:
        ctx.close()

