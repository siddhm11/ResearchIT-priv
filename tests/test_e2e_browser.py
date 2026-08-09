"""
Browser end-to-end tests.

WHY THIS FILE EXISTS
--------------------
Two frontend bugs shipped to production while every server-side check passed:
HTTP 200s, correct rendered markup, correct database writes, full suite green.
Both were only observable in a real browser.

  1. Save did nothing.  hx-target="#actions-1706.03762" — a dot in a CSS id
     selector starts a class token, so that parses as id="actions-1706" AND
     class="03762", and a class token cannot begin with a digit, so
     querySelector raises rather than merely missing.  The POST fired and the
     row was written; only the swap never happened.

  2. Infinite scroll never fired.  hx-trigger="revealed, click" — htmx's
     revealed poller matches the attribute string exactly
     ("[hx-trigger='revealed']"), so a combined trigger can never be selected.

Neither is detectable without executing the page.  These tests drive a real
Chromium, so they fail if a swap does not happen — which is the entire point.

RUNNING
-------
    .venv/bin/python -m pytest tests/test_e2e_browser.py -v

Requires a live server; by default http://127.0.0.1:7860.  Point at the
deployed Space with:

    E2E_BASE_URL=https://siddhm11-researchit.hf.space .venv/bin/python -m pytest tests/test_e2e_browser.py -v

Skips itself (rather than failing) when playwright is absent or no server is
reachable, so it never blocks the unit suite in an environment without them.
"""
from __future__ import annotations

import os
import uuid

import pytest

playwright_api = pytest.importorskip(
    "playwright.sync_api", reason="playwright not installed"
)
sync_playwright = playwright_api.sync_playwright

BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

# The feed pipeline is genuinely slow on a cold cache (Ward clustering, ANN over
# binary-quantised vectors, metadata fetch), so waits here are generous. These
# are correctness tests, not latency tests.
FEED_TIMEOUT_MS = 90_000
SWAP_TIMEOUT_MS = 30_000


def _server_up() -> bool:
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(f"{BASE_URL}/healthz/reranker", timeout=15)
        return True
    except (urllib.error.URLError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(), reason=f"no server reachable at {BASE_URL}"
)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    # A fresh user per test: cookie identity drives the whole recommendation
    # cascade, so sharing one would let tests leak state into each other.
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    ctx.add_cookies([{
        "name": "arxiv_user_id",
        "value": f"e2e-{uuid.uuid4()}",
        "url": BASE_URL,
    }])
    pg = ctx.new_page()
    yield pg
    ctx.close()


def _onboard(page, categories=("nlp",)):
    """Complete onboarding fully, so `/` serves a feed instead of redirecting.

    Onboarding is only 'complete' once /api/onboarding/complete has been
    POSTed — selecting categories is not enough. Until then `/` answers 302
    back to /onboarding, so any test that skips the final step waits forever
    for a card that is never rendered.
    """
    page.goto(f"{BASE_URL}/onboarding", wait_until="domcontentloaded")
    page.wait_for_selector(".cat-opt", timeout=SWAP_TIMEOUT_MS)
    for key in categories:
        page.click(f'.cat-opt[data-key="{key}"]')
    page.click("#continue-btn")
    page.wait_for_selector("#step-2 .searchbar", timeout=SWAP_TIMEOUT_MS)

    # Step 2's footer submits the form that marks onboarding done.
    page.click(".ob-footer button[type=submit]")
    page.wait_for_url(f"{BASE_URL}/", timeout=SWAP_TIMEOUT_MS)


# ── Bug 1: the Save swap ─────────────────────────────────────────────────────

def test_save_button_swaps_to_saved_state(page):
    """Clicking Save must visibly become 'Saved'.

    Regression test for the dot-in-id selector bug: the POST succeeded and the
    interaction row was written even while this was broken, so only asserting
    on the response would have passed.
    """
    page.goto(f"{BASE_URL}/search?q=chain+of+thought+prompting",
              wait_until="domcontentloaded")
    card = page.locator("article.card").first
    card.wait_for(timeout=FEED_TIMEOUT_MS)

    save = card.locator("button.btn-save")
    assert save.inner_text().strip().startswith("Save"), "expected an unsaved card"

    save.click()

    # The swap replaces the actions row; wait for the new state to appear.
    saved = card.locator("button.btn-save[disabled]")
    saved.wait_for(timeout=SWAP_TIMEOUT_MS)
    assert "Saved" in saved.inner_text()
    assert card.locator("button.btn-pass").inner_text().strip() == "Remove"


def test_save_target_selectors_are_resolvable(page):
    """Every hx-target on the page must actually resolve in the DOM.

    This is the generalised form of bug 1 — it catches ANY id whose characters
    break a CSS selector, not just the dot case, and it does so without having
    to click each control.
    """
    page.goto(f"{BASE_URL}/search?q=transformer", wait_until="domcontentloaded")
    page.locator("article.card").first.wait_for(timeout=FEED_TIMEOUT_MS)

    bad = page.evaluate(
        """() => {
            const out = [];
            document.querySelectorAll('[hx-target]').forEach(el => {
                const sel = el.getAttribute('hx-target');
                // htmx's extended syntax is not a plain CSS selector
                if (/^(this|closest |find |next|previous)/.test(sel)) return;
                try {
                    if (document.querySelectorAll(sel).length !== 1) {
                        out.push([sel, 'matched ' + document.querySelectorAll(sel).length]);
                    }
                } catch (e) {
                    out.push([sel, 'SyntaxError: ' + e.message]);
                }
            });
            return out;
        }"""
    )
    assert bad == [], f"unresolvable hx-target selectors: {bad}"


def test_saved_paper_appears_in_library(page):
    """Save must persist: the paper shows up in the Library on a fresh load."""
    page.goto(f"{BASE_URL}/search?q=chain+of+thought+prompting",
              wait_until="domcontentloaded")
    card = page.locator("article.card").first
    card.wait_for(timeout=FEED_TIMEOUT_MS)
    arxiv_id = card.get_attribute("data-arxiv-id")

    card.locator("button.btn-save").click()
    card.locator("button.btn-save[disabled]").wait_for(timeout=SWAP_TIMEOUT_MS)

    page.goto(f"{BASE_URL}/saved", wait_until="domcontentloaded")
    page.locator("article.card").first.wait_for(timeout=SWAP_TIMEOUT_MS)
    assert page.locator(f'article.card[data-arxiv-id="{arxiv_id}"]').count() == 1


# ── Bug 2: infinite scroll ───────────────────────────────────────────────────

def test_scrolling_appends_more_papers(page):
    """Scrolling to the bottom must load another page WITHOUT any click.

    Regression test for the htmx trigger bug. The loader element rendered
    correctly and its endpoint returned 200 the whole time it was broken, so
    markup and HTTP assertions both passed.
    """
    _onboard(page)
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.locator("article.card").first.wait_for(timeout=FEED_TIMEOUT_MS)

    before = page.locator("article.card").count()
    assert before > 0

    if page.locator(".feed-more").count() == 0:
        pytest.skip("feed had only one page for this user")

    page.mouse.wheel(0, 40_000)
    # The observer uses an 800px rootMargin, so this should fire before the
    # loader is even fully on screen.
    page.wait_for_function(
        f"document.querySelectorAll('article.card').length > {before}",
        timeout=SWAP_TIMEOUT_MS,
    )
    after = page.locator("article.card").count()
    assert after > before, f"scroll appended nothing ({before} -> {after})"


def test_feed_has_no_duplicate_papers_across_pages(page):
    """Paging must not repeat a paper the user has already been shown."""
    _onboard(page)
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.locator("article.card").first.wait_for(timeout=FEED_TIMEOUT_MS)

    for _ in range(3):
        if page.locator(".feed-more").count() == 0:
            break
        n = page.locator("article.card").count()
        page.mouse.wheel(0, 40_000)
        try:
            page.wait_for_function(
                f"document.querySelectorAll('article.card').length > {n}",
                timeout=SWAP_TIMEOUT_MS,
            )
        except Exception:
            break

    ids = page.eval_on_selector_all(
        "article.card", "els => els.map(e => e.dataset.arxivId)"
    )
    assert len(ids) == len(set(ids)), f"duplicate papers in feed: {ids}"


# ── Dismiss + undo ───────────────────────────────────────────────────────────

def test_dismiss_shows_undo_and_undo_cancels_the_request(page):
    """Undo must CANCEL the dismissal, not compensate for it.

    This matters beyond UX: a committed dismissal is folded into the user's
    negative EWMA profile, and an EWMA is a lossy running average with no exact
    inverse — so an undone dismissal must never have been sent at all.
    """
    page.goto(f"{BASE_URL}/search?q=transformer", wait_until="domcontentloaded")
    page.locator("article.card").first.wait_for(timeout=FEED_TIMEOUT_MS)

    posts: list[str] = []
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST" else None)

    card = page.locator("article.card").first
    arxiv_id = card.get_attribute("data-arxiv-id")
    card.locator("button.btn-pass").click()

    undo = page.locator(".toast .undo")
    undo.wait_for(timeout=5_000)
    assert "Removed" in page.locator(".toast .msg").inner_text()
    undo.click()

    # Wait past the commit window to prove the POST was truly cancelled.
    page.wait_for_timeout(6_500)
    assert page.locator(f'article.card[data-arxiv-id="{arxiv_id}"]').count() == 1, \
        "undo did not restore the card"
    assert not any("not-interested" in u for u in posts), \
        f"undo failed to cancel the dismissal POST: {posts}"


def test_dismiss_commits_when_not_undone(page):
    """Left alone, a dismissal must actually be sent after the undo window."""
    page.goto(f"{BASE_URL}/search?q=transformer", wait_until="domcontentloaded")
    page.locator("article.card").first.wait_for(timeout=FEED_TIMEOUT_MS)

    posts: list[str] = []
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST" else None)

    page.locator("article.card").first.locator("button.btn-pass").click()
    page.wait_for_timeout(7_000)
    assert any("not-interested" in u for u in posts), \
        f"dismissal was never committed: {posts}"


# ── Theme, layout, console health ────────────────────────────────────────────

def test_theme_toggle_switches_and_persists(page):
    """Toggling theme must stamp the root element and survive a reload."""
    # /search rather than / — a brand-new user is redirected off the feed into
    # onboarding, which has no topbar.
    page.goto(f"{BASE_URL}/search", wait_until="domcontentloaded")
    page.wait_for_selector(".topbar", timeout=SWAP_TIMEOUT_MS)

    page.click(".theme-toggle")
    stamped = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert stamped in ("light", "dark"), f"toggle did not stamp a theme: {stamped}"

    page.reload(wait_until="domcontentloaded")
    assert page.evaluate(
        "document.documentElement.getAttribute('data-theme')") == stamped, \
        "theme did not persist across reload"


def test_text_is_legible_in_both_themes(page):
    """Body text must never resolve to the same colour as its background.

    Catches the classic broken-artifact bug where a colour token is defined only
    inside a media query, so one theme renders its text on the other's ground.
    """
    page.goto(f"{BASE_URL}/search?q=transformer", wait_until="domcontentloaded")
    page.locator("article.card").first.wait_for(timeout=FEED_TIMEOUT_MS)

    for theme in ("light", "dark"):
        page.evaluate(f"document.documentElement.setAttribute('data-theme','{theme}')")
        colours = page.evaluate(
            """() => {
                const t = document.querySelector('.card-title');
                const c = document.querySelector('article.card');
                const s = getComputedStyle(t), cs = getComputedStyle(c);
                return {text: s.color, cardBg: cs.backgroundColor,
                        bodyBg: getComputedStyle(document.body).backgroundColor};
            }"""
        )
        assert colours["text"] != colours["cardBg"], \
            f"{theme}: title colour equals card background ({colours})"
        assert colours["bodyBg"] not in ("rgba(0, 0, 0, 0)", "transparent"), \
            f"{theme}: body background is transparent ({colours})"


def test_no_console_errors_on_feed(page):
    """A clean console. Both shipped bugs were silent, but this is cheap."""
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    _onboard(page)
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.locator("article.card").first.wait_for(timeout=FEED_TIMEOUT_MS)
    page.mouse.wheel(0, 40_000)
    page.wait_for_timeout(4_000)

    # Ignore network noise from third-party/offline resources; we care about JS.
    real = [e for e in errors if "net::" not in e and "favicon" not in e]
    assert real == [], f"console errors: {real}"


def test_no_horizontal_overflow_on_mobile(page, browser):
    """The page body must never scroll sideways on a phone viewport."""
    ctx = browser.new_context(viewport={"width": 390, "height": 844},
                              is_mobile=True, has_touch=True)
    ctx.add_cookies([{"name": "arxiv_user_id",
                      "value": f"e2e-{uuid.uuid4()}", "url": BASE_URL}])
    p = ctx.new_page()
    try:
        p.goto(f"{BASE_URL}/search?q=transformer", wait_until="domcontentloaded")
        p.locator("article.card").first.wait_for(timeout=FEED_TIMEOUT_MS)
        overflow = p.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 1, f"page scrolls horizontally by {overflow}px"

        # Bottom nav must not cover the last card's controls.
        assert p.locator(".botnav").is_visible()
    finally:
        ctx.close()
