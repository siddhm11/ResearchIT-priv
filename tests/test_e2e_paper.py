"""
Browser tests for the redesigned paper page, against the seeded test world.

RUNNING
-------
Build the world and start a server on it (see scripts/seed_e2e.py for the
full command, which blanks every outside service):

    .venv/bin/python scripts/seed_e2e.py --db /tmp/e2e.db --site /tmp/e2e_arxiv \\
        --sidecar /tmp/e2e_sidecar.sqlite
    .venv/bin/python -m http.server 8799 --directory /tmp/e2e_arxiv &
    <the server command from seed_e2e.py> &
    E2E_BASE_URL=http://127.0.0.1:7861 .venv/bin/python -m pytest tests/test_e2e_paper.py -v

Skips itself when playwright is missing, no server answers, or the server was
not started on the seeded world with PAPER_PAGE_V2 and FIGURES on.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
import uuid

import pytest

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:7861").rstrip("/")
SATNAV, MATHS, NO_SUMMARY, LEAKSCALE = "2609.31507", "2609.99001", "2609.99002", "2609.27176"


def _redesign_world() -> bool:
    try:
        html = urllib.request.urlopen(f"{BASE_URL}/p/{SATNAV}", timeout=15).read().decode()
    except (urllib.error.URLError, OSError):
        return False
    return "data-sumbox" in html and "/fig/" in html


pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(not _redesign_world(),
                       reason=f"no seeded redesign server at {BASE_URL} (see scripts/seed_e2e.py)"),
]


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def open_page(browser, path, width=1440, height=900, theme="light"):
    ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=theme)
    ctx.add_cookies([{"name": "arxiv_user_id", "value": f"e2e-{uuid.uuid4()}", "url": BASE_URL}])
    page = ctx.new_page()
    page.errors = []
    page.on("console", lambda m: page.errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    page.goto(BASE_URL + path, wait_until="networkidle")
    return ctx, page


@pytest.fixture
def satnav(browser):
    ctx, page = open_page(browser, f"/p/{SATNAV}")
    yield page
    ctx.close()


def test_the_plain_summary_shows_first_and_the_abstract_is_one_keypress_away(satnav):
    satnav.wait_for_selector("#panel-plain .tx-body")
    assert "satellite photos" in satnav.inner_text("#panel-plain")
    satnav.focus("#tab-plain")
    satnav.keyboard.press("ArrowRight")
    assert satnav.evaluate("document.activeElement.id") == "tab-abs"
    assert satnav.is_visible("#panel-abs") and not satnav.is_visible("#panel-plain")
    satnav.keyboard.press("Home")
    assert satnav.is_visible("#panel-plain")


def test_the_tab_choice_is_remembered(browser):
    ctx, page = open_page(browser, f"/p/{SATNAV}")
    page.click("#tab-abs")
    page.goto(f"{BASE_URL}/p/{LEAKSCALE}", wait_until="networkidle")
    assert page.is_visible("#panel-abs")
    ctx.close()


def test_figures_load_from_our_own_route(satnav):
    sizes = satnav.evaluate("[...document.querySelectorAll('img[src^=\"/fig/\"]')].map(i => i.naturalWidth)")
    assert len(sizes) >= 2 and all(w > 0 for w in sizes)
    assert satnav.evaluate("[...document.images].every(i => !i.src.startsWith('https://arxiv.org'))")


def test_the_lightbox_opens_closes_and_returns_focus(satnav):
    satnav.click("#sec-fig [data-fig]")
    satnav.wait_for_function("document.getElementById('lb').open")
    assert "Figure 1" in satnav.inner_text("#lb-cap")
    assert "CC BY 4.0" in satnav.inner_text("#lb-credit")
    satnav.keyboard.press("Escape")
    satnav.wait_for_function("!document.getElementById('lb').open")
    assert satnav.evaluate("document.activeElement.closest('#sec-fig') !== null")


def test_the_overflow_menu_works_by_keyboard(satnav):
    satnav.click(".inline-acts [data-menu]")
    assert satnav.is_visible(".inline-acts .menu")
    assert satnav.evaluate("document.activeElement.closest('.menu') !== null")
    satnav.keyboard.press("Escape")
    assert not satnav.is_visible(".inline-acts .menu")
    assert satnav.evaluate("document.activeElement.hasAttribute('data-menu')")


def test_the_outline_marks_the_section_in_view(satnav):
    def active():
        return satnav.evaluate("[...document.querySelectorAll('.outline nav a.is-active')].map(a => a.dataset.sec)")
    satnav.evaluate("document.getElementById('sec-fig').scrollIntoView()")
    satnav.wait_for_timeout(400)
    assert active() == ["sec-fig"]
    # At the very bottom the last section is the one in view, however short.
    satnav.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
    satnav.wait_for_timeout(400)
    last = satnav.evaluate("[...document.querySelectorAll('.outline nav a')].filter(a => !a.hidden).at(-1).dataset.sec")
    assert active() == [last]


def test_saving_from_the_new_page_swaps_the_button(satnav):
    satnav.click(".inline-acts .btn-save")
    satnav.wait_for_selector(".inline-acts .btn-saved", timeout=15_000)


def test_no_console_errors_in_either_theme(browser):
    for theme in ("light", "dark"):
        ctx, page = open_page(browser, f"/p/{SATNAV}", theme=theme)
        page.wait_for_timeout(500)
        assert page.errors == [], page.errors
        ctx.close()


def test_layout_does_not_jump_while_figures_load(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.add_init_script("""
      window.__cls = 0;
      new PerformanceObserver(l => { for (const e of l.getEntries()) if (!e.hadRecentInput) window.__cls += e.value; })
        .observe({type: 'layout-shift', buffered: true});
    """)
    page.goto(f"{BASE_URL}/p/{SATNAV}", wait_until="networkidle")
    page.wait_for_timeout(800)
    assert page.evaluate("window.__cls") < 0.1
    ctx.close()


@pytest.mark.parametrize("width", [390, 900, 1180, 1440])
def test_no_sideways_scrolling_at_any_width(browser, width):
    ctx, page = open_page(browser, f"/p/{SATNAV}", width=width)
    assert not page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
    outline_shown = page.is_visible(".outline")
    assert outline_shown == (width >= 1180)
    ctx.close()


def test_maths_renders_and_a_markup_payload_stays_text(browser):
    ctx, page = open_page(browser, f"/p/{MATHS}")
    page.click("#tab-abs")
    page.wait_for_selector("#panel-abs .katex")
    assert page.locator("#panel-abs .katex-display").count() == 1      # the $$...$$ block
    assert page.locator("h1 .katex").count() == 1                      # maths in the title
    assert page.evaluate("window.__xss") is None
    assert page.locator("#panel-abs img").count() == 0
    ctx.close()


def test_a_paper_with_no_plain_summary_opens_on_its_abstract(browser):
    ctx, page = open_page(browser, f"/p/{NO_SUMMARY}")
    page.wait_for_function("document.getElementById('tab-plain').hidden")
    assert page.is_visible("#panel-abs")
    assert "stands alone" in page.inner_text("#panel-abs")
    ctx.close()


def test_a_paper_without_arxiv_html_shows_no_figure_sections(browser):
    ctx, page = open_page(browser, f"/p/{MATHS}")
    assert page.locator("#sec-fig").count() == 0 and page.locator("#sec-figs").count() == 0
    ctx.close()


# ── briefing and Ask (server started with BRIEFING=on ASK=on) ────────────────

def _needs(page, selector, switch):
    if page.locator(selector).count() == 0:
        pytest.skip(f"server not started with {switch}=on")


def test_the_briefing_arrives_with_links_to_each_passage(satnav):
    _needs(satnav, ".brief-slot, #sec-things", "BRIEFING")
    satnav.wait_for_selector("#sec-things li")
    assert satnav.locator("#sec-things li").count() == 3
    href = satnav.get_attribute("#sec-things a.src", "href")
    assert href == f"https://arxiv.org/html/{SATNAV}v1#abstract"
    assert satnav.locator("#sec-how li").count() == 3 and satnav.is_visible("#sec-use")
    # The outline's briefing links appear once the sections exist.
    satnav.wait_for_function("!document.querySelector('.outline a[data-sec=\"sec-things\"]').hidden")


def test_a_suggested_question_is_echoed_then_answered_with_a_cite(satnav):
    _needs(satnav, "#chat", "ASK")
    satnav.click('[data-ask-key="limits"]')
    satnav.wait_for_selector("#chat-log .msg-user")
    assert satnav.inner_text("#chat-log .msg-user") == "What are the limitations?"
    satnav.wait_for_selector("#chat-log .cite")
    assert "three limitations" in satnav.inner_text("#chat-log")
    assert satnav.get_attribute("#chat-log .cite", "href") == f"https://arxiv.org/html/{SATNAV}v1#S6.p1"
    assert satnav.locator('[data-ask-key="limits"]').count() == 0      # the chip is used up


def test_a_typed_question_sends_on_enter_and_a_missing_model_says_busy(satnav):
    _needs(satnav, "#chat", "ASK")
    satnav.fill("#chat-input", "How many cities are there?")
    satnav.press("#chat-input", "Enter")
    satnav.wait_for_selector("#chat-log .msg-bot >> nth=1")
    assert "busy right now" in satnav.inner_text("#chat-log")
    assert satnav.input_value("#chat-input") == ""


def test_shift_enter_starts_a_new_line_instead_of_sending(satnav):
    _needs(satnav, "#chat", "ASK")
    satnav.fill("#chat-input", "First line")
    satnav.press("#chat-input", "Shift+Enter")
    satnav.wait_for_timeout(300)
    assert satnav.locator("#chat-log .msg-user").count() == 0


def test_on_a_phone_ask_opens_as_a_sheet_and_closes_on_escape(browser):
    ctx, page = open_page(browser, f"/p/{SATNAV}", width=390, height=844)
    _needs(page, "#chat", "ASK")
    assert not page.is_visible("#chat") and page.is_visible(".ask-fab")
    page.click(".ask-fab")
    assert page.is_visible("#chat") and page.evaluate("document.activeElement.id") == "chat-input"
    page.keyboard.press("Escape")
    assert not page.is_visible("#chat")
    assert page.evaluate("document.activeElement.classList.contains('ask-fab')")
    assert page.errors == []
    ctx.close()


def test_ask_about_this_figure_asks_in_the_panel(satnav):
    _needs(satnav, "#chat", "ASK")
    satnav.click("#sec-fig [data-fig]")
    satnav.wait_for_function("document.getElementById('lb').open")
    satnav.click("#lb-ask")
    satnav.wait_for_selector("#chat-log .msg-user")
    assert satnav.inner_text("#chat-log .msg-user") == "Explain figure 1"
