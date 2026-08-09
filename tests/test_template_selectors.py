"""
Static guard against htmx selector bugs that only manifest in a browser.

Background
----------
htmx passes hx-target / hx-select / hx-include values straight to
document.querySelectorAll with NO escaping. An arXiv id contains a dot, and in
a CSS selector a dot starts a class token, so

    #actions-1706.03762   parses as  id="actions-1706" AND class="03762"

and because a class token may not begin with a digit, querySelector *raises*
rather than merely failing to match. The result is a control that silently does
nothing while the server still returns 200 and writes its row.

This is a known, unfixed defect in the 1.x line — htmx issues #1537 and #2601.
Maintainer position is that escaping is the application's responsibility;
CSS.escape was rejected for 1.x over IE11 compatibility and only landed in
2.0.5+. This project is pinned to htmx 1.9.12, so the fix will never arrive
upstream and these rules have to be enforced here.

The sanctioned workaround is the attribute form, [id='…'], which matches any id
regardless of its characters.

These are plain string checks over the templates: no server, no browser, fast
enough to run on every commit. tests/test_e2e_browser.py covers the same class
dynamically, but needs a live server and a real Chromium.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"

# htmx attributes whose value is fed to querySelectorAll.
SELECTOR_ATTRS = (
    "hx-target", "hx-select", "hx-select-oob",
    "hx-include", "hx-indicator", "hx-disabled-elt",
)

# htmx's extended, non-CSS selector syntax — exempt from the id rules.
_EXTENDED = re.compile(r"^\s*(this\b|closest\s|find\s|next\b|previous\b|body\b|window\b|document\b)")


_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _template_files() -> list[Path]:
    return sorted(TEMPLATES.rglob("*.html"))


def _live_markup(path: Path) -> str:
    """Template text with comments stripped.

    These rules describe attributes the browser actually sees. Comments in
    this codebase deliberately quote the broken forms to explain why they are
    banned, so scanning raw text flags its own documentation.
    """
    text = path.read_text()
    text = _JINJA_COMMENT.sub("", text)
    return _HTML_COMMENT.sub("", text)


def _selector_attrs(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r'(' + "|".join(SELECTOR_ATTRS) + r')\s*=\s*"([^"]*)"'
    )
    return pattern.findall(text)


def test_no_id_selector_interpolates_a_template_variable():
    """`hx-target="#thing-{{ var }}"` is banned outright.

    Any Jinja value can contain a dot, colon or leading digit; each of those
    breaks or throws inside a `#id` selector. Use `[id='thing-{{ var }}']`.
    """
    offenders: list[str] = []
    for path in _template_files():
        for attr, value in _selector_attrs(_live_markup(path)):
            if _EXTENDED.match(value):
                continue
            if value.lstrip().startswith("#") and "{{" in value:
                offenders.append(f"{path.name}: {attr}=\"{value}\"")

    assert offenders == [], (
        "id selectors built from template variables must use the attribute "
        "form [id='…'] — htmx 1.x does not escape them (issues #1537, #2601):\n  "
        + "\n  ".join(offenders)
    )


def test_static_id_selectors_are_valid_css():
    """Hard-coded `#foo` targets must be selectors a browser accepts."""
    bad: list[str] = []
    valid = re.compile(r"^#[A-Za-z_-][A-Za-z0-9_-]*$")
    for path in _template_files():
        for attr, value in _selector_attrs(_live_markup(path)):
            v = value.strip()
            if _EXTENDED.match(v) or "{{" in v or not v.startswith("#"):
                continue
            if not valid.match(v):
                bad.append(f"{path.name}: {attr}=\"{v}\"")

    assert bad == [], "invalid static id selectors:\n  " + "\n  ".join(bad)


def test_attribute_form_targets_are_quoted():
    """`[id=…]` must quote its value, or a dot/digit still breaks parsing."""
    bad: list[str] = []
    for path in _template_files():
        for attr, value in _selector_attrs(_live_markup(path)):
            v = value.strip()
            if not v.startswith("[id="):
                continue
            if not re.match(r"^\[id=['\"].*['\"]\]$", v):
                bad.append(f"{path.name}: {attr}=\"{v}\"")

    assert bad == [], "unquoted attribute selectors:\n  " + "\n  ".join(bad)


def test_revealed_trigger_is_never_combined_with_another_event():
    """`hx-trigger="revealed, …"` can never fire.

    htmx's reveal poller collects elements with an exact attribute match:

        querySelectorAll("[hx-trigger='revealed'],[data-hx-trigger='revealed']")

    so any combined spec is invisible to it. htmx's own infinite-scroll recipe
    uses a bare `revealed` and never combines it. This project drives scrolling
    from its own IntersectionObserver in app.js instead — see rec_page.html.
    """
    offenders: list[str] = []
    for path in _template_files():
        for m in re.finditer(r'hx-trigger\s*=\s*"([^"]*)"', _live_markup(path)):
            spec = m.group(1)
            if "revealed" in spec and "," in spec:
                offenders.append(f"{path.name}: hx-trigger=\"{spec}\"")

    assert offenders == [], (
        "`revealed` cannot be combined with another trigger:\n  "
        + "\n  ".join(offenders)
    )


def test_every_hx_target_id_has_a_matching_element_somewhere():
    """Catch targets pointing at ids no template ever renders.

    Only checks static ids — variable ones are resolved at request time and are
    covered dynamically by the browser suite.
    """
    all_text = "\n".join(_live_markup(p) for p in _template_files())
    rendered_ids = set(re.findall(r'id="([A-Za-z_-][A-Za-z0-9_-]*)"', all_text))

    missing: list[str] = []
    for path in _template_files():
        for attr, value in _selector_attrs(_live_markup(path)):
            v = value.strip()
            if _EXTENDED.match(v) or "{{" in v or not v.startswith("#"):
                continue
            if v[1:] not in rendered_ids:
                missing.append(f"{path.name}: {attr}=\"{v}\"")

    assert missing == [], (
        "hx-target points at an id no template renders:\n  " + "\n  ".join(missing)
    )
