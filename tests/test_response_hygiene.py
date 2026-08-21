"""
Baseline response hygiene: compression, caching, security headers, HTML 404,
Open Graph, and token contrast.

All of these were flagged by a site audit against the live deployment. They are
small individually; together they are the difference between a site that looks
maintained and one that does not.
"""
import re

from fastapi.testclient import TestClient

from app.main import app


def _client() -> TestClient:
    return TestClient(app)


# ── Compression ──────────────────────────────────────────────────────────────

def test_static_css_is_compressed():
    """117KB of CSS+JS was shipped uncompressed on every load; CSS alone
    compresses 72%."""
    with _client() as c:
        r = c.get("/static/styles.css", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"


def test_static_js_is_compressed():
    with _client() as c:
        r = c.get("/static/htmx.min.js", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"


def test_a_client_that_cannot_gunzip_still_gets_the_asset():
    with _client() as c:
        r = c.get("/static/styles.css", headers={"Accept-Encoding": "identity"})
    assert r.status_code == 200
    assert "content-encoding" not in r.headers
    assert len(r.content) > 1000


# ── Caching ──────────────────────────────────────────────────────────────────

def test_static_assets_are_cacheable():
    with _client() as c:
        r = c.get("/static/app.js")
    cc = r.headers.get("cache-control", "")
    assert "max-age" in cc, f"no max-age on a static asset: {cc!r}"


def test_static_cache_is_revalidated_not_immutable():
    """Filenames are stable across deploys, so `immutable` would pin a stale
    stylesheet in returning visitors' caches until a hard reload."""
    with _client() as c:
        r = c.get("/static/styles.css")
    assert "immutable" not in r.headers.get("cache-control", "")


# ── Security headers ─────────────────────────────────────────────────────────

def test_nosniff_is_set():
    with _client() as c:
        r = c.get("/static/styles.css")
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_referrer_policy_protects_search_queries():
    """Every card links out to arxiv.org; without this the full referring URL
    — including the user's query — travels with the click."""
    with _client() as c:
        r = c.get("/search?q=test")
    assert r.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


# ── HTML 404 ─────────────────────────────────────────────────────────────────

def test_unknown_page_returns_html_not_raw_json():
    with _client() as c:
        r = c.get("/nonexistent-page")
    assert r.status_code == 404
    assert "text/html" in r.headers.get("content-type", "")
    assert '{"detail"' not in r.text
    assert "/search" in r.text, "404 page offers no way back"


def test_unknown_api_route_still_returns_json():
    """htmx and the map client parse JSON; content negotiation is by path
    because htmx sends `Accept: */*`."""
    with _client() as c:
        r = c.get("/api/nonexistent")
    assert r.status_code == 404
    assert "application/json" in r.headers.get("content-type", "")


# ── Open Graph ───────────────────────────────────────────────────────────────

def test_pages_carry_open_graph_tags():
    """Without these a shared link renders as a naked URL everywhere."""
    with _client() as c:
        r = c.get("/search")
    for prop in ("og:title", "og:description", "og:type", "og:site_name"):
        assert f'property="{prop}"' in r.text, f"missing {prop}"
    assert 'name="twitter:card"' in r.text


def test_og_title_is_not_empty():
    with _client() as c:
        r = c.get("/search")
    m = re.search(r'<meta property="og:title" content="([^"]*)"', r.text)
    assert m and m.group(1).strip(), "og:title rendered empty"


# ── Contrast ─────────────────────────────────────────────────────────────────

def _relative_luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(fg: str, bg: str) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def test_secondary_text_passes_aa_in_both_themes():
    """--text-3 carries the card's provenance line at 0.7rem, so the
    large-text exemption does not apply. It failed in BOTH themes: light
    4.49:1 on a white card, dark 4.28:1 on --surface."""
    import pathlib
    css = pathlib.Path("app/static/styles.css").read_text()
    values = re.findall(r"--text-3:\s*(#[0-9A-Fa-f]{6})", css)
    assert len(values) >= 2, "expected a light and at least one dark definition"

    light, darks = values[0], values[1:]
    assert _contrast(light, "#FFFFFF") >= 4.5, (
        f"light --text-3 {light} on a white card is "
        f"{_contrast(light, '#FFFFFF'):.2f}:1")
    for d in darks:
        assert _contrast(d, "#1B1815") >= 4.5, (
            f"dark --text-3 {d} on --surface is {_contrast(d, '#1B1815'):.2f}:1")


def test_dark_theme_definitions_agree():
    """The media-query and [data-theme=dark] blocks must not drift apart."""
    import pathlib
    css = pathlib.Path("app/static/styles.css").read_text()
    darks = re.findall(r"--text-3:\s*(#[0-9A-Fa-f]{6})", css)[1:]
    assert len(set(darks)) == 1, f"dark --text-3 defined inconsistently: {darks}"


# ── Mobile ───────────────────────────────────────────────────────────────────

def test_theme_toggle_is_reachable_on_a_phone():
    """`.nav` is display:none below 720px and the toggle used to live inside it.

    So a phone had no way to change theme at all — on a product whose whole
    premise is reading on a phone. The bottom nav is a four-item grid and a
    fifth would unbalance it, so the control lives in the top bar instead.
    """
    import pathlib
    html = pathlib.Path("app/templates/base.html").read_text()

    nav = html[html.index('<nav class="nav"'):html.index("</nav>")]
    assert "theme-toggle" not in nav, (
        "theme toggle is inside .nav, which is hidden below 720px")

    topbar = html[html.index('<header class="topbar">'):html.index("</header>")]
    assert "theme-toggle" in topbar, "theme toggle left the top bar entirely"


def test_the_toggle_is_pinned_right_on_mobile():
    """.nav carries margin-left:auto; with it hidden, the toggle needs its own."""
    import pathlib
    css = pathlib.Path("app/static/styles.css").read_text()
    assert ".topbar .theme-toggle" in css
