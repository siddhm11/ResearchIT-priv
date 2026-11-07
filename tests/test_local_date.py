"""The feed masthead shows the reader's date, not the server's (audit F12).

The server writes the date in UTC, so a reader in India saw yesterday's date
until 05:30. The page now carries the exact time, and app.js rewrites it in
the browser's own time zone.
"""
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "app" / "static" / "app.js"
# 2026-11-08 20:30 UTC: already 9 November in India, still 8 November in New York.
STAMP = "2026-11-08T20:30:00+00:00"


def test_the_masthead_carries_the_exact_time(tmp_path, monkeypatch):
    import asyncio
    import re

    from fastapi.testclient import TestClient

    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    from app.config import COOKIE_NAME
    from app.main import app
    asyncio.run(db_mod.init_db())
    asyncio.run(db_mod.complete_onboarding("reader"))
    with TestClient(app, cookies={COOKIE_NAME: "reader"}) as c:
        html = c.get("/").text
    stamps = re.findall(r'<time class="local-date" datetime="([^"]+)">', html)
    assert stamps and all(s.endswith("+00:00") for s in stamps)


@pytest.mark.browser
@pytest.mark.parametrize("zone, expected", [
    ("Asia/Kolkata", "Monday 9 November 2026"),
    ("America/New_York", "Sunday 8 November 2026"),
])
def test_the_browser_shows_its_own_date(zone, expected):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_context(timezone_id=zone).new_page()
            page.set_content(f'<body><time class="local-date" datetime="{STAMP}">'
                             'Sunday 8 November 2026</time></body>')
            page.add_script_tag(content=APP_JS.read_text())
            assert page.text_content("time") == expected
        finally:
            browser.close()
