"""
Dated daily pages (/day/<yyyy-mm-dd>, app/daily_pages.py, app/routers/daily.py):
one day's Hugging Face Daily Papers list, most upvoted first, from lists already
in memory; prev/next days; a share card; exposures for people, none for bots.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app import daily_pages, trending_shelf, trending_topics

UA = {"user-agent": "Mozilla/5.0 (Macintosh) Chrome/131 Safari/537.36"}


@pytest.fixture
def days(monkeypatch, tmp_path):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "t.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    asyncio.run(db_mod.init_db())
    monkeypatch.setenv(trending_topics.SWITCH, "on")
    monkeypatch.setitem(trending_topics._state, "days", {
        "2026-11-20": [{"arxiv_id": "2611.00001", "title": "Old day paper", "summary": "", "upvotes": 5,
                        "github": "", "stars": 0}],
        "2026-11-24": [{"arxiv_id": "2611.00002", "title": "Less upvoted", "summary": "S.", "upvotes": 3,
                        "github": "", "stars": 0},
                       {"arxiv_id": "2611.00003", "title": "Stale upvotes", "summary": "", "upvotes": 1,
                        "github": "", "stars": 0}]})
    # The shelf's fresher count for the same paper wins.
    monkeypatch.setitem(trending_shelf._state, "papers", {
        "2611.00003": {"arxiv_id": "2611.00003", "title": "Stale upvotes", "listed_on": "2026-11-24",
                       "upvotes": 40, "summary": ""}})
    monkeypatch.setitem(trending_shelf._state, "code", {"2611.00003": ("org/repo", 12)})


def test_a_days_list_is_most_upvoted_first_with_fresh_counts(days):
    got = daily_pages.papers("2026-11-24")
    assert [p["arxiv_id"] for p in got] == ["2611.00003", "2611.00002"]
    assert got[0]["upvotes"] == 40 and got[0]["github"] == "org/repo"
    assert daily_pages.available()[0] == "2026-11-24"
    assert daily_pages.neighbours("2026-11-24") == ("2026-11-20", None)


def test_day_pages(days):
    from app.main import app
    with TestClient(app) as c:
        r = c.get("/day", headers=UA, follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == "/day/2026-11-24"
        page = c.get("/day/2026-11-24", headers=UA)
        assert page.status_code == 200
        assert "Tuesday 24 November 2026" in page.text
        assert page.text.index("Stale upvotes") < page.text.index("Less upvoted")
        assert "sf=daily_page" in page.text and "pol=daily_page_v1" in page.text
        assert 'href="/day/2026-11-20"' in page.text
        assert "/og/day/2026-11-24.png" in page.text
        assert c.get("/day/2026-11-25", headers=UA).status_code == 404
        assert c.get("/day/nonsense", headers=UA).status_code == 404
        card = c.get("/og/day/2026-11-24.png")
        assert card.status_code == 200 and card.headers["content-type"] == "image/png"
        bot = c.get("/day/2026-11-24", headers={"user-agent": "Googlebot/2.1"})
        assert bot.status_code == 200 and "set-cookie" not in bot.headers
