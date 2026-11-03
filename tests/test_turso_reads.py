"""Turso reads that must stay cheap (Turso's free plan allows 500M rows read a month,
and it blocked every read on 2026-10-24 once that was gone).

What must hold: the figure warm-up asks for the newest rows by rowid, never a scan
of update_date; the starter feed's Turso fallback (a ~2M-row LIKE scan) runs only on
the Space or when TURSO_SCAN_FALLBACK=on; and /healthz/deep reports Turso's in-band
errors such as BLOCKED instead of "ok".
"""
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app import config, figure_prewarm, local_meta, turso_sync, turso_svc


async def test_figure_warm_up_reads_only_the_newest_rows(monkeypatch):
    monkeypatch.setattr(config, "TURSO_URL", "https://example.turso.io")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "t")
    sent = []

    async def execute(stmts, timeout=60):
        sent.extend(s["sql"] for s in stmts)
        return [[["2610.00001", "CC-BY-4.0"]]]

    monkeypatch.setattr(turso_sync, "_execute", execute)
    assert await figure_prewarm.recent_ids() == [("2610.00001", "CC-BY-4.0")]
    sql = sent[0]
    assert "p.rowid > (SELECT MAX(rowid) FROM papers) - 5000" in sql and "update_date >=" in sql


async def test_starter_feed_never_scans_turso_off_the_space(monkeypatch):
    monkeypatch.setattr(config, "TURSO_URL", "https://example.turso.io")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "t")
    monkeypatch.setattr(local_meta, "is_available", lambda: False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("TURSO_SCAN_FALLBACK", raising=False)
    turso_svc._TRENDING_CACHE.clear()
    post = AsyncMock(side_effect=AssertionError("Turso must not be called"))
    with patch.object(httpx.AsyncClient, "post", post):
        assert await turso_svc.fetch_trending_by_categories({"cs.LG"}, limit=5) == []
    assert not post.called


@pytest.mark.parametrize("env, allowed", [({"SPACE_ID": "siddhm11/ResearchIT"}, True),
                                          ({"TURSO_SCAN_FALLBACK": "on"}, True),
                                          ({"SPACE_ID": "x", "TURSO_SCAN_FALLBACK": "off"}, False),
                                          ({}, False)])
def test_scan_fallback_runs_only_on_the_space_or_when_asked(monkeypatch, env, allowed):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("TURSO_SCAN_FALLBACK", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert turso_svc._scan_fallback_allowed() is allowed


async def test_health_reports_a_blocked_turso(monkeypatch):
    from app.main import app
    monkeypatch.setattr(config, "TURSO_URL", "https://example.turso.io")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "t")
    real_post = httpx.AsyncClient.post

    async def post(self, url, *a, **kw):
        if "example.turso.io" in str(url):
            return httpx.Response(200, request=httpx.Request("POST", url), json={"results": [
                {"type": "error", "error": {"message": "Operation was blocked: SQL read operations are forbidden",
                                            "code": "BLOCKED"}}, {"type": "ok"}]})
        return await real_post(self, url, *a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        data = (await c.get("/healthz/deep")).json()
    turso = data["services"]["turso"]
    assert turso["status"] == "error" and turso["error"].startswith("BLOCKED")
