"""
/healthz/deep reports whether Groq still serves the configured LLM models.

The keepalive workflow fails when this says "error", which turns the next
model retirement into a GitHub notification instead of weeks of silently
missing rewrites, overviews and explanations. No network: the client is fake.
"""
import asyncio
from types import SimpleNamespace

import pytest

from app import config, groq_svc, local_meta, map_locate_svc, qdrant_svc, zilliz_svc
from app.routers import health

CHAIN = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def _client(served: list[str] | Exception):
    def list_models(timeout=None):
        if isinstance(served, Exception):
            raise served
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in served])
    return SimpleNamespace(models=SimpleNamespace(list=list_models))


@pytest.fixture(autouse=True)
def _chain(monkeypatch):
    monkeypatch.setattr(config, "GROQ_MODELS", list(CHAIN))
    groq_svc._benched.clear()
    yield
    groq_svc._benched.clear()


def test_all_models_served(monkeypatch):
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _client(CHAIN + ["other"]))
    r = groq_svc.probe_models()
    assert r["status"] == "ok" and r["primary_available"] is True
    assert r["active"] == CHAIN[0] and r["benched"] == {}


def test_retired_primary_is_ok_but_flagged_and_benched(monkeypatch):
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _client([CHAIN[1]]))
    r = groq_svc.probe_models()
    assert r["status"] == "ok"                  # the fallback is serving
    assert r["primary_available"] is False      # ...but someone should know
    assert r["active"] == CHAIN[1]
    assert CHAIN[0] in r["benched"]
    # Requests skip it without paying a 404 first.
    assert groq_svc._ready_models() == [CHAIN[1]]


def test_no_configured_model_served_is_an_error(monkeypatch):
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _client(["llama-3.3-70b-versatile"]))
    r = groq_svc.probe_models()
    assert r["status"] == "error" and r["active"] is None


def test_groq_unreachable_is_an_error(monkeypatch):
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _client(ConnectionError("down")))
    assert groq_svc.probe_models()["status"] == "error"


def test_no_key_is_skipped(monkeypatch):
    monkeypatch.setattr(groq_svc, "_get_client", lambda: None)
    assert groq_svc.probe_models()["status"] == "skipped"


def _healthy_everything_else(monkeypatch):
    monkeypatch.setattr(config, "map_store_is_dedicated", lambda: True)
    monkeypatch.setattr(config, "qdrant_recent_configured", lambda: False)
    monkeypatch.setattr(config, "TURSO_URL", "")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "")
    monkeypatch.setattr(qdrant_svc, "_client", lambda: SimpleNamespace(
        get_collection=lambda name: SimpleNamespace(points_count=100)))
    monkeypatch.setattr(zilliz_svc, "_get_client", lambda: SimpleNamespace(
        list_collections=lambda: ["papers"]))
    monkeypatch.setattr(local_meta, "stats", lambda: {"available": True})

    async def map_stats():
        return {"collection": "m", "points": 1, "status": "green", "sample_valid": True}
    monkeypatch.setattr(map_locate_svc, "collection_stats", map_stats)


@pytest.mark.parametrize(("served", "overall"), [
    (CHAIN, "healthy"),
    ([CHAIN[1]], "healthy"),
    (["llama-3.3-70b-versatile"], "degraded"),
])
def test_deep_health_includes_llm(monkeypatch, served, overall):
    _healthy_everything_else(monkeypatch)
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _client(served))
    result = asyncio.run(health.healthz_deep())
    assert result["services"]["llm"]["configured"] == CHAIN
    assert result["overall"] == overall
