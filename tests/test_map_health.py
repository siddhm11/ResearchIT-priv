"""The keepalive must expose a failed map store even when search is healthy."""

import asyncio
from types import SimpleNamespace

import pytest

from app import config, local_meta, map_locate_svc, qdrant_svc, zilliz_svc
from app.routers import health


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"collection": "arxiv_map_positions", "points": 100, "status": "green", "sample_valid": True}, "healthy"),
        ({"collection": "arxiv_map_positions", "points": 0, "status": "green", "sample_valid": False}, "degraded"),
        ({"collection": "arxiv_map_positions", "points": 100, "status": "red", "sample_valid": True}, "degraded"),
        ({"collection": "arxiv_map_positions", "points": 100, "status": "green", "sample_valid": False}, "degraded"),
    ],
)
def test_deep_health_reports_map_collection_failure(monkeypatch, stats, expected):
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
        return stats

    monkeypatch.setattr(map_locate_svc, "collection_stats", map_stats)
    result = asyncio.run(health.healthz_deep())

    assert result["overall"] == expected
    assert result["services"]["map_positions"]["points_count"] == stats["points"]
    assert result["services"]["map_positions"]["status"] == (
        "ok" if expected == "healthy" else "error"
    )


@pytest.mark.parametrize(
    ("stats", "expected_status"),
    [
        ({"collection": "arxiv_map_positions", "points": 100, "status": "green", "sample_valid": True}, 200),
        ({"collection": "arxiv_map_positions", "points": 0, "status": "green", "sample_valid": False}, 503),
        ({"collection": "arxiv_map_positions", "points": 100, "status": "red", "sample_valid": True}, 503),
        ({"collection": "arxiv_map_positions", "points": 100, "status": "green", "sample_valid": False}, 503),
    ],
)
def test_map_api_health_reflects_collection_state(monkeypatch, stats, expected_status):
    from fastapi.testclient import TestClient
    from app.main import app

    async def map_stats():
        return stats

    monkeypatch.setattr(map_locate_svc, "collection_stats", map_stats)
    response = TestClient(app).get("/api/space/locate/health")
    assert response.status_code == expected_status
    assert response.json() == stats


def test_map_api_health_returns_503_on_connection_failure(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    async def map_stats():
        raise ConnectionError("map collection unavailable")

    monkeypatch.setattr(map_locate_svc, "collection_stats", map_stats)
    response = TestClient(app).get("/api/space/locate/health")
    assert response.status_code == 503
    assert response.json()["collection"] == map_locate_svc.MAP_COLLECTION


def test_collection_stats_reads_and_validates_a_map_point(monkeypatch):
    calls = []

    def scroll(**kwargs):
        calls.append(kwargs)
        return ([SimpleNamespace(
            payload={"arxiv_id": "1706.03762"}, vector=[1.0, 2.0, 3.0],
        )], None)

    client = SimpleNamespace(
        get_collection=lambda name: SimpleNamespace(points_count=100, status="green"),
        scroll=scroll,
    )
    monkeypatch.setattr(map_locate_svc, "_map_client", lambda: client)
    stats = asyncio.run(map_locate_svc.collection_stats())

    assert stats["sample_valid"] is True
    assert calls == [{
        "collection_name": map_locate_svc.MAP_COLLECTION,
        "limit": 1,
        "with_payload": ["arxiv_id"],
        "with_vectors": True,
    }]


def test_collection_stats_rejects_non_3d_data(monkeypatch):
    client = SimpleNamespace(
        get_collection=lambda name: SimpleNamespace(points_count=100, status="green"),
        scroll=lambda **kwargs: ([SimpleNamespace(
            payload={"arxiv_id": "1706.03762"}, vector=[1.0, 2.0],
        )], None),
    )
    monkeypatch.setattr(map_locate_svc, "_map_client", lambda: client)
    assert asyncio.run(map_locate_svc.collection_stats())["sample_valid"] is False


def test_deep_health_does_not_silently_skip_missing_map_config(monkeypatch):
    monkeypatch.setattr(config, "map_store_is_dedicated", lambda: False)
    monkeypatch.setattr(config, "qdrant_recent_configured", lambda: False)
    monkeypatch.setattr(config, "TURSO_URL", "")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "")
    monkeypatch.setattr(qdrant_svc, "_client", lambda: SimpleNamespace(
        get_collection=lambda name: SimpleNamespace(points_count=100)))
    monkeypatch.setattr(zilliz_svc, "_get_client", lambda: SimpleNamespace(
        list_collections=lambda: ["papers"]))
    monkeypatch.setattr(local_meta, "stats", lambda: {"available": True})

    result = asyncio.run(health.healthz_deep())
    assert result["services"]["map_positions"]["status"] == "error"
    assert result["overall"] == "degraded"
