"""
Per-shard search latency.

/healthz/shards asserts the shards share a retrieval CONFIG; it says nothing
about what that config costs. The two clusters hold near-identical volume —
arxiv_dense_a 899,456 points, arxiv_dense_b + arxiv_recent 899,382 between them
— yet shard a runs `hnsw_on_disk: true` and the other two do not. Whether that
costs anything is a measurement, not an inference.
"""
import inspect

from fastapi.testclient import TestClient

from app.main import app
from app.routers import health


def test_parameters_are_clamped():
    """/healthz/ab is an unauthenticated BGE-M3 amplifier. This must not be."""
    src = inspect.getsource(health.healthz_shard_latency)
    assert "min(int(samples), 10)" in src
    assert "min(int(limit), 50)" in src


def test_it_takes_no_free_text():
    """No query string in means no encoder work per request."""
    sig = inspect.signature(health.healthz_shard_latency)
    assert "q" not in sig.parameters, "free-text input would make this an amplifier"


def test_the_probe_vector_is_deterministic():
    """Numbers taken days apart have to be comparable, so the query is fixed."""
    src = inspect.getsource(health.healthz_shard_latency)
    assert "default_rng(20260824)" in src


def test_the_first_sample_is_discarded():
    """An earlier version reported sample one as a 'cold' reading.

    It was measuring client construction and TLS setup on OUR side: a fresh
    process gave 1743-5891ms, while the second and third calls in the same
    process gave 366ms and 470ms — indistinguishable from the median. Every
    reported number must be steady state.
    """
    src = inspect.getsource(health.healthz_shard_latency)
    assert "Discarded" in src, "no warm-up call before timing"
    assert "first_ms" not in src, (
        "still reporting a first-sample figure that conflates process-cold "
        "with graph-cold")


def test_endpoint_responds_and_reports_the_on_disk_flag():
    with TestClient(app) as c:
        r = c.get("/healthz/shard-latency?samples=1&limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "shards" in body and "verdict" in body
    for shard in body["shards"].values():
        if "median_ms" in shard:
            # The correlation this exists to expose must be in one response.
            assert "hnsw_on_disk" in shard
            assert "spread_ms" in shard


def test_clamping_actually_applies():
    with TestClient(app) as c:
        r = c.get("/healthz/shard-latency?samples=999&limit=999")
    assert r.status_code == 200
    assert r.json()["limit"] == 50


def test_verdict_is_absent_when_there_is_nothing_to_compare():
    """With one shard reachable there is no on-disk vs in-RAM comparison, and
    inventing one would be worse than saying nothing."""
    with TestClient(app) as c:
        body = c.get("/healthz/shard-latency?samples=1&limit=5").json()
    reachable = [s for s in body["shards"].values() if "median_ms" in s]
    if len({s.get("hnsw_on_disk") for s in reachable}) < 2:
        assert body["verdict"] is None
