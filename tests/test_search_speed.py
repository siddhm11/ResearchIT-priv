"""Search latency and concurrency plumbing (2026-10-14, doc 06 changelog).

Covers: retrieval for the original query no longer waits on the Groq rewrite;
the cross-encoder window deepens only when nothing else is waiting on the
inference pool; the reranker always hands the pipeline logits; and the pool is
sized from the CPUs the container may actually use.
"""
import asyncio
import types
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app import compute_pool, config, hybrid_search_svc as hs, reranker_bge_svc


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(config, "SPARSE_BACKEND", "fts")
    monkeypatch.setattr(config, "SEARCH_BGE_RERANK", False)
    monkeypatch.setattr(hs.fts_svc, "is_available", lambda: True)
    monkeypatch.setattr(hs.embed_svc, "encode_query", lambda q: (np.ones(1024, np.float32), {}))
    monkeypatch.setattr(hs.turso_svc, "fetch_metadata_batch", AsyncMock(return_value={}))
    monkeypatch.setattr(hs.arxiv_svc, "fetch_metadata_batch", AsyncMock(return_value={}))
    monkeypatch.setattr(hs.zilliz_svc, "search_sparse", AsyncMock(return_value=[]))
    monkeypatch.setattr(hs.fts_svc, "search_sparse",
                        AsyncMock(return_value=[{"arxiv_id": "2601.00002", "score": 1.0}]))


async def test_original_query_retrieval_starts_before_the_rewrite_returns(world, monkeypatch):
    """The rewrite only returns once dense retrieval for the ORIGINAL query has
    started. Under the old ordering (retrieval after the rewrite) this deadlocks
    and the wait_for times out."""
    dense_started = asyncio.Event()
    seen = []

    async def dense(vec, limit):
        seen.append(limit)
        dense_started.set()
        return [{"arxiv_id": "2601.00001", "score": 0.9}]

    async def rewrite(q):
        await dense_started.wait()
        return q            # unchanged: the common case

    monkeypatch.setattr(hs.qdrant_svc, "search_dense_merged", dense)
    monkeypatch.setattr(hs.groq_svc, "rewrite", rewrite)
    ids, meta = await asyncio.wait_for(hs.search("some long casual topic query", return_meta=True), 5)
    assert set(ids) == {"2601.00001", "2601.00002"}
    assert meta["retrieval_mode"] == "hybrid"
    assert seen == [10 * config.SEARCH_FETCH_K_MULTIPLIER]


async def test_rewritten_query_is_still_searched(world, monkeypatch):
    dense = AsyncMock(return_value=[{"arxiv_id": "2601.00001", "score": 0.9}])
    monkeypatch.setattr(hs.qdrant_svc, "search_dense_merged", dense)
    monkeypatch.setattr(hs.groq_svc, "rewrite", AsyncMock(return_value="better phrasing"))
    _, meta = await hs.search("some long casual topic query", return_meta=True)
    assert dense.await_count == 2
    assert hs.fts_svc.search_sparse.await_args_list[1].args[0] == "better phrasing"
    assert meta["retrieval_sources"] == ["fts_q0", "qdrant_q0", "fts_q1", "qdrant_q1"]


async def test_encode_time_is_model_time_not_rewrite_wait(world, monkeypatch):
    async def slow_rewrite(q):
        await asyncio.sleep(0.3)
        return q
    monkeypatch.setattr(hs.qdrant_svc, "search_dense_merged", AsyncMock(return_value=[]))
    monkeypatch.setattr(hs.groq_svc, "rewrite", slow_rewrite)
    _, meta = await hs.search("some long casual topic query", return_meta=True)
    assert meta["groq_time_ms"] >= 300
    assert meta["encode_time_ms"] < 200


def test_rerank_depth_deepens_only_when_idle(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_RERANK_TOP_N", 10)
    monkeypatch.setattr(config, "SEARCH_RERANK_TOP_N_IDLE", 20)
    monkeypatch.setattr(compute_pool, "_pending", 0)
    assert hs.rerank_depth() == 20
    monkeypatch.setattr(compute_pool, "_pending", 1)
    assert hs.rerank_depth() == 10


def test_rerank_depth_never_below_base(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_RERANK_TOP_N", 10)
    monkeypatch.setattr(config, "SEARCH_RERANK_TOP_N_IDLE", 5)
    monkeypatch.setattr(compute_pool, "_pending", 0)
    assert hs.rerank_depth() == 10


async def test_pending_counts_running_inference_and_recovers_after_errors():
    started, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()

    def work():
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result(5)
        return 7

    job = asyncio.create_task(compute_pool.run_inference(work))
    await asyncio.wait_for(started.wait(), 5)
    assert compute_pool.pending() == 1
    release.set()
    assert await job == 7
    assert compute_pool.pending() == 0

    def boom():
        raise ValueError("model failed")
    with pytest.raises(ValueError):
        await compute_pool.run_inference(boom)
    assert compute_pool.pending() == 0


def test_reranker_asks_for_logits(monkeypatch):
    """mxbai returns probabilities by default; the pipeline sigmoids again, which
    squeezed every score into 0.5-0.73 and let the boosts take over."""
    calls = {}

    def predict(pairs, **kw):
        calls.update(kw)
        return np.array([2.0, -1.0])

    monkeypatch.setattr(reranker_bge_svc, "_reranker", types.SimpleNamespace(predict=predict))
    sentinel = object()
    monkeypatch.setattr(reranker_bge_svc, "_identity", lambda: sentinel)
    papers = [{"arxiv_id": "a", "title": "t", "abstract": "x"},
              {"arxiv_id": "b", "title": "u", "abstract": "y"}]
    out, _ = reranker_bge_svc.rerank("q", papers, top_n=2)
    assert calls["activation_fn"] is sentinel
    assert [p["arxiv_id"] for p in out] == ["a", "b"]
    assert out[0]["bge_rerank_score"] == 2.0


def test_usable_cpus_honours_cgroup_quota(monkeypatch):
    """A 2-vCPU container commonly reports the host's cores via os.cpu_count()."""
    monkeypatch.setattr(compute_pool.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(compute_pool, "cgroup_cpu_quota", lambda: 2.0)
    if hasattr(compute_pool.os, "sched_getaffinity"):
        monkeypatch.setattr(compute_pool.os, "sched_getaffinity", lambda pid: set(range(16)))
    assert compute_pool.usable_cpus() == 2


def test_usable_cpus_without_quota_uses_cpu_count(monkeypatch):
    monkeypatch.setattr(compute_pool.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(compute_pool, "cgroup_cpu_quota", lambda: None)
    if hasattr(compute_pool.os, "sched_getaffinity"):
        monkeypatch.setattr(compute_pool.os, "sched_getaffinity", lambda pid: set(range(4)))
    assert compute_pool.usable_cpus() == 4


def test_runtime_stats_summarise_recent_requests(monkeypatch):
    from app import runtime_stats
    monkeypatch.setattr(runtime_stats, "_buffers", {})
    for ms in (100, 200, 300, 400):
        runtime_stats.record("search", {"total_ms": ms, "encode_time_ms": ms // 10,
                                        "groq_status": "rewritten"})
    t = runtime_stats.timings()["search"]
    assert t["n"] == 4
    assert t["total_ms"] == {"p50": 300, "p95": 400, "max": 400}
    assert "groq_status" not in t


def test_runtime_endpoint_reports_memory_cpu_models_and_timings():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        r = c.get("/healthz/runtime")
    assert r.status_code == 200
    body = r.json()
    assert body["cpu"]["usable_cpus"] >= 1
    assert body["cpu"]["inference_workers"] >= 1
    assert body["models"]["search_reranker"] == config.RERANKER_MODEL
    assert "peak_rss_gb" in body["memory"]
    assert isinstance(body["timings"], dict)


def test_docker_image_bakes_in_the_configured_reranker():
    from pathlib import Path
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    for model, rev in config.RERANKER_PINS.items():
        assert f"snapshot_download('{model}', revision='{rev}'" in dockerfile
    assert config.RERANKER_MODEL in config.RERANKER_PINS


def test_bge_m3_pin_matches_dockerfile_and_ingest():
    """Query vectors, ingest vectors and the image must all use one BGE-M3 commit."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    rev = config.BGE_M3_PINS["BAAI/bge-m3"]
    assert config.BGE_M3_REVISION == rev
    assert f"snapshot_download('BAAI/bge-m3', revision='{rev}'" in (root / "Dockerfile").read_text()
    assert rev in (root / "scripts" / "ingest_backends.py").read_text()


def test_embed_model_path_resolves_pinned_snapshot(monkeypatch, tmp_path):
    import sys, types
    from app import embed_svc
    calls = []
    def fake_snapshot(**kw):
        calls.append(kw)
        if kw.get("local_files_only"):
            raise FileNotFoundError("not cached")
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot))
    assert embed_svc.model_path() == str(tmp_path)
    assert [c["revision"] for c in calls] == [config.BGE_M3_REVISION] * 2
    assert calls[0]["local_files_only"] is True


def test_dockerfile_has_no_recursive_chmod_and_skips_onnx():
    """A `chmod -R` layer duplicated every file under /app, and the ONNX copies
    added ~2.7 GB nothing loads; together they made the image too large for the
    HF builder's push step (exit 137, 2026-10-16)."""
    from pathlib import Path
    d = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    lines = [l for l in d.splitlines() if not l.lstrip().startswith("#")]
    assert not any("chmod -R" in l for l in lines)
    assert "COPY --chown=1000:1000 . ." in d
    assert d.count("'onnx/*'") >= 2
    assert not any("BGEM3FlagModel(" in l or "CrossEncoder(" in l for l in lines if l.startswith("RUN"))
