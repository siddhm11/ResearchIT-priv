"""app/keep_warm.py: a pass reads the model weights (never a cached answer), runs
only after the models have been idle, and records the faults that show whether
the weights had left RAM."""
import asyncio
import time

import numpy as np
import pytest

from app import compute_pool, embed_svc, keep_warm, reranker_bge_svc, runtime_stats


class _Encoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return {"dense_vecs": [np.zeros(1024, dtype=np.float32)], "lexical_weights": [{7: 0.5}]}


class _CrossEncoder:
    def __init__(self):
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1
        return [0.0] * len(pairs)


@pytest.fixture
def models(monkeypatch):
    enc, ce = _Encoder(), _CrossEncoder()
    monkeypatch.setattr(embed_svc, "_model", enc)
    monkeypatch.setattr(embed_svc, "get_model", lambda: enc)
    monkeypatch.setattr(reranker_bge_svc, "_reranker", ce)
    embed_svc._encode_cached.cache_clear()
    yield enc, ce
    embed_svc._encode_cached.cache_clear()


def test_a_pass_reads_the_weights_even_when_the_text_is_cached(models):
    enc, ce = models
    embed_svc.encode_query(keep_warm.TEXT)
    embed_svc.encode_query(keep_warm.TEXT)
    assert enc.calls == 1                       # a search for the same text is cached
    keep_warm._pass()
    keep_warm._pass()
    assert (enc.calls, ce.calls) == (3, 4)     # short and long abstracts: two batches a pass


def test_the_rerank_looks_like_a_search_top_10():
    long = [len(d) > reranker_bge_svc._LONG_PAIR_CHARS for d in keep_warm.DOCS]
    assert len(long) == 10 and sum(long) == 5


def test_every_page_of_every_weight_is_read_and_left_unchanged(monkeypatch):
    torch = pytest.importorskip("torch")

    class Holder:
        def __init__(self, module):
            self.model = module

    enc = torch.nn.Sequential(torch.nn.Embedding(3000, 64), torch.nn.Linear(64, 64))
    enc.register_buffer("position_ids", torch.arange(512))
    ce = torch.nn.Linear(64, 1)
    ce.extra = enc[1]                                     # shared with the encoder: counted once
    before = [p.clone() for p in enc.parameters()]
    monkeypatch.setattr(embed_svc, "_model", Holder(enc))
    monkeypatch.setattr(reranker_bge_svc, "_reranker", Holder(ce))
    size = sum(t.numel() * t.element_size() for t in [*enc.parameters(), *enc.buffers(), ce.weight, ce.bias])
    assert keep_warm._touch() == size
    assert all(torch.equal(a, b) for a, b in zip(before, enc.parameters()))


def test_without_torch_models_nothing_is_touched(models):
    assert keep_warm._touch() == 0


def test_a_pass_never_loads_a_model(monkeypatch):
    monkeypatch.setattr(embed_svc, "_model", None)
    monkeypatch.setattr(embed_svc, "get_model", lambda: pytest.fail("loaded the encoder"))
    monkeypatch.setattr(reranker_bge_svc, "_reranker", None)
    assert keep_warm._pass()["major_faults"] >= 0


async def test_idle_time_restarts_after_any_model_work(monkeypatch):
    monkeypatch.setattr(compute_pool, "_last_done", time.monotonic() - 500)
    assert compute_pool.idle_s() >= 500
    await compute_pool.run_inference(lambda: None)
    assert compute_pool.idle_s() < 5
    monkeypatch.setattr(compute_pool, "_pending", 1)
    assert compute_pool.idle_s() == 0


@pytest.mark.parametrize("mode, idle, expected", [
    ("on", keep_warm.INTERVAL_S, True), ("on", keep_warm.INTERVAL_S - 1, False), ("off", 10**6, False),
])
def test_due(monkeypatch, mode, idle, expected):
    monkeypatch.setenv(keep_warm.SWITCH, mode)
    monkeypatch.setattr(compute_pool, "idle_s", lambda: idle)
    assert keep_warm.due() is expected


async def test_a_pass_with_many_faults_counts_as_cold_after_the_first(models, monkeypatch):
    monkeypatch.setitem(keep_warm._state, "passes", 0)
    monkeypatch.setitem(keep_warm._state, "cold", 0)
    before = runtime_stats.counters().get("keep_warm_cold", 0)
    monkeypatch.setattr(keep_warm, "_pass", lambda: {"encode_ms": 4100, "rerank_ms": 2300, "touch_ms": 900,
                                                     "weights_gb": 2.56, "major_faults": 9000})
    await keep_warm.run_once()                  # the first pass reads the word tables in
    assert runtime_stats.counters().get("keep_warm_cold", 0) == before
    out = await keep_warm.run_once()
    assert out["major_faults"] == 9000
    assert runtime_stats.counters()["keep_warm_cold"] == before + 1
    assert keep_warm.status()["cold"] == 1
    assert keep_warm.status()["last_major_faults"] == 9000
    assert runtime_stats.timings()["keep_warm"]["encode_ms"]["max"] >= 4100


async def test_nothing_runs_without_models(monkeypatch):
    monkeypatch.setattr(embed_svc, "_model", None)
    monkeypatch.setattr(reranker_bge_svc, "_reranker", None)
    assert await keep_warm.run_once() is None


async def test_the_loop_starts_only_when_switched_on(monkeypatch):
    monkeypatch.setenv(keep_warm.SWITCH, "off")
    await keep_warm.start()
    assert keep_warm._task is None
    monkeypatch.setenv(keep_warm.SWITCH, "on")
    await keep_warm.start()
    try:
        await asyncio.sleep(0)
        assert keep_warm.status()["scheduler"] == "running"
    finally:
        await keep_warm.stop()
    assert keep_warm._task is None and keep_warm.status()["scheduler"] == "off"


def test_memory_says_where_the_weights_went(monkeypatch):
    files = {
        "/proc/self/status": "VmHWM:\t3128000 kB\nVmRSS:\t 450000 kB\nRssAnon:\t 300000 kB\n"
                             "RssFile:\t 150000 kB\nVmSwap:\t2600000 kB\n",
        "/sys/fs/cgroup/memory.current": "538000000",
        "/sys/fs/cgroup/memory.max": "16000000000",
        "/sys/fs/cgroup/memory.swap.current": "2700000000",
    }
    monkeypatch.setattr(runtime_stats, "_read", files.get)
    m = runtime_stats.memory()
    assert (m["rss_gb"], m["anon_gb"], m["file_gb"], m["swap_gb"]) == (0.45, 0.3, 0.15, 2.6)
    assert m["container_swap_gb"] == 2.7 and m["major_faults"] >= 0


def test_memory_says_whether_the_machine_swaps_and_reclaims(monkeypatch):
    files = {
        "/proc/meminfo": "MemTotal:       263000000 kB\nSwapTotal:       8000000 kB\nSwapFree:        2500000 kB\n",
        "/sys/fs/cgroup/memory.stat": "anon 450000000\nfile 90000000\nfile_mapped 40000000\n"
                                      "workingset_refault_anon 120000\nworkingset_refault_file 880000\n"
                                      "pgmajfault 51000\nslab 1000\n",
        "/sys/fs/cgroup/memory.pressure": "some avg10=0.00 avg60=1.50 avg300=2.25 total=900\n"
                                          "full avg10=0.00 avg60=0.40 avg300=0.75 total=300",
    }
    monkeypatch.setattr(runtime_stats, "_read", files.get)
    m = runtime_stats.memory()
    assert (m["host_total_swap_gb"], m["host_free_swap_gb"]) == (8.0, 2.5)
    assert m["cgroup"] == {"anon_gb": 0.45, "page_cache_gb": 0.09, "mapped_gb": 0.04, "refaults_anon": 120000,
                           "refaults_file": 880000, "major_faults": 51000}
    assert (m["pressure_some_avg300"], m["pressure_full_avg300"]) == (2.25, 0.75)
    assert m["memlock_limit_gb"] == "unlimited" or m["memlock_limit_gb"] >= 0
