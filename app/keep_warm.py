"""
Keep the search models in memory while nobody is searching (switch MODEL_KEEP_WARM).

Measured on the Space: after a quiet spell the process held 0.45 GB (peak
3.1 GB) and the next search took 9.9 s on the server (2026-11-02); 3.3 hours
after the last search, one took 10.6 s end to end against 3-4 s warm, and the
process grew from 1.0 to 2.4 GB while the weights were read back (2026-11-03).
With nothing else running it then lost them again: 1.1 GB at 54 minutes and
0.66 GB at 92, in sudden drops about half an hour apart, the container's
memory falling with it. A pass every few idle minutes is well inside that.
Nothing here unloads a model: the system took the memory back. It can do that
in two ways, and the models are exposed to both (laptop, same library pins):

- BGE-M3's 2.27 GB of weights are never copied into the process: the tensors
  are the memory-mapped pytorch_model.bin, so the system can drop any page of
  them at any time and read it back from disk on the next search.
- The reranker's 283 MB, like the rest of the process's private memory
  (1.1 GB once both models are loaded, laptop), can only leave RAM through
  swap. The Space's whole process held 0.45 GB, so the Space very probably
  swaps; /healthz/runtime now reports swap to settle it.

Both kinds of reclaim take the pages used longest ago first. A pass therefore
reads every page of every weight of both models (one value per 4 KB; 5-8 ms
on the laptop once in RAM) and runs both models the way a search does: one
query encode, and the cross-encoder on a top 10 of short and long abstracts,
so the same kernels run. A forward pass alone would leave most of the two
word-embedding tables (1 GB of BGE-M3's 2.27 GB), of which a search reads only
its own tokens' rows, to be dropped.

A pass runs through the inference pool like a search, and only after
INTERVAL_S without any model work, so a busy Space runs none. A search that
arrives during one waits for it: 0.21 s on the laptop warm, which should be
about 1 s on the Space (it runs these models about 4.5 times slower).

Each pass records its times and the major page faults it caused. A warm pass
causes almost none; a pass with many means the weights left RAM within the
interval anyway, and /healthz/runtime counts those as keep_warm_cold.

Modes: off; on.
"""
from __future__ import annotations

import asyncio
import os
import resource
import time

from app import compute_pool, embed_svc, errors, features, reranker_bge_svc, runtime_stats

SWITCH = "MODEL_KEEP_WARM"
INTERVAL_S = int(os.getenv("MODEL_KEEP_WARM_INTERVAL_S", "120"))
COLD_FAULTS = 256           # pages read back from disk or swap (1 MB) before a pass counts as cold
TEXT = "keeping the search models in memory"
_SENTENCE = ("We study how retrieval models rank scientific papers for a short query, "
             "comparing dense, lexical and cross-encoder scores on held-out searches. ")
# A search's top 10 mixes abstracts the reranker scores in two batches
# (reranker_bge_svc._LONG_PAIR_CHARS): five of each length.
DOCS = [f"Title: Ranking papers for short queries, part {i}\nAbstract: {_SENTENCE * (4 if i < 5 else 10)}"
        for i in range(10)]
PAGE = 4096

_state: dict = {"scheduler": "off", "passes": 0, "cold": 0, "last_at": None,
                "last_encode_ms": None, "last_rerank_ms": None, "last_touch_ms": None,
                "last_major_faults": None, "weights_gb": None}
_task: asyncio.Task | None = None


def _faults() -> int:
    """Major page faults of the calling thread (Linux), else of the whole process."""
    return resource.getrusage(getattr(resource, "RUSAGE_THREAD", resource.RUSAGE_SELF)).ru_majflt


def _weights() -> list:
    """Every parameter and buffer of the loaded models, once each. The traced
    graphs hold these same tensors (checked on the laptop: no weight copies)."""
    try:
        import torch
    except ImportError:
        return []
    seen: dict = {}
    for holder in (embed_svc._model, reranker_bge_svc._reranker):
        module = getattr(holder, "model", None)
        if isinstance(module, torch.nn.Module):
            for t in list(module.parameters()) + list(module.buffers()):
                seen.setdefault(t.data_ptr(), t)
    return list(seen.values())


def _touch() -> int:
    """Read one value from every page of every weight. Returns the bytes covered."""
    tensors = _weights()
    if not tensors:
        return 0
    import torch
    with torch.inference_mode():
        for t in tensors:
            t.detach().reshape(-1)[:: max(1, PAGE // t.element_size())].sum()
    return sum(t.numel() * t.element_size() for t in tensors)


def _pass() -> dict:
    """One encode, one top-10 rerank and a read of every weight page, on the
    inference thread, for whichever model is loaded (never loads one)."""
    f0, t0 = _faults(), time.perf_counter()
    if embed_svc._model is not None:
        # Around the LRU cache: an answer from the cache would read no weights.
        embed_svc._encode_cached.__wrapped__(TEXT)
    t1 = time.perf_counter()
    if reranker_bge_svc._reranker is not None:
        reranker_bge_svc._score_in_length_groups(reranker_bge_svc._reranker, [[TEXT, d] for d in DOCS])
    t2 = time.perf_counter()
    covered = _touch()
    t3 = time.perf_counter()
    return {"encode_ms": round((t1 - t0) * 1000), "rerank_ms": round((t2 - t1) * 1000),
            "touch_ms": round((t3 - t2) * 1000), "weights_gb": round(covered / 1e9, 2),
            "major_faults": _faults() - f0}


def due() -> bool:
    """Whether to run a pass now: switch on and no model work for INTERVAL_S."""
    return features.mode(SWITCH) == "on" and compute_pool.idle_s() >= INTERVAL_S


async def run_once() -> dict | None:
    """Run one pass and record it. None when neither model is loaded."""
    if embed_svc._model is None and reranker_bge_svc._reranker is None:
        return None
    out = await compute_pool.run_inference(_pass)
    runtime_stats.record("keep_warm", {k: v for k, v in out.items() if k != "weights_gb"})
    runtime_stats.count("keep_warm_pass")
    # The first pass reads in the word-table rows no search has needed yet
    # (laptop: 1 GB, 64,000 faults), which says nothing about reclaim.
    first = _state["passes"] == 0
    _state["passes"] += 1
    if out["major_faults"] >= COLD_FAULTS and not first:
        runtime_stats.count("keep_warm_cold")
        _state["cold"] += 1
    _state.update(last_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                  last_encode_ms=out["encode_ms"], last_rerank_ms=out["rerank_ms"],
                  last_touch_ms=out["touch_ms"], last_major_faults=out["major_faults"],
                  weights_gb=out["weights_gb"])
    return out


async def _loop() -> None:
    # Checking every half interval keeps the longest gap without model work
    # under 1.5 x INTERVAL_S.
    while True:
        _state["scheduler"] = "running"
        await asyncio.sleep(INTERVAL_S / 2)
        if not due():
            continue
        try:
            await run_once()
        except Exception as e:                       # a failed pass never stops the loop
            errors.report("keep_warm", "a keep-warm pass failed", e)


async def start() -> None:
    global _task
    if features.mode(SWITCH) != "on" or _task is not None:
        return
    _task = asyncio.create_task(_loop(), name="keep-warm")


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
    _task = None
    _state["scheduler"] = "off"


def status() -> dict:
    return {"switch": features.mode(SWITCH), "interval_s": INTERVAL_S,
            "idle_s": round(compute_pool.idle_s()), **_state}
