"""
Keep the search models in memory while nobody is searching (switch MODEL_KEEP_WARM).

Measured on the Space 2026-11-02: after a quiet spell the process held 0.45 GB
(peak 3.1 GB), so the model weights were no longer in RAM. The next search took
9.9 s on the server (cross-encoder 4.8 s) while they were read back, and the
process returned to 1.85 GB; the searches after it took 1.4-2.1 s. Of the 20
searches before that, the slowest took 9.1 s. Nothing here unloads a model:
the system reclaimed the memory.

A forward pass reads every weight of every layer (the word-embedding table only
row by row, which costs a search just a few pages), so one uncached encode and
one reranked pair keep both models in recent use. A pass runs through the
inference pool like a search, and only after INTERVAL_S without any model work:
a reader's search is never queued behind one, and a busy Space runs none.

Each pass records its time and the major page faults it caused. A warm pass
takes well under a second with almost no faults; a slow pass with many faults
means the weights left RAM within the interval anyway, and /healthz/runtime
counts those as keep_warm_cold.

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
DOC = "Title: Keeping models in memory\nAbstract: A short passage, scored so the reranker's weights stay in RAM."

_state: dict = {"scheduler": "off", "passes": 0, "cold": 0, "last_at": None,
                "last_encode_ms": None, "last_rerank_ms": None, "last_major_faults": None}
_task: asyncio.Task | None = None


def _faults() -> int:
    """Major page faults of the calling thread (Linux), else of the whole process."""
    return resource.getrusage(getattr(resource, "RUSAGE_THREAD", resource.RUSAGE_SELF)).ru_majflt


def _pass() -> dict:
    """One encode and one reranked pair, on the inference thread, for whichever
    model is loaded (never loads one)."""
    f0, t0 = _faults(), time.perf_counter()
    if embed_svc._model is not None:
        # Around the LRU cache: an answer from the cache would read no weights.
        embed_svc._encode_cached.__wrapped__(TEXT)
    t1 = time.perf_counter()
    if reranker_bge_svc._reranker is not None:
        reranker_bge_svc._score_in_length_groups(reranker_bge_svc._reranker, [[TEXT, DOC]])
    t2 = time.perf_counter()
    return {"encode_ms": round((t1 - t0) * 1000), "rerank_ms": round((t2 - t1) * 1000),
            "major_faults": _faults() - f0}


def due() -> bool:
    """Whether to run a pass now: switch on and no model work for INTERVAL_S."""
    return features.mode(SWITCH) == "on" and compute_pool.idle_s() >= INTERVAL_S


async def run_once() -> dict | None:
    """Run one pass and record it. None when neither model is loaded."""
    if embed_svc._model is None and reranker_bge_svc._reranker is None:
        return None
    out = await compute_pool.run_inference(_pass)
    runtime_stats.record("keep_warm", out)
    runtime_stats.count("keep_warm_pass")
    _state["passes"] += 1
    if out["major_faults"] >= COLD_FAULTS:
        runtime_stats.count("keep_warm_cold")
        _state["cold"] += 1
    _state.update(last_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                  last_encode_ms=out["encode_ms"], last_rerank_ms=out["rerank_ms"],
                  last_major_faults=out["major_faults"])
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
