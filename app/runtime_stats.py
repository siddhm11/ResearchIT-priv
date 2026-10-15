"""
What the deployed process is actually doing: memory, CPUs, thread pools, and
recent stage timings for search and the feed. Served at /healthz/runtime.

Every capacity decision before this (how much RAM is left for a second vector
index, whether torch oversubscribes the Space's 2 vCPUs, where a search's
1.5 s goes) rested on estimates, because nothing reported the live process.
Timings are kept in memory only: the last 200 of each kind, lost on restart.
"""
from __future__ import annotations

import resource
import sys
import threading
from collections import deque

_N = 200
_buffers: dict[str, deque] = {}
_lock = threading.Lock()


def record(kind: str, timings: dict) -> None:
    """Keep the numeric *_ms fields of one request's timings."""
    row = {k: v for k, v in timings.items()
           if k.endswith("_ms") and isinstance(v, (int, float))}
    if not row:
        return
    with _lock:
        _buffers.setdefault(kind, deque(maxlen=_N)).append(row)


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def timings() -> dict:
    """p50 / p95 / max per stage, per request kind."""
    with _lock:
        snap = {k: list(v) for k, v in _buffers.items()}
    out = {}
    for kind, rows in snap.items():
        keys = sorted({k for r in rows for k in r})
        out[kind] = {"n": len(rows)}
        for k in keys:
            xs = [r[k] for r in rows if k in r]
            out[kind][k] = {"p50": _pct(xs, 0.5), "p95": _pct(xs, 0.95), "max": max(xs)}
    return out


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def memory() -> dict:
    """Resident and peak memory of this process, and the container's limit."""
    out: dict = {}
    status = _read("/proc/self/status")
    if status:
        for line in status.splitlines():
            key, _, val = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                out[{"VmRSS": "rss_gb", "VmHWM": "peak_rss_gb"}[key]] = round(
                    int(val.split()[0]) / 1e6, 3)
    else:
        # macOS reports bytes, Linux kilobytes; only the peak is available here.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["peak_rss_gb"] = round(peak / (1e9 if sys.platform == "darwin" else 1e6), 3)
    # cgroup v2, then v1. memory.current includes reclaimable page cache (the
    # metadata sidecar, any memory-mapped vectors), so it overstates pressure.
    for cur, lim in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                     ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
                      "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        c, l_ = _read(cur), _read(lim)
        if c is not None:
            out["container_used_gb"] = round(int(c) / 1e9, 3)
            out["container_limit_gb"] = None if l_ in (None, "max") else round(int(l_) / 1e9, 3)
            break
    return out


def models() -> dict:
    from app import config, embed_svc, reranker_bge_svc
    try:
        import torch
        torch_threads = torch.get_num_threads()
    except ImportError:
        torch_threads = None
    return {
        "embedder": config.BGE_M3_MODEL,
        "embedder_loaded": embed_svc._model is not None,
        "search_reranker": config.RERANKER_MODEL,
        "search_reranker_loaded": reranker_bge_svc._reranker is not None,
        "reranker_max_length": config.RERANKER_MAX_LENGTH,
        "rerank_depth": {"base": config.SEARCH_RERANK_TOP_N,
                         "idle": config.SEARCH_RERANK_TOP_N_IDLE},
        "torch_threads": torch_threads,
    }
