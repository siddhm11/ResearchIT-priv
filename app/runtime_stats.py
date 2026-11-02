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
_counters: dict[str, int] = {}
_lock = threading.Lock()


def record(kind: str, timings: dict) -> None:
    """Keep the numeric *_ms fields of one request's timings."""
    row = {k: v for k, v in timings.items()
           if k.endswith("_ms") and isinstance(v, (int, float))}
    if not row:
        return
    with _lock:
        _buffers.setdefault(kind, deque(maxlen=_N)).append(row)


def count(name: str, n: int = 1) -> None:
    """Bump an event counter (since the last restart)."""
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


def counters() -> dict[str, int]:
    with _lock:
        return dict(sorted(_counters.items()))


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


# /proc/self/status fields, in kB. RSS splits into anonymous memory (the
# Python heap, torch tensors) and file pages (anything read through mmap); swap
# is anonymous memory the system moved out of RAM. Together they say where model
# weights went when RSS fell from 3.1 GB to 0.45 GB on a quiet Space (2026-11-02).
_STATUS = {"VmRSS": "rss_gb", "VmHWM": "peak_rss_gb", "RssAnon": "anon_gb",
           "RssFile": "file_gb", "VmSwap": "swap_gb"}


def memory() -> dict:
    """Resident and peak memory of this process, and the container's limit."""
    out: dict = {}
    status = _read("/proc/self/status")
    if status:
        for line in status.splitlines():
            key, _, val = line.partition(":")
            if key in _STATUS:
                out[_STATUS[key]] = round(int(val.split()[0]) / 1e6, 3)
    else:
        # macOS reports bytes, Linux kilobytes; only the peak is available here.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["peak_rss_gb"] = round(peak / (1e9 if sys.platform == "darwin" else 1e6), 3)
    # Pages read back from disk or swap since the process started.
    out["major_faults"] = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
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
    swap = _read("/sys/fs/cgroup/memory.swap.current")
    if swap is not None:
        out["container_swap_gb"] = round(int(swap) / 1e9, 3)
    out.update(_reclaim())
    return out


# memory.stat fields: what the container holds, and how often pages it had
# lost were needed again (refaults) - the cost a quiet spell leaves behind.
_CGROUP_STAT = {"anon": "anon_gb", "file": "page_cache_gb", "file_mapped": "mapped_gb",
                "zswapped": "zswapped_gb", "workingset_refault_anon": "refaults_anon",
                "workingset_refault_file": "refaults_file", "pgmajfault": "major_faults"}


def _reclaim() -> dict:
    """Whether the machine swaps and how hard it is reclaiming memory, for
    app/keep_warm.py: swap on the host (/proc/meminfo inside a container shows
    the host's), the container's memory.stat, the share of time its tasks
    waited on memory (PSI, %, 5-minute average), and how much memory this
    process may lock in RAM (mlock), the one way to keep pages there for good."""
    out: dict = {}
    meminfo = _read("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            key, _, val = line.partition(":")
            if key in ("SwapTotal", "SwapFree"):
                out[f"host_{key[4:].lower()}_swap_gb"] = round(int(val.split()[0]) / 1e6, 3)
    stat = _read("/sys/fs/cgroup/memory.stat")
    if stat:
        cg = {}
        for line in stat.splitlines():
            key, _, val = line.partition(" ")
            if key in _CGROUP_STAT:
                name = _CGROUP_STAT[key]
                cg[name] = round(int(val) / 1e9, 3) if name.endswith("_gb") else int(val)
        out["cgroup"] = cg
    pressure = _read("/sys/fs/cgroup/memory.pressure")
    if pressure:
        for line in pressure.splitlines():
            kind, _, rest = line.partition(" ")
            fields = dict(f.split("=") for f in rest.split())
            if "avg300" in fields:
                out[f"pressure_{kind}_avg300"] = float(fields["avg300"])
    soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    out["memlock_limit_gb"] = "unlimited" if soft == resource.RLIM_INFINITY else round(soft / 1e9, 6)
    return out


def models() -> dict:
    from app import config, embed_svc, keep_warm, model_trace, reranker_bge_svc
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
        "traced": model_trace.status(),
        "keep_warm": keep_warm.status(),
    }
