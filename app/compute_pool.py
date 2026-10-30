"""
Where blocking work runs, so that network waits and model inference stop
competing for the same threads.

Before this, everything went through asyncio's default executor: BGE-M3
encoding, the search cross-encoder, FTS5 and every sync Qdrant call. A home
feed load fans out to up to ~24 blocking Qdrant calls (3 shards x (clusters + 1)),
each 200-700 ms of mostly network wait. On a small default pool those calls
occupied every thread, so a search arriving during a feed load queued for a
thread while both CPUs sat idle.

Two pools now:

  * inference  -- model forward passes. Sized from the CPUs this container may
    actually use (cgroup quota / affinity, not the host's core count), with
    torch's intra-op threads set to match. On the 2-vCPU Space that is ONE
    worker running torch on 2 threads: concurrent requests queue and each
    runs at full speed, instead of four torch threads thrashing two cores.
  * io         -- blocking network clients (Qdrant). Its threads wait rather
    than compute, but it is capped (config.IO_POOL_WORKERS = 12) because the
    free Qdrant clusters lose throughput past ~4 concurrent fan-outs.

`pending()` reports queued + running inference jobs; search uses it to rerank
deeper when nobody else is waiting (see config.SEARCH_RERANK_TOP_N_IDLE).
"""
from __future__ import annotations

import asyncio
import contextvars
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from app import config


def usable_cpus() -> int:
    """CPUs this process may use: the tightest of the cgroup quota, the affinity
    mask and os.cpu_count(). Containers usually report the HOST's cores through
    os.cpu_count(), which is how torch ends up oversubscribing a 2-vCPU box."""
    counts = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            counts.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    q = cgroup_cpu_quota()
    if q:
        counts.append(max(1, math.ceil(q)))
    return max(1, min(counts))


def cgroup_cpu_quota() -> float | None:
    """CPU quota in cores from cgroup v2 (cpu.max) or v1 (cfs_quota/period)."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
            if quota != "max":
                return int(quota) / int(period)
            return None
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return quota / period if quota > 0 else None
    except (OSError, ValueError):
        return None


CPUS = usable_cpus()
INFERENCE_WORKERS = config.INFERENCE_WORKERS or max(1, CPUS // 2)
TORCH_THREADS = config.INFERENCE_THREADS or max(1, CPUS // INFERENCE_WORKERS)

_inference = ThreadPoolExecutor(max_workers=INFERENCE_WORKERS, thread_name_prefix="inference")
_io = ThreadPoolExecutor(max_workers=config.IO_POOL_WORKERS, thread_name_prefix="io")

_pending = 0
_pending_lock = threading.Lock()
_torch_configured = False


def configure_torch() -> int | None:
    """Set torch's intra-op thread count once. Returns the count, or None when
    torch is not installed (CI, local dev without the ML stack)."""
    global _torch_configured
    try:
        import torch
    except ImportError:
        return None
    if not _torch_configured:
        torch.set_num_threads(TORCH_THREADS)
        _torch_configured = True
        print(f"[compute_pool] torch threads={TORCH_THREADS}, inference workers="
              f"{INFERENCE_WORKERS}, usable CPUs={CPUS} (os.cpu_count={os.cpu_count()})")
    return torch.get_num_threads()


def configure_gil() -> float:
    """Apply config.GIL_SWITCH_INTERVAL_MS (0 keeps Python's default). Returns
    the interval in force, in ms. See app/model_trace.py for why it matters."""
    if config.GIL_SWITCH_INTERVAL_MS > 0:
        sys.setswitchinterval(config.GIL_SWITCH_INTERVAL_MS / 1000)
        print(f"[compute_pool] GIL switch interval {config.GIL_SWITCH_INTERVAL_MS} ms")
    return sys.getswitchinterval() * 1000


def pending() -> int:
    """Inference jobs queued or running right now."""
    return _pending


async def run_inference(fn, *args):
    """Run a model forward pass on the inference pool."""
    global _pending
    with _pending_lock:
        _pending += 1
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_inference, lambda: fn(*args))
    finally:
        with _pending_lock:
            _pending -= 1


async def run_io(fn, *args):
    """Run a blocking network call on the io pool, carrying contextvars across
    (loop.run_in_executor does not propagate them by itself)."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(_io, lambda: ctx.run(fn, *args))


def stats() -> dict:
    """Diagnostics for /healthz/runtime."""
    return {
        "usable_cpus": CPUS,
        "os_cpu_count": os.cpu_count(),
        "cgroup_cpu_quota": cgroup_cpu_quota(),
        "inference_workers": INFERENCE_WORKERS,
        "torch_threads_target": TORCH_THREADS,
        "io_workers": config.IO_POOL_WORKERS,
        "inference_pending": _pending,
        "gil_switch_interval_ms": round(sys.getswitchinterval() * 1000, 3),
    }
