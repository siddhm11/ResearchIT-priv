"""
Nothing may block the event loop with sidecar I/O.

`app/local_meta.py` is synchronous sqlite3 over a 2.7 GB file. A blocking call
inside an `async def` stalls EVERY in-flight request, including ones that never
touch the sidecar — and this runs as a single uvicorn worker on 2 vCPUs, so
there is no other worker to absorb it.

Measured on a 300k-row stand-in: a 72 ms query held the loop for 78 ms; the
same query via a thread held it for 7 ms. The real sidecar has 1.8M rows.

The shared connection is opened read-only with check_same_thread=False
precisely so it can be used from a thread (local_meta._probe).
"""
import ast
import asyncio
import pathlib
import sqlite3
import tempfile
import time

import pytest

# Functions that touch the sidecar's connection and therefore must not be
# called directly from a coroutine.
BLOCKING = {"fetch_rows", "fetch_trending", "pub_year_month", "newest_update_date"}

# is_available() and stats() are cheap after the first probe and are excluded.


def _async_callers_of_blocking_sidecar_fns():
    """Every `local_meta.<blocking>()` call lexically inside an async def."""
    offenders = []
    for path in sorted(pathlib.Path("app").rglob("*.py")):
        if path.name == "local_meta.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr in BLOCKING
                        and isinstance(f.value, ast.Name)
                        and f.value.id == "local_meta"):
                    offenders.append(f"{path}:{node.lineno} in {fn.name}()")
    return offenders


def test_no_coroutine_calls_the_sidecar_directly():
    offenders = _async_callers_of_blocking_sidecar_fns()
    assert not offenders, (
        "sidecar I/O called directly from a coroutine — wrap in "
        "asyncio.to_thread: " + ", ".join(offenders))


def test_the_check_would_catch_a_direct_call():
    """Guard against the guard silently matching nothing."""
    src = (
        "import asyncio\n"
        "from app import local_meta\n"
        "async def handler(ids):\n"
        "    return local_meta.fetch_rows(ids)\n"
    )
    tree = ast.parse(src)
    hits = [
        n.lineno for fn in ast.walk(tree) if isinstance(fn, ast.AsyncFunctionDef)
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in BLOCKING
    ]
    assert hits == [4]


async def test_a_threaded_query_does_not_stall_the_loop():
    """The property itself, not just the call shape."""
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/t.db"
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE papers (arxiv_id TEXT, blob TEXT)")
        c.executemany("INSERT INTO papers VALUES (?,?)",
                      [(f"p{i}", "x" * 200) for i in range(120_000)])
        c.commit()
        c.close()
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                               check_same_thread=False)

        def slow():
            return conn.execute(
                "SELECT COUNT(*) FROM papers WHERE blob LIKE '%zz%'").fetchone()

        gaps = []
        stop = asyncio.Event()

        async def heartbeat():
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.03)
        await asyncio.to_thread(slow)
        stop.set()
        await hb
        conn.close()

    assert max(gaps) < 0.05, (
        f"loop stalled {max(gaps)*1000:.0f} ms even through a thread")
