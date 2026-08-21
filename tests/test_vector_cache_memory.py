"""
The vector cache must cost what its comment says it costs.

qdrant_svc documents "Vectors are 1024 floats = 4KB each. A 25K cap = ~100MB
RAM ceiling." That is true of a packed buffer and false of a Python list, which
boxes every element: measured, one 1024-float list is 32,824 bytes, so the 25K
cap was really 0.82 GB — 8x over budget, on a 16 GB box already shared with
BGE-M3, a cross-encoder and a 2.7 GB metadata sidecar.

Storing float32 arrays makes the documented figure true. The risk it introduces
is that `if not vec` raises on an array where it used to be a valid emptiness
check, so the guards are pinned here too.
"""
import pathlib
import re
import sys

import numpy as np

from app import qdrant_svc


def test_a_cached_vector_is_packed_not_boxed():
    qdrant_svc._VECTOR_CACHE.clear()
    qdrant_svc._vec_cache_put("1706.03762", [0.1] * 1024)

    cached = qdrant_svc._vec_cache_get("1706.03762")
    assert isinstance(cached, np.ndarray)
    assert cached.dtype == np.float32
    assert cached.nbytes == 4096, "a 1024-dim float32 vector must be 4KB"


def test_the_documented_ceiling_is_now_accurate():
    per_entry = np.zeros(1024, dtype=np.float32).nbytes
    ceiling_mb = per_entry * qdrant_svc._VECTOR_CACHE_MAX / 1e6
    assert ceiling_mb < 110, f"cache ceiling is {ceiling_mb:.0f} MB, not ~100 MB"

    # And the boxed form really was the 8x it is claimed to be.
    boxed = sys.getsizeof([0.0] * 1024) + 1024 * sys.getsizeof(0.1)
    assert boxed / per_entry > 5, (
        "a Python list is no longer materially larger; the rationale in "
        "qdrant_svc may need revisiting")


def test_cache_still_evicts_at_the_cap():
    qdrant_svc._VECTOR_CACHE.clear()
    try:
        qdrant_svc._VECTOR_CACHE_MAX_ORIG = qdrant_svc._VECTOR_CACHE_MAX
        qdrant_svc._VECTOR_CACHE_MAX = 3
        for i in range(5):
            qdrant_svc._vec_cache_put(f"p{i}", [float(i)] * 1024)
        assert len(qdrant_svc._VECTOR_CACHE) <= 3
        assert qdrant_svc._vec_cache_get("p0") is None, "LRU eviction stopped working"
        assert qdrant_svc._vec_cache_get("p4") is not None
    finally:
        qdrant_svc._VECTOR_CACHE_MAX = qdrant_svc._VECTOR_CACHE_MAX_ORIG
        qdrant_svc._VECTOR_CACHE.clear()


def test_stats_report_the_footprint():
    qdrant_svc._VECTOR_CACHE.clear()
    qdrant_svc._vec_cache_put("a", [0.0] * 1024)
    stats = qdrant_svc.vector_cache_stats()
    assert stats["size"] == 1
    assert stats["approx_bytes"] == 4096


def test_no_caller_uses_truthiness_on_a_fetched_vector():
    """`if not vec` raises on a numpy array.

    Two call sites used it as an emptiness check and would now throw:
    map_locate_svc and the collections router.

    Parsed with `ast` rather than grepped, so prose in a docstring that merely
    mentions the pattern is not mistaken for the pattern.
    """
    import ast

    offenders = []
    for path in sorted(pathlib.Path("app").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (isinstance(test, ast.UnaryOp)
                    and isinstance(test.op, ast.Not)
                    and isinstance(test.operand, ast.Name)
                    and test.operand.id in {"vec", "vector"}):
                offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "truthiness test on a vector, which raises for numpy arrays: "
        + ", ".join(offenders))
