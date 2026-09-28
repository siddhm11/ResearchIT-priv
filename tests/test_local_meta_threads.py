"""The sidecar must serve concurrent threads correctly.

Regression: one sqlite3 connection shared across asyncio.to_thread workers
returned empty results ("bad parameter or other API misuse") and, once, another
thread's rows. starter_papers fans out per category, so 9 of 10 categories came
back empty and each fell through to a Turso scan that timed out after 30s.
"""
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import local_meta

CODES = [f"cs.T{i}" for i in range(8)]


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    path = tmp_path / "meta.sqlite"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT,
                 authors TEXT, abstract_preview TEXT, categories TEXT,
                 primary_topic TEXT, update_date TEXT, citation_count INTEGER,
                 influential_citations INTEGER)""")
    c.execute("CREATE TABLE paper_categories (arxiv_id TEXT, code TEXT, citation_count INTEGER)")
    c.execute("CREATE INDEX ix_pc ON paper_categories(code, citation_count DESC)")
    papers, cats = [], []
    for i in range(40_000):
        aid, code = f"2601.{i:05d}", CODES[i % len(CODES)]
        papers.append((aid, f"t{i}", "[]", "a", code, "", "2026-01-15", i % 997, 0))
        cats.append((aid, code, i % 997))
    c.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?)", papers)
    c.executemany("INSERT INTO paper_categories VALUES (?,?,?)", cats)
    c.commit()
    c.close()
    monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(path))
    monkeypatch.setattr(local_meta, "_conn", None)
    monkeypatch.setattr(local_meta, "_probed", False)
    monkeypatch.setattr(local_meta, "_available", False)
    monkeypatch.setattr(local_meta, "_max_date", None)
    assert local_meta.is_available()
    return path


def _hammer():
    jobs = [code for _ in range(6) for code in CODES]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(zip(jobs, pool.map(
            lambda code: local_meta.fetch_trending({code}, limit=200), jobs)))


def test_concurrent_trending_queries_each_get_their_own_rows(sidecar):
    for code, rows in _hammer():
        assert len(rows) == 200, f"{code}: {len(rows)} rows"
        assert all(r["categories"] == code for r in rows), f"{code}: foreign rows"


def test_each_thread_gets_its_own_handle(sidecar):
    barrier = threading.Barrier(4)

    def grab(_):
        conn = local_meta.connection()
        barrier.wait(timeout=10)      # all four alive at once: distinct threads
        return conn

    with ThreadPoolExecutor(max_workers=4) as pool:
        conns = list(pool.map(grab, range(4)))
    assert len({id(c) for c in conns}) == 4
    assert local_meta.connection() is local_meta.connection()
