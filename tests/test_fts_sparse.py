"""Local FTS5 sparse retrieval.

Builds a small real sidecar rather than mocking sqlite: the failure modes worth
covering here are FTS5 syntax errors from user text and the AND/OR fallback, and
neither survives a mock.
"""
import asyncio
import sqlite3

import pytest

from app import config, fts_svc, local_meta


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    path = tmp_path / "metadata.sqlite"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE papers (
                     arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT,
                     abstract_preview TEXT, categories TEXT, primary_topic TEXT,
                     update_date TEXT, citation_count INTEGER,
                     influential_citations INTEGER)""")
    rows = [
        ("2506.00001", "Diffusion transformers for video generation",
         "A diffusion transformer that generates video frames.", "2025-06-01"),
        ("0704.0002", "Sparsity-certifying graph decompositions",
         "A pebble game algorithm for sparse graphs.", "2007-04-01"),
        ("2606.00001", "Retrieval augmented generation for agents",
         "An agent framework using retrieval augmented generation.", "2026-06-01"),
    ]
    c.executemany("INSERT INTO papers (arxiv_id,title,abstract_preview,update_date,"
                  "citation_count,influential_citations) VALUES (?,?,?,?,0,0)",
                  [(a, t, ab, d) for a, t, ab, d in rows])
    c.execute("""CREATE VIRTUAL TABLE papers_fts USING fts5(
                     title, abstract_preview, content='papers',
                     content_rowid='rowid', tokenize='porter unicode61')""")
    c.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    c.commit(); c.close()

    # Reset both modules' cached probes so each test opens this sidecar.
    monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(path))
    monkeypatch.setattr(local_meta, "_conn", None)
    monkeypatch.setattr(local_meta, "_probed", False)
    monkeypatch.setattr(local_meta, "_available", False)
    monkeypatch.setattr(fts_svc, "_probed", False)
    monkeypatch.setattr(fts_svc, "_available", False)
    return path


def run(q, limit=10):
    return asyncio.run(fts_svc.search_sparse(q, limit=limit))


def test_finds_paper_by_title(sidecar):
    out = run("diffusion transformer video")
    assert [r["arxiv_id"] for r in out][:1] == ["2506.00001"]


def test_covers_papers_newer_than_the_snapshot(sidecar):
    """The whole point of the switch: post-2025-06 papers must be reachable."""
    out = run("retrieval augmented generation agents")
    assert "2606.00001" in [r["arxiv_id"] for r in out]


def test_falls_back_from_and_to_or(sidecar):
    """A term absent from the corpus must not empty the result set."""
    out = run("diffusion zzzznotacorpusterm")
    assert [r["arxiv_id"] for r in out][:1] == ["2506.00001"]


@pytest.mark.parametrize("q", [
    'quantum "unbalanced',          # unterminated quote
    "graph AND OR NOT",             # bare operators
    "sparse NEAR/2 graph",          # NEAR syntax
    "video*",                       # prefix operator
    "^anchored",                    # column-anchor operator
    "it's a graph",                 # apostrophe
    "-- drop table papers",         # sql-ish noise
])
def test_user_text_is_never_parsed_as_fts_syntax(sidecar, q):
    """Any of these raises OperationalError if passed to MATCH unescaped,
    which would silently kill the sparse arm for that query."""
    out = run(q)
    assert isinstance(out, list)


def test_empty_and_punctuation_only_queries(sidecar):
    assert run("") == []
    assert run("   ") == []
    assert run("!!! ??? ...") == []


def test_single_characters_are_dropped(sidecar):
    assert fts_svc._terms("a b diffusion c") == ["diffusion"]


def test_scores_are_higher_is_better(sidecar):
    """Negated from bm25()'s more-negative-is-better so it matches the dense arm."""
    out = run("diffusion transformer video")
    assert len(out) >= 1
    assert out == sorted(out, key=lambda r: r["score"], reverse=True)


def test_respects_limit(sidecar):
    assert len(run("generation OR graph OR diffusion", limit=2)) <= 2


def test_unavailable_without_fts_index(tmp_path, monkeypatch):
    """An older sidecar has no papers_fts; degrade instead of raising."""
    path = tmp_path / "old.sqlite"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT)")
    c.execute("INSERT INTO papers VALUES ('1','t')")
    c.commit(); c.close()
    monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(path))
    monkeypatch.setattr(local_meta, "_conn", None)
    monkeypatch.setattr(local_meta, "_probed", False)
    monkeypatch.setattr(local_meta, "_available", False)
    monkeypatch.setattr(fts_svc, "_probed", False)
    monkeypatch.setattr(fts_svc, "_available", False)
    assert fts_svc.is_available() is False
    assert run("anything") == []


def test_stats_reports_index_size(sidecar):
    s = fts_svc.stats()
    assert s["available"] is True
    assert s["indexed"] == 3
