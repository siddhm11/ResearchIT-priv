"""scripts/sidecar_full_abstracts.py and the sidecar reader that serves its output.

73% of sidecar abstracts were cut at 500 characters; the search reranker and
the plain-language summaries read the stump. The script fills papers.
abstract_full from the Kaggle CSV and indexes authors; local_meta serves the
full text when the column exists and the stump otherwise.
"""
import csv
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from app import local_meta

_spec = importlib.util.spec_from_file_location(
    "sidecar_full_abstracts",
    Path(__file__).resolve().parents[1] / "scripts" / "sidecar_full_abstracts.py")
sfa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sfa)

FULL = ("We describe a new algorithm, the pebble game with colors, and use it to "
        "obtain a characterization of the family of sparse graphs. ") * 6
STUMP = FULL[:500]

SCHEMA = """
CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT, abstract_preview TEXT,
  categories TEXT, primary_topic TEXT, update_date TEXT, citation_count INTEGER, influential_citations INTEGER);
CREATE TABLE build_state (key TEXT PRIMARY KEY, value TEXT);
CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract_preview, content='papers', content_rowid='rowid',
  tokenize='porter unicode61');
"""


def _sidecar(tmp_path) -> Path:
    path = tmp_path / "m.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    rows = [("0704.0002", "Sparsity-certifying Graph Decompositions", "Ileana Streinu; Louis Theran", STUMP),
            ("hep-th/9901001", "Strings", "Juan Maldacena", STUMP),          # CSV text differs
            ("2607.00001", "A recent paper", "Jiajun Jiang", "Short and complete.")]
    conn.executemany("INSERT INTO papers (arxiv_id, title, authors, abstract_preview, categories, "
                     "update_date, citation_count) VALUES (?,?,?,?,'cs.DS','2026-01-01',1)", rows)
    conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return path


def _csv(tmp_path) -> Path:
    path = tmp_path / "k.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "abstract"])
        # Raw arXiv text: leading spaces and hard line breaks.
        w.writerow(["0704.0002", "x", "  " + FULL.replace("algorithm, the", "algorithm,\nthe")])
        w.writerow(["hep-th/9901001", "x", "A different paper entirely. " * 30])
    return path


def test_stumps_get_their_full_text_and_only_when_the_text_agrees(tmp_path):
    side = _sidecar(tmp_path)
    conn = sqlite3.connect(side)
    stats = sfa.add_full_abstracts(conn, str(_csv(tmp_path)), dry_run=False)
    assert stats == {"cut": 2, "filled": 1}
    got = dict(conn.execute("SELECT arxiv_id, abstract_full FROM papers"))
    assert got["0704.0002"] == " ".join(FULL.split())
    assert got["hep-th/9901001"] is None
    assert got["2607.00001"] is None


def test_a_dry_run_changes_nothing(tmp_path):
    side = _sidecar(tmp_path)
    conn = sqlite3.connect(side)
    assert sfa.add_full_abstracts(conn, str(_csv(tmp_path)), dry_run=True)["filled"] == 1
    assert "abstract_full" not in {r[1] for r in conn.execute("PRAGMA table_info(papers)")}


def test_the_rebuilt_index_finds_authors(tmp_path):
    side = _sidecar(tmp_path)
    conn = sqlite3.connect(side)
    sfa.rebuild_fts(conn)
    hits = conn.execute("SELECT p.arxiv_id FROM papers_fts f JOIN papers p ON p.rowid = f.rowid "
                        "WHERE papers_fts MATCH 'Jiajun Jiang'").fetchall()
    assert hits == [("2607.00001",)]
    conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('integrity-check')")


@pytest.fixture
def served(tmp_path, monkeypatch):
    def point_at(path):
        monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(path))
        monkeypatch.setattr(local_meta, "_conn", None)
        monkeypatch.setattr(local_meta, "_probed", False)
        monkeypatch.setattr(local_meta, "_available", False)
        monkeypatch.setattr(local_meta, "_max_date", None)
        assert local_meta.is_available()
    return point_at


def test_the_reader_serves_the_full_abstract(tmp_path, served):
    side = _sidecar(tmp_path)
    sfa.add_full_abstracts(sqlite3.connect(side), str(_csv(tmp_path)), dry_run=False)
    served(side)
    rows = {r["arxiv_id"]: r for r in local_meta.fetch_rows(["0704.0002", "hep-th/9901001"])}
    assert rows["0704.0002"]["abstract_preview"] == " ".join(FULL.split())
    assert rows["hep-th/9901001"]["abstract_preview"] == STUMP


def test_an_older_sidecar_still_serves_its_stumps(tmp_path, served):
    served(_sidecar(tmp_path))
    assert local_meta.fetch_rows(["0704.0002"])[0]["abstract_preview"] == STUMP
