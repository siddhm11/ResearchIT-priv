"""scripts/append_to_sidecar.py keeps papers, categories and the FTS index in step."""
import importlib.util
import sqlite3
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "append_to_sidecar", Path(__file__).resolve().parents[1] / "scripts" / "append_to_sidecar.py")
ats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ats)

SCHEMA = """
CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT, abstract_preview TEXT,
  categories TEXT, primary_topic TEXT, update_date TEXT, citation_count INTEGER, influential_citations INTEGER);
CREATE TABLE paper_categories (code TEXT NOT NULL, arxiv_id TEXT NOT NULL, citation_count INTEGER, update_date TEXT);
CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract_preview, content='papers', content_rowid='rowid',
  tokenize='porter unicode61');
"""


def _row(aid, title, cats="cs.RO cs.LG"):
    return {"arxiv_id": aid, "title": title, "authors": "A. Author", "abstract_preview": "An abstract " * 10,
            "categories": cats, "primary_topic": "Robotics", "update_date": "2026-09-01",
            "citation_count": "3", "influential_citations": "0"}


def _sidecar(tmp_path):
    conn = sqlite3.connect(tmp_path / "m.sqlite")
    conn.executescript(SCHEMA)
    ats.append_rows(conn, [_row("2607.00001", "Old legged locomotion"), _row("2607.00002", "Old grasping")])
    return conn


def test_new_papers_get_categories_and_are_searchable(tmp_path):
    conn = _sidecar(tmp_path)
    stats = ats.append_rows(conn, [_row("2609.00001", "Quadruped parkour policies", "cs.RO"),
                                   _row("2609.00002", "Humanoid whole body control")])
    assert stats == {"inserted": 2, "skipped_existing": 0, "category_rows": 3}
    hits = conn.execute("SELECT p.arxiv_id FROM papers_fts f JOIN papers p ON p.rowid = f.rowid "
                        "WHERE papers_fts MATCH 'parkour'").fetchall()
    assert hits == [("2609.00001",)]
    assert ats.verify(conn)["fts_integrity"] == "ok"
    assert ats.verify(conn)["papers_without_categories"] == 0


def test_existing_paper_is_skipped_not_renumbered(tmp_path):
    conn = _sidecar(tmp_path)
    before = conn.execute("SELECT rowid FROM papers WHERE arxiv_id='2607.00001'").fetchone()
    stats = ats.append_rows(conn, [_row("2607.00001", "Replaced title")])
    assert stats["skipped_existing"] == 1 and stats["inserted"] == 0
    assert conn.execute("SELECT rowid, title FROM papers WHERE arxiv_id='2607.00001'").fetchone() == (before[0], "Old legged locomotion")
    assert ats.verify(conn)["fts_integrity"] == "ok"
