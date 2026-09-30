"""scripts/refresh_citations.py applies staged counts without inventing zeros."""
import importlib.util
import sqlite3
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    "refresh_citations", Path(__file__).resolve().parents[1] / "scripts" / "refresh_citations.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def _sidecar(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, citation_count INTEGER, influential_citations INTEGER);
        CREATE TABLE paper_categories (code TEXT, arxiv_id TEXT, citation_count INTEGER, update_date TEXT);
        INSERT INTO papers VALUES ('2506.00001', 0, 0), ('2401.00001', 50, 5), ('2401.00002', 7, 1);
        INSERT INTO paper_categories VALUES ('cs.LG','2506.00001',0,''), ('cs.CL','2506.00001',0,''),
                                            ('cs.LG','2401.00001',50,''), ('cs.LG','2401.00002',7,'');
    """)
    db.commit()
    db.close()


def _stage(path):
    db = rc._staging(str(path))
    db.executemany("INSERT INTO citations VALUES (?,?,?,?,?)", [
        ("2506.00001", 1488, 158, 1, "t"),   # new paper, had 0
        ("2401.00001", 50, 5, 1, "t"),       # unchanged
        ("2401.00002", None, None, 0, "t"),  # S2 does not know it
    ])
    db.commit()


def test_apply_sidecar_updates_both_tables_and_keeps_unknown_counts(tmp_path):
    side, stage = tmp_path / "m.sqlite", tmp_path / "c.sqlite"
    _sidecar(side)
    _stage(stage)
    rc.apply_sidecar(SimpleNamespace(sidecar=str(side), staging=str(stage)))
    db = sqlite3.connect(side)
    assert dict(db.execute("SELECT arxiv_id, citation_count FROM papers")) == {
        "2506.00001": 1488, "2401.00001": 50, "2401.00002": 7}
    assert {r[0] for r in db.execute(
        "SELECT citation_count FROM paper_categories WHERE arxiv_id='2506.00001'")} == {1488}


def test_only_changed_known_rows_go_to_turso(tmp_path):
    side, stage = tmp_path / "m.sqlite", tmp_path / "c.sqlite"
    _sidecar(side)
    _stage(stage)
    db = rc._staging(str(stage))
    db.execute(f"ATTACH DATABASE '{side}' AS s")
    assert db.execute(rc._changed(db)).fetchall() == [("2506.00001", 1488, 158)]
