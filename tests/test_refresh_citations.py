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


def test_only_changed_known_rows_go_to_turso():
    current = {"2506.00001": (0, 0), "2401.00001": (50, 5), "2401.00002": (7, 1)}
    staged = [("2506.00001", 1488, 158, 1),   # was 0: write
              ("2401.00001", 50, 5, 1),       # unchanged: skip
              ("2401.00002", None, None, 0),  # unknown to S2: keep stored count
              ("2609.99999", 3, 0, 1)]        # not in Turso: not an update
    assert rc.diff_against(current, staged) == [("2506.00001", 1488, 158)]


def test_fetch_stages_every_batch_in_order_with_workers(tmp_path, monkeypatch):
    side, stage = tmp_path / "m.sqlite", tmp_path / "c.sqlite"
    db = sqlite3.connect(side)
    db.executescript("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, citation_count INTEGER, influential_citations INTEGER);")
    db.executemany("INSERT INTO papers VALUES (?,0,0)", [(f"2601.{i:05d}",) for i in range(1200)])
    db.commit()
    monkeypatch.setattr(rc, "BATCH", 100)
    monkeypatch.setitem(rc._KEY, "value", "k")
    monkeypatch.setattr(rc, "_KEYED_INTERVAL_S", 0.0)
    calls = []

    def fake_post(client, ids):
        calls.append(ids[0])
        return [{"citationCount": 1, "influentialCitationCount": 0} for _ in ids]

    monkeypatch.setattr(rc, "_post", fake_post)
    assert rc.fetch(SimpleNamespace(sidecar=str(side), staging=str(stage), since="", workers=4)) == 0
    got = sqlite3.connect(stage).execute("SELECT COUNT(*), SUM(found) FROM citations").fetchone()
    assert got == (1200, 1200) and len(calls) == 12


def test_fetch_stops_cleanly_on_a_failed_batch(tmp_path, monkeypatch):
    side, stage = tmp_path / "m.sqlite", tmp_path / "c.sqlite"
    db = sqlite3.connect(side)
    db.executescript("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, citation_count INTEGER, influential_citations INTEGER);")
    db.executemany("INSERT INTO papers VALUES (?,0,0)", [(f"2601.{i:05d}",) for i in range(1000)])
    db.commit()
    monkeypatch.setattr(rc, "BATCH", 100)
    monkeypatch.setitem(rc._KEY, "value", "k")
    monkeypatch.setattr(rc, "_KEYED_INTERVAL_S", 0.0)
    calls = []

    def fake_post(client, ids):
        calls.append(ids[0])
        return None if len(calls) == 3 else [{"citationCount": 1} for _ in ids]

    monkeypatch.setattr(rc, "_post", fake_post)
    assert rc.fetch(SimpleNamespace(sidecar=str(side), staging=str(stage), since="", workers=2)) == 1
    staged = sqlite3.connect(stage).execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    assert staged == 200          # the two batches before the failure, in order
    assert len(calls) <= 5        # nothing like the full 10 batches was queued
