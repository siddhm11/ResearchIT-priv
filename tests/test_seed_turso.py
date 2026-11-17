"""scripts/seed_turso_from_sidecar.py: rebuilding Turso's papers from the sidecar.

What must hold: every row arrives with its sidecar rowid (new papers must keep the
highest rowids), a re-run after a stop resumes without duplicating anything, licences
come from the listing only for papers we hold, and verify() catches a wrong row.
Turso is stood in for by a local SQLite database behind the same `run()` interface.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import seed_turso_from_sidecar as seed  # noqa: E402


def value(c):
    if c["type"] == "null":
        return None
    return int(c["value"]) if c["type"] == "integer" else c["value"]


class LocalTarget(seed.Target):
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.calls = 0

    def run(self, stmts):
        self.calls += 1
        out = []
        for s in stmts:
            cur = self.db.execute(s["sql"], [value(c) for c in s.get("args", [])])
            out.append([[None if v is None else str(v) for v in row] for row in cur.fetchall()])
        self.db.commit()
        return out


@pytest.fixture
def sidecar(tmp_path):
    path = tmp_path / "side.sqlite"
    db = sqlite3.connect(path)
    db.execute(seed.PAPERS_DDL)
    ids = [f"0704.{i:04d}" for i in range(1, 6)] + [f"2506.{i:05d}" for i in range(1, 2400)] + ["hep-ph/0101001"]
    db.executemany(f"INSERT INTO papers ({', '.join(seed.COLUMNS)}) VALUES (?,?,?,?,?,?,?,?,?)",
                   [(a, f"Title {a}", "A B", "Abstract " * 5, "cs.LG", "Machine learning", "2026-01-01", i, 0)
                    for i, a in enumerate(ids)])
    db.commit()
    return path, ids


def test_seeds_every_row_with_its_rowid_and_resumes(sidecar, tmp_path):
    path, ids = sidecar
    side = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    target = LocalTarget(tmp_path / "turso.sqlite")
    assert seed.seed_papers(side, target, dry_run=False) == len(ids)
    got = target.db.execute("SELECT rowid, arxiv_id FROM papers ORDER BY rowid").fetchall()
    assert got == list(side.execute("SELECT rowid, arxiv_id FROM papers ORDER BY rowid"))
    assert seed.verify(side, target, n=50) == []
    # A stop part-way: drop the last 1,000 rows, run again, nothing doubles.
    target.db.execute("DELETE FROM papers WHERE rowid > ?", (len(ids) - 1000,))
    target.db.commit()
    assert seed.seed_papers(side, target, dry_run=False) == 1000
    assert target.db.execute("SELECT COUNT(*), MAX(rowid) FROM papers").fetchone() == (len(ids), len(ids))


def test_verify_catches_a_wrong_row(sidecar, tmp_path):
    path, ids = sidecar
    side = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    target = LocalTarget(tmp_path / "turso.sqlite")
    seed.seed_papers(side, target, dry_run=False)
    target.db.execute("UPDATE papers SET citation_count = 999")
    target.db.commit()
    assert any("differs" in p for p in seed.verify(side, target, n=20))


def test_licences_only_for_held_papers_from_the_listing(sidecar, tmp_path):
    path, ids = sidecar
    listing = tmp_path / "listing.json"
    with open(listing, "w") as f:
        for a in ["2506.00001", "2506.00002", "2507.99999", "0704.0001"]:
            f.write(json.dumps({"id": a, "license": "http://creativecommons.org/licenses/by/4.0/"}) + "\n")
        f.write(json.dumps({"id": "2506.00003", "license": None}) + "\n")
    side = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    target = LocalTarget(tmp_path / "turso.sqlite")
    assert seed.seed_licences(side, target, str(listing), "2506", dry_run=False) == 2
    rows = target.db.execute("SELECT arxiv_id, licence, licence_url FROM paper_licences ORDER BY arxiv_id").fetchall()
    assert [r[0] for r in rows] == ["2506.00001", "2506.00002"] and rows[0][1] == "CC-BY-4.0"
