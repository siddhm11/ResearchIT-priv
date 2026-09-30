"""Safety properties of scripts/ingest_arxiv.py and scripts/ingest_backends.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ingest_arxiv  # noqa: E402
import ingest_backends  # noqa: E402


def test_topic_label_matches_existing_rows():
    assert ingest_backends.topic_label("cs.CL cs.LG") == "NLP/Computational Linguistics"
    assert ingest_backends.topic_label("cs.RO") == "Robotics"
    assert ingest_backends.topic_label("zz.XX") == "Other Technical"
    assert ingest_backends.topic_label("") == "Other Technical"


def _upserter(monkeypatch, present):
    up = object.__new__(ingest_backends.Upserter)
    up._next_id = 500
    calls = {"q": [], "z": [], "t": []}

    def fake_q(path, body=None, method="GET", timeout=180):
        calls["q"].append((path, method, body))
        if path.endswith("/points/scroll"):
            return {"result": {"points": [{"payload": {"arxiv_id": a}} for a in present]}}
        return {}

    monkeypatch.setattr(up, "_q", fake_q, raising=False)
    monkeypatch.setattr(up, "_z", lambda *a, **k: calls["z"].append(a), raising=False)
    monkeypatch.setattr(up, "_t", lambda stmts, **k: calls["t"].append(stmts), raising=False)
    return up, calls


def _papers(*ids):
    return [{"arxiv_id": i, "title": f"T {i}", "authors": "A", "abstract": "x" * 40,
             "categories": "cs.CV cs.LG", "primary_topic": "cs.CV", "update_date": "2026-08-01"} for i in ids]


def test_papers_already_in_qdrant_are_not_written_again(monkeypatch):
    """A rerun after a Qdrant-ok/Turso-failed batch must not duplicate vectors."""
    up, calls = _upserter(monkeypatch, present={"2608.00001"})
    up.upsert(_papers("2608.00001", "2608.00002"), [([0.1] * 4, {1: 0.5})] * 2)
    puts = [b for p, m, b in calls["q"] if m == "PUT"]
    assert [pt["payload"]["arxiv_id"] for pt in puts[0]["points"]] == ["2608.00002"]
    assert puts[0]["points"][0]["id"] == 500 and up._next_id == 501
    # Turso still gets both rows, so the resume marker catches up.
    assert len(calls["t"][0]) == 2


def test_nothing_new_means_no_qdrant_write(monkeypatch):
    up, calls = _upserter(monkeypatch, present={"2608.00001"})
    up.upsert(_papers("2608.00001"), [([0.1] * 4, {})])
    assert not [1 for _p, m, _b in calls["q"] if m == "PUT"]


def test_turso_rows_carry_the_friendly_label_and_zilliz_is_skipped(monkeypatch):
    monkeypatch.setattr(ingest_backends, "WRITE_ZILLIZ", False)
    up, calls = _upserter(monkeypatch, present=set())
    up.upsert(_papers("2608.00003"), [([0.1] * 4, {1: 0.5})])
    label = calls["t"][0][0]["args"][5]["value"]
    assert label == "Computer Vision"
    assert calls["z"] == []


def test_coverage_report_flags_short_and_unknown_categories(capsys):
    state = {"cs.CV": 100, "cs.CV#total": 100, "cs.RO": 40, "cs.RO#total": 90,
             "econ.TH": 0, "econ.TH#total": 0}
    assert ingest_arxiv.report_coverage(["cs.CV", "econ.TH"], state) == 0
    assert ingest_arxiv.report_coverage(["cs.CV", "cs.RO", "cs.SE"], state) == 3
    out = capsys.readouterr().out
    assert "cs.RO 40/90" in out and "cs.SE 0/None" in out


def test_failed_fetch_reports_unknown_total(monkeypatch):
    def boom(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(ingest_arxiv.urllib.request, "urlopen", boom)
    monkeypatch.setattr(ingest_arxiv.time, "sleep", lambda s: None)
    assert ingest_arxiv.fetch_page("cs.CV", "2026-08-01", "2026-08-02", 0) == ([], None)
