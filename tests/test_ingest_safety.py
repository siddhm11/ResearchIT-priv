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


def test_windows_split_until_each_fits_one_query(monkeypatch):
    # 11,092 papers over 64 days, spread evenly: two halves of ~5.5k fit.
    def probe(cat, since, until, start):
        from datetime import date
        days = (date.fromisoformat(until) - date.fromisoformat(since)).days
        return [], round(11092 * days / 64)
    w = ingest_arxiv.plan_windows("cs.AI", "2026-07-29", "2026-10-01", probe=probe)
    assert w == [["2026-07-29", "2026-08-30"], ["2026-08-30", "2026-10-01"]]


def test_small_or_failed_probe_keeps_one_window():
    assert ingest_arxiv.plan_windows("cs.RO", "2026-07-29", "2026-10-01",
                                     probe=lambda *a: ([], 3273)) == [["2026-07-29", "2026-10-01"]]
    assert ingest_arxiv.plan_windows("cs.RO", "2026-07-29", "2026-10-01",
                                     probe=lambda *a: ([], None)) == [["2026-07-29", "2026-10-01"]]


def test_coverage_checks_every_window(capsys):
    w = [["2026-07-29", "2026-08-30"], ["2026-08-30", "2026-10-01"]]
    k1, k2 = (ingest_arxiv.window_key("cs.AI", w, x) for x in w)
    state = {"cs.AI#windows": w, k1: 5342, f"{k1}#total": 5342, k2: 4000, f"{k2}#total": 5750}
    assert ingest_arxiv.report_coverage(["cs.AI"], state) == 3
    assert "cs.AI@2026-08-30..2026-10-01 4000/5750" in capsys.readouterr().out
    state[k2] = 5750
    assert ingest_arxiv.report_coverage(["cs.AI"], state) == 0
