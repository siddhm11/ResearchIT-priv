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
    assert puts[0]["points"][0]["id"] == ingest_backends.point_id("2608.00002")
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


class FakeArxiv:
    """Papers spread evenly over days; fails pages that pass arXiv's 10k cap."""
    def __init__(self, per_day: dict, fail=()):
        self.per_day, self.fail, self.calls = per_day, set(fail), []

    def __call__(self, cat, since, until, start):
        from datetime import date, timedelta
        self.calls.append((cat, since, until, start))
        if cat in self.fail:
            return [], None
        a, b = date.fromisoformat(since), date.fromisoformat(until)
        ids = [f"{cat}:{(a + timedelta(d)).isoformat()}:{i}"
               for d in range((b - a).days) for i in range(self.per_day[cat])]
        if start + 200 > 10_000 and len(ids) > 10_000:
            return [], None                                   # the real API's 500
        return [{"arxiv_id": x} for x in ids[start:start + 200]], len(ids)


def _walk(fake, cats, state=None):
    state = {} if state is None else state
    handled = []
    seen, new = ingest_arxiv.walk(cats, "2026-07-29", "2026-10-01", state,
                                  lambda ps: handled.extend(p["arxiv_id"] for p in ps) or len(ps),
                                  lambda: None, fetch=fake, delay=0, log=lambda m: None)
    return state, handled, seen, new


def test_small_category_costs_only_its_own_pages():
    fake = FakeArxiv({"cs.RO": 10})                           # 640 papers, 4 pages
    state, handled, seen, new = _walk(fake, ["cs.RO"])
    assert len(fake.calls) == 4 and seen == new == 640 == len(set(handled))
    assert ingest_arxiv.report_coverage(["cs.RO"], state) == 0


def test_oversized_window_splits_before_processing_and_covers_everything():
    fake = FakeArxiv({"cs.AI": 175})                          # 11,200 papers in 64 days
    state, handled, seen, _ = _walk(fake, ["cs.AI"])
    assert state["cs.AI#windows"] == [["2026-07-29", "2026-08-30"], ["2026-08-30", "2026-10-01"]]
    assert seen == len(handled) == len(set(handled)) == 11_200   # nothing twice, nothing lost
    assert ingest_arxiv.report_coverage(["cs.AI"], state) == 0


def test_failed_requests_leave_a_coverage_gap():
    fake = FakeArxiv({"cs.CL": 5, "cs.CV": 5}, fail={"cs.CL"})
    state, _h, _s, _n = _walk(fake, ["cs.CL", "cs.CV"])
    assert ingest_arxiv.report_coverage(["cs.CL", "cs.CV"], state) == 3


def test_resumed_run_keeps_the_recorded_split():
    fake = FakeArxiv({"cs.AI": 175})
    state, _h, _s, _n = _walk(fake, ["cs.AI"])
    fake2 = FakeArxiv({"cs.AI": 175})
    _st, handled, seen, _n = _walk(fake2, ["cs.AI"], state=state)
    assert seen == 0 and all(c[3] > 0 for c in fake2.calls)      # resumes at the end, no re-split


def test_coverage_checks_every_window(capsys):
    w = [["2026-07-29", "2026-08-30"], ["2026-08-30", "2026-10-01"]]
    k1, k2 = (ingest_arxiv.window_key("cs.AI", w, x) for x in w)
    state = {"cs.AI#windows": w, k1: 5342, f"{k1}#total": 5342, k2: 4000, f"{k2}#total": 5750}
    assert ingest_arxiv.report_coverage(["cs.AI"], state) == 3
    assert "cs.AI@2026-08-30..2026-10-01 4000/5750" in capsys.readouterr().out
    state[k2] = 5750
    assert ingest_arxiv.report_coverage(["cs.AI"], state) == 0


def test_point_ids_are_stable_disjoint_and_fit_uint64():
    pid = ingest_backends.point_id
    assert pid("2609.01004") == pid("2609.01004")
    ids = [pid(f"{yymm}.{n:05d}") for yymm in ("2608", "2609", "2610") for n in range(100_000)]
    assert len(set(ids)) == len(ids)                      # 300k ids, no collision
    assert min(ids) >= 1 << 62 > 300_000                  # never meets sequential ids
    assert max(ids) < 1 << 63                             # Qdrant ids are unsigned 64-bit


def test_rate_limits_back_off_for_minutes_and_honour_retry_after():
    import urllib.error
    def http(code, retry_after=None):
        headers = {"Retry-After": retry_after} if retry_after else {}
        return urllib.error.HTTPError("u", code, "x", headers, None)
    assert ingest_arxiv.retry_wait(http(429), 0) == 30
    assert ingest_arxiv.retry_wait(http(429), 2) == 120
    assert ingest_arxiv.retry_wait(http(429), 5) == 300           # capped
    assert ingest_arxiv.retry_wait(http(503, "45"), 0) == 45      # server's own advice
    assert ingest_arxiv.retry_wait(TimeoutError("x"), 0) == 5     # transient: retry soon


def test_fetch_page_retries_a_rate_limit_then_succeeds(monkeypatch):
    import io, urllib.error
    calls, waits = [], []
    feed = (b'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            b'<opensearch:totalResults>0</opensearch:totalResults></feed>')

    def urlopen(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return io.BytesIO(feed)

    monkeypatch.setattr(ingest_arxiv.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ingest_arxiv.time, "sleep", waits.append)
    assert ingest_arxiv.fetch_page("cs.RO", "2026-09-28", "2026-10-02", 0) == ([], 0)
    assert waits == [30, 60] and len(calls) == 3
