"""
scripts/baseline_metrics.py: the numbers every later change is compared with.

All on a synthetic interactions table built in tmp_path. The edge cases are the
ones a reader of the report would otherwise have to guess at: where a session
ends, which days count as a return, and what counts as a Pass.
"""
import datetime as dt
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "baseline_metrics", Path(__file__).resolve().parents[1] / "scripts" / "baseline_metrics.py")
bm = importlib.util.module_from_spec(_spec)
sys.modules["baseline_metrics"] = bm   # dataclasses look their module up
_spec.loader.exec_module(bm)

T0 = dt.datetime(2026, 9, 1, 9, 0, tzinfo=dt.timezone.utc)   # a Tuesday


def ev(user, kind, minutes=0, paper="2609.00001", source=None, days=0):
    return bm.Event(user, paper, kind, source, T0 + dt.timedelta(days=days, minutes=minutes))


def make_db(tmp_path, rows):
    path = tmp_path / "baseline.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE interactions ({', '.join(bm.COLUMNS)})")
    for i, (user, kind, ts, source) in enumerate(rows, 1):
        conn.execute("INSERT INTO interactions (id, user_id, paper_id, event_type, source, timestamp) "
                     "VALUES (?, ?, ?, ?, ?, ?)", (i, user, "2609.00001", kind, source, ts))
    conn.commit()
    conn.close()
    return str(path)


def test_a_29_minute_gap_is_one_session_and_31_minutes_is_two():
    one = bm.sessions([ev("a", "save", 0), ev("a", "save", 29)])
    two = bm.sessions([ev("a", "save", 0), ev("a", "save", 31)])
    assert len(one) == 1 and len(two) == 2


def test_sessions_are_per_reader():
    out = bm.sessions([ev("a", "save", 0), ev("b", "save", 1), ev("a", "save", 2)])
    assert sorted(len(s) for s in out) == [1, 2]


def test_a_return_on_day_seven_counts_and_day_eight_does_not():
    seven = bm.compute([ev("a", "view"), ev("a", "view", days=7)], today=T0.date() + dt.timedelta(days=30))
    eight = bm.compute([ev("a", "view"), ev("a", "view", days=8)], today=T0.date() + dt.timedelta(days=30))
    week = bm.iso_week(T0.date())
    assert seven["weeks"][week]["seven_day_return"] == 1.0
    assert eight["weeks"][week]["seven_day_return"] == 0.0


def test_days_whose_return_window_is_still_open_are_left_out():
    out = bm.compute([ev("a", "view")], today=T0.date() + dt.timedelta(days=3))
    assert out["weeks"][bm.iso_week(T0.date())]["seven_day_return"] is None


def test_open_then_save_in_one_session_counts_and_across_sessions_does_not():
    same = bm.compute([ev("a", "click", 0), ev("a", "save", 5)])
    split = bm.compute([ev("a", "click", 0), ev("a", "save", 90)])
    week = bm.iso_week(T0.date())
    assert same["weeks"][week]["open_to_save"] == 1.0
    assert split["weeks"][week]["open_to_save"] == 0.0


def test_saving_a_different_paper_is_not_open_to_save():
    out = bm.compute([ev("a", "click", 0, paper="x"), ev("a", "save", 5, paper="y")])
    assert out["weeks"][bm.iso_week(T0.date())]["open_to_save"] == 0.0


def test_unsave_is_not_a_pass():
    out = bm.compute([ev("a", "save"), ev("a", "unsave", 1), ev("a", "not_interested", 2)])
    week = out["weeks"][bm.iso_week(T0.date())]
    assert week["pass_share"] == 0.5
    assert week["unsaves"] == 1


def test_a_reader_with_600_events_in_a_day_is_excluded():
    bot = [ev("bot", "view", i % 600) for i in range(600)]
    out = bm.compute(bot + [ev("a", "save")])
    assert out["excluded_users"] == 1
    assert out["weeks"][bm.iso_week(T0.date())]["readers"] == 1


def test_clicks_are_split_by_surface_only_from_the_fix_date():
    events = [ev("a", "click", source="recommendation"),
              ev("a", "click", source="search", days=2),
              ev("a", "click", source="recommendation", days=3)]
    out = bm.compute(events, fix_date=T0.date() + dt.timedelta(days=2))
    assert out["weeks"][bm.iso_week(T0.date())]["clicks_by_surface"] == {"search": 1, "recommendation": 1}


def test_an_empty_database_gives_an_empty_report(tmp_path):
    db = make_db(tmp_path, [])
    assert bm.compute(bm.load_events(db)) == {"weeks": {}, "excluded_users": 0}


def test_timestamps_without_a_zone_are_utc(tmp_path):
    db = make_db(tmp_path, [("a", "save", "2026-09-01 23:59:59", None)])
    (e,) = bm.load_events(db)
    assert e.ts == dt.datetime(2026, 9, 1, 23, 59, 59, tzinfo=dt.timezone.utc)


def test_the_report_holds_no_user_ids(tmp_path):
    db = make_db(tmp_path, [("reader-secret-id", "save", "2026-09-01 10:00:00", None),
                            ("reader-secret-id", "click", "2026-09-01 10:05:00", "search")])
    out = tmp_path / "out"
    assert bm.main(["--db", db, "--out", str(out), "--fix-date", "2026-09-01"]) == 0
    files = list(out.iterdir())
    assert {f.suffix for f in files} == {".md", ".json"}
    for f in files:
        assert "reader-secret-id" not in f.read_text()
    data = json.loads(next(f for f in files if f.suffix == ".json").read_text())
    assert data["weeks"]["2026-W36"]["saves"] == 1
    assert "## Definitions" in next(f for f in files if f.suffix == ".md").read_text()
