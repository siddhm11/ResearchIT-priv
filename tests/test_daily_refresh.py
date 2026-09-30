"""Pure logic of scripts/daily_refresh.py."""
import datetime as dt
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "daily_refresh", Path(__file__).resolve().parents[1] / "scripts" / "daily_refresh.py")
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)


def test_window_overlaps_and_includes_today():
    assert dr.window(dt.date(2026, 10, 1), 3) == ("2026-09-28", "2026-10-02")
    assert dr.window(dt.date(2026, 1, 1), 3) == ("2025-12-29", "2026-01-02")


def test_recent_prefix_wraps_years():
    assert dr.recent_prefix(dt.date(2026, 10, 1), 3) == "2608"
    assert dr.recent_prefix(dt.date(2026, 2, 15), 3) == "2512"
    assert dr.recent_prefix(dt.date(2026, 1, 5), 1) == "2601"


def test_ingest_count_is_parsed_from_the_last_summary():
    out = "  cs.CV  200/300 ...\ndone: examined 1,204, new 873 in 4.2 min\ncoverage: all 43 categories"
    assert dr.parse_ingest_new(out) == 873
    assert dr.parse_ingest_new("no summary") is None


def test_only_changed_known_counts_are_written():
    current = {"a": (5, 1), "b": (0, 0), "c": (7, 0), "d": (3, 0)}
    fetched = {"a": {"citationCount": 5, "influentialCitationCount": 1},   # unchanged
               "b": {"citationCount": 12, "influentialCitationCount": 2},  # changed
               "c": None,                                                   # unknown to S2: keep
               "d": {"citationCount": None}}                                # no count: keep
    assert dr.changed_counts(current, fetched) == [("b", 12, 2)]


def test_verify_counts_catches_mismatches():
    before = {"recent_points": 100, "turso_rows": 1000, "turso_max_rowid": 1000}
    ok = {"recent_points": 150, "turso_rows": 1050, "turso_max_rowid": 1050}
    assert dr.verify_counts(before, ok, 50) == []
    assert "Qdrant" in dr.verify_counts(before, {**ok, "recent_points": 149}, 50)[0]
    assert "reported" in dr.verify_counts(before, ok, 49)[0]
    assert "contiguous" in dr.verify_counts(before, {**ok, "turso_max_rowid": 1051}, 50)[0]


def test_workflow_is_gated_serialised_and_passes_every_required_secret():
    """Plain-text checks: CI deliberately has no YAML parser installed."""
    import re
    text = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily-refresh.yml").read_text()
    assert re.search(r"(?m)^    if: vars\.DAILY_REFRESH == 'on'$", text)      # merging must not start it
    assert re.search(r"(?m)^  group: daily-refresh$", text)
    assert re.search(r"(?m)^  cancel-in-progress: false$", text)
    assert re.search(r'(?m)^    - cron: "30 2 \* \* \*"$', text)
    assert re.search(r"(?m)^  contents: read$", text)
    for key in dr.REQUIRED_ENV + ("S2_API_KEY",):
        assert re.search(rf"(?m)^          {key}: \$\{{\{{ secrets\.{key} \}}\}}$", text), key
