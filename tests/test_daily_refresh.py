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


def test_history_rows_keep_the_count_being_overwritten():
    current = {"a": (5, 1), "b": (0, 0)}
    assert dr.history_rows(current, [("b", 12, 2)], "2026-10-24") == [("b", "2026-10-24", 0, 12, 0, 2)]


def _run(conn, stmt):
    conn.execute(stmt["sql"], [int(a["value"]) if a["type"] == "integer" else a["value"]
                               for a in stmt.get("args", [])])


def test_history_upsert_spans_the_whole_day_and_sums_to_the_growth():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute(dr.HISTORY_DDL)
    _run(conn, dr.history_insert([("b", "2026-10-24", 0, 12, 0, 2), ("c", "2026-10-24", 3, 4, 0, 0)]))
    _run(conn, dr.history_insert([("b", "2026-10-24", 12, 15, 2, 3)]))     # second run, same day
    _run(conn, dr.history_insert([("b", "2026-10-25", 15, 16, 3, 3)]))
    assert conn.execute("SELECT citations_before, citations FROM citation_history "
                        "WHERE arxiv_id = 'b' ORDER BY checked_on").fetchall() == [(0, 15), (15, 16)]
    gain = conn.execute("SELECT SUM(citations - citations_before) FROM citation_history "
                        "WHERE arxiv_id = 'b'").fetchone()[0]
    assert gain == 16


def _run_batch(conn, batch):
    """What Turso does with a Hrana batch: a step runs only if its condition holds."""
    import sqlite3
    ok = []

    def holds(cond):
        if cond is None:
            return True
        if cond["type"] == "ok":
            return ok[cond["step"]]
        if cond["type"] == "not":
            return not holds(cond["cond"])
        raise ValueError(cond)

    for step in batch["steps"]:
        ran = holds(step.get("condition"))
        if ran:
            try:
                _run(conn, step["stmt"])
            except sqlite3.Error:
                ran = False
        ok.append(ran)


def test_a_failed_statement_rolls_back_the_whole_batch():
    import sqlite3
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute(dr.HISTORY_DDL)
    row = dr.history_insert([("b", "2026-10-24", 0, 12, 0, 2)])
    _run_batch(conn, dr.tx_batch([row, {"sql": "UPDATE no_such_table SET x = 1"}]))
    assert conn.execute("SELECT COUNT(*) FROM citation_history").fetchone()[0] == 0
    assert not conn.in_transaction
    _run_batch(conn, dr.tx_batch([row]))
    assert conn.execute("SELECT COUNT(*) FROM citation_history").fetchone()[0] == 1


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
    # a catch-up day reaches the script through env, never pasted into the shell
    assert "TODAY: ${{ inputs.today }}" in text and '${TODAY:+--today "$TODAY"}' in text
    assert "inputs.today }} \\" not in text
    # catch-up runs can raise the ingest limit; the job ceiling sits above it
    assert "INGEST_TIMEOUT_MIN: ${{ inputs.ingest_timeout_min || '60' }}" in text
    assert '--ingest-timeout-min "$INGEST_TIMEOUT_MIN"' in text
    assert re.search(r"(?m)^    timeout-minutes: 300$", text)


def test_ingest_that_overruns_fails_the_run_with_its_output(monkeypatch, capsys):
    import subprocess
    import pytest

    def overrun(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ingest", timeout=60, output="partial log", stderr="Thread 0x1 stack")

    monkeypatch.setattr(dr.subprocess, "run", overrun)
    with pytest.raises(SystemExit, match="exceeded 1 min"):
        dr.stage_ingest("2026-09-28", "2026-10-02", timeout_s=60)
    assert "Thread 0x1 stack" in capsys.readouterr().out


def test_the_daily_ingest_uses_the_oai_harvest(monkeypatch):
    import subprocess
    seen = []

    def run(cmd, **k):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "done: examined 1,204, new 873 in 9.1 min\n", "")

    monkeypatch.setattr(dr.subprocess, "run", run)
    assert dr.stage_ingest("2026-10-17", "2026-10-21")[0] == 873
    cmd = seen[0]
    assert cmd[cmd.index("--source") + 1] == "oai"
    assert cmd[cmd.index("--since") + 1] == "2026-10-17" and cmd[cmd.index("--until") + 1] == "2026-10-21"


def test_licence_coverage_counts_new_papers_with_a_licence(monkeypatch):
    monkeypatch.setenv("FIGURE_LICENCES", "CC-BY-4.0,CC0-1.0")
    got = dr.licence_coverage(["CC-BY-4.0", "ARXIV-NONEXCLUSIVE-1.0", "CC-BY-NC-SA-4.0", None, "CC0-1.0"])
    assert got["new_papers"] == 5 and got["with_licence"] == 4 and got["with_licence_share"] == 0.8
    assert got["open"] == 3 and got["allow_figures"] == 2 and got["allow_figures_share"] == 0.4
    assert dr.licence_coverage([])["with_licence_share"] is None      # a quiet day is not an error


def _hf_record(aid, upvotes, title="A paper", **paper):
    return {"numComments": 2, "paper": {"id": aid, "title": title, "upvotes": upvotes, **paper}}


def test_hf_rows_rank_by_upvotes_and_drop_bad_records():
    payload = [_hf_record("2610.00001v2", 5, ai_keywords=[" video world models ", "", 3],
                          githubRepo="https://github.com/a/b", githubStars=40, projectPage="javascript:x"),
               _hf_record("2610.00002", 90, title="  Two\n lines  ", ai_summary="One line.",
                          projectPage="https://example.org"),
               _hf_record("2610.00001", 7),                      # same paper, another version
               _hf_record("not-an-id", 50), {"paper": "broken"}, _hf_record("2610.00003", "many")]
    papers, history = dr.hf_rows(payload, "2026-10-22", "2026-10-24")
    assert [p[0] for p in papers] == ["2610.00001", "2610.00002", "2610.00003"]
    assert papers[0][3] == '["video world models"]' and papers[0][5] == "https://github.com/a/b"
    assert papers[0][6] == "" and papers[1][6] == "https://example.org"
    assert papers[1][2] == "Two lines" and papers[1][4] == "One line."
    assert [(h[2], h[5], h[6]) for h in history] == [(5, 2, 3), (90, 1, 3), (0, 3, 3)]
    assert history[0][4] == 40 and history[0][3] == 2


def test_hf_history_keeps_the_first_listing_and_the_latest_count_of_the_day():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute(dr.HF_PAPERS_DDL)
    conn.execute(dr.HF_HISTORY_DDL)

    def record(listed_on, checked_on, upvotes):
        papers, history = dr.hf_rows([_hf_record("2610.00001", upvotes)], listed_on, checked_on)
        for stmt in dr.hf_statements(papers, history):
            _run(conn, stmt)

    record("2026-10-22", "2026-10-22", 10)
    record("2026-10-22", "2026-10-22", 25)      # a second run the same day
    record("2026-10-23", "2026-10-23", 60)      # listed again the next day: keeps the first
    assert conn.execute("SELECT listed_on, first_seen FROM hf_daily_papers").fetchall() == [
        ("2026-10-22", "2026-10-22")]
    assert conn.execute("SELECT checked_on, upvotes FROM hf_upvote_history ORDER BY checked_on").fetchall() == [
        ("2026-10-22", 25), ("2026-10-23", 60)]


def test_hf_stage_skips_a_failed_day_and_never_fails_the_run():
    import sqlite3
    import httpx

    conn = sqlite3.connect(":memory:")

    class FakeStores:
        def t(self, sql, args=()):
            conn.execute(sql, args)
            return []

        def t_many(self, stmts):
            for s in stmts:
                _run(conn, s)

    def handler(request):
        day = request.url.params["date"]
        assert request.url.params["limit"] == "100"
        if request.url.params["p"] != "0":
            return httpx.Response(200, json=[])
        if day == "2026-10-23":
            return httpx.Response(503)
        if day == "2026-10-22":
            return httpx.Response(200, json=[])                 # a quiet day
        return httpx.Response(200, json=[_hf_record("2610.00001", 12), _hf_record("2610.00002", 30)])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    got = dr.stage_hf_daily(FakeStores(), dt.date(2026, 10, 24), days=3, client=client)
    assert got == {"days": 3, "lists": 1, "empty": 1, "papers": 2, "failed": ["2026-10-23"]}
    assert conn.execute("SELECT arxiv_id, day_rank FROM hf_upvote_history ORDER BY day_rank").fetchall() == [
        ("2610.00002", 1), ("2610.00001", 2)]


def test_hf_list_pages_past_the_hundred_paper_limit():
    import httpx

    def handler(request):
        page = int(request.url.params["p"])
        return httpx.Response(200, json=[_hf_record(f"2609.{page}{i:04d}", 1) for i in range([3, 1, 0][page])])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert len(dr._hf_list(client, "2026-09-29", page_size=3)) == 4
