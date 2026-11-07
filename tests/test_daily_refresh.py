"""Pure logic of scripts/daily_refresh.py."""
import datetime as dt
import importlib.util
import sqlite3
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
    before = {"recent_points": 100, "turso_max_rowid": 1000}
    ok = {"recent_points": 150, "turso_max_rowid": 1050, "turso_new_rows": 50}
    assert dr.verify_counts(before, ok, 50) == []
    assert "Qdrant" in dr.verify_counts(before, {**ok, "recent_points": 149}, 50)[0]
    assert "reported" in dr.verify_counts(before, ok, 49)[0]
    assert "contiguous" in dr.verify_counts(before, {**ok, "turso_max_rowid": 1051}, 50)[0]
    # A paper written twice moves to a new rowid: a Turso row with no Qdrant point.
    replaced = {"recent_points": 150, "turso_max_rowid": 1051, "turso_new_rows": 51}
    assert "Qdrant" in dr.verify_counts(before, replaced, 50)[0]


def test_counts_never_scan_the_whole_papers_table():
    """COUNT(*) over papers reads ~2M rows on Turso's metered free plan; only the
    highest rowid and the rows above a given one may be read."""
    sent = []

    class Fake(dr.Stores):
        def __init__(self):
            pass

        def q(self, path, body=None):
            return {"count": 7}

        def t(self, sql, args=()):
            sent.append((sql, tuple(args)))
            return [[1000]] if "MAX(rowid)" in sql else [[3]]

    st = Fake()
    assert st.counts() == {"recent_points": 7, "turso_max_rowid": 1000}
    assert st.counts(since_rowid=997)["turso_new_rows"] == 3
    whole = [sql for sql, args in sent if "COUNT(*)" in sql and "WHERE rowid >" not in sql]
    assert not whole, whole


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
    # the input default and the scheduled run's agree; the ingest stops itself inside the ceiling
    assert "INGEST_TIMEOUT_MIN: ${{ inputs.ingest_timeout_min || '240' }}" in text
    assert re.search(r'(?m)^        description: "Stop the ingest after this long[^"]*"\n        default: "240"$', text)
    assert '--ingest-timeout-min "$INGEST_TIMEOUT_MIN"' in text
    ceiling = int(re.search(r"(?m)^    timeout-minutes: (\d+)$", text)[1])
    assert 240 + dr.INGEST_OVERRUN_S / 60 + 45 <= ceiling <= 360


def _fake_ingest(monkeypatch, tmp_path, body: str):
    """A real child process in place of ingest_arxiv.py: the real command line, the
    real environment and real exit codes, without the model or the stores."""
    (tmp_path / "ingest_arxiv.py").write_text(
        "import argparse, json, os, sys, time\n"
        "ap = argparse.ArgumentParser()\n"
        "for a in ('--source', '--since', '--until', '--state', '--time-limit-min'):\n"
        "    ap.add_argument(a)\n"
        "args = ap.parse_args()\n"
        "def save(**k):\n"
        "    json.dump(k, open(args.state, 'w'))\n" + body)
    monkeypatch.setattr(dr, "HERE", tmp_path)


def test_the_daily_ingest_uses_the_oai_harvest_within_its_limit(monkeypatch, tmp_path):
    _fake_ingest(monkeypatch, tmp_path,
                 "assert args.source == 'oai' and args.since == '2026-10-17' and args.until == '2026-10-21'\n"
                 "assert args.time_limit_min == '90.0'\n"
                 "assert os.environ['INGEST_STACK_FILE'].endswith('stacks.txt')\n"
                 "save(**{'oai:cs': 'done', 'oai_failed_tries': 2, 'result': {'examined': 1204, 'new': 873}})\n")
    assert dr.stage_ingest("2026-10-17", "2026-10-21", 5400) == {"new_papers": 873, "failed_tries": 2, "gap": None}


def test_a_coverage_gap_names_the_sets_and_keeps_the_count(monkeypatch, tmp_path):
    _fake_ingest(monkeypatch, tmp_path,
                 "save(**{'oai:cs': 'unfinished 4/9', 'oai:econ': 'done', 'oai:stat': 'failed',\n"
                 "        'oai_failed_tries': 14, 'result': {'examined': 64, 'new': 24}})\n"
                 "sys.exit(3)\n")
    got = dr.stage_ingest("2026-11-03", "2026-11-07", 60)
    assert got == {"new_papers": 24, "failed_tries": 14,
                   "gap": "ingest incomplete: cs=unfinished 4/9, stat=failed"}


def test_a_crash_is_a_gap_with_no_count(monkeypatch, tmp_path):
    _fake_ingest(monkeypatch, tmp_path, "sys.exit(1)\n")
    assert dr.stage_ingest("2026-11-03", "2026-11-07", 60) == {
        "new_papers": None, "failed_tries": 0, "gap": "ingest crashed with exit 1"}


def test_an_ingest_that_ignores_its_limit_is_killed_with_its_stacks(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dr, "INGEST_OVERRUN_S", 1)
    _fake_ingest(monkeypatch, tmp_path,
                 "open(os.environ['INGEST_STACK_FILE'], 'w').write('Thread 0x1 stack')\n"
                 "time.sleep(60)\n")
    got = dr.stage_ingest("2026-11-03", "2026-11-07", 1)
    assert got["new_papers"] is None and "killed" in got["gap"]
    assert "Thread 0x1 stack" in capsys.readouterr().out


def test_ingest_output_reaches_the_log_while_it_runs(monkeypatch, tmp_path):
    """It used to be captured and printed at the end, so nobody could follow a run,
    and a killed run lost it. The child now writes straight to the job's stdout."""
    seen = {}

    def run(cmd, **k):
        seen.update(k)
        return dr.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(dr.subprocess, "run", run)
    dr.stage_ingest("2026-11-03", "2026-11-07", 60)
    assert "capture_output" not in seen and "stdout" not in seen and "stderr" not in seen


def test_catch_up_reaches_back_past_the_lookback_only_when_behind():
    assert dr.catch_up_since("2026-11-03", None) == "2026-11-03"              # nothing recorded yet
    assert dr.catch_up_since("2026-11-03", "2026-11-05") == "2026-11-03"      # a normal day
    assert dr.catch_up_since("2026-11-10", "2026-11-04") == "2026-11-03"      # six bad days
    assert dr.catch_up_since("2026-11-03", "2026-11-04") == "2026-11-03"      # the overlap day itself


class _SqlStores:
    """Stores.t against a real SQLite, so the SQL itself is under test."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:")

    def t(self, sql, args=()):
        try:
            return [list(r) for r in self.db.execute(sql, args).fetchall()]
        except sqlite3.Error as e:
            raise RuntimeError(str(e)) from e


def test_the_watermark_starts_unknown_and_only_moves_forward():
    st = _SqlStores()
    assert dr.read_complete_through(st) is None           # no table yet: not an error
    dr.record_complete_through(st, "2026-11-04")
    assert dr.read_complete_through(st) == "2026-11-04"
    dr.record_complete_through(st, "2026-11-06")
    dr.record_complete_through(st, "2026-10-07")          # a manual run for a past day
    assert dr.read_complete_through(st) == "2026-11-06"


def _main_with(monkeypatch, ingest, *, citations=None, argv=("--today", "2026-11-10")):
    """dr.main() with every stage stubbed but the order, the problems and the exit code real."""
    st = _SqlStores()
    st.counts = lambda since_rowid=None: {"recent_points": 10, "turso_max_rowid": 5,
                                          **({} if since_rowid is None else {"turso_new_rows": 0})}
    calls = {"citations": 0, "since": None}

    def stage_ingest(since, until, limit):
        calls["since"] = since
        return ingest

    def stage_citations(s, prefix):
        calls["citations"] += 1
        if citations:
            raise citations
        return {"changed": 0}

    for k, v in dict(Stores=lambda: st, stage_ingest=stage_ingest, stage_citations=stage_citations,
                     stage_hf_daily=lambda *a, **k: {}, stage_hn=lambda *a, **k: {},
                     stage_licences=lambda *a, **k: {}, stage_sample_new=lambda *a, **k: []).items():
        monkeypatch.setattr(dr, k, v)
    for k in dr.REQUIRED_ENV:
        monkeypatch.setenv(k, "x")
    monkeypatch.setattr(dr.sys, "argv", ["daily_refresh.py", *argv])
    return dr.main(), st, calls


OK_INGEST = {"new_papers": 0, "failed_tries": 0, "gap": None}


def test_a_complete_run_moves_the_watermark(monkeypatch, capsys):
    code, st, calls = _main_with(monkeypatch, OK_INGEST)
    assert code == 0 and dr.read_complete_through(st) == "2026-11-10"
    assert calls["since"] == "2026-11-07" and calls["citations"] == 1


def test_an_ingest_gap_fails_the_run_after_citations_and_checks_and_holds_the_watermark(monkeypatch, capsys):
    gap = {"new_papers": 0, "failed_tries": 84, "gap": "ingest incomplete: cs=failed"}
    code, st, calls = _main_with(monkeypatch, gap)
    out = capsys.readouterr().out
    assert code == 1 and calls["citations"] == 1          # Semantic Scholar does not depend on arXiv
    assert dr.read_complete_through(st) is None
    assert '"problems"' in out and "ingest incomplete: cs=failed" in out and '"after"' in out


def test_after_an_outage_the_next_run_starts_from_the_last_complete_day(monkeypatch, capsys):
    monkeypatch.setattr(dr, "read_complete_through", lambda s: "2026-11-04")
    _, _, calls = _main_with(monkeypatch, OK_INGEST)
    assert calls["since"] == "2026-11-03"                 # not 7 Nov, which would lose 4 days


def test_failed_citations_are_reported_and_the_checks_still_run(monkeypatch, capsys):
    code, _, _ = _main_with(monkeypatch, OK_INGEST, citations=SystemExit("Semantic Scholar batch failed at 2611.0"))
    out = capsys.readouterr().out
    assert code == 1 and "Semantic Scholar batch failed" in out and '"after"' in out


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
