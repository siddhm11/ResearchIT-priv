"""
scripts/turso_copy.py: the daily job's copy of its writes onto the mirror
database, against two real SQLite databases built with production's schema.

What must hold: after a run the mirror holds the same papers (same rowids),
licences, citation counts and history, and Hugging Face lists as the primary; a
batch that fails leaves a history row and its count both absent, never one
without the other; rows already equal are not written again (the free plan's
writes are the scarce side).
"""
import importlib.util
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tc = _load("turso_copy")
dr = _load("daily_refresh")

PAPERS_DDL = ("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT, "
              "abstract_preview TEXT, categories TEXT, primary_topic TEXT, update_date TEXT, "
              "citation_count INTEGER, influential_citations INTEGER)")
LICENCE_DDL = ("CREATE TABLE IF NOT EXISTS paper_licences (arxiv_id TEXT PRIMARY KEY, "
               "licence TEXT NOT NULL, licence_url TEXT, source TEXT, updated_at TEXT)")
DDL = [dr.HISTORY_DDL, dr.HF_PAPERS_DDL, dr.HF_HISTORY_DDL, LICENCE_DDL]


class FakeDb:
    """The two calls turso_copy makes, over SQLite: rows() and an
    all-or-nothing tx()."""

    def __init__(self, name):
        self.name = name
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        self.conn.execute(PAPERS_DDL)
        for d in DDL:
            self.conn.execute(d)
        self.writes = 0
        self.fail_tx_after: int | None = None

    def rows(self, sql, args=()):
        return [list(r) for r in self.conn.execute(sql, list(args)).fetchall()]

    def tx(self, stmts):
        if not stmts:
            return
        if self.fail_tx_after is not None:
            if self.fail_tx_after == 0:
                raise RuntimeError(f"{self.name}: HTTP 503")
            self.fail_tx_after -= 1
        self.conn.execute("BEGIN")
        try:
            for s in stmts:
                self.conn.execute(s["sql"], [tc.uncell(a) for a in s.get("args", [])])
        except sqlite3.Error:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")
        self.writes += sum(1 for s in stmts if not s["sql"].startswith(("CREATE", "INSERT OR REPLACE INTO mirror_state")))

    def table(self, name):
        """papers with its rowids (the copy keeps them); other tables by content,
        since a replaced row there may take a new rowid and nothing keys on it."""
        if name == "papers":
            return self.conn.execute("SELECT rowid, * FROM papers ORDER BY rowid").fetchall()
        return self.conn.execute(f"SELECT * FROM {name} ORDER BY 1, 2").fetchall()


def _paper(i, cites=0):
    return (f"2610.{i:05d}", f"Paper {i}", "A; B", "abstract " * 5, "cs.LG", "AI/ML", "2026-10-24", cites, 0)


def _add_papers(db, ids, cites=0):
    db.conn.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?)", [_paper(i, cites) for i in ids])


def _pair(n_shared=3):
    src, dst = FakeDb("primary"), FakeDb("mirror")
    _add_papers(src, range(1, n_shared + 1))
    _add_papers(dst, range(1, n_shared + 1))
    for db in (src, dst):
        db.tx([{"sql": tc.STATE_DDL}])
    return src, dst


def _set_wm(dst, key, value):
    dst.tx([tc.set_state_stmt(key, value)])


# ── papers ───────────────────────────────────────────────────────────────────

def test_new_papers_are_copied_with_their_rowids():
    src, dst = _pair()
    _add_papers(src, range(4, 9))
    out = tc.copy_papers(src, dst)
    assert out["copied"] == 5
    assert dst.table("papers") == src.table("papers")
    assert tc.copy_papers(src, dst)["copied"] == 0


def test_papers_that_no_longer_line_up_are_not_appended():
    src, dst = _pair()
    dst.conn.execute("UPDATE papers SET arxiv_id = 'x' WHERE rowid = 3")
    _add_papers(src, [4])
    assert "differ at rowid 3" in tc.copy_papers(src, dst)["problem"]
    assert len(dst.table("papers")) == 3


def test_a_mirror_ahead_of_the_primary_is_reported_not_overwritten():
    src, dst = _pair()
    _add_papers(dst, [4])
    assert "above the primary" in tc.copy_papers(src, dst)["problem"]


# ── citations ────────────────────────────────────────────────────────────────

def _history(db, rows):
    db.tx([dr.history_insert(rows)])
    db.tx([tc.stmt("UPDATE papers SET citation_count = ?, influential_citations = ? WHERE arxiv_id = ?",
                   (c, f, a)) for a, _d, _cb, c, _fb, f in rows])


def test_citation_changes_reach_the_mirror_with_their_counts():
    src, dst = _pair()
    _set_wm(dst, "citation_history_rowid", 0)
    _history(src, [("2610.00001", "2026-10-24", 0, 7, 0, 1), ("2610.00002", "2026-10-24", 0, 3, 0, 0)])
    out = tc.copy_by_watermark(src, dst, "citation_history", tc.HISTORY_COLS,
                               ["arxiv_id", "checked_on"], "citation_history_rowid", extra=tc._count_update)
    assert out["copied"] == 2
    assert dst.rows("SELECT arxiv_id, citation_count FROM papers WHERE citation_count > 0 ORDER BY 1") == [
        ["2610.00001", 7], ["2610.00002", 3]]
    assert tc.get_state(dst, "citation_history_rowid") == "2"


def test_a_same_day_rerun_that_updated_a_row_in_place_is_picked_up():
    src, dst = _pair()
    _set_wm(dst, "citation_history_rowid", 0)
    _history(src, [("2610.00001", "2026-10-24", 0, 7, 0, 1)])
    args = (src, dst, "citation_history", tc.HISTORY_COLS, ["arxiv_id", "checked_on"], "citation_history_rowid")
    tc.copy_by_watermark(*args, extra=tc._count_update)
    _history(src, [("2610.00001", "2026-10-24", 0, 9, 0, 1)])        # same key, same rowid
    assert src.rows("SELECT MAX(rowid) FROM citation_history") == [[1]]
    tc.copy_by_watermark(*args, extra=tc._count_update)
    assert dst.rows("SELECT citation_count FROM papers WHERE arxiv_id = '2610.00001'") == [[9]]


def test_rows_already_equal_are_not_written_again():
    src, dst = _pair()
    _set_wm(dst, "citation_history_rowid", 0)
    _history(src, [("2610.00001", "2026-10-24", 0, 7, 0, 1)])
    args = (src, dst, "citation_history", tc.HISTORY_COLS, ["arxiv_id", "checked_on"], "citation_history_rowid")
    tc.copy_by_watermark(*args, extra=tc._count_update)
    dst.writes = 0
    assert tc.copy_by_watermark(*args, extra=tc._count_update)["copied"] == 0
    assert dst.writes == 0


def test_a_failed_batch_leaves_no_history_row_without_its_count(monkeypatch):
    monkeypatch.setattr(tc, "BATCH", 4)                 # two statements per row: two rows a batch
    src, dst = _pair(n_shared=6)
    _set_wm(dst, "citation_history_rowid", 0)
    _history(src, [(f"2610.0000{i}", "2026-10-24", 0, 10 + i, 0, 0) for i in range(1, 7)])
    dst.fail_tx_after = 1                               # first batch lands, the second fails
    args = (src, dst, "citation_history", tc.HISTORY_COLS, ["arxiv_id", "checked_on"], "citation_history_rowid")
    with pytest.raises(RuntimeError):
        tc.copy_by_watermark(*args, extra=tc._count_update)
    copied = {r[0] for r in dst.rows("SELECT arxiv_id FROM citation_history")}
    counted = {r[0] for r in dst.rows("SELECT arxiv_id FROM papers WHERE citation_count > 0")}
    assert copied == counted and len(copied) == 2
    assert tc.get_state(dst, "citation_history_rowid") == "0", "watermark moved past unwritten rows"

    dst.fail_tx_after = None
    tc.copy_by_watermark(*args, extra=tc._count_update)
    assert dst.table("citation_history") == src.table("citation_history")
    assert dst.rows("SELECT arxiv_id, citation_count FROM papers ORDER BY 1") == \
        src.rows("SELECT arxiv_id, citation_count FROM papers ORDER BY 1")


def test_no_watermark_means_no_guessing():
    src, dst = _pair()
    assert "run scripts/turso_catchup.py" in tc.copy_by_watermark(
        src, dst, "citation_history", tc.HISTORY_COLS, ["arxiv_id", "checked_on"], "citation_history_rowid")["problem"]


# ── the whole stage ──────────────────────────────────────────────────────────

def _hf(db, aid, day, upvotes):
    papers = [(aid, day, f"Title {aid}", "[]", "", "", "", day)]
    history = [(aid, day, upvotes, 0, 0, 1, 1)]
    db.tx(dr.hf_statements(papers, history))


def test_a_daily_run_leaves_the_mirror_equal():
    src, dst = _pair()
    _set_wm(dst, "citation_history_rowid", 0)
    _set_wm(dst, "hf_upvote_history_rowid", 0)
    _add_papers(src, range(4, 7))
    src.conn.execute("INSERT INTO paper_licences VALUES ('2610.00005', 'CC-BY-4.0', 'u', 'oai', 't')")
    _history(src, [("2610.00002", "2026-10-25", 0, 4, 0, 0)])
    _hf(src, "2610.00004", "2026-10-25", 12)

    out = tc.mirror_daily(src, dst, DDL)
    assert out["problems"] == []
    for table in ("papers", "paper_licences", "citation_history", "hf_upvote_history", "hf_daily_papers"):
        assert dst.table(table) == src.table(table), table

    # The next day: the paper's upvotes grow and its title is corrected.
    _hf(src, "2610.00004", "2026-10-26", 40)
    src.conn.execute("UPDATE hf_daily_papers SET title = 'Fixed' WHERE arxiv_id = '2610.00004'")
    assert tc.mirror_daily(src, dst, DDL)["problems"] == []
    assert dst.table("hf_daily_papers") == src.table("hf_daily_papers")
    assert dst.table("hf_upvote_history") == src.table("hf_upvote_history")


def test_the_stage_is_off_without_its_secrets(monkeypatch):
    monkeypatch.delenv("TURSO_MIRROR_URL", raising=False)
    monkeypatch.delenv("TURSO_MIRROR_DB_TOKEN", raising=False)
    assert "skipped" in dr.stage_mirror(object())


def test_a_mirror_failure_is_a_problem_of_the_run(monkeypatch):
    monkeypatch.setenv("TURSO_MIRROR_URL", "https://m.turso.io")
    monkeypatch.setenv("TURSO_MIRROR_DB_TOKEN", "m")

    class St:
        turl, ttok, http = "https://p.turso.io", "p", None

    def boom(*a, **k):
        raise RuntimeError("mirror: HTTP 503")

    real = dr._load

    def load(name):
        mod = real(name)
        if name == "turso_copy":
            mod.mirror_daily = boom
        return mod

    monkeypatch.setattr(dr, "_load", load)
    out = dr.stage_mirror(St())
    assert out["problems"] and "503" in out["problems"][0]


def test_workflow_passes_the_mirror_secrets():
    text = (ROOT / ".github" / "workflows" / "daily-refresh.yml").read_text()
    for key in ("TURSO_MIRROR_URL", "TURSO_MIRROR_DB_TOKEN"):
        assert f"          {key}: ${{{{ secrets.{key} }}}}" in text


def test_the_http_client_decodes_values_and_raises_on_a_failed_step():
    import httpx
    sent = []

    def handler(request):
        body = __import__("json").loads(request.content)
        sent.append(body)
        req = body["requests"][0]
        if req["type"] == "execute":
            return httpx.Response(200, json={"results": [{"type": "ok", "response": {"type": "execute", "result": {
                "cols": [], "rows": [[{"type": "integer", "value": "7"}, {"type": "text", "value": "x"},
                                      {"type": "null"}, {"type": "blob", "base64": "AAE"}]]}}}, {"type": "ok"}]})
        return httpx.Response(200, json={"results": [{"type": "ok", "response": {"type": "batch", "result": {
            "step_results": [], "step_errors": [None, {"message": "UNIQUE constraint failed"}, None]}}},
            {"type": "ok"}]})

    db = tc.Db("mirror", "libsql://m.turso.io", "tok", httpx.Client(transport=httpx.MockTransport(handler)))
    assert db.rows("SELECT ?", (5,)) == [[7, "x", None, b"\x00\x01"]]
    assert sent[0]["requests"][0]["stmt"]["args"] == [{"type": "integer", "value": "5"}]
    with pytest.raises(RuntimeError, match="UNIQUE"):
        db.tx([tc.stmt("INSERT INTO t VALUES (1)")])
    steps = sent[1]["requests"][0]["batch"]["steps"]
    assert steps[0]["stmt"]["sql"] == "BEGIN" and steps[-1]["stmt"]["sql"] == "ROLLBACK"
