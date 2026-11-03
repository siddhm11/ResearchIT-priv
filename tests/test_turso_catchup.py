"""
scripts/turso_catchup.py: making the two Turso databases equal before the app
mirrors to both, replayed on the 2026-10-24 situation.

The old database ("mirror") holds the history: interactions, caches, the
day's counters up to the switch. The new one ("primary") has the papers and
citation changes the daily job wrote since, and a few user and cache rows the
Space pushed after the switch, numbered from 1 again, so their ids collide with
the old ones. After `apply` both must hold exactly the same rows, nothing lost,
nothing doubled, and the daily mirror copy must start cleanly from there.
"""
import importlib.util
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cu = _load("turso_catchup")
tc = cu.tc
dr = _load("daily_refresh")

from app import turso_sync  # noqa: E402

PAPERS_DDL = ("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT, "
              "abstract_preview TEXT, categories TEXT, primary_topic TEXT, update_date TEXT, "
              "citation_count INTEGER, influential_citations INTEGER)")


class FakeDb:
    def __init__(self, name):
        self.name = name
        self.url = f"https://{name}.example"
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        self.conn.execute(PAPERS_DDL)
        for d in turso_sync.REMOTE_DDL:
            self.conn.execute(d)

    def rows(self, sql, args=()):
        try:
            return [list(r) for r in self.conn.execute(sql, list(args)).fetchall()]
        except sqlite3.Error as e:
            raise RuntimeError(f"{self.name}: {e}") from None

    def tx(self, stmts):
        if not stmts:
            return
        self.conn.execute("BEGIN")
        try:
            for s in stmts:
                self.conn.execute(s["sql"], [tc.uncell(a) for a in s.get("args", [])])
        except sqlite3.Error as e:
            self.conn.execute("ROLLBACK")
            raise RuntimeError(f"{self.name}: {e}") from None
        self.conn.execute("COMMIT")

    def ins(self, table, row):
        self.conn.execute(f"INSERT INTO {table} VALUES ({', '.join('?' * len(row))})", row)

    def all(self, table, cols="*"):
        return sorted(self.conn.execute(f"SELECT {cols} FROM {table}").fetchall(), key=repr)


def _paper(i, cites=0):
    return (f"2610.{i:05d}", f"Paper {i}", "A; B", "abstract", "cs.LG", "AI/ML", "2026-10-20", cites, 0)


def _interaction(i, user, paper, ts):
    return (i, user, paper, "save", "feed", 0, f"q{i}", "v", "dense", None, 1.0, "p", ts, None)


def _source(i, aid, fetched):
    return (i, aid, "ok", "v1", "cc-by", "", "[]", 1, 2, 3, None, None, None, 1, fetched, None)


@pytest.fixture
def pair(monkeypatch):
    old, new = FakeDb("mirror"), FakeDb("primary")
    for db in (old, new):
        cu.ensure_schema(db)
    for i in range(1, 11):
        old.ins("papers", _paper(i, cites=1))
        new.ins("papers", _paper(i, cites=1))
    for i in (11, 12):                                   # the daily job, after the switch
        new.ins("papers", _paper(i))
    new.tx([dr.history_insert([("2610.00011", "2026-10-25", 0, 6, 0, 1)])])
    new.conn.execute("UPDATE papers SET citation_count = 6, influential_citations = 1 WHERE arxiv_id = '2610.00011'")
    new.ins("paper_licences", ("2610.00011", "CC-BY-4.0", "u", "listing", "2026-10-25"))
    old.ins("paper_licences", ("2610.00009", "CC0", "u", "oai", "2026-10-21"))
    new.tx(dr.hf_statements([("2610.00012", "2026-10-25", "T", "[]", "", "", "", "2026-10-25")],
                            [("2610.00012", "2026-10-25", 30, 0, 0, 1, 1)]))

    # user data: the history on the old side
    for i in range(1, 6):
        old.ins("user_interactions", _interaction(i, "u1", f"2610.0000{i}", f"2026-10-2{i}T00:00:00"))
    old.ins("user_profiles", ("u1", "longterm", b"\x01" * 8, 5, "2026-10-24T10:00:00"))
    old.ins("user_onboarding", ("u1", '["cs.LG"]', 1, "2026-10-01", "2026-10-01"))
    old.ins("user_clusters", ("u1", 0, "2610.00001", b"\x00", 0.6, "a", "2026-10-24T10:00:00"))
    old.ins("user_clusters", ("u1", 1, "2610.00002", b"\x00", 0.4, "b", "2026-10-24T10:00:00"))
    for i, aid in enumerate(["2610.00001", "2610.00002", "2610.00003"], start=1):
        old.ins("paper_sources", _source(i, aid, "2026-10-20"))
    old.ins("reader_days", (1, "2026-10-24", "u1", 0, 5, 1, 2, 3, 1, 0, 0))
    old.ins("ai_daily", (1, "2026-10-24", "explain", "calls", 3))

    # what the Space pushed to the new side after the switch, ids from 1 again
    new.ins("user_interactions", _interaction(1, "u2", "2610.00011", "2026-10-25T08:00:00"))
    new.ins("user_clusters", ("u1", 0, "2610.00005", b"\x02", 1.0, "c", "2026-10-25T09:00:00"))
    new.ins("paper_sources", _source(1, "2610.00003", "2026-10-25"))     # newer fetch of a known one
    new.ins("paper_sources", _source(2, "2610.00011", "2026-10-25"))     # a new one
    new.ins("reader_days", (1, "2026-10-24", "u1", 0, 2, 0, 1, 1, 0, 0, 0))   # the rest of the 24th
    new.ins("reader_days", (2, "2026-10-25", "u2", 0, 1, 0, 0, 0, 0, 0, 0))
    new.ins("ai_daily", (1, "2026-10-24", "explain", "calls", 1))
    return new, old


def _apply(new, old):
    cu.daily_tables_pass(new, old, write=True)
    cu.user_tables_pass(new, old, write=True)


def test_apply_makes_every_table_equal(pair):
    new, old = pair
    _apply(new, old)
    out = cu.verify(new, old, full=True)
    assert out["equal"], out["different"]


def test_colliding_ids_are_renumbered_not_overwritten(pair):
    new, old = pair
    _apply(new, old)
    rows = new.all("user_interactions", "id, user_id, paper_id")
    assert rows == [(1, "u1", "2610.00001"), (2, "u1", "2610.00002"), (3, "u1", "2610.00003"),
                    (4, "u1", "2610.00004"), (5, "u1", "2610.00005"), (6, "u2", "2610.00011")]


def test_a_cache_key_on_both_sides_keeps_the_newer_row_once(pair):
    new, old = pair
    _apply(new, old)
    assert new.all("paper_sources", "id, arxiv_id, fetched_at") == [
        (1, "2610.00001", "2026-10-20"), (2, "2610.00002", "2026-10-20"),
        (3, "2610.00003", "2026-10-25"), (4, "2610.00011", "2026-10-25")]


def test_a_day_split_across_the_switch_adds_up(pair):
    new, old = pair
    _apply(new, old)
    assert new.all("reader_days", "day, user_id, pages, searches") == [
        ("2026-10-24", "u1", 7, 1), ("2026-10-25", "u2", 1, 0)]
    assert new.all("ai_daily", "day, n") == [("2026-10-24", 4)]


def test_a_users_clusters_come_whole_from_the_later_side(pair):
    new, old = pair
    _apply(new, old)
    assert old.all("user_clusters", "user_id, cluster_idx, medoid_paper_id") == [("u1", 0, "2610.00005")]


def test_daily_job_data_reaches_the_old_side_with_counts_and_watermarks(pair):
    new, old = pair
    _apply(new, old)
    assert old.rows("SELECT rowid, arxiv_id FROM papers WHERE rowid > 10 ORDER BY rowid") == [
        [11, "2610.00011"], [12, "2610.00012"]]
    assert old.rows("SELECT citation_count FROM papers WHERE arxiv_id = '2610.00011'") == [[6]]
    assert {r[0] for r in new.all("paper_licences", "arxiv_id")} == {"2610.00009", "2610.00011"}
    assert tc.get_state(old, "citation_history_rowid") == "1"
    # the daily copy now starts cleanly
    new.ins("papers", _paper(13))
    out = tc.mirror_daily(new, old, [])
    assert out["problems"] == [] and out["papers"]["copied"] == 1


def test_apply_twice_changes_nothing(pair):
    new, old = pair
    _apply(new, old)
    before = {t: (new.all(t), old.all(t)) for t in ("user_interactions", "paper_sources", "reader_days")}
    report = cu.user_tables_pass(new, old, write=True)
    assert all(s["primary"]["upsert"] == s["mirror"]["upsert"] == 0 for s in report.values())
    assert {t: (new.all(t), old.all(t)) for t in before} == before


def test_plan_writes_nothing(pair):
    new, old = pair
    snap = (new.all("user_interactions"), old.all("papers"))
    report = cu.user_tables_pass(new, old, write=False)
    cu.daily_tables_pass(new, old, write=False)
    assert report["user_interactions"]["primary"]["upsert"] == 6
    assert (new.all("user_interactions"), old.all("papers")) == snap


def test_identical_rows_are_matched_and_duplicates_kept():
    cols = ["id", "user_id", "paper_id"]
    a = [[1, "u", "p"], [2, "u", "p"]]                  # a genuine double save
    b = [[1, "u", "p"], [2, "v", "q"]]
    assert cu.merge_id_table("user_interactions", cols, a, b) == [[1, "u", "p"], [2, "u", "p"], [3, "v", "q"]]


def test_misaligned_papers_are_left_alone(pair):
    new, old = pair
    old.conn.execute("UPDATE papers SET arxiv_id = 'moved' WHERE rowid = 5")
    report = cu.daily_tables_pass(new, old, write=True)
    assert report["papers"].get("problem") and tc.max_rowid(old, "papers") == 10


def test_natural_keys_match_the_replicated_schema():
    ddl = " ".join(turso_sync.REMOTE_DDL)
    for spec in turso_sync.TABLES.values():
        table = spec["remote"]
        body = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\)\s*$|CREATE TABLE IF NOT EXISTS {table} \((.*)",
                         next(d for d in turso_sync.REMOTE_DDL if f"EXISTS {table} (" in d), re.S)
        text = body.group(0)
        uniques = re.findall(r"(\w+) TEXT NOT NULL UNIQUE", text)
        composite = re.findall(r"UNIQUE \(([^)]*)\)", text)
        expected = uniques or ([c.strip() for c in composite[0].split(",")] if composite else None)
        assert cu.NATURAL.get(table) == expected, table
    assert ddl


def test_apply_refuses_while_the_space_may_be_pushing(monkeypatch):
    monkeypatch.setenv("TURSO_URL", "https://p.turso.io")
    monkeypatch.setenv("TURSO_DB_TOKEN", "p")
    monkeypatch.setenv("TURSO_OLD_URL", "https://o.turso.io")
    monkeypatch.setenv("TURSO_OLD_DB_TOKEN", "o")
    monkeypatch.setattr(sys, "argv", ["turso_catchup.py", "apply", "--mirror-prefix", "TURSO_OLD"])
    with pytest.raises(SystemExit, match="space-sync-stopped"):
        cu.main()
