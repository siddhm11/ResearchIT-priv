"""Local FTS5 sparse retrieval.

Builds a small real sidecar rather than mocking sqlite: the failure modes worth
covering here are FTS5 syntax errors from user text and the AND/OR fallback, and
neither survives a mock.
"""
import asyncio
import sqlite3

import pytest

from app import config, fts_svc, local_meta


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    path = tmp_path / "metadata.sqlite"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE papers (
                     arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT,
                     abstract_preview TEXT, categories TEXT, primary_topic TEXT,
                     update_date TEXT, citation_count INTEGER,
                     influential_citations INTEGER)""")
    rows = [
        ("2506.00001", "Diffusion transformers for video generation",
         "A diffusion transformer that generates video frames.", "2025-06-01"),
        ("0704.0002", "Sparsity-certifying graph decompositions",
         "A pebble game algorithm for sparse graphs.", "2007-04-01"),
        ("2606.00001", "Retrieval augmented generation for agents",
         "An agent framework using retrieval augmented generation.", "2026-06-01"),
    ]
    c.executemany("INSERT INTO papers (arxiv_id,title,abstract_preview,update_date,"
                  "citation_count,influential_citations) VALUES (?,?,?,?,0,0)",
                  [(a, t, ab, d) for a, t, ab, d in rows])
    c.execute("""CREATE VIRTUAL TABLE papers_fts USING fts5(
                     title, abstract_preview, content='papers',
                     content_rowid='rowid', tokenize='porter unicode61')""")
    c.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    c.commit(); c.close()

    # Reset both modules' cached probes so each test opens this sidecar.
    monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(path))
    monkeypatch.setattr(local_meta, "_conn", None)
    monkeypatch.setattr(local_meta, "_probed", False)
    monkeypatch.setattr(local_meta, "_available", False)
    monkeypatch.setattr(fts_svc, "_probed", False)
    monkeypatch.setattr(fts_svc, "_available", False)
    return path


def run(q, limit=10):
    return asyncio.run(fts_svc.search_sparse(q, limit=limit))


def test_finds_paper_by_title(sidecar):
    out = run("diffusion transformer video")
    assert [r["arxiv_id"] for r in out][:1] == ["2506.00001"]


def test_covers_papers_newer_than_the_snapshot(sidecar):
    """The whole point of the switch: post-2025-06 papers must be reachable."""
    out = run("retrieval augmented generation agents")
    assert "2606.00001" in [r["arxiv_id"] for r in out]


def test_falls_back_from_and_to_or(sidecar):
    """A term absent from the corpus must not empty the result set."""
    out = run("diffusion zzzznotacorpusterm")
    assert [r["arxiv_id"] for r in out][:1] == ["2506.00001"]


def test_results_say_whether_every_word_matched(sidecar):
    """hybrid_search_svc tells a title from a question by this mark."""
    assert {r["match"] for r in run("diffusion transformer video")} == {"all"}
    assert {r["match"] for r in run("diffusion zzzznotacorpusterm")} == {"any"}


@pytest.mark.parametrize("q", [
    'quantum "unbalanced',          # unterminated quote
    "graph AND OR NOT",             # bare operators
    "sparse NEAR/2 graph",          # NEAR syntax
    "video*",                       # prefix operator
    "^anchored",                    # column-anchor operator
    "it's a graph",                 # apostrophe
    "-- drop table papers",         # sql-ish noise
])
def test_user_text_is_never_parsed_as_fts_syntax(sidecar, q):
    """Any of these raises OperationalError if passed to MATCH unescaped,
    which would silently kill the sparse arm for that query."""
    out = run(q)
    assert isinstance(out, list)


def test_empty_and_punctuation_only_queries(sidecar):
    assert run("") == []
    assert run("   ") == []
    assert run("!!! ??? ...") == []


def test_single_characters_are_dropped(sidecar):
    assert fts_svc._terms("a b diffusion c") == ["diffusion"]


def test_scores_are_higher_is_better(sidecar):
    """Negated from bm25()'s more-negative-is-better so it matches the dense arm."""
    out = run("diffusion transformer video")
    assert len(out) >= 1
    assert out == sorted(out, key=lambda r: r["score"], reverse=True)


def test_respects_limit(sidecar):
    assert len(run("generation OR graph OR diffusion", limit=2)) <= 2


def test_unavailable_without_fts_index(tmp_path, monkeypatch):
    """An older sidecar has no papers_fts; degrade instead of raising."""
    path = tmp_path / "old.sqlite"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT)")
    c.execute("INSERT INTO papers VALUES ('1','t')")
    c.commit(); c.close()
    monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(path))
    monkeypatch.setattr(local_meta, "_conn", None)
    monkeypatch.setattr(local_meta, "_probed", False)
    monkeypatch.setattr(local_meta, "_available", False)
    monkeypatch.setattr(fts_svc, "_probed", False)
    monkeypatch.setattr(fts_svc, "_available", False)
    assert fts_svc.is_available() is False
    assert run("anything") == []


def test_stats_reports_index_size(sidecar):
    s = fts_svc.stats()
    assert s["available"] is True
    assert s["indexed"] == 3


def test_filler_words_are_not_matched(sidecar):
    """Filler words would turn the OR fallback into a scan of most of the corpus."""
    assert fts_svc._terms("how do diffusion transformers handle video") == [
        "diffusion", "transformers", "handle", "video"]
    out = run("what are the papers about diffusion video")
    assert [r["arxiv_id"] for r in out][:1] == ["2506.00001"]


def test_all_filler_query_still_searches(sidecar):
    assert fts_svc._terms("what is it") == ["what", "is", "it"]


def test_over_budget_search_returns_nothing_instead_of_waiting(sidecar, monkeypatch):
    """A late lexical list blocks the whole search; a missing one only costs RRF
    one input. Past the budget the arm must give up, not finish."""
    monkeypatch.setattr(config, "SEARCH_FTS_BUDGET_MS", -1)
    monkeypatch.setattr(fts_svc, "_PROGRESS_STEPS", 1)
    assert run("diffusion transformer video") == []


def test_budget_handler_is_cleared_after_search(sidecar, monkeypatch):
    """The per-thread connection is shared with metadata reads, which must not
    inherit an expired deadline."""
    monkeypatch.setattr(fts_svc, "_PROGRESS_STEPS", 1)
    assert fts_svc._probe()
    assert fts_svc._search_sync("diffusion transformer video", 10)   # same thread, same handle
    monkeypatch.setattr(fts_svc.time, "monotonic", lambda: 1e18)     # any old deadline is now past
    conn = local_meta.connection()
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3


# ── Rank inside FTS before touching `papers` (flight test 2026-10-16) ─────────

_JOIN_FIRST = ("SELECT p.arxiv_id, bm25(papers_fts, ?, ?) AS s "
               "FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid "
               "WHERE papers_fts MATCH ? ORDER BY s LIMIT ?")


@pytest.fixture
def big_sidecar(tmp_path, monkeypatch):
    """Enough rows that LIMIT cuts the match set, with exact-score ties."""
    import random
    rnd = random.Random(7)
    words = ["graph", "neural", "network", "learning", "quantum", "diffusion",
             "model", "sparse", "agent", "protein", "theory", "field"]
    path = tmp_path / "big.sqlite"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT,
                 abstract_preview TEXT, categories TEXT, primary_topic TEXT,
                 update_date TEXT, citation_count INTEGER, influential_citations INTEGER)""")
    rows = []
    for i in range(1500):
        title = " ".join(rnd.choices(words, k=rnd.randint(2, 6)))
        abstract = " ".join(rnd.choices(words, k=rnd.randint(5, 40)))
        rows.append((f"2401.{i:05d}", title, abstract))
    rows += [(f"2402.{i:05d}", "graph learning", "graph learning") for i in range(30)]  # ties
    c.executemany("INSERT INTO papers (arxiv_id,title,abstract_preview,citation_count,"
                  "influential_citations) VALUES (?,?,?,0,0)", rows)
    c.execute("""CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract_preview,
                 content='papers', content_rowid='rowid', tokenize='porter unicode61')""")
    c.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    c.commit(); c.close()
    monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(path))
    monkeypatch.setattr(local_meta, "_conn", None)
    monkeypatch.setattr(local_meta, "_probed", False)
    monkeypatch.setattr(local_meta, "_available", False)
    monkeypatch.setattr(fts_svc, "_probed", False)
    monkeypatch.setattr(fts_svc, "_available", False)
    return path


@pytest.mark.parametrize("match", ['"graph"', '"graph" AND "learning"',
                                   '"quantum" OR "protein" OR "agent"', '"field"'])
@pytest.mark.parametrize("limit", [1, 10, 60])
def test_rank_first_query_returns_the_join_first_rows(big_sidecar, match, limit):
    """Same ids, same scores, same order as the old join-then-sort query."""
    c = sqlite3.connect(big_sidecar)
    args = (fts_svc._TITLE_WEIGHT, fts_svc._ABSTRACT_WEIGHT, match, limit)
    assert c.execute(fts_svc._SQL, args).fetchall() == c.execute(_JOIN_FIRST, args).fetchall()


def test_papers_is_looked_up_only_after_the_limit(big_sidecar):
    """The plan must sort inside the FTS co-routine and join afterwards; joining
    first reads one `papers` row per match (280k for "learning")."""
    c = sqlite3.connect(big_sidecar)
    plan = [r[-1] for r in c.execute("EXPLAIN QUERY PLAN " + fts_svc._SQL,
                                     (2.0, 1.0, '"graph"', 60))]
    assert any("CO-ROUTINE" in step for step in plan), plan
    join = next(i for i, step in enumerate(plan) if step.startswith("SEARCH p"))
    scan_f = next(i for i, step in enumerate(plan) if step.startswith("SCAN f"))
    assert join > scan_f, plan


def test_records_timing_and_outcome(sidecar, monkeypatch):
    from app import runtime_stats
    monkeypatch.setattr(runtime_stats, "_buffers", {})
    monkeypatch.setattr(runtime_stats, "_counters", {})
    run("diffusion transformer video")                 # AND hit
    run("diffusion zzzznotacorpusterm")                # AND empty, OR hit
    run("zzzznotacorpusterm qqqqalsonot")              # nothing
    t = runtime_stats.timings()["fts"]
    assert t["n"] == 3 and "fts_ms" in t and "and_ms" in t and "or_ms" in t
    assert runtime_stats.counters() == {"fts_and_hit": 1, "fts_empty": 1, "fts_or_hit": 1}


def test_counts_budget_skips(sidecar, monkeypatch):
    from app import runtime_stats
    monkeypatch.setattr(runtime_stats, "_counters", {})
    monkeypatch.setattr(config, "SEARCH_FTS_BUDGET_MS", -1)
    monkeypatch.setattr(fts_svc, "_PROGRESS_STEPS", 1)
    assert run("diffusion transformer video") == []
    assert runtime_stats.counters() == {"fts_budget_skip": 1}


# ── Sidecar page-cache warm-up ───────────────────────────────────────────────

def test_warmup_reads_the_whole_file(sidecar, monkeypatch):
    monkeypatch.setattr(local_meta, "_warmup", {"status": "not started"})
    local_meta._warm(str(sidecar), chunk=4096)
    st = local_meta.warmup_status()
    assert st["status"] == "done"
    assert st["read_mb"] == round(sidecar.stat().st_size / 1e6)
    assert "seconds" in st and "mb_per_s" in st


def test_warmup_respects_the_kill_switch_and_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(local_meta, "_warmup", {"status": "not started"})
    monkeypatch.setattr(local_meta, "_warmup_thread", None)
    monkeypatch.setenv("SIDECAR_WARMUP", "0")
    assert local_meta.start_warmup() is False
    assert local_meta.warmup_status()["status"] == "disabled"
    monkeypatch.setenv("SIDECAR_WARMUP", "1")
    monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(tmp_path / "absent.sqlite"))
    assert local_meta.start_warmup() is False
    assert local_meta.warmup_status()["status"] == "no sidecar"


def test_warmup_runs_once_in_the_background(sidecar, monkeypatch):
    monkeypatch.setattr(local_meta, "_warmup", {"status": "not started"})
    monkeypatch.setattr(local_meta, "_warmup_thread", None)
    monkeypatch.setenv("SIDECAR_WARMUP", "1")
    assert local_meta.start_warmup() is True
    assert local_meta.start_warmup() is False          # idempotent
    local_meta._warmup_thread.join(timeout=10)
    assert local_meta.warmup_status()["status"] == "done"
