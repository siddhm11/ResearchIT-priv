"""
app/ai_usage.py: where the app's AI goes.

Every model call is one llm_calls row with its tokens, the answering model's
place in the chain, whether the reply was cut off, and Groq's rate-limit
window at that moment. These tests pin what a row holds for each kind of
call, that the budget cap still covers only the briefing and Ask, and that the
new columns replicate.
"""
import aiosqlite
import pytest

from app import ai_usage, config, groq_svc, llm, turso_sync
from app.groq_svc import Completion


async def rows(sql="SELECT feature, model, status, tokens_in, tokens_out, model_rank, truncated, "
                   "requests_left, tokens_left FROM llm_calls ORDER BY id"):
    async with aiosqlite.connect(config.DB_PATH) as conn:
        return await (await conn.execute(sql)).fetchall()


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(groq_svc, "_windows", {"qwen/qwen3.8-27b": {"requests_left": 987, "tokens_left": 6512}})


async def test_a_direct_call_is_logged_with_tokens_rank_and_window(window):
    await ai_usage.log_call("search_overview", Completion("text", "qwen/qwen3.8-27b", False, 1312, 140, 0),
                            latency_ms=812, chars_in=5200)
    assert await rows() == [("search_overview", "qwen/qwen3.8-27b", "ok", 1312, 140, 0, 0, 987, 6512)]


async def test_a_call_no_model_answered_is_logged_without_tokens():
    await ai_usage.log_call("search_rewrite", None, latency_ms=3000, status="unavailable")
    assert await rows() == [("search_rewrite", "", "unavailable", None, None, None, 0, None, None)]


async def test_briefing_and_ask_calls_carry_tokens_through_llm(monkeypatch, window):
    async def answer(system, user, max_tokens, timeout):
        return Completion('{"a": 1}', "qwen/qwen3.8-27b", False, 3300, 410, 1)

    monkeypatch.setattr(llm, "_provider", lambda: "groq")
    monkeypatch.setattr(llm, "_ask_groq", answer)
    data, call = await llm.complete_json("briefing", "s", "u")
    assert data == {"a": 1} and (call.tokens_in, call.tokens_out, call.model_rank) == (3300, 410, 1)
    assert await rows() == [("briefing", "qwen/qwen3.8-27b", "ok", 3300, 410, 1, 0, 987, 6512)]


async def test_a_reply_cut_off_at_the_cap_is_failed_but_marked_truncated(monkeypatch):
    async def cut(system, user, max_tokens, timeout):
        raise llm.Truncated(Completion('{"a": ', "openai/gpt-oss-20b", True, 3100, 900, 2))

    monkeypatch.setattr(llm, "_provider", lambda: "groq")
    monkeypatch.setattr(llm, "_ask_groq", cut)
    data, call = await llm.complete_json("ask", "s", "u")
    assert data is None and call.status == "failed"         # what Ask and the briefing already handle
    assert await rows() == [("ask", "openai/gpt-oss-20b", "failed", 3100, 900, 2, 1, None, None)]


async def test_search_and_plain_summary_calls_do_not_use_up_the_briefing_budget(monkeypatch):
    monkeypatch.setattr(config, "LLM_DAILY_CALLS", 3)
    for _ in range(5):
        await ai_usage.log_call("search_overview", Completion("x", "m", False))
    assert await llm.within_budget()
    for _ in range(3):
        await ai_usage.log_call("briefing", Completion("x", "m", False))
    assert not await llm.within_budget()


def test_the_new_columns_replicate_and_are_added_remotely():
    new = ["tokens_in", "tokens_out", "model_rank", "truncated", "requests_left", "tokens_left"]
    assert turso_sync.TABLES["llm_calls"]["cols"][-6:] == new
    ddl = next(d for d in turso_sync.REMOTE_DDL if "TABLE IF NOT EXISTS llm_calls" in d)
    assert all(c in ddl for c in new)
    assert {("llm_calls", c) for c in new} <= {(t, c) for t, c, _ in turso_sync.REMOTE_COLUMNS}


# ── the three features that call groq_svc directly ───────────────────────────

@pytest.fixture
def groq_answers(monkeypatch):
    monkeypatch.setattr(groq_svc, "_get_client", lambda: object())
    answer = {"value": Completion("an answer.", "qwen/qwen3.8-27b", False, 500, 40, 0)}

    def run(*args):
        if isinstance(answer["value"], Exception):
            raise answer["value"]
        return answer["value"]

    for name in ("_run_rewrite", "_run_summary", "_run_explain"):
        monkeypatch.setattr(groq_svc, name, run)
    return answer


async def test_search_and_plain_summaries_log_each_call_under_its_feature(groq_answers):
    await groq_svc.rewrite("how do computers learn from examples")
    await groq_svc.generate_search_summary("q", [{"title": "A", "abstract": "x"}, {"title": "B", "abstract": "y"}])
    await groq_svc.explain_paper("T", "An abstract long enough to explain. " * 6)
    got = await rows("SELECT feature, status, tokens_in, model_rank FROM llm_calls ORDER BY id")
    assert got == [("search_rewrite", "ok", 500, 0), ("search_overview", "ok", 500, 0), ("explain", "ok", 500, 0)]


async def test_a_search_that_looks_academic_makes_no_call(groq_answers):
    await groq_svc.rewrite("LLM hallucination")
    assert await rows("SELECT COUNT(*) FROM llm_calls") == [(0,)]


async def test_a_failed_or_benched_call_is_logged_and_the_feature_still_degrades(groq_answers):
    groq_answers["value"] = RuntimeError("no Groq model answered (configured: [...])")
    assert await groq_svc.rewrite("how do computers learn from examples") == "how do computers learn from examples"
    groq_answers["value"] = ValueError("bad response")
    assert await groq_svc.explain_paper("T", "An abstract long enough to explain. " * 6) is None
    assert await rows("SELECT feature, status FROM llm_calls ORDER BY id") == [
        ("search_rewrite", "unavailable"), ("explain", "failed")]


# ── daily counts ─────────────────────────────────────────────────────────────

async def daily(feature=None):
    sql = "SELECT feature, metric, n FROM ai_daily" + (" WHERE feature = ?" if feature else "") + " ORDER BY metric"
    async with aiosqlite.connect(config.DB_PATH) as conn:
        return await (await conn.execute(sql, (feature,) if feature else ())).fetchall()


@pytest.fixture
def no_pending(monkeypatch):
    monkeypatch.setattr(ai_usage, "_pending", {})


async def test_counts_add_up_across_flushes_and_each_flush_gets_a_new_id(no_pending):
    ai_usage.count("explain", "cache_hit")
    ai_usage.count("explain", "cache_hit", 2)
    ai_usage.count("explain", "cache_miss")
    ai_usage.count("explain", "cache_miss", 0)                 # nothing to add
    await ai_usage.flush()
    first = await rows("SELECT id FROM ai_daily WHERE metric = 'cache_hit'")
    ai_usage.count("explain", "cache_hit")
    await ai_usage.flush()
    assert await daily("explain") == [("explain", "cache_hit", 4), ("explain", "cache_miss", 1)]
    assert (await rows("SELECT id FROM ai_daily WHERE metric = 'cache_hit'"))[0][0] > first[0][0]   # pushed again


async def test_counts_survive_a_failed_flush(no_pending, monkeypatch):
    ai_usage.count("ask", "answer")
    monkeypatch.setattr(config, "DB_PATH", "/nonexistent/dir/x.db")
    await ai_usage.flush()
    monkeypatch.undo()
    assert ai_usage._pending                                   # kept for the next flush


def test_daily_counts_replicate_one_row_per_day_feature_and_metric():
    assert turso_sync.TABLES["ai_daily"]["cols"] == ["id", "day", "feature", "metric", "n"]
    ddl = next(d for d in turso_sync.REMOTE_DDL if "TABLE IF NOT EXISTS ai_daily" in d)
    assert "UNIQUE (day, feature, metric)" in ddl


async def test_search_and_plain_summary_outcomes_are_counted(groq_answers, no_pending):
    await groq_svc.rewrite("LLM hallucination")                                   # looks academic
    await groq_svc.rewrite("how do computers learn from examples")               # rewritten
    groq_answers["value"] = Completion("INSUFFICIENT", "m", False)
    await groq_svc.explain_paper("T", "An abstract long enough to explain. " * 6)
    await ai_usage.flush()
    got = {(f, m): n for f, m, n in await daily()}
    assert got[("search_rewrite", "skipped")] == 1 and got[("search_rewrite", "rewritten")] == 1
    assert got[("explain", "declined")] == 1
