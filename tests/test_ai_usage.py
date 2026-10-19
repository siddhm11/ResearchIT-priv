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
