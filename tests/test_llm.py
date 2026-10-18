"""app/llm.py: one JSON-only way to call a model, logged and budgeted."""
import asyncio

import aiosqlite
import pytest

from app import config, llm


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(config, "RESEARCHIT_ENV", "test")
    monkeypatch.setattr(config, "LLM_PROVIDER", "fake")
    llm._fake_replies.clear()
    yield
    llm._fake_replies.clear()


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure! Here it is:\n```json\n{"a": {"b": [1, 2]}}\n```\nAnything else?', {"a": {"b": [1, 2]}}),
    ('{"s": "a } brace and a \\" quote"}', {"s": 'a } brace and a " quote'}),
    ('not json {broken} then {"ok": true}', {"ok": True}),
    ('[1, 2, 3]', None),
    ('no braces at all', None),
    ('', None),
])
def test_first_json_object(text, expected):
    assert llm.first_json_object(text) == expected


async def test_a_call_returns_parsed_json_and_is_logged(fake):
    llm.fake_reply('{"answer": 42}')
    data, call = await llm.complete_json("briefing", "system", "user")
    assert data == {"answer": 42} and call.status == "ok" and call.model == "fake"
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute("SELECT feature, status, chars_in FROM llm_calls")
        assert await cur.fetchall() == [("briefing", "ok", len("system") + len("user"))]


async def test_an_unparsable_reply_is_none_and_logged_as_such(fake):
    llm.fake_reply("I cannot help with that.")
    data, call = await llm.complete_json("ask", "s", "u")
    assert data is None and call.status == "unparsable"


async def test_a_failing_provider_is_none_never_an_exception(fake):
    llm.fake_reply(RuntimeError("provider down"))
    data, call = await llm.complete_json("ask", "s", "u")
    assert data is None and call.status == "failed"


async def test_the_daily_call_cap_stops_new_calls(fake, monkeypatch):
    monkeypatch.setattr(config, "LLM_DAILY_CALLS", 2)
    llm.fake_reply('{"n": 1}', '{"n": 2}', '{"n": 3}')
    assert (await llm.complete_json("ask", "s", "u"))[0] == {"n": 1}
    assert (await llm.complete_json("ask", "s", "u"))[0] == {"n": 2}
    data, call = await llm.complete_json("ask", "s", "u")
    assert data is None and call.status == "over_budget"
    assert len(llm._fake_replies) == 1                    # the third reply was never requested


async def test_the_spend_cap_counts_estimated_dollars(fake, monkeypatch):
    monkeypatch.setattr(config, "LLM_PRICES", "fake=1000000:1000000")   # $1 per character
    monkeypatch.setattr(config, "LLM_DAILY_USD_CAP", 5.0)
    llm.fake_reply('{"a": 1}', '{"a": 2}')
    await llm.complete_json("ask", "ab", "cd")            # 4 in + 8 out characters = $12
    data, call = await llm.complete_json("ask", "s", "u")
    assert call.status == "over_budget"
    assert (await llm.today())["usd"] >= 5.0


def test_prices_parse_and_unknown_models_cost_nothing(monkeypatch):
    monkeypatch.setattr(config, "LLM_PRICES", "claude-x=1.0:5.0; broken; other=2:3")
    assert llm.estimate_usd("claude-x", 1_000_000, 1_000_000) == 6.0
    assert llm.estimate_usd("qwen/qwen3.8-27b", 10**9, 10**9) == 0.0


def test_the_fake_provider_refuses_outside_tests(monkeypatch):
    monkeypatch.setattr(config, "RESEARCHIT_ENV", "")
    monkeypatch.setattr(config, "LLM_PROVIDER", "fake")
    with pytest.raises(RuntimeError):
        llm.fake_reply('{"a": 1}')
    assert llm._provider() == "groq"


async def test_groq_without_a_key_fails_quietly(monkeypatch):
    from app import groq_svc
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(groq_svc, "_get_client", lambda: None)
    data, call = await llm.complete_json("ask", "s", "u")
    assert data is None and call.status == "failed"
