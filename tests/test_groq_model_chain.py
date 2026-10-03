"""
The Groq model chain in groq_svc.

Groq shut down llama-3.3-70b-versatile on 2026-08-16. It was the only model
this module knew, every caller degrades silently by design, and so search
rewrite, the search overview and paper explanations all stopped for seven
weeks with nothing visibly wrong. These tests pin the replacement: an ordered
list of models where a retired or rate-limited model is skipped, not fatal.
No network: the Groq client is a fake.
"""
from types import SimpleNamespace

import groq
import httpx
import pytest

from app import config, groq_svc

CHAIN = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


def _status_error(cls, status: int, code: str = "", headers: dict | None = None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {})
    body = {"error": {"message": f"HTTP {status}", "code": code}}
    return cls(f"HTTP {status}", response=response, body=body)


def _gone():
    return _status_error(groq.NotFoundError, 404, "model_not_found")


def _completion(text: str, finish: str = "stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason=finish)])


class FakeClient:
    """Answers per model: an exception to raise or a completion to return."""

    def __init__(self, behaviour: dict):
        self.behaviour = behaviour
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.behaviour[kwargs["model"]]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _chain(monkeypatch):
    monkeypatch.setattr(config, "GROQ_MODELS", list(CHAIN))
    groq_svc._benched.clear()
    yield
    groq_svc._benched.clear()


def _complete(client, timeout: float = 5.0) -> str:
    return groq_svc._complete(client, [{"role": "user", "content": "q"}],
                              temperature=0.1, max_tokens=60, timeout=timeout)


def test_first_model_answers_when_healthy():
    client = FakeClient({m: _completion(f"from {m}") for m in CHAIN})
    assert _complete(client) == "from openai/gpt-oss-120b"
    assert [c["model"] for c in client.calls] == ["openai/gpt-oss-120b"]


def test_retired_model_fails_over_and_is_benched():
    client = FakeClient({CHAIN[0]: _gone(), CHAIN[1]: _completion("qwen answer"),
                         CHAIN[2]: _completion("unused")})
    assert _complete(client) == "qwen answer"
    # The second call must not pay for the dead model again.
    client.calls.clear()
    assert _complete(client) == "qwen answer"
    assert [c["model"] for c in client.calls] == [CHAIN[1]]
    assert CHAIN[0] in groq_svc.model_status()["benched"]


def test_decommissioned_400_counts_as_gone():
    err = _status_error(groq.BadRequestError, 400, "model_decommissioned")
    client = FakeClient({CHAIN[0]: err, CHAIN[1]: _completion("ok"),
                         CHAIN[2]: _completion("unused")})
    assert _complete(client) == "ok"
    assert CHAIN[0] in groq_svc.model_status()["benched"]


def test_rate_limit_benches_for_retry_after():
    err = _status_error(groq.RateLimitError, 429, "rate_limit_exceeded",
                        headers={"retry-after": "42"})
    client = FakeClient({CHAIN[0]: err, CHAIN[1]: _completion("ok"),
                         CHAIN[2]: _completion("unused")})
    assert _complete(client) == "ok"
    benched = groq_svc.model_status()["benched"][CHAIN[0]]
    assert benched["reason"] == "rate limited"
    assert 30 <= benched["seconds_left"] <= 42


def test_ordinary_bad_request_is_not_swallowed():
    # A malformed request is our bug, not the model's; failing over would hide it.
    err = _status_error(groq.BadRequestError, 400, "invalid_request_error")
    client = FakeClient({CHAIN[0]: err, CHAIN[1]: _completion("unused"),
                         CHAIN[2]: _completion("unused")})
    with pytest.raises(groq.BadRequestError):
        _complete(client)
    assert groq_svc.model_status()["benched"] == {}


def test_server_error_tries_next_model_without_benching():
    err = _status_error(groq.InternalServerError, 503)
    client = FakeClient({CHAIN[0]: err, CHAIN[1]: _completion("ok"),
                         CHAIN[2]: _completion("unused")})
    assert _complete(client) == "ok"
    assert groq_svc.model_status()["benched"] == {}


def test_every_model_gone_raises_so_callers_fall_back():
    client = FakeClient({m: _gone() for m in CHAIN})
    with pytest.raises(RuntimeError, match="no Groq model answered"):
        _complete(client)
    # Once all are benched, the next call fails fast without any request.
    client.calls.clear()
    with pytest.raises(RuntimeError):
        _complete(client)
    assert client.calls == []


def test_whole_chain_shares_one_timeout_budget():
    client = FakeClient({m: _completion("ok") for m in CHAIN})
    _complete(client, timeout=2.0)
    assert 0 < client.calls[0]["timeout"] <= 2.0


async def test_rewrite_survives_a_retired_model(monkeypatch):
    client = FakeClient({CHAIN[0]: _gone(),
                         CHAIN[1]: _completion("LLM hallucination factual errors"),
                         CHAIN[2]: _completion("unused")})
    monkeypatch.setattr(groq_svc, "_get_client", lambda: client)
    assert await groq_svc.rewrite("when AI makes up fake facts") == \
        "LLM hallucination factual errors"


async def test_rewrite_returns_query_when_no_model_answers(monkeypatch):
    client = FakeClient({m: _gone() for m in CHAIN})
    monkeypatch.setattr(groq_svc, "_get_client", lambda: client)
    assert await groq_svc.rewrite("when AI makes up fake facts") == \
        "when AI makes up fake facts"


def test_request_options_per_family():
    oss = groq_svc._request_options("openai/gpt-oss-120b", 60)
    assert oss["reasoning_effort"] == "low" and oss["include_reasoning"] is False
    # Reasoning is spent from the same budget; the answer must keep its 60.
    assert oss["max_completion_tokens"] > 60
    qwen = groq_svc._request_options("qwen/qwen3.8-27b", 60)
    assert qwen == {"reasoning_effort": "none", "max_completion_tokens": 60}
    assert groq_svc._request_options("some/future-model", 60) == {"max_completion_tokens": 60}


def test_explain_cache_key_follows_the_head_of_the_chain(monkeypatch):
    before = groq_svc.explain_cache_key("1706.03762", "abstract")
    monkeypatch.setattr(config, "GROQ_MODELS", ["qwen/qwen3.8-27b"])
    assert groq_svc.explain_model() == "qwen/qwen3.8-27b"
    assert groq_svc.explain_cache_key("1706.03762", "abstract") != before

