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
                              temperature=0.1, max_tokens=60, timeout=timeout).text


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



# ── Output hygiene ───────────────────────────────────────────────────────────

def _only(client_text: str, finish: str = "stop") -> FakeClient:
    return FakeClient({m: _completion(client_text, finish) for m in CHAIN})


def test_completion_reports_model_and_truncation():
    result = groq_svc._complete(_only("abc", "length"), [{"role": "user", "content": "q"}],
                                temperature=0.1, max_tokens=60, timeout=5.0)
    assert result == groq_svc.Completion("abc", CHAIN[0], True)


def test_inline_reasoning_is_stripped():
    assert groq_svc._clean("<think>plan the answer</think>LLM hallucination") == \
        "LLM hallucination"
    # Cut off before the closing tag: nothing after <think> is an answer.
    assert groq_svc._clean("<think>still thinking when the cap hit") == ""


def test_typographic_hyphens_become_ascii():
    assert groq_svc._clean("Large\u2011language\u00a0model") == "Large-language model"


async def test_truncated_rewrite_is_discarded(monkeypatch):
    # gpt-oss at a too-small cap returned "LLa" for "the llama model by facebook".
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _only("LLa", "length"))
    assert await groq_svc.rewrite("the llama model by facebook") == \
        "the llama model by facebook"


async def test_a_rewrite_that_answers_the_reader_is_discarded(monkeypatch):
    # Production, 2026-10-22: the model answered a greeting instead of rewriting it.
    monkeypatch.setattr(groq_svc, "_get_client",
                        lambda: _only("I'm doing well, thank you! How can I help you today?"))
    assert await groq_svc.rewrite("HI HOW ARE YOU >") == "HI HOW ARE YOU >"


@pytest.mark.parametrize("query,rewritten,reply", [
    ("when AI makes up fake facts", "LLM hallucination factual errors", False),
    ("papers like attention is all you need", "attention is all you need", False),
    ("automating know your business verification", "automated KYB verification", False),
    ("false positives in statistics", "type I error rate", False),
    ("hey whats up there", "Hey there!", True),
    ("how are u today", "I am fine, thanks for asking", True),
    ("tell me about llms", "large language models " * 5, True),
])
def test_what_reads_like_a_reply(query, rewritten, reply):
    assert groq_svc._reads_like_a_reply(query, rewritten) is reply


async def test_truncated_overview_keeps_whole_sentences(monkeypatch):
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _only(
        "**Transformers** replaced recurrence. Scaling laws predict loss. Underpinning these", "length"))
    papers = [{"title": "A", "abstract": "x"}, {"title": "B", "abstract": "y"}]
    html = await groq_svc.generate_search_summary("llm pretraining", papers)
    assert html == "<strong>Transformers</strong> replaced recurrence. Scaling laws predict loss."


async def test_truncated_explanation_without_a_full_sentence_is_dropped(monkeypatch):
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _only("The problem is that", "length"))
    assert await groq_svc.explain_paper("Title", "a" * 200) is None


async def test_complete_explanation_is_returned_untouched(monkeypatch):
    text = "One problem. One method. One finding."
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _only(text))
    assert await groq_svc.explain_paper("Title", "a" * 200) == text


# ── The overview is rendered with |safe ──────────────────────────────────────

async def test_overview_markup_from_the_model_is_escaped(monkeypatch):
    # The prompt carries the URL's query, so a crafted link can ask the model
    # to echo markup. Only the **bold** conversion may produce tags.
    monkeypatch.setattr(groq_svc, "_get_client", lambda: _only(
        'See **this** <img src=x onerror="alert(1)"> and <script>x()</script>.'))
    papers = [{"title": "A", "abstract": "x"}, {"title": "B", "abstract": "y"}]
    html = await groq_svc.generate_search_summary("q", papers)
    assert "<img" not in html and "<script" not in html
    assert "&lt;img" in html and "<strong>this</strong>" in html


# ── usage and rate-limit windows (app/ai_usage.py reads these) ───────────────

class RawFakeClient(FakeClient):
    """Like FakeClient, but through with_raw_response, carrying Groq's headers."""

    def __init__(self, behaviour: dict, headers: dict):
        super().__init__(behaviour)
        self.headers = headers
        raw = SimpleNamespace(create=self._raw_create)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create, with_raw_response=raw))

    def _raw_create(self, **kwargs):
        result = self._create(**kwargs)
        return SimpleNamespace(headers=self.headers, parse=lambda: result)


HEADERS = {"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "987",
           "x-ratelimit-reset-requests": "1m12.4s", "x-ratelimit-limit-tokens": "8000",
           "x-ratelimit-remaining-tokens": "6512", "x-ratelimit-reset-tokens": "11.2s"}


def _with_usage(text, prompt, completion):
    c = _completion(text)
    c.usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    return c


def test_a_completion_carries_tokens_and_the_models_place_in_the_chain():
    groq_svc._windows.clear()
    client = RawFakeClient({CHAIN[0]: _status_error(groq.InternalServerError, 500),
                            CHAIN[1]: _with_usage("ok", 812, 143)}, HEADERS)
    result = groq_svc._complete(client, [{"role": "user", "content": "q"}],
                                temperature=0.1, max_tokens=60, timeout=5)
    assert (result.model, result.tokens_in, result.tokens_out, result.rank) == (CHAIN[1], 812, 143, 1)


def test_rate_limit_headers_are_kept_per_model():
    groq_svc._windows.clear()
    client = RawFakeClient({CHAIN[0]: _with_usage("ok", 10, 2)}, HEADERS)
    groq_svc._complete(client, [{"role": "user", "content": "q"}], temperature=0.1, max_tokens=60, timeout=5)
    w = groq_svc.windows()[CHAIN[0]]
    assert (w["requests_left"], w["requests_limit"], w["tokens_left"], w["tokens_limit"]) == (987, 1000, 6512, 8000)
    assert w["tokens_reset"] == "11.2s" and w["at"] > 0


def test_a_429_still_records_the_window_it_ran_into():
    groq_svc._windows.clear()
    limited = _status_error(groq.RateLimitError, 429, headers={**HEADERS, "x-ratelimit-remaining-tokens": "0",
                                                               "retry-after": "7"})
    client = RawFakeClient({CHAIN[0]: limited, CHAIN[1]: _completion("ok")}, HEADERS)
    groq_svc._complete(client, [{"role": "user", "content": "q"}], temperature=0.1, max_tokens=60, timeout=5)
    assert groq_svc.windows()[CHAIN[0]]["tokens_left"] == 0


def test_models_limit_stops_at_the_first_model():
    """The named-paper lookup asks only the primary model: no fallback answer."""
    err = _status_error(groq.RateLimitError, 429, "rate_limit_exceeded", headers={"retry-after": "42"})
    client = FakeClient({CHAIN[0]: err, CHAIN[1]: _completion("fallback answer"),
                         CHAIN[2]: _completion("unused")})
    with pytest.raises(RuntimeError, match="no Groq model answered"):
        groq_svc._complete(client, [{"role": "user", "content": "q"}], temperature=0.1,
                           max_tokens=60, timeout=5.0, models=1)
    assert [c["model"] for c in client.calls] == [CHAIN[0]]
