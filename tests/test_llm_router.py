"""Failure injection: no credentials, models, network, or production storage."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from app import config, groq_svc, llm_router as lr

QWEN, FALLBACK = lr.DEFAULT_MODELS


class Failure(Exception):
    def __init__(self, status, code="", headers=None):
        super().__init__("SECRET must never be logged")
        self.status_code = status
        self.body = {"error": {"code": code}}
        self.response = NS(headers=headers or {})


def answer(text="A useful answer.", finish="stop", refusal=None):
    return NS(choices=[NS(message=NS(content=text, refusal=refusal), finish_reason=finish)],
              usage=NS(total_tokens=12))


def keys(shared=False):
    return tuple(lr.Credential(alias, f"SECRET-{alias}", "shared" if shared else alias, "app")
                 for alias in ("a", "b"))


class Script:
    def __init__(self, results):
        self.results, self.calls = iter(results), []

    async def __call__(self, key, model, task, messages, timeout):
        self.calls.append((model, key.alias))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


async def complete(router, **kwargs):
    return await router.complete("explain", [], (QWEN, FALLBACK), bool, **kwargs)


async def test_other_key_on_qwen_precedes_fallback(capsys):
    script = Script([Failure(401), answer()])
    router = lr.Router(keys(), script)
    result = await complete(router)
    assert result.model == QWEN and result.credential == "b" and result.attempts == 2
    assert script.calls == [(QWEN, "a"), (QWEN, "b")]
    assert "SECRET" not in capsys.readouterr().out
    script.results = iter([answer()])
    await complete(router)
    assert script.calls[-1] == (QWEN, "b")  # invalid A stays disabled


async def test_all_qwen_routes_before_fallback_and_recover_priority():
    clock = [100.0]
    script = Script([Failure(403), Failure(403), answer(), answer()])
    router = lr.Router(keys(), script, clock=lambda: clock[0])
    result = await complete(router)
    assert result.model == FALLBACK
    assert script.calls == [(QWEN, "a"), (QWEN, "b"), (FALLBACK, "a")]
    clock[0] += 32
    result = await complete(router)
    assert result.model == QWEN  # last successful model never sticks


async def test_shared_quota_skips_sibling_but_not_other_model():
    script = Script([Failure(429, headers={"retry-after": "120"}), answer()])
    router = lr.Router(keys(shared=True), script)
    result = await complete(router)
    assert result.model == FALLBACK
    assert script.calls == [(QWEN, "a"), (FALLBACK, "b")]


async def test_independent_quota_can_still_serve_qwen():
    script = Script([Failure(429), answer()])
    result = await complete(lr.Router(keys(), script))
    assert result.model == QWEN and result.credential == "b"


async def test_unknown_membership_does_not_invent_quota():
    script = Script([Failure(429), answer()])
    router = lr.Router((lr.Credential("a", "a"), lr.Credential("b", "b", "known")), script)
    assert (await complete(router)).model == FALLBACK
    assert script.calls[1][0] == FALLBACK


@pytest.mark.parametrize("code", ["model_permission_blocked_org", "model_permission_blocked_project"])
async def test_permission_is_scoped_to_model(code):
    script = Script([Failure(403, code), answer()])
    result = await complete(lr.Router(keys(shared=True), script))
    assert result.model == FALLBACK and result.attempts == 2


async def test_account_exhaustion_skips_all_models_for_shared_account():
    script = Script([Failure(429, "insufficient_quota")])
    result = await complete(lr.Router(keys(shared=True), script))
    assert result.outcome == "all_routes_unavailable" and result.attempts == 1


async def test_retired_model_does_not_sweep_keys():
    script = Script([Failure(400, "model_decommissioned"), answer()])
    result = await complete(lr.Router(keys(), script))
    assert result.model == FALLBACK
    assert len(script.calls) == 2


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_bad_request_is_terminal(status):
    script = Script([Failure(status)])
    result = await complete(lr.Router(keys(), script))
    assert result.outcome == "request_error" and result.attempts == 1


@pytest.mark.parametrize("response", [answer("INSUFFICIENT"), answer(refusal="No"), answer(finish="content_filter")])
async def test_abstention_does_not_shop_for_answers(response):
    script = Script([response])
    assert (await complete(lr.Router(keys(), script))).outcome == "abstained"
    assert len(script.calls) == 1


@pytest.mark.parametrize("response", [answer(""), answer(finish="length"), answer("<think>hidden</think>Answer")])
async def test_bad_output_changes_model_not_key(response):
    script = Script([response, answer()])
    assert (await complete(lr.Router(keys(), script))).model == FALLBACK
    assert script.calls == [(QWEN, "a"), (FALLBACK, "b")]


async def test_outage_stops_after_evidence_across_models():
    script = Script([Failure(503), Failure(503), Failure(503)])
    router = lr.Router(keys(), script)
    result = await complete(router)
    assert result.attempts == 3
    assert (await complete(router)).attempts == 0


async def test_total_deadline_cancels_transport_and_releases_capacity():
    cancelled = asyncio.Event()

    async def hanging(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    router = lr.Router(keys(), hanging)
    result = await complete(router, budget=.02)
    assert result.outcome == "deadline_exhausted" and cancelled.is_set()
    assert not router.inflight
    assert not router.gates  # caller deadline is not a credential failure


async def test_deadline_includes_capacity_queue():
    started = asyncio.Event()

    async def hanging(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    router = lr.Router(keys(), hanging, max_inflight=1)
    first = asyncio.create_task(complete(router))
    await started.wait()
    second = await complete(router, budget=.02)
    assert second.outcome == "deadline_exhausted" and second.attempts == 0
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not router.inflight


async def test_only_one_recovery_probe_and_busy_qwen_not_demoted():
    now = [0.0]
    script = Script([Failure(429)])
    router = lr.Router(keys(shared=True), script, clock=lambda: now[0])
    await router.complete("explain", [], (QWEN,), bool)
    now[0] = 32
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def delayed(key, model, *args):
        calls.append((model, key.alias))
        entered.set()
        await release.wait()
        return answer()

    router.transport = delayed
    first = asyncio.create_task(complete(router))
    await entered.wait()
    second = asyncio.create_task(complete(router))
    await asyncio.sleep(0)
    assert len(calls) == 1
    release.set()
    results = await asyncio.gather(first, second)
    assert all(r.model == QWEN for r in results)


async def test_older_success_cannot_clear_new_shared_quota_failure():
    started, release = asyncio.Event(), asyncio.Event()

    async def transport(key, *args):
        if key.alias == "a":
            started.set()
            await release.wait()
            return answer()
        raise Failure(429)

    router = lr.Router(keys(shared=True), transport)
    first = asyncio.create_task(router.complete("explain", [], (QWEN,), bool))
    await started.wait()
    await router.complete("explain", [], (QWEN,), bool)
    release.set()
    await first
    assert (await router.complete("explain", [], (QWEN,), bool)).attempts == 0


def test_retry_headers_cover_both_exhausted_limits():
    assert lr.retry_delay({"retry-after": "2", "x-ratelimit-remaining-tokens": "0",
                           "x-ratelimit-reset-tokens": "1m2.5s"}) == 62.5
    assert lr.retry_delay({"retry-after": "nan"}) == 0
    assert lr.retry_delay({"retry-after": "bad"}) == 0


def test_pool_deduplicates_secrets_and_ignores_blanks(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "same")
    monkeypatch.setenv("GROQ_API_KEY_2", "same")
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    monkeypatch.setattr(config, "GROQ_KEY_POOL_JSON", json.dumps([
        {"id": "a", "secret_env": "GROQ_API_KEY"},
        {"id": "b", "secret_env": "GROQ_API_KEY_2"},
        {"id": "c", "secret_env": "GROQ_API_KEY_3"}]))
    pool = lr.credentials_from_config()
    assert len(pool) == 1 and "same" not in repr(pool)


@pytest.mark.parametrize("raw", ['{}', '[{"id":"secret","key":"SECRET"}]', 'invalid'])
def test_bad_configuration_has_sanitized_error(monkeypatch, raw):
    monkeypatch.setattr(config, "GROQ_KEY_POOL_JSON", raw)
    with pytest.raises(ValueError, match="Invalid GROQ_KEY_POOL_JSON") as exc:
        lr.credentials_from_config()
    assert "SECRET" not in str(exc.value)


def test_model_adapter_parameters_are_not_shared():
    qwen = lr.model_options(QWEN, "rewrite")
    fallback = lr.model_options(FALLBACK, "rewrite")
    assert qwen["reasoning_effort"] == "none" and "include_reasoning" not in qwen
    assert fallback["include_reasoning"] is False and "reasoning_format" not in fallback


async def test_sdk_retries_disabled_and_clients_closed(monkeypatch):
    groq = pytest.importorskip("groq")
    client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=answer()))), close=AsyncMock())
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return client

    monkeypatch.setattr(groq, "AsyncGroq", factory)
    router = lr.Router(keys())
    assert (await complete(router)).outcome == "success"
    assert created[0]["max_retries"] == 0
    await router.aclose()
    client.close.assert_awaited_once()


async def test_services_preserve_output_contract_and_escape_html(monkeypatch):
    monkeypatch.setattr(config, "GROQ_FAILOVER_ENABLED", True)
    fake = AsyncMock(side_effect=[lr.Generation("all_routes_unavailable"),
                                 lr.Generation("success", '<script>bad</script> **safe**')])
    monkeypatch.setattr(groq_svc, "_routed", fake)
    assert await groq_svc.rewrite("when AI makes stuff up") == "when AI makes stuff up"
    text = await groq_svc.generate_search_summary("topic", [{"title": "t", "abstract": "a"}] * 2)
    assert "<script>" not in text and "<strong>safe</strong>" in text


async def test_explanation_singleflight_survives_one_cancelled_waiter(monkeypatch):
    monkeypatch.setattr(config, "GROQ_FAILOVER_ENABLED", True)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def routed(*args, **kwargs):
        calls.append(args)
        entered.set()
        await release.wait()
        return lr.Generation("success", "Explanation", FALLBACK)

    monkeypatch.setattr(groq_svc, "_routed", routed)
    first = asyncio.create_task(groq_svc.explain_paper_result("title", "a" * 200))
    await entered.wait()
    second = asyncio.create_task(groq_svc.explain_paper_result("title", "a" * 200))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert (await second).model == FALLBACK and len(calls) == 1
    assert not groq_svc._explanation_tasks


def test_cache_identity_includes_title_and_actual_model():
    key = groq_svc.explain_cache_key("id", "abstract", title="t", model=QWEN)
    assert key != groq_svc.explain_cache_key("id", "abstract", title="changed", model=QWEN)
    assert key != groq_svc.explain_cache_key("id", "abstract", title="t", model=FALLBACK)


async def test_no_keys_and_zero_budget_make_no_requests():
    script = Script([])
    assert (await complete(lr.Router((), script))).attempts == 0
    result = await complete(lr.Router(keys(), script), budget=0)
    assert result.outcome == "deadline_exhausted" and result.attempts == 0


async def test_missing_sdk_degrades_without_repeating(monkeypatch):
    async def missing(*args):
        raise ImportError("No SDK")
    result = await complete(lr.Router(keys(), missing))
    assert result.outcome == "sdk_unavailable" and result.attempts == 1


async def test_project_restriction_keeps_other_project_eligible():
    credentials = (lr.Credential("a", "a", "org", "first"),
                   lr.Credential("b", "b", "org", "second"))
    script = Script([Failure(403, "model_permission_blocked_project"), answer()])
    assert (await complete(lr.Router(credentials, script))).model == QWEN


async def test_cancelling_last_explanation_waiter_stops_generation(monkeypatch):
    monkeypatch.setattr(config, "GROQ_FAILOVER_ENABLED", True)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def hanging(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(groq_svc, "_routed", hanging)
    waiter = asyncio.create_task(groq_svc.explain_paper_result("cancel", "a" * 200))
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert cancelled.is_set() and not groq_svc._explanation_tasks


async def test_explanations_only_canary_keeps_other_tasks_on_legacy_path(monkeypatch):
    monkeypatch.setattr(config, "GROQ_FAILOVER_ENABLED", True)
    monkeypatch.setattr(config, "GROQ_FAILOVER_TASKS", frozenset({"explain"}))
    monkeypatch.setattr(groq_svc, "_get_client", lambda: None)
    routed = AsyncMock(return_value=lr.Generation("success", "Explanation", QWEN))
    monkeypatch.setattr(groq_svc, "_routed", routed)
    assert await groq_svc.rewrite("when AI makes stuff up") == "when AI makes stuff up"
    assert await groq_svc.generate_search_summary("topic", [{"title": "t", "abstract": "a"}] * 2) is None
    assert await groq_svc.explain_paper("title", "a" * 200) == "Explanation"
    routed.assert_awaited_once()


@pytest.mark.parametrize("raw", ['[]', '{"rewrite":["unknown"]}', '{"rewrite":[]}', '{"bad":[]}'])
def test_unknown_models_and_bad_model_config_fail_closed(monkeypatch, raw):
    monkeypatch.setattr(config, "GROQ_MODEL_ORDER_JSON", raw)
    with pytest.raises(ValueError, match="Invalid GROQ_MODEL_ORDER_JSON"):
        lr.configured_models("rewrite")


async def test_inventory_command_does_not_probe_by_default(monkeypatch):
    from scripts.check_groq_routes import inspect_routes
    monkeypatch.setattr(config, "GROQ_KEY_POOL_JSON", "")
    monkeypatch.setattr(config, "GROQ_MODEL_ORDER_JSON", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "SECRET")
    report = await inspect_routes()
    assert report["routes"][0]["catalog"] == "not_checked"
    assert "SECRET" not in json.dumps(report)


async def test_provider_reset_controls_quota_recovery(monkeypatch):
    monkeypatch.setattr(lr.random, "uniform", lambda *args: 0)
    now = [0.0]
    script = Script([Failure(429, headers={"retry-after": "2"}), answer()])
    router = lr.Router(keys(shared=True), script, clock=lambda: now[0])
    await router.complete("rewrite", [], (QWEN,), bool)
    now[0] = 1.9
    assert (await router.complete("rewrite", [], (QWEN,), bool)).attempts == 0
    now[0] = 2.0
    assert (await router.complete("rewrite", [], (QWEN,), bool)).outcome == "success"


async def test_routed_generation_preserves_ai_usage(monkeypatch):
    from app import ai_usage
    monkeypatch.setattr(config, "GROQ_MODELS", [QWEN, FALLBACK, "openai/gpt-oss-120b"])
    monkeypatch.setattr(config, "GROQ_MODEL_ORDER_JSON", "")
    generated = lr.Generation("success", "An explanation.", FALLBACK,
                              tokens_in=20, tokens_out=30)
    router = NS(complete=AsyncMock(return_value=generated))
    monkeypatch.setattr(lr, "get_router", lambda: router)
    log = AsyncMock()
    monkeypatch.setattr(ai_usage, "log_call", log)
    count = []
    monkeypatch.setattr(groq_svc, "_count", lambda *args: count.append(args))
    await groq_svc._routed("explain", [{"role": "user", "content": "text"}], subject="paper-id")
    completion = log.call_args.args[1]
    assert completion.tokens_in == 20 and completion.tokens_out == 30
    assert completion.model == FALLBACK and completion.rank == 1
    assert log.call_args.kwargs["subject"] == "paper-id"
    assert count == [("explain", "written")]
    assert router.complete.call_args.args[2] == (QWEN, FALLBACK)


async def test_real_sdk_raw_response_preserves_rate_headers(monkeypatch):
    groq = pytest.importorskip("groq")
    import httpx
    original = groq.AsyncGroq
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "7900"}, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": QWEN,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    def client(**kwargs):
        return original(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(groq, "AsyncGroq", client)
    headers = []
    router = lr.Router(keys(), header_observer=lambda m, h: headers.append((m, h)))
    try:
        result = await complete(router)
    finally:
        await router.aclose()
    assert result.text == "OK" and result.tokens_in == 10 and result.tokens_out == 2
    assert headers[0][1]["x-ratelimit-remaining-tokens"] == "7900"
    assert requests[0]["reasoning_effort"] == "none"
