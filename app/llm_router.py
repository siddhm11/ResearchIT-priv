"""Bounded, model-first Groq failover for one serving event loop.

Only aliases and normalized outcomes leave this module; never log provider
exception text (it can contain request content). No network work at import.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

from app import config

DEFAULT_MODELS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")
SUPPORTED_MODELS = {*DEFAULT_MODELS, "openai/gpt-oss-120b"}
TASK_BUDGETS = {"rewrite": 2.0, "summary": 4.0, "explain": 8.0}


@dataclass(frozen=True)
class Credential:
    alias: str
    secret: str = field(repr=False)
    org: str = "unknown"
    project: str = "unknown"


@dataclass(frozen=True)
class Generation:
    outcome: str
    text: str | None = None
    model: str | None = None
    credential: str | None = None
    attempts: int = 0
    elapsed_ms: int = 0
    total_tokens: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None


def credentials_from_config() -> tuple[Credential, ...]:
    """Resolve references to environment secrets; malformed pools fail closed."""
    if not config.GROQ_KEY_POOL_JSON.strip():
        return ((Credential("primary", config.GROQ_API_KEY),)
                if config.GROQ_API_KEY else ())
    try:
        rows = json.loads(config.GROQ_KEY_POOL_JSON)
        if not isinstance(rows, list) or len(rows) > 16:
            raise ValueError
        result, aliases, secrets = [], set(), set()
        for row in rows:
            if not isinstance(row, dict) or set(row) - {"id", "secret_env", "org", "project"}:
                raise ValueError
            alias, name = row["id"], row["secret_env"]
            if not isinstance(alias, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", alias):
                raise ValueError
            if alias in aliases or not isinstance(name, str) or not re.fullmatch(r"GROQ_API_KEY(?:_[A-Z0-9]+)?", name):
                raise ValueError
            aliases.add(alias)
            org, project = row.get("org", "unknown"), row.get("project", "unknown")
            if not all(isinstance(v, str) and v.strip() for v in (org, project)):
                raise ValueError
            secret = (config.GROQ_API_KEY if name == "GROQ_API_KEY"
                      else os.environ.get(name, "")).strip()
            if secret and secret not in secrets:
                result.append(Credential(alias, secret, org, project))
                secrets.add(secret)
        return tuple(result)
    except (ValueError, TypeError, KeyError):
        raise ValueError("Invalid GROQ_KEY_POOL_JSON; check credential references") from None


def configured_models(task: str) -> tuple[str, ...]:
    """Only models with explicit parameter adapters are eligible."""
    try:
        data = json.loads(config.GROQ_MODEL_ORDER_JSON or "{}")
        if not isinstance(data, dict) or set(data) - set(TASK_BUDGETS):
            raise ValueError
        default = list(config.GROQ_MODELS[:2])
        for models in [default, *data.values()]:
            if (not isinstance(models, list) or not 1 <= len(models) <= 2
                    or any(not isinstance(m, str) or m not in SUPPORTED_MODELS for m in models)
                    or len(set(models)) != len(models)):
                raise ValueError
        return tuple(data.get(task, default))
    except (ValueError, TypeError):
        raise ValueError("Invalid GROQ_MODEL_ORDER_JSON; use supported task/model IDs") from None


def model_options(model: str, task: str) -> dict:
    """Short tasks use Qwen instruct mode; GPT-OSS needs reasoning headroom."""
    tokens = {"rewrite": 80, "summary": 220, "explain": 280}[task]
    if model == "qwen/qwen3.8-27b":
        return {"reasoning_effort": "none", "reasoning_format": "hidden",
                "max_completion_tokens": tokens, "temperature": 0.2}
    if model in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
        return {"reasoning_effort": "low", "include_reasoning": False,
                "max_completion_tokens": tokens + 1024, "temperature": 0.2}
    raise ValueError("Model has no parameter adapter")


def _duration(value: str) -> float:
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", str(value))
        if not parts or "".join(n + unit for n, unit in parts) != value:
            return 0.0
        seconds = sum(float(n) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[u]
                      for n, u in parts)
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


def retry_delay(headers: dict) -> float:
    """Respect the longest applicable exhausted limit, including HTTP dates."""
    value = headers.get("retry-after", "")
    seconds = _duration(value)
    if value and not seconds:
        try:
            seconds = max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            pass
    waits = [seconds]
    for kind in ("requests", "tokens"):
        if headers.get(f"x-ratelimit-remaining-{kind}") == "0":
            waits.append(_duration(headers.get(f"x-ratelimit-reset-{kind}", "")))
    return max(waits)


@dataclass
class Gate:
    until: float = 0
    failures: int = 0
    probing: bool = False


class Router:
    """State changes are atomic between awaits; I/O never holds a state lock."""

    def __init__(self, credentials: tuple[Credential, ...], transport=None,
                 clock: Callable[[], float] = time.monotonic, max_inflight: int = 4,
                 header_observer=None):
        self.credentials = credentials
        self.transport = transport or self._send
        self.clock = clock
        self.header_observer = header_observer
        self.clients = {}
        self.gates: dict[tuple, Gate] = {}
        self.inflight: set[str] = set()
        self.changed = asyncio.Event()
        self.capacity = asyncio.Semaphore(max_inflight)
        self.cursor = 0
        self.provider_failures: set[tuple[str, str]] = set()
        self.failure_epoch = 0
        self.unknown_membership = any(c.org == "unknown" for c in credentials)

    def _scopes(self, key: Credential, model: str) -> list[tuple]:
        org = "unknown" if self.unknown_membership else key.org
        project = key.project if key.project != "unknown" else "*"
        return [("provider",), ("model", model), ("key", key.alias),
                ("org", org), ("org_model", org, model),
                ("project", org, project), ("project_model", org, project, model),
                ("route", key.alias, model)]

    def _open(self, scope: tuple, delay: float | None = None) -> None:
        gate = self.gates.setdefault(scope, Gate())
        gate.failures += 1
        backoff = min(300, 30 * 2 ** min(gate.failures - 1, 4))
        wait = delay if delay is not None and delay > 0 else backoff
        gate.until = max(gate.until, self.clock() + wait + random.uniform(0, 1))
        self.changed.set()

    def _disable(self, scope: tuple) -> None:
        self.gates.setdefault(scope, Gate()).until = math.inf
        self.changed.set()

    async def _reserve(self, model: str, tried: set[str]):
        while True:
            self.changed.clear()
            waiting = False
            for offset in range(len(self.credentials)):
                index = (self.cursor + offset) % len(self.credentials)
                key = self.credentials[index]
                if key.alias in tried:
                    continue
                scopes = self._scopes(key, model)
                gates = [self.gates[s] for s in scopes if s in self.gates]
                if any(g.until > self.clock() for g in gates):
                    continue
                if key.alias in self.inflight or any(g.probing for g in gates):
                    waiting = True
                    continue
                probes = [g for g in gates if g.until > 0]
                for gate in probes:
                    gate.probing = True
                self.inflight.add(key.alias)
                self.cursor = (index + 1) % len(self.credentials)
                snapshot = [(s, self.gates[s], self.gates[s].failures, self.gates[s].until)
                            for s in scopes if s in self.gates]
                return key, probes, snapshot, self.failure_epoch
            if not waiting:
                return None
            # Busy healthy keys must not demote Qwen merely due to local load.
            await self.changed.wait()

    def _failure(self, exc: Exception, key: Credential, model: str) -> str:
        scopes = self._scopes(key, model)
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", {})
        body = body.get("error", body) if isinstance(body, dict) else {}
        code = body.get("code", "") if isinstance(body, dict) else ""
        response = getattr(exc, "response", None)
        headers = {str(k).lower(): str(v) for k, v in getattr(response, "headers", {}).items()}
        if self.header_observer:
            self.header_observer(model, headers)
        if status == 401:
            self._disable(scopes[2])
            return "invalid_key"
        if code in {"model_decommissioned", "model_deprecated"}:
            self._disable(scopes[1])
            return "model_unavailable"
        if code == "model_permission_blocked_org":
            self._disable(scopes[4])
            return "permission"
        if code == "model_permission_blocked_project":
            # Unknown project membership requires the wider org/model scope.
            self._disable(scopes[4] if key.project == "unknown" else scopes[6])
            return "permission"
        if status == 402 or code in {"insufficient_quota", "billing_hard_limit_reached", "spend_limit_exceeded"}:
            self._disable(scopes[3])
            return "account_unavailable"
        if status == 429:
            self._open(scopes[4], retry_delay(headers))
            return "rate_limit"
        if status in {403, 404} or code == "model_not_found":
            self._open(scopes[-1])
            return "route_unavailable"
        if status in {408, 409} or (isinstance(status, int) and status >= 500) or isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}:
            self.failure_epoch += 1
            self._open(scopes[-1])
            model_gate = self.gates.setdefault(scopes[1], Gate())
            model_gate.failures += 1
            if model_gate.failures >= 3:
                self._open(scopes[1])
            self.provider_failures.add((key.alias, model))
            if len({m for _, m in self.provider_failures}) >= 2:
                self._open(scopes[0])
            return "transient"
        return "request_error"

    async def _send(self, key: Credential, model: str, task: str,
                    messages: list[dict], timeout: float):
        from groq import AsyncGroq
        if key.alias not in self.clients:
            self.clients[key.alias] = AsyncGroq(api_key=key.secret, max_retries=0)
        api = self.clients[key.alias].chat.completions
        options = dict(model=model, messages=messages, timeout=timeout,
                       **model_options(model, task))
        raw_api = getattr(api, "with_raw_response", None)
        if raw_api is None:
            return await api.create(**options)
        raw = await raw_api.create(**options)
        if self.header_observer:
            self.header_observer(model, raw.headers)
        return await raw.parse()

    async def complete(self, task: str, messages: list[dict], models: tuple[str, ...],
                       validator: Callable[[str], bool], budget: float | None = None) -> Generation:
        """One total deadline; each eligible key once per model, no SDK retries."""
        started = self.clock()
        budget = TASK_BUDGETS[task] if budget is None else budget
        deadline, attempts, total_tokens = started + budget, 0, 0
        tokens_in, tokens_out = None, None

        def result(outcome, text=None, model=None, key=None):
            return Generation(outcome, text, model, key, attempts,
                              int((self.clock() - started) * 1000), total_tokens,
                              tokens_in, tokens_out)

        if budget <= 0:
            return result("deadline_exhausted")
        try:
            async with asyncio.timeout(budget):
                async with self.capacity:
                    for model in models:
                        tried: set[str] = set()
                        while reservation := await self._reserve(model, tried):
                            key, probes, snapshot, failure_epoch = reservation
                            tried.add(key.alias)
                            try:
                                remaining = deadline - self.clock()
                                if remaining <= 0:
                                    return result("deadline_exhausted")
                                attempts += 1
                                response = await self.transport(key, model, task, messages,
                                                                remaining)
                            except ImportError:
                                return result("sdk_unavailable")
                            except Exception as exc:
                                outcome = self._failure(exc, key, model)
                                print(f"[llm_router] task={task} model={model} key={key.alias} attempt={attempts} outcome={outcome}")
                                if outcome == "request_error":
                                    return result(outcome)
                                continue
                            finally:
                                self.inflight.discard(key.alias)
                                for gate in probes:
                                    gate.probing = False
                                self.changed.set()
                            # Transport success proves availability, even on abstention.
                            for scope, gate, failures, until in snapshot:
                                # Another in-flight key may just have exhausted a
                                # shared quota. An older success must not erase it.
                                if (self.gates.get(scope) is gate and gate.failures == failures
                                        and gate.until == until):
                                    self.gates.pop(scope, None)
                            if failure_epoch == self.failure_epoch:
                                self.provider_failures.clear()
                            total_tokens += getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
                            tokens_in = getattr(getattr(response, "usage", None), "prompt_tokens", None)
                            tokens_out = getattr(getattr(response, "usage", None), "completion_tokens", None)
                            choices = getattr(response, "choices", None)
                            choice = choices[0] if choices else None
                            message = getattr(choice, "message", None)
                            text = getattr(message, "content", None)
                            text = text.strip() if isinstance(text, str) else ""
                            finish = getattr(choice, "finish_reason", None)
                            if getattr(message, "refusal", None) or finish == "content_filter" or text.upper().startswith("INSUFFICIENT"):
                                return result("abstained", model=model, key=key.alias)
                            if (finish != "stop" or not text or "<think" in text.lower()
                                    or "</think" in text.lower() or not validator(text)):
                                break  # Different key cannot fix this model's output.
                            return result("success", text, model, key.alias)
            return result("all_routes_unavailable")
        except TimeoutError:
            return result("deadline_exhausted")

    async def aclose(self) -> None:
        """Close every lazily created connection pool."""
        clients, self.clients = self.clients, {}
        await asyncio.gather(*(client.close() for client in clients.values()), return_exceptions=True)


_router: Router | None = None


def get_router() -> Router:
    """Configuration changes take effect on restart, including secret replacement."""
    global _router
    if _router is None:
        from app.groq_svc import _note_window
        _router = Router(credentials_from_config(), header_observer=_note_window)
    return _router


async def aclose() -> None:
    global _router
    router, _router = _router, None
    if router is not None:
        await router.aclose()
