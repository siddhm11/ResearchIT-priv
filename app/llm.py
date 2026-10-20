"""
One way for the redesign's features to call a language model.

    data, call = await llm.complete_json("briefing", system, user, max_tokens=900, timeout=12)

Returns the first JSON object in the reply (or None) and a record of the call.
Every call is logged to `llm_calls`, which turso_sync replicates, and new calls
stop for the UTC day once LLM_DAILY_CALLS or LLM_DAILY_USD_CAP is reached; the
log survives restarts, so a redeploy doesn't reset the day's spend.

Providers (LLM_PROVIDER):
    groq   the GROQ_MODELS failover chain in app/groq_svc.py (the default).
           JSON comes from the prompt, not Groq's JSON mode, which not every
           model in the chain supports; the reply is parsed here instead.
    fake   canned replies for tests, set with fake_reply(). Refuses to load
           unless RESEARCHIT_ENV=test, so production can never use it.

Model output is untrusted: callers check it before showing anything.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

import aiosqlite

from app import config, errors


@dataclass
class Call:
    feature: str
    provider: str
    model: str = ""
    chars_in: int = 0
    chars_out: int = 0
    usd_est: float = 0.0
    latency_ms: int = 0
    status: str = "ok"          # ok | failed | unparsable | over_budget | unavailable


class BudgetExceeded(Exception):
    pass


# ── fake provider (tests only) ───────────────────────────────────────────────

_fake_replies: list = []


def fake_reply(*replies) -> None:
    """Queue replies for the fake provider: strings, or callables(system, user) -> str."""
    if config.RESEARCHIT_ENV != "test":
        raise RuntimeError("the fake model provider is for tests only")
    _fake_replies.extend(replies)


def _provider() -> str:
    p = config.LLM_PROVIDER
    if p == "fake" and config.RESEARCHIT_ENV != "test":
        print("[llm] WARNING: LLM_PROVIDER=fake outside tests; using groq")
        return "groq"
    return p if p in ("groq", "fake") else "groq"


def configured_model() -> str:
    """The model a cached result is attributed to: the head of the chain."""
    if _provider() == "fake":
        return "fake"
    return config.GROQ_MODELS[0] if config.GROQ_MODELS else ""


# ── budget ───────────────────────────────────────────────────────────────────

def _prices() -> dict[str, tuple[float, float]]:
    out = {}
    for part in filter(None, config.LLM_PRICES.split(";")):
        try:
            model, rates = part.split("=", 1)
            cin, cout = rates.split(":", 1)
            out[model.strip()] = (float(cin), float(cout))
        except ValueError:
            continue
    return out


def estimate_usd(model: str, chars_in: int, chars_out: int) -> float:
    cin, cout = _prices().get(model, (0.0, 0.0))
    return round((chars_in * cin + chars_out * cout) / 1e6, 6)


async def today() -> dict:
    """Calls and estimated spend so far this UTC day."""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        cur = await conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(usd_est), 0) FROM llm_calls "
            "WHERE created_at >= date('now') AND status != 'over_budget'")
        calls, usd = await cur.fetchone()
    return {"calls": calls, "usd": round(usd, 4),
            "call_cap": config.LLM_DAILY_CALLS, "usd_cap": config.LLM_DAILY_USD_CAP}


async def within_budget() -> bool:
    t = await today()
    if t["calls"] >= config.LLM_DAILY_CALLS:
        return False
    return not (config.LLM_DAILY_USD_CAP > 0 and t["usd"] >= config.LLM_DAILY_USD_CAP)


async def _log(call: Call) -> None:
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            await conn.execute(
                "INSERT INTO llm_calls (feature, provider, model, chars_in, chars_out, usd_est, "
                "latency_ms, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (call.feature, call.provider, call.model, call.chars_in, call.chars_out,
                 call.usd_est, call.latency_ms, call.status))
            await conn.commit()
    except Exception as e:
        errors.report("llm", "logging a model call failed", e)


# ── parsing ──────────────────────────────────────────────────────────────────

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


def first_json_object(text: str) -> dict | None:
    """The first balanced {...} in a reply that parses as a JSON object."""
    text = _FENCE.sub("", text or "")
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:i + 1])
                    except ValueError:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


# ── the call ─────────────────────────────────────────────────────────────────

async def _ask_groq(system: str, user: str, max_tokens: int, timeout: float) -> tuple[str, str]:
    from app import groq_svc
    client = groq_svc._get_client()
    if client is None:
        raise RuntimeError("no Groq client (GROQ_API_KEY unset or groq not installed)")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    result = await asyncio.to_thread(groq_svc._complete, client, messages,
                                     temperature=0.2, max_tokens=max_tokens, timeout=timeout)
    if result.truncated:
        raise RuntimeError("reply cut off at the token cap")
    return result.text, result.model


async def _ask_fake(system: str, user: str) -> tuple[str, str]:
    if not _fake_replies:
        raise RuntimeError("no fake reply queued")
    reply = _fake_replies.pop(0)
    if isinstance(reply, Exception):
        raise reply
    text = reply(system, user) if callable(reply) else reply
    return text, "fake"


async def complete_json(feature: str, system: str, user: str, *, max_tokens: int = 900,
                        timeout: float = 12.0) -> tuple[dict | None, Call]:
    """The model's reply as a JSON object, or None; never raises."""
    provider = _provider()
    call = Call(feature=feature, provider=provider, chars_in=len(system) + len(user))
    if not await within_budget():
        call.status = "over_budget"
        await _log(call)
        return None, call
    t0 = time.perf_counter()
    try:
        if provider == "fake":
            text, model = await _ask_fake(system, user)
        else:
            text, model = await asyncio.wait_for(_ask_groq(system, user, max_tokens, timeout), timeout + 1)
    except Exception as e:
        call.latency_ms = int((time.perf_counter() - t0) * 1000)
        # Every model benched (rate limits on Groq's free tier) or no key: the
        # paper is not the problem, so callers retry soon instead of caching it.
        text = str(e)
        call.status = "unavailable" if ("no Groq model answered" in text or "no Groq client" in text) else "failed"
        print(f"[llm] {feature} call {call.status}: {type(e).__name__}: {text[:160]}")
        await _log(call)
        return None, call
    call.latency_ms = int((time.perf_counter() - t0) * 1000)
    call.model, call.chars_out = model, len(text or "")
    call.usd_est = estimate_usd(model, call.chars_in, call.chars_out)
    data = first_json_object(text)
    if data is None:
        call.status = "unparsable"
    await _log(call)
    return data, call
