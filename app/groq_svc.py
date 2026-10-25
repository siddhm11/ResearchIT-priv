"""
Groq LLM query rewriter — Phase 3.

Responsibilities:
  - Rewrite casual user queries into dense academic keyword strings
  - Uses the first available model in config.GROQ_MODELS (see _complete)
  - Falls back to original query on ANY error or timeout
  - Skips rewriting for queries that already look academic
  - This is an ENHANCEMENT, not a dependency — search works without it
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import NamedTuple

from app import config, llm_router

def _failover_enabled(task: str) -> bool:
    return config.GROQ_FAILOVER_ENABLED and task in config.GROQ_FAILOVER_TASKS


async def _routed(task: str, messages: list[dict], *, subject: str | None = None) -> llm_router.Generation:
    """The feature flag controls all network failover, never retrieval behavior."""
    def valid(text: str) -> bool:
        if task == "rewrite":
            return len(text) <= 200 and len(text.split()) <= 8 and "\n" not in text
        return len(text) <= 2400

    try:
        result = await llm_router.get_router().complete(
            task, messages, llm_router.configured_models(task), valid)
    except ValueError:
        result = llm_router.Generation("configuration_error")
    print(f"[groq_svc] task={task} outcome={result.outcome} model={result.model} "
          f"attempts={result.attempts} elapsed_ms={result.elapsed_ms} tokens={result.total_tokens}")
    from app import ai_usage
    completion = (Completion(result.text or "", result.model, False,
                             result.tokens_in, result.tokens_out,
                             llm_router.configured_models(task).index(result.model))
                  if result.model else None)
    feature = {"rewrite": "search_rewrite", "summary": "search_overview", "explain": "explain"}[task]
    await ai_usage.log_call(feature, completion, latency_ms=result.elapsed_ms,
                            status="ok" if result.outcome == "success" else "unavailable",
                            chars_in=sum(len(m.get("content", "")) for m in messages), subject=subject)
    if task == "explain":
        _count("explain", "written" if result.text else "declined")
    return result


# ── Client singleton ─────────────────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()


def _get_client():
    """Lazy Groq client init — only connects when first query arrives."""
    global _client
    if _client is not None:
        return _client

    if not config.GROQ_API_KEY:
        return None

    with _client_lock:
        if _client is not None:
            return _client

        # Returning None rather than raising when the package is absent. Every
        # other optional dependency here degrades quietly — BGE-M3, Zilliz,
        # LightGBM and the metadata sidecar all fall back — but this one raised
        # ModuleNotFoundError straight out of the request, so /api/search/summary
        # answered 500 and the page showed "Something went wrong" instead of
        # simply omitting the overview. The rewriter has the same call path, so
        # a missing groq also broke search itself, not just the summary.
        try:
            from groq import Groq
        except ImportError:
            print("[groq_svc] groq not installed -- summaries and rewrite disabled")
            return None

        # max_retries=0: the SDK's own retry loop would re-send a 429 or 5xx to
        # the SAME model with backoff, spending the rewrite's 2s budget on a
        # model that already said no. _complete() moves to the next model.
        _client = Groq(api_key=config.GROQ_API_KEY, max_retries=0)
        print("[groq_svc] Groq client initialized")
        return _client


# ── Model chain ──────────────────────────────────────────────────────────────
#
# Groq retires models every few months. This module used to hard-code
# llama-3.3-70b-versatile; Groq shut it down on 2026-08-16 and every call
# returned 404 for seven weeks with no visible symptom, because each caller
# degrades by design to "no rewrite / no overview / no explanation". So the
# model is now an ordered list (config.GROQ_MODELS): a model Groq reports as
# gone or rate-limited is benched for a while and the next one answers.

_GONE_COOLDOWN_S = 3600.0       # re-check a retired model hourly, not per call
_RATE_LIMIT_COOLDOWN_S = 20.0   # when a 429 carries no Retry-After
_MAX_COOLDOWN_S = 300.0
_MIN_ATTEMPT_S = 0.25           # too little budget left to be worth a request

# gpt-oss always reasons, and its reasoning is spent from the same completion
# budget as the answer: at the rewrite's 60-token cap it returned empty or
# cut-off strings ("LLa") in testing on 2026-10-04. It gets headroom and its
# reasoning is kept out of the response. Qwen can switch reasoning off.
_REASONING_HEADROOM = 512

_benched: dict[str, tuple[float, str]] = {}   # model -> (until, reason)
_bench_lock = threading.Lock()


def _request_options(model: str, max_tokens: int) -> dict:
    """Per-family parameters; max_tokens is the budget for the visible answer."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False,
                "max_completion_tokens": max_tokens + _REASONING_HEADROOM}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none", "max_completion_tokens": max_tokens}
    return {"max_completion_tokens": max_tokens}


def _bench(model: str, seconds: float, reason: str) -> None:
    with _bench_lock:
        _benched[model] = (time.monotonic() + seconds, reason)


def _ready_models() -> list[str]:
    """Configured models in order, minus any still benched."""
    now = time.monotonic()
    with _bench_lock:
        return [m for m in config.GROQ_MODELS
                if _benched.get(m, (0.0, ""))[0] <= now]


def _error_code(exc) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("code") or "")
    return ""


def _model_gone(exc) -> bool:
    """True when Groq says this model no longer exists for this key."""
    return (getattr(exc, "status_code", None) == 404
            or _error_code(exc) in ("model_not_found", "model_decommissioned"))


def _retry_after(exc) -> float:
    response = getattr(exc, "response", None)
    try:
        seconds = float(response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        seconds = _RATE_LIMIT_COOLDOWN_S
    return min(max(seconds, 1.0), _MAX_COOLDOWN_S)


def model_status() -> dict:
    """The configured chain and which models are benched right now."""
    now = time.monotonic()
    with _bench_lock:
        benched = {
            m: {"seconds_left": int(until - now), "reason": reason}
            for m, (until, reason) in _benched.items() if until > now
        }
    return {"configured": list(config.GROQ_MODELS), "benched": benched}


_PROBE_TIMEOUT_S = 10.0


def probe_models() -> dict:
    """Which configured models Groq still serves to this key (one request).

    Synchronous; run it in a thread. A configured model Groq no longer lists
    is benched straight away, so requests stop paying a 404 to find out. This
    is what /healthz/deep reports and the keepalive workflow alerts on --
    the check that would have caught the 2026-08-16 Llama shutdown the day it
    happened instead of seven weeks later.
    """
    configured = list(config.GROQ_MODELS)
    client = _get_client()
    if client is None:
        return {"status": "skipped", "configured": configured,
                "reason": "GROQ_API_KEY not set or groq not installed"}
    t0 = time.perf_counter()
    try:
        served = {m.id for m in client.models.list(timeout=_PROBE_TIMEOUT_S).data}
    except Exception as e:
        return {"status": "error", "configured": configured,
                "error": str(e)[:200],
                "time_ms": int((time.perf_counter() - t0) * 1000)}

    available = {m: m in served for m in configured}
    for model, ok in available.items():
        if not ok:
            _bench(model, _GONE_COOLDOWN_S, "not listed by Groq")
    ready = _ready_models()
    return {
        # Degraded but serving is still "ok": a fallback answering is the
        # chain doing its job. "primary_available" is the early warning.
        "status": "ok" if ready else "error",
        "configured": configured,
        "available": available,
        "primary_available": bool(configured) and available[configured[0]],
        "active": ready[0] if ready else None,
        "benched": model_status()["benched"],
        "time_ms": int((time.perf_counter() - t0) * 1000),
    }


class Completion(NamedTuple):
    text: str
    model: str
    truncated: bool                 # stopped at the token cap, mid-thought
    tokens_in: int | None = None    # Groq's usage, when the response carries it
    tokens_out: int | None = None
    rank: int = 0                   # place of `model` in GROQ_MODELS: 0 is the primary


# The newest rate-limit headers Groq sent for each model (app/ai_usage.py shows
# them). Groq reports requests left today and tokens left this MINUTE; tokens
# per day are not in any header, so app/ai_usage.py counts them from usage.
_windows: dict[str, dict] = {}
_WINDOW_HEADERS = {
    "requests_limit": "x-ratelimit-limit-requests", "requests_left": "x-ratelimit-remaining-requests",
    "requests_reset": "x-ratelimit-reset-requests", "tokens_limit": "x-ratelimit-limit-tokens",
    "tokens_left": "x-ratelimit-remaining-tokens", "tokens_reset": "x-ratelimit-reset-tokens",
}


def _note_window(model: str, headers) -> None:
    """Keep the rate-limit headers of a response (or of a 429) for `model`."""
    if not headers:
        return
    window = {}
    for key, header in _WINDOW_HEADERS.items():
        value = headers.get(header)
        if value is None:
            continue
        try:
            window[key] = int(value) if key.endswith(("_limit", "_left")) else str(value)
        except ValueError:
            window[key] = str(value)
    if window:
        window["at"] = time.time()
        _windows[model] = window


def windows() -> dict[str, dict]:
    """A copy of the newest rate-limit headers per model."""
    return {m: dict(w) for m, w in _windows.items()}


async def _run_logged(feature: str, fn, *args, chars_in: int = 0, subject: str | None = None) -> Completion:
    """Run a blocking completion off the event loop and log it (app/ai_usage.py).

    Raises what the completion raised, after logging it as failed, or as
    unavailable when every model in the chain was benched or refused.
    """
    from app import ai_usage
    t0 = time.perf_counter()
    try:
        result = await asyncio.get_running_loop().run_in_executor(None, fn, *args)
    except Exception as e:
        status = "unavailable" if "no Groq model answered" in str(e) else "failed"
        await ai_usage.log_call(feature, None, latency_ms=(time.perf_counter() - t0) * 1000,
                                status=status, chars_in=chars_in, subject=subject)
        raise
    await ai_usage.log_call(feature, result, latency_ms=(time.perf_counter() - t0) * 1000, chars_in=chars_in,
                            subject=subject)
    return result


def ai_usage_query(query: str) -> str | None:
    from app import ai_usage
    return ai_usage.query_subject(query)


def _count(feature: str, metric: str) -> None:
    from app import ai_usage
    ai_usage.count(feature, metric)


def _rank(model: str) -> int:
    try:
        return list(config.GROQ_MODELS).index(model)
    except ValueError:
        return len(config.GROQ_MODELS)


# Reasoning a model emits inline instead of in its own field, including an
# opening tag the token cap cut off before it closed.
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S | re.I)
# gpt-oss writes typographic hyphens and spaces (U+2011 "Large‑language‑model");
# they render, but break copy-paste into a search box and FTS5 tokenisation.
_TYPOGRAPHIC = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-",
                              "\u00a0": " ", "\u202f": " ", "\u2009": " "})


def _clean(text: str) -> str:
    text = _THINK.sub("", text or "").translate(_TYPOGRAPHIC)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


_SENTENCE_END = re.compile(r"[.!?][\"')\]]?(?=\s|$)")


def _trim_to_sentence(text: str) -> str:
    """Drop a trailing fragment the token cap cut off mid-sentence."""
    ends = list(_SENTENCE_END.finditer(text))
    return text[:ends[-1].end()].strip() if ends else ""


def _complete(client, messages: list[dict], *, temperature: float,
              max_tokens: int, timeout: float) -> Completion:
    """One chat completion from the first configured model that answers.

    ``timeout`` is the budget for the whole call, failover included. Retired
    and rate-limited models are benched and skipped; any other failure raises,
    and every caller already degrades to its no-LLM path on an exception.
    """
    import groq

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    for model in _ready_models():
        remaining = deadline - time.monotonic()
        if remaining < _MIN_ATTEMPT_S:
            break
        kwargs = dict(messages=messages, model=model, temperature=temperature, timeout=remaining,
                      **_request_options(model, max_tokens))
        try:
            # The raw form exposes the rate-limit headers; a client without it
            # (the tests' fake) is called the plain way.
            raw_api = getattr(client.chat.completions, "with_raw_response", None)
            if raw_api is not None:
                raw = raw_api.create(**kwargs)
                _note_window(model, raw.headers)
                response = raw.parse()
            else:
                response = client.chat.completions.create(**kwargs)
        except groq.RateLimitError as e:
            _note_window(model, getattr(getattr(e, "response", None), "headers", None))
            _bench(model, _retry_after(e), "rate limited")
            last_error = e
            continue
        except groq.InternalServerError as e:
            last_error = e          # one bad response; not worth benching
            continue
        except groq.APIStatusError as e:
            if not _model_gone(e):
                raise
            _bench(model, _GONE_COOLDOWN_S, f"unavailable (HTTP {e.status_code})")
            print(f"[groq_svc] WARNING: Groq no longer serves {model!r} -- "
                  f"skipping it; update GROQ_MODELS ({e})")
            last_error = e
            continue
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(_clean(choice.message.content), model, choice.finish_reason == "length",
                          getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
                          _rank(model))

    raise RuntimeError(
        f"no Groq model answered (configured: {config.GROQ_MODELS}, "
        f"status: {model_status()['benched']})") from last_error


# ── Rewrite prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an academic search query optimizer for arXiv papers.

Your job: Convert casual or conversational user queries into academic search strings.

Rules:
1. Output ONLY the rewritten query string — no explanation, no quotes, no preamble.
2. If the user's query is casual or conversational, rewrite it using standard academic terms.
3. CRITICAL: If the query is ALREADY a precise technical term, a single keyword, an acronym, or a known paper title (e.g., "perplexity", "transformers", "Adam optimizer"), DO NOT expand it. Return it EXACTLY AS IS. Do NOT add random related words.
4. Never output more than 8 words.

Examples:
User: "when AI makes up fake facts"
Output: LLM hallucination factual errors

User: "the llama model by facebook"
Output: LLaMA foundation language model Meta AI

User: "perplexity"
Output: perplexity

User: "attention is all you need"
Output: attention is all you need

User: "gradient descent"
Output: gradient descent"""


# ── Heuristic: should we skip rewriting? ─────────────────────────────────────

_ACADEMIC_PATTERN = re.compile(
    r"""(?:
        \d{4}\.\d{4,5}          |   # arXiv ID
        [A-Z]{2,}               |   # Acronyms like LLM, NLP, BERT
        transformer|attention   |
        neural|network          |
        \b(?:et\s+al|arxiv)\b
    )""",
    re.VERBOSE | re.IGNORECASE,
)


def _looks_academic(query: str) -> bool:
    """Heuristic: skip rewriting if query already looks academic or is very short."""
    words = query.split()
    
    # 1-2 word queries are usually precise keywords or author names (e.g., "perplexity", "lecun")
    # Expanding them almost always ruins the precision.
    if len(words) <= 2:
        return True
        
    if len(words) > 6:
        matches = len(_ACADEMIC_PATTERN.findall(query))
        if matches >= 2:
            return True
    return False


# ── Public API ───────────────────────────────────────────────────────────────

async def rewrite(query: str) -> str:
    """
    Rewrite a user query into an academic search string using Groq LLM.

    Falls back to the original query on ANY error — this function never
    raises exceptions and never blocks the search pipeline.

    Args:
        query: Raw user search query.

    Returns:
        Rewritten academic query string, or original query on error.
    """
    query = query.strip()
    if not query:
        return query

    # Skip if already academic-looking
    if _looks_academic(query):
        _count("search_rewrite", "skipped")
        return query

    if _failover_enabled("rewrite"):
        result = await _routed("rewrite", [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ], subject=ai_usage_query(query))
        rewritten = _clean(result.text or "").strip('"').strip("'").strip()
        _count("search_rewrite", "discarded" if not rewritten else
               "unchanged" if rewritten.lower() == query.lower() else "rewritten")
        return rewritten or query

    client = _get_client()
    if client is None:
        return query  # No API key configured — skip

    try:
        result = await _run_logged("search_rewrite", _run_rewrite, client, query,
                                   chars_in=len(_SYSTEM_PROMPT) + len(query),
                                   subject=ai_usage_query(query))
        # A rewrite cut off at the cap ("LLa") is worse than none: it would be
        # embedded and searched as if it were the reader's intent.
        if result.truncated:
            _count("search_rewrite", "discarded")
            return query
        rewritten = result.text.strip('"').strip("'").strip()

        # Sanity check: rewritten should be non-empty and not absurdly long
        if not rewritten or len(rewritten) > 200:
            _count("search_rewrite", "discarded")
            return query

        _count("search_rewrite", "unchanged" if rewritten.lower() == query.lower() else "rewritten")
        return rewritten

    except Exception as e:
        print(f"[groq_svc] Rewrite failed, using original query: {e}")
        return query


def _run_rewrite(client, query: str) -> Completion:
    """Sync helper: call Groq chat completion with timeout."""
    return _complete(
        client,
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=60,
        timeout=2.0,  # Hard 2s timeout — search must not stall
    )


# ── AI Search Summaries ──────────────────────────────────────────────────────

def _summary_prompt(query: str, papers: list[dict]) -> str:
    """The search-overview prompt over the top five results."""
    # Build context from top 5 papers max
    context_lines = []
    for i, p in enumerate(papers[:5]):
        context_lines.append(f"Paper {i+1}: {p['title']}\nAbstract: {p['abstract'][:800]}...")
    context_str = "\n\n".join(context_lines)

    prompt = f"""You are an expert AI research assistant. 
The user searched for: "{query}"

Here are the top papers returned for this query:
{context_str}

Task: Write a concise, synthesized overview (3-4 sentences max) that answers the user's query based ONLY on these papers. 
Format: Return plain text with basic markdown (bolding key terms is good). DO NOT start with "Here is a summary" or similar filler. DO NOT output bullet points. Be direct, educational, and authoritative."""
    return prompt


async def generate_search_summary(query: str, papers: list[dict]) -> str | None:
    """
    Generate a short 3-4 sentence AI summary synthesizing the top papers.
    
    Returns:
        Summary HTML string, or None if error or not enough papers.
    """
    if not papers or len(papers) < 2:
        return None
        
    client = None if _failover_enabled("summary") else _get_client()
    if client is None and not _failover_enabled("summary"):
        return None

    prompt = _summary_prompt(query, papers)

    try:
        if _failover_enabled("summary"):
            generated = await _routed("summary", [{"role": "user", "content": prompt}],
                                      subject=ai_usage_query(query))
            result = Completion(_clean(generated.text or ""), generated.model or "", False)
        else:
            result = await _run_logged("search_overview", _run_summary, client, prompt, chars_in=len(prompt),
                                       subject=ai_usage_query(query))
        
        summary = _trim_to_sentence(result.text) if result.truncated else result.text
        if not summary:
            _count("search_overview", "empty")
            return None
        _count("search_overview", "trimmed" if result.truncated else "shown")
            
        # Escape first, then bold. ai_summary.html renders this with |safe, and
        # the model's text is steered by the search query -- which arrives in
        # the URL -- and by paper abstracts, so it must never reach the page
        # as markup. Only the <strong> tags added here are HTML.
        import html
        summary_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>',
                              html.escape(summary, quote=False))
        
        return summary_html
        
    except Exception as e:
        print(f"[groq_svc] Summary generation failed: {e}")
        return None


def _run_summary(client, prompt: str) -> Completion:
    """Sync helper: call Groq chat completion for summaries with 4s timeout."""
    return _complete(
        client,
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
        timeout=4.0,  # 4s timeout so it doesn't hang indefinitely
    )


# ── Plain-language paper explanation ─────────────────────────────────────────
#
# An arXiv abstract is written for peers. A reader from an adjacent field — the
# person this product is FOR, per doc 01 — bounces off the vocabulary before
# reaching the idea. This turns one abstract into three plain sentences.
#
# Hallucination control follows doc 07 §A.6: the prompt is constrained to the
# supplied text, temperature is low, and the output is short enough that there
# is little room to wander. There is no Citations API on Groq, so the mitigation
# is prompt-level, and the UI labels the result as generated rather than
# presenting it as the paper's own words.

_EXPLAIN_PROMPT_VERSION = "v1"

_EXPLAIN_SYSTEM = """You explain research papers to capable readers who work in \
a DIFFERENT field. They are not beginners — do not talk down — but they do not \
share this paper's vocabulary.

Write exactly three sentences:
1. The problem, in ordinary language.
2. What the authors actually did.
3. What they found, and why it matters.

RULES:
- Use ONLY what the title and abstract state. Introduce no method, dataset, \
number or claim that is not there.
- Expand or avoid jargon and acronyms. If a term is unavoidable, gloss it in \
the same sentence.
- No LaTeX, no notation, no markdown, no preamble, no bullet points.
- If the abstract is too truncated or vague to summarise honestly, reply with \
exactly: INSUFFICIENT"""


def explain_model() -> str:
    """The model an explanation is attributed to: the head of the chain.

    A fallback model may have written a given explanation, but the cache key
    tracks the configured policy, so changing GROQ_MODELS regenerates them.
    """
    return config.GROQ_MODELS[0] if config.GROQ_MODELS else ""


def explain_cache_key(arxiv_id: str, abstract: str, *, title: str = "",
                      model: str | None = None) -> str:
    """Content-addressed, per doc 07 §A.4.

    Keyed on the abstract TEXT rather than just the id, so the 500-char stored
    truncation and a later backfilled full abstract are different cache
    entries — otherwise repairing the corpus would silently keep serving
    explanations generated from stumps. The prompt version and model are in the
    key for the same reason.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(arxiv_id.encode())
    h.update(b"\x00")
    h.update((abstract or "").encode())
    h.update(b"\x00")
    h.update(_EXPLAIN_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update((model or explain_model()).encode())
    h.update(b"\x00")
    h.update(title.encode())
    return h.hexdigest()


def explanation_models() -> tuple[str, ...]:
    """Only currently accepted producing models may be read from cache."""
    if not _failover_enabled("explain"):
        return (explain_model(),)
    try:
        return llm_router.configured_models("explain")
    except ValueError:
        return ()


_explanation_tasks: dict[tuple[str, str], tuple[asyncio.Task, int]] = {}


async def explain_paper_result(title: str, abstract: str, arxiv_id: str | None = None) -> llm_router.Generation:
    """Carry actual model provenance; share concurrent work for identical input."""
    if not _failover_enabled("explain"):
        text = await explain_paper(title, abstract, arxiv_id)
        return llm_router.Generation("success" if text else "unavailable", text, explain_model())
    if not abstract or len(abstract.strip()) < 120:
        return llm_router.Generation("abstained")
    identity = (title, abstract)
    if identity not in _explanation_tasks:
        prompt = f"<title>{title.strip()}</title>\n<abstract>{abstract.strip()}</abstract>"
        task = asyncio.create_task(_routed("explain", [
            {"role": "system", "content": _EXPLAIN_SYSTEM},
            {"role": "user", "content": prompt},
        ], subject=arxiv_id))
        _explanation_tasks[identity] = (task, 0)
    task, count = _explanation_tasks[identity]
    _explanation_tasks[identity] = (task, count + 1)
    try:
        return await asyncio.shield(task)
    finally:
        _, count = _explanation_tasks[identity]
        if count == 1:
            del _explanation_tasks[identity]
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        else:
            _explanation_tasks[identity] = (task, count - 1)


async def explain_paper(title: str, abstract: str, arxiv_id: str | None = None) -> str | None:
    """Three plain sentences, or None when it cannot be done honestly."""
    if _failover_enabled("explain"):
        result = await explain_paper_result(title, abstract, arxiv_id)
        return result.text
    if not abstract or len(abstract.strip()) < 120:
        return None
    client = _get_client()
    if client is None:
        return None

    prompt = (f"<title>{title.strip()}</title>\n"
              f"<abstract>{abstract.strip()}</abstract>")

    try:
        result = await _run_logged("explain", _run_explain, client, prompt,
                                   chars_in=len(_EXPLAIN_SYSTEM) + len(prompt), subject=arxiv_id)
    except Exception as e:
        print(f"[groq_svc] explain failed: {e}")
        return None

    # Cut off at the cap: keep the complete sentences rather than cache a
    # fragment, which would be served to every later reader of this paper.
    text = _trim_to_sentence(result.text) if result.truncated else result.text
    # The model's own refusal path. Honoured rather than second-guessed: a
    # summary of a mutilated abstract is worse than no summary.
    if not text or text.upper().startswith("INSUFFICIENT"):
        _count("explain", "declined")
        return None
    _count("explain", "trimmed" if result.truncated else "written")
    return text


def _run_explain(client, prompt: str) -> Completion:
    return _complete(
        client,
        [
            {"role": "system", "content": _EXPLAIN_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=220,
        timeout=8.0,
    )
