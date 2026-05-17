"""
Groq LLM query rewriter — Phase 3.

Responsibilities:
  - Rewrite casual user queries into dense academic keyword strings
  - Uses llama-3.3-70b-versatile via Groq's ultra-fast inference
  - Falls back to original query on ANY error or timeout
  - Skips rewriting for queries that already look academic
  - This is an ENHANCEMENT, not a dependency — search works without it
"""
from __future__ import annotations

import re
import threading

from app import config

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

        from groq import Groq

        _client = Groq(api_key=config.GROQ_API_KEY)
        print("[groq_svc] Groq client initialized")
        return _client


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
        return query

    client = _get_client()
    if client is None:
        return query  # No API key configured — skip

    try:
        import asyncio

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_rewrite, client, query)
        rewritten = result.strip().strip('"').strip("'").strip()

        # Sanity check: rewritten should be non-empty and not absurdly long
        if not rewritten or len(rewritten) > 200:
            return query

        return rewritten

    except Exception as e:
        print(f"[groq_svc] Rewrite failed, using original query: {e}")
        return query


def _run_rewrite(client, query: str) -> str:
    """Sync helper: call Groq chat completion with timeout."""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=60,
        timeout=2.0,  # Hard 2s timeout — search must not stall
    )
    return response.choices[0].message.content
