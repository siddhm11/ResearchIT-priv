"""
Put the paper a short search names first (switch SEARCH_NAMED_PAPER).

"resnet", "lora", "gpt-3 paper": the reader means one paper, whose title
usually lacks the name ("Deep Residual Learning for Image Recognition"), while
hundreds of later papers that use the name fill every retrieval list. Measured
2026-10-30 on 30 such nicknames: 2 originals on page 1 today, and in most cases
the original is in none of the four top-60 lists, so no reranking or rewording
can reach it.

So the model is asked which paper the search names (groq_svc.name_paper),
and the answer counts only if a corpus title matches it almost exactly
(difflib ratio >= MIN_RATIO after lowercasing and dropping punctuation; the
most cited wins when several do). A recalled title that matches nothing is
dropped ("React: The Library for Web and Native User Interfaces").

Looser matching (the part of a title before its colon, word prefixes) found
3 more papers on 140 held-out searches but pinned 4 topic searches to minor
papers ("simultaneous localization and mapping" -> a 28-citation dataset
list), because the model sometimes answers with the topic's own name.

Only searches of up to MAX_WORDS words ask: on 60 descriptive LitSearch
questions the model named the right paper for none and a wrong one for 4.

Measured 2026-10-30 (doc 06), Qwen, strict check, up to 4 words:
  302 named searches (Papers with Code method/dataset names, nicknames, two
  logged titles): paper at #1 60 -> 132; 7 pins were another paper.
  140 held-out names never looked at while building this: 47 -> 64; 6 pins
  were another paper, 2 of them pushing down a correct #1.
  Most wrong pins are siblings (Llama 2 for "llama", the other word2vec
  paper); clearly wrong: "trpo" -> PPO, "deepmind control suite" -> DreamerV2.
  238 topic searches: 7 pinned, all the field's canonical paper ("knowledge
  distillation" -> Hinton et al.; "panoptic segmentation" -> the paper that
  defined it, which is not on page 1 today); 2 of them were #1 already.

Modes: off; shadow (ask and check, record in search meta, pin nothing); on.
"""
from __future__ import annotations

import asyncio
import difflib
import re
import time

from app import features, groq_svc, local_meta, runtime_stats

SWITCH = "SEARCH_NAMED_PAPER"
MAX_WORDS = 4
WAIT_S = 1.6            # from the start of the search; the model call itself times out at 1.5 s
MIN_RATIO = 0.92
_ARXIV_ID = re.compile(r"\b\d{4}\.\d{4,5}\b")


def wanted(query: str) -> bool:
    """Whether this search should ask: switch active, 1 to MAX_WORDS words, not an arXiv id."""
    n = len(query.split())
    return features.active(SWITCH) and 1 <= n <= MAX_WORDS and not _ARXIV_ID.search(query)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split())


def find_title(title: str) -> tuple[str, str] | None:
    """(arxiv_id, corpus title) for a recalled title, or None. Synchronous sqlite:
    call it through asyncio.to_thread."""
    want = _norm(title)
    if len(want.split()) < 2:
        return None
    conn = local_meta.connection()
    if conn is None:
        return None
    try:
        rows = conn.execute(
            "SELECT p.arxiv_id, p.title, p.citation_count FROM papers_fts f "
            "JOIN papers p ON p.rowid = f.rowid WHERE papers_fts MATCH ? LIMIT 200",
            ('title:"' + want + '"',)).fetchall()
    except Exception as e:
        print(f"[named_paper] title lookup failed: {e}")
        return None
    best = None
    for arxiv_id, found, cites in rows:
        if difflib.SequenceMatcher(None, _norm(found), want).ratio() >= MIN_RATIO:
            if best is None or (cites or 0) > best[2]:
                best = (arxiv_id, " ".join((found or "").split()), cites or 0)
    return (best[0], best[1]) if best else None


async def resolve(query: str) -> dict | None:
    """{arxiv_id, title, recalled} for the paper this search names, or None."""
    recalled = await groq_svc.name_paper(query)
    if not recalled:
        runtime_stats.count("named_paper_none")
        return None
    hit = await asyncio.to_thread(find_title, recalled)
    if not hit:
        runtime_stats.count("named_paper_not_in_corpus")
        return None
    runtime_stats.count("named_paper_found")
    return {"arxiv_id": hit[0], "title": hit[1], "recalled": recalled}


async def collect(task: asyncio.Task | None, started: float) -> dict | None:
    """The task's answer if it is ready by WAIT_S after `started` (a perf_counter
    time), else None; a late lookup is cancelled rather than waited for."""
    if task is None:
        return None
    try:
        return await asyncio.wait_for(task, timeout=max(0.0, started + WAIT_S - time.perf_counter()))
    except asyncio.TimeoutError:
        runtime_stats.count("named_paper_late")
        return None
    except Exception as e:
        print(f"[named_paper] lookup failed: {e}")
        return None


def pin(ranked: list[dict], found: dict) -> list[dict]:
    """`ranked` with the named paper first (added when retrieval missed it)."""
    rest = [it for it in ranked if it.get("arxiv_id") != found["arxiv_id"]]
    item = next((it for it in ranked if it.get("arxiv_id") == found["arxiv_id"]), None) or {
        "arxiv_id": found["arxiv_id"], "rrf_score": 0.0}
    item["named_paper"] = True
    return [item] + rest
