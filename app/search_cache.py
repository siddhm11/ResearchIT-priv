"""
Recent search results, in memory, so going back to a search, or running the
same search again, doesn't run the pipeline again: no BGE-M3 encode, no Groq
rewrite, no Qdrant or FTS, no metadata fetch. Nothing is written anywhere; a
deploy or restart empties it.

    results   normalised query -> the ranked papers and the search's meta.
              Kept SEARCH_CACHE_TTL_S for a full (hybrid) search, and
              SEARCH_CACHE_DEGRADED_TTL_S when one retrieval arm was missing,
              since a later search may do better. Shared by every reader:
              results don't depend on who asked. At most MAX_ENTRIES.
    repeats   (reader cookie, query) -> the query_id that reader was shown,
              for as long as the same results entry lives. A reader coming
              back to a search keeps its query_id (clicks stay tied to the
              search that showed them) and isn't counted or logged twice.
              Memory only: search_log stays unlinked from readers.

Callers get copies, so per-reader fields (saved, query_id, position) never
leak into the shared entry.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app import config, runtime_stats

MAX_ENTRIES = 256
MAX_REPEATS = 2048

_now = time.monotonic


@dataclass
class Entry:
    papers: list[dict]
    meta: dict
    created: float
    expires: float
    serial: int = field(default=0)

    def copies(self) -> tuple[list[dict], dict]:
        return [dict(p) for p in self.papers], dict(self.meta)


_results: OrderedDict[str, Entry] = OrderedDict()
_repeats: OrderedDict[tuple[str, str], tuple[int, str]] = OrderedDict()
_serial = 0


def key(query: str) -> str:
    """Case and spacing don't change a search."""
    return " ".join((query or "").split()).lower()


def _ttl(meta: dict) -> int:
    full = (meta or {}).get("retrieval_mode") == "hybrid"
    return config.SEARCH_CACHE_TTL_S if full else min(config.SEARCH_CACHE_DEGRADED_TTL_S,
                                                       config.SEARCH_CACHE_TTL_S)


def get(query: str) -> Entry | None:
    """The live entry for this query, or None."""
    k = key(query)
    if not k or config.SEARCH_CACHE_TTL_S <= 0:
        return None
    entry = _results.get(k)
    if entry is None or entry.expires <= _now():
        if entry is not None:
            del _results[k]
        runtime_stats.count("search_cache_miss")
        return None
    _results.move_to_end(k)
    runtime_stats.count("search_cache_hit")
    return entry


def put(query: str, papers: list[dict], meta: dict) -> Entry | None:
    """Keep a finished search. Empty results are not kept."""
    global _serial
    k = key(query)
    if not k or not papers or config.SEARCH_CACHE_TTL_S <= 0:
        return None
    now = _now()
    _serial += 1
    entry = Entry(papers=[dict(p) for p in papers], meta=dict(meta or {}), created=now,
                  expires=now + _ttl(meta), serial=_serial)
    _results[k] = entry
    _results.move_to_end(k)
    while len(_results) > MAX_ENTRIES:
        _results.popitem(last=False)
    return entry


def repeat_query_id(user_id: str | None, query: str, entry: Entry | None) -> str | None:
    """The query_id this reader was shown these same results under, if any."""
    if not user_id or entry is None:
        return None
    seen = _repeats.get((user_id, key(query)))
    if seen is None or seen[0] != entry.serial:
        return None
    return seen[1]


def remember(user_id: str | None, query: str, entry: Entry | None, query_id: str) -> None:
    if not user_id or entry is None:
        return
    k = (user_id, key(query))
    _repeats[k] = (entry.serial, query_id)
    _repeats.move_to_end(k)
    while len(_repeats) > MAX_REPEATS:
        _repeats.popitem(last=False)


def clear() -> None:
    _results.clear()
    _repeats.clear()


def stats() -> dict:
    now = _now()
    return {"entries": sum(1 for e in _results.values() if e.expires > now), "readers": len(_repeats)}
