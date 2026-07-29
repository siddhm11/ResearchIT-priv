"""
In-memory user state cache (hot path).

Keeps the last N positive/negative paper IDs per user so that
recommendation requests don't need a DB round-trip on every page load.
The cache is populated lazily on first access from the SQLite interactions
table, then kept up-to-date by the event endpoints.

Thread-safety: asyncio is single-threaded; no locks needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

from app import db, config

MAX_POSITIVES = config.REC_POSITIVE_LIMIT   # max positive IDs kept in memory per user
MAX_NEGATIVES = 50                          # max negative IDs kept in memory per user


@dataclass
class UserState:
    # Most-recently-interacted first
    positives: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_POSITIVES))
    negatives: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_NEGATIVES))
    loaded: bool = False  # True once hydrated from DB

    def add_positive(self, paper_id: str) -> None:
        # Remove from negatives if it was there
        try:
            self.negatives.remove(paper_id)
        except ValueError:
            pass
        # Prepend (deque handles maxlen eviction automatically)
        if paper_id not in self.positives:
            self.positives.appendleft(paper_id)

    def add_negative(self, paper_id: str) -> None:
        try:
            self.positives.remove(paper_id)
        except ValueError:
            pass
        if paper_id not in self.negatives:
            self.negatives.appendleft(paper_id)

    @property
    def positive_list(self) -> list[str]:
        return list(self.positives)

    @property
    def negative_list(self) -> list[str]:
        return list(self.negatives)

    def has_enough_for_recs(self) -> bool:
        from app.config import REC_MIN_POSITIVES
        return len(self.positives) >= REC_MIN_POSITIVES


# ── Global in-process cache ───────────────────────────────────────────────────

_cache: dict[str, UserState] = {}


def get_user_state(user_id: str) -> UserState:
    """Return the in-memory state for a user (creates if missing)."""
    if user_id not in _cache:
        _cache[user_id] = UserState()
    return _cache[user_id]


async def ensure_loaded(user_id: str) -> UserState:
    """
    Return the user state, loading from DB the first time.
    Subsequent calls are O(1) dict lookup.
    """
    state = get_user_state(user_id)
    if state.loaded:
        return state

    rows = await db.get_user_interactions(
        user_id,
        event_types=["save", "not_interested"],
        limit=MAX_POSITIVES + MAX_NEGATIVES,
    )

    # Rows are ordered newest-first; we want newest in the front of the deque
    # Process oldest-first so that appendleft ends with newest at front.
    for row in reversed(rows):
        if row["event_type"] == "save":
            state.add_positive(row["paper_id"])
        elif row["event_type"] == "not_interested":
            state.add_negative(row["paper_id"])

    state.loaded = True
    return state


def record_positive(user_id: str, paper_id: str) -> None:
    """Update in-memory state synchronously (DB write happens separately)."""
    get_user_state(user_id).add_positive(paper_id)


def record_negative(user_id: str, paper_id: str) -> None:
    get_user_state(user_id).add_negative(paper_id)


def all_seen(user_id: str) -> set[str]:
    """All paper IDs this user has interacted with (used to filter recs)."""
    state = get_user_state(user_id)
    return set(state.positive_list) | set(state.negative_list)
