"""
Unit tests for user_state.py — pure in-memory logic, no DB/network.
"""
import pytest
from collections import OrderedDict, deque

# We test the UserState dataclass directly
from app.user_state import UserState, MAX_POSITIVES, MAX_NEGATIVES


def test_add_positive_basic():
    s = UserState()
    s.add_positive("1706.03762")
    assert "1706.03762" in s.positives
    assert "1706.03762" not in s.negatives


def test_add_negative_basic():
    s = UserState()
    s.add_negative("1706.03762")
    assert "1706.03762" in s.negatives
    assert "1706.03762" not in s.positives


def test_save_moves_from_negatives_to_positives():
    s = UserState()
    s.add_negative("abc")
    s.add_positive("abc")
    assert "abc" in s.positives
    assert "abc" not in s.negatives


def test_dismiss_moves_from_positives_to_negatives():
    s = UserState()
    s.add_positive("abc")
    s.add_negative("abc")
    assert "abc" in s.negatives
    assert "abc" not in s.positives


def test_no_duplicates_in_positives():
    s = UserState()
    s.add_positive("abc")
    s.add_positive("abc")
    assert list(s.positives).count("abc") == 1


def test_positive_list_most_recent_first():
    s = UserState()
    for i in range(3):
        s.add_positive(f"paper{i}")
    # appendleft → most recent is first
    assert s.positive_list[0] == "paper2"
    assert s.positive_list[-1] == "paper0"


def test_maxlen_positives_evicts_oldest():
    s = UserState()
    for i in range(MAX_POSITIVES + 5):
        s.add_positive(f"paper{i}")
    assert len(s.positives) == MAX_POSITIVES
    # Newest are kept (appendleft evicts right end)
    assert "paper4" not in s.positives   # oldest evicted


def test_has_enough_for_recs_false_when_empty():
    s = UserState()
    assert not s.has_enough_for_recs()


def test_has_enough_for_recs_true_after_one_save():
    s = UserState()
    s.add_positive("1706.03762")
    assert s.has_enough_for_recs()


def test_all_seen_union():
    from app.user_state import _cache, record_positive, record_negative, all_seen
    uid = "test-seen-user"
    _cache.pop(uid, None)
    record_positive(uid, "pos1")
    record_negative(uid, "neg1")
    seen = all_seen(uid)
    assert "pos1" in seen
    assert "neg1" in seen
    _cache.pop(uid, None)


# ── Bounded cache (audit F9) ──────────────────────────────────────────────────
# Every cookieless visit mints a user id, so crawlers on public paper pages
# grew the cache for the life of the process: 2,000 visits, 2,000 states.

def test_the_user_cache_is_bounded(monkeypatch):
    from app import user_state as us
    monkeypatch.setattr(us, "_cache", OrderedDict())
    monkeypatch.setattr(us, "MAX_CACHED_USERS", 100)
    for i in range(250):
        us.get_user_state(f"crawler-{i}")
    assert len(us._cache) == 100


def test_a_recently_used_reader_outlives_the_crawlers(monkeypatch):
    from app import user_state as us
    monkeypatch.setattr(us, "_cache", OrderedDict())
    monkeypatch.setattr(us, "MAX_CACHED_USERS", 100)
    us.record_positive("reader", "1706.03762")
    for i in range(250):
        us.get_user_state(f"crawler-{i}")
        us.get_user_state("reader")
    assert "1706.03762" in us.get_user_state("reader").positives
