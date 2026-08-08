"""
Tests for EWMA profile embedding computation.

Covers:
  - ewma_update produces L2-normalised output
  - First interaction sets the profile directly
  - Multiple updates blend correctly
  - Negative dismiss pushes vector away
  - Storage round-trip (save + load)
"""
import asyncio
import pytest
import numpy as np

from app.recommend.profiles import (
    ewma_update,
    EMBEDDING_DIM,
    ALPHA_LONG_TERM,
    ALPHA_SHORT_TERM,
    ALPHA_NEGATIVE,
    _to_bytes,
    _from_bytes,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _random_unit_vec(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _assert_unit(v: np.ndarray, tol: float = 1e-5):
    assert abs(np.linalg.norm(v) - 1.0) < tol, f"norm = {np.linalg.norm(v)}"


# ── ewma_update unit tests ───────────────────────────────────────────────────

def test_ewma_first_interaction_sets_profile():
    """First interaction: profile == normalised input."""
    embed = _random_unit_vec(1)
    result = ewma_update(None, embed, ALPHA_LONG_TERM)
    _assert_unit(result)
    # Should be very close to input (already unit-norm)
    assert np.allclose(result, embed, atol=1e-5)


def test_ewma_update_is_normalised():
    """EWMA output is always L2-normalised."""
    current = _random_unit_vec(10)
    new = _random_unit_vec(20)
    result = ewma_update(current, new, ALPHA_LONG_TERM)
    _assert_unit(result)


def test_ewma_long_term_alpha_is_stable():
    """With α=0.03, a single new interaction should only move the
    profile slightly — cosine similarity to old profile should be high."""
    current = _random_unit_vec(100)
    new = _random_unit_vec(200)  # different direction
    result = ewma_update(current, new, ALPHA_LONG_TERM)
    sim = float(np.dot(current, result))
    # At α=0.03, should preserve >97% of old direction
    assert sim > 0.97, f"cosine sim = {sim}"


def test_ewma_short_term_alpha_is_responsive():
    """With α=0.40, the profile should shift significantly toward the new input."""
    current = _random_unit_vec(100)
    new = _random_unit_vec(200)
    result = ewma_update(current, new, ALPHA_SHORT_TERM)
    sim_to_old = float(np.dot(current, result))
    sim_to_new = float(np.dot(new, result))
    # Short-term should move meaningfully toward new
    assert sim_to_new > 0.3, f"sim to new = {sim_to_new}"


def test_ewma_multiple_updates_converge():
    """Repeated identical inputs should converge the profile to that input.
    With α=0.03 (Doc 06 correction), convergence is slower — need ~200 updates."""
    target = _random_unit_vec(42)
    profile = _random_unit_vec(99)  # start far away
    for _ in range(200):
        profile = ewma_update(profile, target, ALPHA_LONG_TERM)
    sim = float(np.dot(profile, target))
    assert sim > 0.99, f"after 200 updates, sim = {sim}"


def test_ewma_dissimilar_input_shifts_profile():
    """Feeding a dissimilar vector should shift the profile away from original."""
    current = _random_unit_vec(10)
    dissimilar = _random_unit_vec(999)  # a genuinely different direction
    result = ewma_update(current, dissimilar, ALPHA_SHORT_TERM)
    sim_to_old = float(np.dot(current, result))
    sim_to_new = float(np.dot(dissimilar, result))
    # With α=0.40, profile should move toward new input
    assert sim_to_old < 1.0, f"profile didn't move, sim to old = {sim_to_old}"
    assert sim_to_new > 0.0, f"profile should have some similarity to new, got {sim_to_new}"


# ── Binary storage round-trip ─────────────────────────────────────────────────

def test_bytes_roundtrip():
    """to_bytes → from_bytes preserves data exactly."""
    original = _random_unit_vec(77)
    recovered = _from_bytes(_to_bytes(original))
    assert np.allclose(original, recovered, atol=1e-7)


def test_bytes_size():
    """Each profile vector should be exactly 4096 bytes."""
    v = _random_unit_vec(0)
    b = _to_bytes(v)
    assert len(b) == EMBEDDING_DIM * 4  # float32 = 4 bytes


# ── DB integration tests ─────────────────────────────────────────────────────

@pytest.fixture
def setup_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for each test."""
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test_profiles.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    asyncio.get_event_loop().run_until_complete(db_mod.init_db())
    yield


def test_profile_save_and_load(setup_db):
    """Profile round-trips through SQLite correctly."""
    from app.recommend import profiles

    vec = _random_unit_vec(55)

    async def _run():
        await profiles.save_profile("user-1", "long_term", vec, interaction_count=5)
        loaded = await profiles.load_profile("user-1", "long_term")
        assert loaded is not None
        assert np.allclose(vec, loaded, atol=1e-7)

    asyncio.get_event_loop().run_until_complete(_run())


def test_profile_interaction_count(setup_db):
    """Interaction count persists and retrieves correctly."""
    from app.recommend import profiles

    vec = _random_unit_vec(66)

    async def _run():
        await profiles.save_profile("user-2", "short_term", vec, interaction_count=12)
        count = await profiles.get_interaction_count("user-2", "short_term")
        assert count == 12

    asyncio.get_event_loop().run_until_complete(_run())


def test_profile_not_found_returns_none(setup_db):
    """Missing profile returns None, not an error."""
    from app.recommend import profiles

    async def _run():
        result = await profiles.load_profile("nonexistent", "long_term")
        assert result is None
        count = await profiles.get_interaction_count("nonexistent", "long_term")
        assert count == 0

    asyncio.get_event_loop().run_until_complete(_run())
