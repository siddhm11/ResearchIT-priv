"""
The long-term profile must represent the library, not its first entry.

A plain EWMA seeded with its first observation never escapes it. Because the
profile was L2-normalised after every step it always had unit magnitude, so
each new paper got a fixed 3% pull against a full-strength incumbent. Measured
with the real function over near-orthogonal saves, cos(profile, first_save) was
0.996 after ten saves and still 0.982 after FORTY.

That vector is the largest term in heuristic_score (0.40), the relevance axis
MMR selects against, and the whole of Tier 2 — so "the overall profile built
from your saved papers", which is what the UI calls it, was one arbitrary paper.

Two changes, neither of which touches alpha (doc 06 §3.2):
  * a running-mean warmup, alpha_eff = max(alpha, 1/(count+1)), which converges
    to exactly the documented steady-state EWMA at 33 saves;
  * the stored accumulator keeps its magnitude, which encodes how much the
    user's saves agree, and load_profile normalises on read.
"""
import numpy as np
import pytest

from app.recommend import profiles as P
from app.recommend.profiles import (
    ALPHA_LONG_TERM, ALPHA_NEGATIVE, ALPHA_SHORT_TERM,
    effective_alpha, ewma_update,
)

D = P.EMBEDDING_DIM


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def _noisy(rng, base, sigma):
    """Perturb a unit vector by a noise vector of NORM sigma.

    Scaling per-component would be wrong in 1024 dimensions: sigma=0.1 per
    component has norm ~3.2 and swamps the base entirely.
    """
    n = rng.normal(size=D)
    return _unit(base + n / np.linalg.norm(n) * sigma)


# ── effective_alpha ──────────────────────────────────────────────────────────

def test_warmup_is_a_running_mean():
    assert effective_alpha(ALPHA_LONG_TERM, 0) == 1.0        # first obs
    assert effective_alpha(ALPHA_LONG_TERM, 1) == 0.5        # even blend of 2
    assert effective_alpha(ALPHA_LONG_TERM, 2) == pytest.approx(1 / 3)


def test_warmup_hands_over_to_alpha_and_never_goes_below_it():
    """The steady state must be bit-for-bit the documented EWMA."""
    for count in (33, 50, 100, 10_000):
        assert effective_alpha(ALPHA_LONG_TERM, count) == ALPHA_LONG_TERM


def test_handover_is_continuous():
    """No discontinuity: the two rates are equal where they cross."""
    prev = 1.0
    for count in range(0, 60):
        a = effective_alpha(ALPHA_LONG_TERM, count)
        assert a <= prev + 1e-12, "rate must be non-increasing"
        assert a >= ALPHA_LONG_TERM
        prev = a


@pytest.mark.parametrize("alpha", [ALPHA_LONG_TERM, ALPHA_SHORT_TERM, ALPHA_NEGATIVE])
def test_alpha_itself_is_never_modified(alpha):
    """doc 06 §3.2 fixes these values; the warmup must not redefine them."""
    assert effective_alpha(alpha, None) == alpha
    assert effective_alpha(alpha, 10_000) == alpha


def test_omitting_count_preserves_the_original_behaviour():
    rng = np.random.default_rng(0)
    a, b = _unit(rng.normal(size=D)), _unit(rng.normal(size=D))
    assert np.allclose(
        ewma_update(a, b, ALPHA_LONG_TERM),
        ewma_update(a, b, ALPHA_LONG_TERM, count=None))


# ── The property that matters ────────────────────────────────────────────────

@pytest.fixture
def fake_store(monkeypatch):
    store = {}

    class FakeDB:
        async def get_user_profile(self, uid, ptype):
            return store.get((uid, ptype))

        async def upsert_user_profile(self, user_id, profile_type, vector,
                                      interaction_count):
            store[(user_id, profile_type)] = {
                "vector": vector, "interaction_count": interaction_count}

    monkeypatch.setattr(P, "db", FakeDB())
    return store


def _profile_the_old_way(saves, alpha):
    """The pre-fix update: no warmup, L2-normalised every step."""
    p = None
    for x in saves:
        p = ewma_update(p, x, alpha)          # count=None, normalise=True
    return p


async def test_profile_represents_the_library_not_its_first_entry(fake_store):
    """Asserted against the OLD behaviour rather than a magic threshold.

    An absolute bound would be arbitrary here: with three equally weighted
    clusters, cos(profile, saves[0]) SHOULD land near 1/sqrt(3) = 0.577,
    because saves[0] is one cluster's representative and no more.
    """
    rng = np.random.default_rng(0)
    bases = [_unit(rng.normal(size=D)) for _ in range(3)]
    saves = [_noisy(rng, bases[i % 3], 0.15) for i in range(20)]

    for x in saves:
        await P.update_on_save("u", x)
    new = await P.load_profile("u", "long_term")
    old = _profile_the_old_way(saves, ALPHA_LONG_TERM)

    centroid = _unit(np.mean(saves, axis=0))
    # Margin, not a magic absolute. Tight clusters are the KIND case for the
    # old code -- it reached 0.786 here versus 0.446 on near-orthogonal saves --
    # so the gain is smaller than the worst case and still decisive.
    assert float(new @ centroid) > float(old @ centroid) + 0.15, (
        f"no meaningful gain in representativeness: "
        f"{float(old @ centroid):.3f} -> {float(new @ centroid):.3f}")
    assert float(new @ centroid) > 0.95, "profile should track the centroid closely"

    # And it is no longer pinned to whichever paper happened to be first.
    assert float(new @ saves[0]) < float(old @ saves[0]) - 0.2, (
        f"still dominated by the first save: "
        f"{float(old @ saves[0]):.3f} -> {float(new @ saves[0]):.3f}")
    assert float(new @ saves[0]) == pytest.approx(1 / np.sqrt(3), abs=0.12), (
        "one of three equal clusters should carry about 1/sqrt(3)")


async def test_consumers_still_receive_a_unit_vector(fake_store):
    """The accumulator changed; the contract every consumer relies on did not."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        await P.update_on_save("u", _unit(rng.normal(size=D)))
    for kind in ("long_term", "short_term"):
        p = await P.load_profile("u", kind)
        assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-5)


async def test_accumulator_magnitude_tracks_how_much_the_saves_agree(fake_store):
    """The signal that per-step normalisation was destroying."""
    rng = np.random.default_rng(2)
    base = _unit(rng.normal(size=D))

    async def magnitude(uid, saves):
        for x in saves:
            await P.update_on_save(uid, x)
        raw = await P.load_profile_raw(uid, "long_term")
        return float(np.linalg.norm(raw))

    tight = await magnitude("tight", [_noisy(rng, base, 0.15) for _ in range(20)])
    broad = await magnitude("broad", [_noisy(rng, base, 0.60) for _ in range(20)])
    none_ = await magnitude("none", [_unit(rng.normal(size=D)) for _ in range(20)])

    assert tight > broad > none_, (
        f"magnitude should fall as coherence falls: {tight:.3f} {broad:.3f} {none_:.3f}")
    assert tight > 0.9 and none_ < 0.4


async def test_a_single_save_still_yields_that_paper(fake_store):
    """Cold start is unchanged: one save, and the profile IS that paper."""
    rng = np.random.default_rng(3)
    x = _unit(rng.normal(size=D))
    await P.update_on_save("u", x)
    p = await P.load_profile("u", "long_term")
    assert float(p @ x) == pytest.approx(1.0, abs=1e-5)


async def test_existing_profiles_self_heal_without_migration(fake_store):
    """A stored unit vector is a valid accumulator, so old rows just continue.

    Their interaction_count feeds the warmup, so a profile currently stuck on
    its first paper adapts quickly rather than needing a backfill.
    """
    rng = np.random.default_rng(4)
    first = _unit(rng.normal(size=D))
    # Simulate the OLD state: a unit vector with a small count.
    fake_store[("legacy", "long_term")] = {
        "vector": first.tobytes(), "interaction_count": 3}

    fresh = [_unit(rng.normal(size=D)) for _ in range(6)]
    for x in fresh:
        await P.update_on_save("legacy", x)

    new = await P.load_profile("legacy", "long_term")
    old = _profile_the_old_way([first] + fresh, ALPHA_LONG_TERM)

    # Six new saves against a profile entering with count=3 leave the original
    # about a third of the weight -- correct, since it claims to stand for three
    # interactions. The point is that it MOVES, where before it did not.
    assert float(new @ first) < float(old @ first) - 0.15, (
        f"legacy profile barely moved: "
        f"{float(old @ first):.3f} -> {float(new @ first):.3f}")


# ── Concurrency ──────────────────────────────────────────────────────────────
#
# Every profile update is a read-modify-write across two awaits, and nothing
# serialised them. Measured before the fix: ten concurrent saves for one user
# left interaction_count at 1 — nine updates silently lost. The live database
# showed exactly that signature, with 10-save users carrying profile counts of
# 2, 4 and 6. Saving several papers quickly is the normal way to use the
# product, so most of what a new user told the system was being discarded.

async def test_concurrent_saves_are_not_lost(fake_store):
    import asyncio
    rng = np.random.default_rng(5)
    saves = [_unit(rng.normal(size=D)) for _ in range(10)]

    await asyncio.gather(*(P.update_on_save("racer", x) for x in saves))

    assert fake_store[("racer", "long_term")]["interaction_count"] == 10
    assert fake_store[("racer", "short_term")]["interaction_count"] == 10


async def test_concurrent_dismissals_are_not_lost(fake_store):
    import asyncio
    rng = np.random.default_rng(6)
    dismissals = [_unit(rng.normal(size=D)) for _ in range(8)]

    await asyncio.gather(*(P.update_on_dismiss("racer", x) for x in dismissals))

    assert fake_store[("racer", "negative")]["interaction_count"] == 8


async def test_concurrent_saves_produce_the_sequential_result(fake_store):
    """Serialisation must give the same answer as doing them in order."""
    import asyncio
    rng = np.random.default_rng(7)
    saves = [_unit(rng.normal(size=D)) for _ in range(6)]

    for x in saves:
        await P.update_on_save("sequential", x)
    expected = await P.load_profile("sequential", "long_term")

    await asyncio.gather(*(P.update_on_save("concurrent", x) for x in saves))
    got = await P.load_profile("concurrent", "long_term")

    # Order within the gather is not guaranteed, so compare representativeness
    # rather than demanding an identical vector.
    centroid = _unit(np.mean(saves, axis=0))
    assert float(got @ centroid) == pytest.approx(
        float(expected @ centroid), abs=0.05)


async def test_different_users_do_not_block_each_other(fake_store):
    """The lock is per user; it must not serialise the whole app."""
    import asyncio
    rng = np.random.default_rng(8)
    await asyncio.gather(*(
        P.update_on_save(f"user{i}", _unit(rng.normal(size=D)))
        for i in range(20)))
    for i in range(20):
        assert fake_store[(f"user{i}", "long_term")]["interaction_count"] == 1


def test_background_tasks_are_strongly_referenced():
    """asyncio keeps only a weak reference; an unheld task can be collected."""
    import inspect
    from app.routers import events
    src = inspect.getsource(events)
    assert "asyncio.create_task(_update_profile" not in src, (
        "background profile update is not strongly referenced")
    assert "_pending" in src and "add_done_callback" in src
