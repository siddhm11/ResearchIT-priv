"""Cluster importance must track investment, not save order.

Importance drives quota, and quota is the guarantee that a multi-interest feed
stays multi-interest. The weighting these tests pin used to be `1/(i+1)` over
position in the save list, which is a harmonic decay over ORDER: two interests
of identical size were served 8/2 purely because one was explored earlier, and
the split saturated -- 20 saves to 40 saves moved it three points -- so the
reader had no action available that repaired it.

See clustering.IMPORTANCE_HALF_LIFE_DAYS for the measurements.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.recommend import clustering as C
from app.recommend.fusion import allocate_quotas


_NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)


def _ts(days_ago: float) -> str:
    return (_NOW - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def _two_interests(a_days: list[float], b_days: list[float], seed: int = 11):
    """Two well-separated interests, returned newest-save-first.

    Tight noise on purpose: these tests are about the WEIGHTING, so the
    clustering itself must be unambiguous or the assertion measures the wrong
    thing.
    """
    rng = np.random.default_rng(seed)
    a_anchor = rng.standard_normal(1024)
    a_anchor /= np.linalg.norm(a_anchor)
    b_anchor = rng.standard_normal(1024)
    b_anchor /= np.linalg.norm(b_anchor)

    def make(anchor, k):
        v = anchor + 0.03 * rng.standard_normal((k, 1024))
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    ids = [f"A{i}" for i in range(len(a_days))] + [f"B{i}" for i in range(len(b_days))]
    embs = np.vstack([make(a_anchor, len(a_days)), make(b_anchor, len(b_days))])
    days = list(a_days) + list(b_days)

    order = sorted(range(len(ids)), key=lambda i: days[i])   # newest first
    return (
        [ids[i] for i in order],
        embs[order],
        [_ts(days[i]) for i in order],
    )


def _slots_by_interest(ids, embs, times, total=10) -> dict[str, int]:
    """Feed slots each true interest wins, summed over whatever K Ward picks."""
    clusters = C.compute_clusters(ids, embs, times)
    quotas = allocate_quotas(
        [c.importance for c in clusters], total_slots=total, min_slots=1)
    tally = {"A": 0, "B": 0}
    for cluster, slots in zip(clusters, quotas):
        a_share = sum(1 for p in cluster.paper_ids if p.startswith("A"))
        tally["A" if a_share >= len(cluster.paper_ids) / 2 else "B"] += slots
    return tally


# ── The regression this exists for ───────────────────────────────────────────

def test_equal_interests_explored_at_different_times_get_comparable_slots():
    """The 8/2 case. Ten papers each; B was explored two months earlier."""
    ids, embs, times = _two_interests(
        a_days=list(range(0, 30, 3)), b_days=list(range(60, 90, 3)))

    tally = _slots_by_interest(ids, embs, times)

    # Under the old position weighting this was 8/2. A tilt toward the recent
    # interest is intended -- that is what a recency weighting is FOR -- but the
    # older interest must keep a real share of the page, not a token one.
    assert tally["B"] >= 3, f"minority interest starved: {tally}"
    assert tally["A"] <= 7, f"dominant interest took the page: {tally}"


def test_interleaved_equal_interests_split_evenly():
    """Same size, same period, alternating. There is nothing to prefer."""
    ids, embs, times = _two_interests(
        a_days=list(range(0, 20, 2)), b_days=list(range(1, 21, 2)))

    tally = _slots_by_interest(ids, embs, times)

    assert abs(tally["A"] - tally["B"]) <= 1, f"expected an even split: {tally}"


def test_size_outweighs_recency_when_the_gap_is_modest():
    """A bigger investment wins, even if the other interest is more recent.

    Position weighting could not express this: order dominated everything, so
    an interest twice the size still lost on being explored second.
    """
    ids, embs, times = _two_interests(
        a_days=list(range(0, 20, 4)),        # 5 papers, very recent
        b_days=list(range(30, 90, 3)))       # 20 papers, older

    tally = _slots_by_interest(ids, embs, times)

    assert tally["B"] > tally["A"], f"size should win here: {tally}"


def test_a_genuinely_abandoned_interest_does_decay():
    """Recency still has to mean something -- this is the other failure mode.

    A fix that made importance size-only would serve a year-dead interest as
    eagerly as a live one.
    """
    ids, embs, times = _two_interests(
        a_days=list(range(0, 30, 3)), b_days=list(range(365, 395, 3)))

    tally = _slots_by_interest(ids, embs, times)

    assert tally["A"] > tally["B"], f"stale interest not decayed: {tally}"


def test_importance_does_not_saturate_on_list_position():
    """Doubling the library must not leave the split where it was.

    The old weighting moved 81/19 -> 84/16 between 20 and 40 saves: the reader
    could not repair the imbalance by using the product more. This asserts the
    signal is driven by the data rather than by list index.
    """
    small = _slots_by_interest(*_two_interests(
        a_days=list(range(0, 10, 2)), b_days=list(range(10, 20, 2))))
    grown = _slots_by_interest(*_two_interests(
        a_days=list(range(0, 10, 2)),
        b_days=list(range(10, 20, 2)) + list(range(20, 40, 2))))

    assert grown["B"] > small["B"], (
        f"adding saves to B changed nothing: {small} -> {grown}")


# ── The weighting function itself ────────────────────────────────────────────

def test_weights_use_elapsed_time_not_index():
    """Two saves a day apart and two saves a year apart are different."""
    close = C._recency_weights(2, [_ts(0), _ts(1)])
    far = C._recency_weights(2, [_ts(0), _ts(365)])

    assert close[1] > 0.99, close
    assert far[1] < 0.10, far


def test_reference_point_is_the_newest_save_not_now():
    """A reader returning after a year sees the mix they left with.

    Decaying against wall-clock 'now' would push every cluster toward zero
    together and leave quota dividing noise.
    """
    recent = C._recency_weights(3, [_ts(0), _ts(30), _ts(60)])
    same_spread_but_old = C._recency_weights(
        3, [_ts(400), _ts(430), _ts(460)])

    assert np.allclose(recent, same_spread_but_old)
    assert recent[0] == pytest.approx(1.0)


def test_fallback_is_gentler_than_the_harmonic_decay_it_replaced():
    """No timestamps is a degraded path, not a licence to reinstate the bug."""
    fallback = C._recency_weights(20, None)
    harmonic = np.array([1.0 / (i + 1) for i in range(20)])

    assert fallback[-1] > harmonic[-1] * 5
    # The most recent save must not carry a quarter of all importance on its own.
    assert fallback[0] / fallback.sum() < 0.15
    assert harmonic[0] / harmonic.sum() > 0.25      # what it used to be


@pytest.mark.parametrize("bad", [
    None,
    ["", ""],
    ["not-a-date", "also-not"],
    ["2026-08-28 00:00:00"],                 # wrong length for n=2
])
def test_unusable_timestamps_fall_back_without_raising(bad):
    w = C._recency_weights(2, bad)
    assert w.shape == (2,)
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)


def test_identical_timestamps_fall_back_rather_than_flattening():
    """Every save in the same second carries no recency information at all.

    Decaying over a zero spread would weight every paper identically, which
    silently turns importance into a pure count.
    """
    same = [_ts(5)] * 4
    assert not np.allclose(C._recency_weights(4, same),
                           np.ones(4, dtype=np.float64))


def test_timestamps_remain_optional():
    """compute_clusters is called without timestamps in tests and older paths."""
    ids, embs, _ = _two_interests(a_days=[0, 1, 2], b_days=[3, 4, 5])
    clusters = C.compute_clusters(ids, embs)
    assert clusters
    assert all(c.importance > 0 for c in clusters)
