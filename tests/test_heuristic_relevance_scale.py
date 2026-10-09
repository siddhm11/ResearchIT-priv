"""
heuristic_score, personalised branch: similarities are rescaled across the
candidate set and recency is weighted at 0.45 (doc 06 changelog, 2026-10-09).

Raw profile similarities barely vary across a reader's candidates (~0.05),
so with raw values a small recency edge outvoted any relevance gap. These tests
pin the intended behaviour: relevance decides, recency still counts, and the
cold-start branch is unchanged.
"""
import numpy as np

from app.recommend.reranker import (COLD_START_RECENCY_WEIGHT, PERSONALISED_RECENCY_WEIGHT,
                                    heuristic_score)


def _features(lt, st=None, recency=None, qdrant=None):
    n = len(lt)
    f = np.zeros((n, 37), dtype=np.float32)
    f[:, 20] = lt
    f[:, 21] = lt if st is None else st
    f[:, 6] = 0.0 if recency is None else recency
    f[:, 0] = 0.0 if qdrant is None else qdrant
    f[:, 35] = 0.0                       # neutral position
    return f


def test_weights_are_the_measured_ones():
    assert PERSONALISED_RECENCY_WEIGHT == 0.45
    assert COLD_START_RECENCY_WEIGHT == 0.15


def test_relevance_order_survives_a_compressed_similarity_band():
    # Realistic raw band: every candidate is 0.70-0.75 similar to the reader.
    lt = np.array([0.75, 0.72, 0.70], dtype=np.float32)
    scores = heuristic_score(_features(lt))
    assert list(np.argsort(-scores)) == [0, 1, 2]


def test_a_constant_shift_in_similarity_changes_nothing():
    lt = np.array([0.75, 0.72, 0.70], dtype=np.float32)
    a = heuristic_score(_features(lt))
    b = heuristic_score(_features(lt + 0.2))
    assert np.allclose(a, b)


def test_best_match_beats_a_newer_weak_match():
    # Old formula: 0.40 * (0.75 - 0.70) = 0.02 relevance edge vs a 0.15 recency
    # edge, so the brand-new weak match won. Now the best match wins.
    lt = np.array([0.75, 0.72, 0.70], dtype=np.float32)
    recency = np.array([0.1, 0.5, 1.0], dtype=np.float32)
    scores = heuristic_score(_features(lt, recency=recency))
    assert np.argmax(scores) == 0


def test_recency_still_breaks_near_ties():
    lt = np.array([0.750, 0.749, 0.70], dtype=np.float32)
    recency = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    scores = heuristic_score(_features(lt, recency=recency))
    assert np.argmax(scores) == 1


def test_personalised_recency_weight_applies():
    lt = np.array([0.7, 0.7], dtype=np.float32)          # equal: rescaled relevance is 0
    recency = np.array([1.0, 0.0], dtype=np.float32)
    s = heuristic_score(_features(lt, recency=recency))
    assert np.isclose(s[0] - s[1], 0.45)


def test_cold_start_branch_is_unchanged():
    lt = np.zeros(2, dtype=np.float32)                   # no profile
    recency = np.array([1.0, 0.0], dtype=np.float32)
    qdrant = np.array([0.7, 0.7], dtype=np.float32)
    s = heuristic_score(_features(lt, st=np.zeros(2, dtype=np.float32), recency=recency, qdrant=qdrant))
    assert np.isclose(s[0] - s[1], 0.15)
