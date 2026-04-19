"""
Re-ranking layer for recommendation candidates.

Phase 2c initial: Heuristic scorer using hand-tuned feature weights.
Phase 2c mature:  LightGBM lambdarank trained on save/dismiss data.

The heuristic scorer runs first.  When ≥500 labeled interactions accumulate,
a LightGBM model can be trained offline and loaded here.

Features:
  - cosine_sim_longterm:  dot(user_lt_vec, paper_vec)
  - cosine_sim_shortterm: dot(user_st_vec, paper_vec)
  - paper_age_days:       days since publication
  - rrf_position:         position in the RRF fusion output (lower = better)
  - cosine_sim_negative:  dot(user_neg_vec, paper_vec)  [Doc 06 addition]

Reference: Research-MultiInterest_Recommender_Architecture.md §4
  "LightGBM with a lambdarank objective scores 500 candidates in 2-5ms
   on a single CPU core."

Doc 06 correction: YouTube (2023, Xia et al.) showed a 3x gain from using
  dislikes as both features and labels.  The negative EWMA profile is now
  wired as a penalty feature during reranking.
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np


def _cosine_sim_batch(
    candidate_embeddings: np.ndarray,
    profile_vec: np.ndarray,
) -> np.ndarray:
    """Cosine similarity of each candidate against a single profile vector."""
    pnorm = profile_vec / (np.linalg.norm(profile_vec) + 1e-10)
    cnorms = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
    )
    return cnorms @ pnorm


def compute_features(
    candidate_embeddings: np.ndarray,
    candidate_metadata: list[dict],
    long_term_vec: np.ndarray | None = None,
    short_term_vec: np.ndarray | None = None,
    negative_vec: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract ranking features for each candidate.

    Args:
        candidate_embeddings: shape (N, 1024)
        candidate_metadata: list of dicts with 'published' key (YYYY-MM-DD)
        long_term_vec: user's long-term EWMA profile (1024-dim)
        short_term_vec: user's short-term EWMA profile (1024-dim)
        negative_vec: user's negative EWMA profile (1024-dim) [Doc 06]

    Returns:
        feature matrix of shape (N, num_features)
    """
    n = len(candidate_metadata)
    features = []

    # Feature 1: Cosine similarity to long-term profile
    if long_term_vec is not None:
        lt_sim = _cosine_sim_batch(candidate_embeddings, long_term_vec)
    else:
        lt_sim = np.zeros(n, dtype=np.float32)
    features.append(lt_sim)

    # Feature 2: Cosine similarity to short-term profile
    if short_term_vec is not None:
        st_sim = _cosine_sim_batch(candidate_embeddings, short_term_vec)
    else:
        st_sim = np.zeros(n, dtype=np.float32)
    features.append(st_sim)

    # Feature 3: Paper age in days (0 = today, positive = older)
    now = datetime.now(timezone.utc)
    ages = []
    for meta in candidate_metadata:
        pub = meta.get("published", "")
        try:
            pub_date = datetime.strptime(pub[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            age_days = (now - pub_date).days
        except (ValueError, TypeError):
            age_days = 365  # default to 1 year old if unparseable
        ages.append(age_days)
    features.append(np.array(ages, dtype=np.float32))

    # Feature 4: RRF position (0-indexed, lower = better)
    features.append(np.arange(n, dtype=np.float32))

    # Feature 5: Cosine similarity to negative profile (Doc 06 addition)
    # YouTube (2023): using dislikes as features gives 22% reduction in
    # similar-content; using as both features AND labels gives 60.8%.
    if negative_vec is not None:
        neg_sim = _cosine_sim_batch(candidate_embeddings, negative_vec)
    else:
        neg_sim = np.zeros(n, dtype=np.float32)
    features.append(neg_sim)

    return np.column_stack(features)


def heuristic_score(features: np.ndarray) -> np.ndarray:
    """
    Hand-tuned scoring function.  Used before LightGBM model is trained.

    Weights:
      - 0.40 x long-term similarity     (core relevance)
      - 0.25 x short-term similarity    (session context)
      - 0.15 x recency                  (prefer newer, soft decay)
      - 0.10 x RRF confidence           (prefer higher-ranked candidates)
      - 0.15 x negative penalty         (demote papers like dismissed ones)

    Returns: scores array of shape (N,), higher = better
    """
    lt_sim = features[:, 0]           # cosine sim to long-term
    st_sim = features[:, 1]           # cosine sim to short-term
    age_days = features[:, 2]         # paper age in days
    rrf_pos = features[:, 3]          # RRF rank position
    neg_sim = features[:, 4]          # cosine sim to negative profile

    # Recency: exponential decay with ~365-day half-life
    # Papers from today score 1.0, papers from a year ago score 0.5
    recency = np.exp(-0.002 * age_days)

    # RRF confidence: inverse of position (normalised)
    max_pos = rrf_pos.max() + 1
    rrf_conf = 1.0 - (rrf_pos / max_pos)

    # Negative penalty: papers similar to dismissed papers get demoted
    # Only penalise positive similarity (neg_sim > 0 means similar to disliked)
    neg_penalty = np.clip(neg_sim, 0.0, None)

    scores = (
        0.40 * lt_sim
        + 0.25 * st_sim
        + 0.15 * recency
        + 0.10 * rrf_conf
        - 0.15 * neg_penalty
    )
    return scores


def rerank_candidates(
    candidate_ids: list[str],
    candidate_embeddings: np.ndarray,
    candidate_metadata: list[dict],
    long_term_vec: np.ndarray | None = None,
    short_term_vec: np.ndarray | None = None,
    negative_vec: np.ndarray | None = None,
) -> tuple[list[str], list[float], np.ndarray]:
    """
    Score and re-rank candidates.

    Args:
        negative_vec: user's negative EWMA profile.  Papers similar to this
            get demoted.  (Doc 06: YouTube 3x gain from using dislikes.)

    Returns:
        (sorted_ids, sorted_scores, sorted_embeddings)
        all in descending score order
    """
    features = compute_features(
        candidate_embeddings, candidate_metadata,
        long_term_vec, short_term_vec, negative_vec,
    )
    scores = heuristic_score(features)

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_ids = [candidate_ids[i] for i in order]
    sorted_scores = scores[order].tolist()
    sorted_embs = candidate_embeddings[order]

    return sorted_ids, sorted_scores, sorted_embs
