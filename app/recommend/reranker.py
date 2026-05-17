"""
Re-ranking layer for recommendation candidates.

Phase 2c initial: Heuristic scorer using hand-tuned feature weights.
Phase 6:          LightGBM lambdarank trained on citation pseudo-labels.

The module loads a LightGBM model at import time (if available).
If the model file is missing or LightGBM is not installed, it falls
back gracefully to the heuristic scorer — no crash, no degradation.

Features (37-feature schema):
  0-19:  Content/retrieval (Qdrant score, citations, age, categories, …)
  20-30: User behavior (EWMA profiles, cluster info, interaction counts)
  31-36: Cross features (cosine×recency, cosine×citations, …)

Reference: models/reranker-phase6/production_model/feature_schema.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ── LightGBM model loading (graceful fallback) ──────────────────────────────

_lgb_model = None
_USE_LGB = False

try:
    import lightgbm as lgb

    # Search for model file in several locations
    _MODEL_SEARCH_PATHS = [
        os.environ.get("RERANKER_MODEL_PATH", ""),
        "models/reranker-phase6/production_model/reranker_v1.txt",
        "production_model/reranker_v1.txt",
        str(Path(__file__).resolve().parents[2] / "models" / "reranker-phase6" / "production_model" / "reranker_v1.txt"),
    ]

    for _path in _MODEL_SEARCH_PATHS:
        if _path and os.path.isfile(_path):
            _lgb_model = lgb.Booster(model_file=_path)
            _USE_LGB = True
            print(f"[reranker] SUCCESS: LightGBM model loaded from {_path}")
            print(f"[reranker]   trees={_lgb_model.num_trees()}, features={_lgb_model.num_feature()}")
            break

    if not _USE_LGB:
        print("[reranker] LightGBM installed but model file not found — using heuristic")
        print(f"[reranker]   searched: {[p for p in _MODEL_SEARCH_PATHS if p]}")

except ImportError:
    print("[reranker] LightGBM not installed — using heuristic fallback")
except Exception as e:
    print(f"[reranker] Model load error: {e} — using heuristic fallback")


# ── 37-Feature Schema ────────────────────────────────────────────────────────
# Must match the order in production_model/feature_schema.json exactly.

FEATURE_NAMES = [
    # Content/Retrieval (0-19)
    "qdrant_cosine_score",            # 0
    "candidate_position",             # 1
    "candidate_citation_count",       # 2
    "candidate_log_citations",        # 3
    "candidate_influential_citations",# 4
    "candidate_age_days",             # 5
    "candidate_recency_score",        # 6
    "query_citation_count",           # 7
    "query_age_days",                 # 8
    "year_diff",                      # 9
    "same_primary_category",          # 10
    "co_citation_count",              # 11
    "shared_author_count",            # 12
    "candidate_is_newer",             # 13
    "query_log_citations",            # 14
    "citation_count_ratio",           # 15
    "age_ratio",                      # 16
    "candidate_citations_per_year",   # 17
    "query_num_references",           # 18
    "candidate_num_cited_by",         # 19
    # User behavior (20-30)
    "ewma_longterm_similarity",       # 20
    "ewma_shortterm_similarity",      # 21
    "ewma_negative_similarity",       # 22
    "cluster_importance",             # 23
    "cluster_distance_to_medoid",     # 24
    "is_suppressed_category",         # 25
    "onboarding_category_match",      # 26
    "user_total_saves",               # 27
    "user_total_dismissals",          # 28
    "user_days_since_last_save",      # 29
    "user_session_save_count",        # 30
    # Cross features (31-36)
    "cosine_x_recency",              # 31
    "cosine_x_citations",            # 32
    "category_x_recency",            # 33
    "cosine_x_cocitation",           # 34
    "position_inverse",              # 35
    "citations_x_recency",           # 36
]
NUM_FEATURES = 37


# ── Utility Functions ────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


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


def _parse_year(date_str: str) -> int:
    """Extract year from a YYYY-MM-DD date string."""
    try:
        return int(date_str[:4])
    except (ValueError, TypeError, IndexError):
        return 2020


def _parse_age_days(date_str: str) -> int:
    """Compute age in days from a YYYY-MM-DD date string."""
    now = datetime.now(timezone.utc)
    try:
        pub_date = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return max(0, (now - pub_date).days)
    except (ValueError, TypeError):
        return 365  # default 1 year


def _parse_authors(authors_field) -> list[str]:
    """Parse authors from various formats (JSON string, list, comma-sep)."""
    if isinstance(authors_field, list):
        return authors_field
    if not authors_field:
        return []
    s = str(authors_field)
    if s.startswith("["):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            pass
    return [a.strip() for a in s.split(",") if a.strip()]


# ── Feature Computation (37 features) ────────────────────────────────────────

def compute_features(
    candidate_embeddings: np.ndarray,
    candidate_metadata: list[dict],
    long_term_vec: np.ndarray | None = None,
    short_term_vec: np.ndarray | None = None,
    negative_vec: np.ndarray | None = None,
    *,
    # Phase 6 parameters (optional for backward compat)
    qdrant_scores: np.ndarray | None = None,
    cluster_importance: np.ndarray | float = 0.0,           # (N,) or scalar
    cluster_medoid: np.ndarray | None = None,               # (1024,) or (N, 1024)
    is_suppressed_category: np.ndarray | None = None,       # (N,) pre-computed
    onboarding_category_match: np.ndarray | None = None,    # (N,) pre-computed
    suppressed_categories: set[str] | None = None,          # legacy: compute from set
    onboarding_categories: set[str] | None = None,          # legacy: compute from set
    user_total_saves: int = 0,
    user_total_dismissals: int = 0,
    user_days_since_last_save: float = 0.0,
    user_session_save_count: int = 0,
) -> np.ndarray:
    """
    Compute the full 37-feature matrix for all candidates.

    Backward-compatible: the original 5 parameters still work.
    New Phase 6 keyword args add metadata/user-behavior features.

    Args:
        candidate_embeddings: shape (N, 1024)
        candidate_metadata: list of dicts from Turso (arxiv_id, category,
            published, citation_count, influential_citations, authors, …)
        long_term_vec: user's long-term EWMA profile (1024-dim)
        short_term_vec: user's short-term EWMA profile (1024-dim)
        negative_vec: user's negative EWMA profile (1024-dim)
        qdrant_scores: per-candidate Qdrant cosine scores (N,)
        cluster_importance: importance weight of the serving cluster
        cluster_medoid: cluster medoid embedding (1024-dim)
        suppressed_categories: user's suppressed arXiv categories
        onboarding_categories: user's onboarding category selections
        user_total_saves: total saved papers
        user_total_dismissals: total dismissed papers
        user_days_since_last_save: days since user's most recent save
        user_session_save_count: papers saved in this session

    Returns:
        (N, 37) feature matrix in the schema order
    """
    n = len(candidate_metadata)
    now = datetime.now(timezone.utc)
    features = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    suppressed = suppressed_categories or set()
    onboarding = onboarding_categories or set()
    # Phase 6.1+: pre-computed arrays take priority over set-based computation
    is_suppressed_category_arr = is_suppressed_category
    onboarding_category_match_arr = onboarding_category_match

    # ── Batch cosine similarities (vectorized — fast) ─────────────────────

    # Feature 20: ewma_longterm_similarity
    if long_term_vec is not None:
        features[:, 20] = _cosine_sim_batch(candidate_embeddings, long_term_vec)

    # Feature 21: ewma_shortterm_similarity
    if short_term_vec is not None:
        features[:, 21] = _cosine_sim_batch(candidate_embeddings, short_term_vec)

    # Feature 22: ewma_negative_similarity
    if negative_vec is not None:
        features[:, 22] = _cosine_sim_batch(candidate_embeddings, negative_vec)

    # Feature 24: cluster_distance_to_medoid
    if cluster_medoid is not None:
        if cluster_medoid.ndim == 1:
            # Phase 6.1: single medoid broadcast to all candidates
            features[:, 24] = 1.0 - _cosine_sim_batch(candidate_embeddings, cluster_medoid)
        else:
            # Phase 6.2: per-candidate medoid (N, 1024)
            cand_norms = candidate_embeddings / (
                np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
            )
            med_norms = cluster_medoid / (
                np.linalg.norm(cluster_medoid, axis=1, keepdims=True) + 1e-10
            )
            sims = (cand_norms * med_norms).sum(axis=1)
            features[:, 24] = (1.0 - sims).astype(np.float32)

    # ── Per-candidate features ────────────────────────────────────────────

    for i, meta in enumerate(candidate_metadata):
        # 0: qdrant_cosine_score
        if qdrant_scores is not None and i < len(qdrant_scores):
            features[i, 0] = float(qdrant_scores[i])
        else:
            # Fallback: use long-term similarity as proxy (matches heuristic baseline)
            features[i, 0] = features[i, 20]

        # 1: candidate_position (0-indexed)
        features[i, 1] = float(i)

        # 2: candidate_citation_count
        cand_citations = meta.get("citation_count", 0) or 0
        try:
            cand_citations = int(cand_citations)
        except (ValueError, TypeError):
            cand_citations = 0
        features[i, 2] = float(cand_citations)

        # 3: candidate_log_citations
        features[i, 3] = np.log(cand_citations + 1)

        # 4: candidate_influential_citations
        inf_cit = meta.get("influential_citations", 0) or 0
        try:
            inf_cit = int(inf_cit)
        except (ValueError, TypeError):
            inf_cit = 0
        features[i, 4] = float(inf_cit)

        # 5: candidate_age_days
        pub_str = meta.get("published", "") or meta.get("update_date", "") or ""
        cand_age = _parse_age_days(pub_str)
        features[i, 5] = float(cand_age)

        # 6: candidate_recency_score (matches heuristic decay)
        features[i, 6] = np.exp(-0.002 * cand_age)

        # 7-8: query_citation_count / query_age_days (0 — no seed paper in recs)
        # These will be populated when the user has a clear "query" paper.
        features[i, 7] = 0.0
        features[i, 8] = 0.0

        # 9: year_diff (relative to current year)
        cand_year = _parse_year(pub_str)
        current_year = now.year
        features[i, 9] = abs(current_year - cand_year)

        # 10: same_primary_category (0 — no single query paper)
        c_cat = meta.get("category", "") or meta.get("primary_topic", "") or ""
        features[i, 10] = 0.0

        # 11: co_citation_count (0 — no citation graph in prod yet)
        features[i, 11] = 0.0

        # 12: shared_author_count (0 — no query paper)
        features[i, 12] = 0.0

        # 13: candidate_is_newer (vs current year: always 0 or 1)
        features[i, 13] = 1.0 if cand_year >= current_year else 0.0

        # 14: query_log_citations
        features[i, 14] = 0.0

        # 15: citation_count_ratio
        features[i, 15] = float(cand_citations)  # / (0 + 1) = cand_citations

        # 16: age_ratio (cand_age / 1, since query_age is 0)
        features[i, 16] = 0.0

        # 17: candidate_citations_per_year
        cand_age_years = max(cand_age / 365.0, 0.5)
        features[i, 17] = cand_citations / cand_age_years

        # 18: query_num_references (0 — no citation graph)
        features[i, 18] = 0.0

        # 19: candidate_num_cited_by (0 — no citation graph)
        features[i, 19] = 0.0

        # ── User behavior features (batch values) ────────────────────────

        # 23: cluster_importance (per-candidate array or scalar)
        if isinstance(cluster_importance, np.ndarray):
            features[i, 23] = cluster_importance[i]
        else:
            features[i, 23] = float(cluster_importance)

        # 25: is_suppressed_category (pre-computed array or from set)
        if is_suppressed_category_arr is not None:
            features[i, 25] = is_suppressed_category_arr[i]
        else:
            features[i, 25] = 1.0 if c_cat in suppressed else 0.0

        # 26: onboarding_category_match (pre-computed array or from set)
        if onboarding_category_match_arr is not None:
            features[i, 26] = onboarding_category_match_arr[i]
        else:
            features[i, 26] = 1.0 if c_cat in onboarding else 0.0

        # 27-30: Interaction counts (same for all candidates)
        features[i, 27] = float(user_total_saves)
        features[i, 28] = float(user_total_dismissals)
        features[i, 29] = float(user_days_since_last_save)
        features[i, 30] = float(user_session_save_count)

        # ── Cross features (31-36) ───────────────────────────────────────

        features[i, 31] = features[i, 0] * features[i, 6]   # cosine × recency
        features[i, 32] = features[i, 0] * features[i, 3]   # cosine × log_citations
        features[i, 33] = features[i, 10] * features[i, 6]  # category × recency
        features[i, 34] = features[i, 0] * np.log(features[i, 11] + 1)  # cosine × log_co_citation
        features[i, 35] = 1.0 / (features[i, 1] + 1)        # position_inverse
        features[i, 36] = features[i, 3] * features[i, 6]   # log_citations × recency

    return features


# ── Heuristic Scorer (permanent fallback) ────────────────────────────────────

def heuristic_score(features: np.ndarray) -> np.ndarray:
    """
    Hand-tuned scoring function.  Used before LightGBM model is available
    or as permanent fallback if model fails.

    Uses the EWMA profile similarities + recency + position confidence.
    When no EWMA profiles exist (features 20-22 = 0), falls back to
    Qdrant cosine score + recency + position.

    Returns: scores array of shape (N,), higher = better
    """
    # Use EWMA similarities if available, otherwise Qdrant cosine
    lt_sim = features[:, 20]    # ewma_longterm_similarity
    st_sim = features[:, 21]    # ewma_shortterm_similarity
    neg_sim = features[:, 22]   # ewma_negative_similarity
    qdrant_cosine = features[:, 0]  # qdrant_cosine_score

    # If EWMA profiles are zero (no user data), use Qdrant cosine as proxy
    has_ewma = np.any(lt_sim != 0)
    if has_ewma:
        relevance = 0.40 * lt_sim + 0.25 * st_sim
    else:
        relevance = 0.65 * qdrant_cosine

    # Recency: exponential decay with ~365-day half-life
    recency = features[:, 6]  # candidate_recency_score (pre-computed)

    # Position confidence: inverse of rank position
    position_inv = features[:, 35]  # position_inverse (pre-computed)

    # Negative penalty: demote papers similar to dismissed ones
    neg_penalty = np.clip(neg_sim, 0.0, None)

    scores = (
        relevance
        + 0.15 * recency
        + 0.10 * position_inv
        - 0.15 * neg_penalty
    )
    return scores


# ── Model Accessors (Phase 6.3) ──────────────────────────────────────────────

def is_model_loaded() -> bool:
    """True iff a LightGBM Booster is loaded and active."""
    return _USE_LGB and _lgb_model is not None


def get_loaded_model_path() -> str | None:
    """Return the filesystem path the model was loaded from, or None."""
    if not _USE_LGB:
        return None
    for _p in _MODEL_SEARCH_PATHS:
        if _p and os.path.isfile(_p):
            return _p
    return None


def get_num_trees() -> int:
    """Return number of boosting trees, or 0 if model not loaded."""
    return _lgb_model.num_trees() if _lgb_model is not None else 0


# ── Main Reranking Entry Point ───────────────────────────────────────────────

def rerank_candidates(
    candidate_ids: list[str],
    candidate_embeddings: np.ndarray,
    candidate_metadata: list[dict],
    long_term_vec: np.ndarray | None = None,
    short_term_vec: np.ndarray | None = None,
    negative_vec: np.ndarray | None = None,
    *,
    # Phase 6 additions (all optional — backward compat with legacy callers)
    qdrant_scores: np.ndarray | None = None,
    cluster_importance: np.ndarray | float = 0.0,           # (N,) or scalar
    cluster_medoid: np.ndarray | None = None,               # (1024,) or (N, 1024)
    is_suppressed_category: np.ndarray | None = None,       # (N,) pre-computed
    onboarding_category_match: np.ndarray | None = None,    # (N,) pre-computed
    suppressed_categories: set[str] | None = None,          # legacy path
    onboarding_categories: set[str] | None = None,          # legacy path
    user_total_saves: int = 0,
    user_total_dismissals: int = 0,
    user_days_since_last_save: float = 0.0,
    user_session_save_count: int = 0,
) -> tuple[list[str], list[float], np.ndarray]:
    """
    Score and re-rank candidates.

    Backward-compatible: works with old 6-arg call signature.
    New Phase 6 keyword args enable the full 37-feature LightGBM model.

    Args:
        candidate_ids: arXiv IDs
        candidate_embeddings: (N, 1024) embedding matrix
        candidate_metadata: list of paper dicts from Turso
        long_term_vec: user's long-term EWMA profile
        short_term_vec: user's short-term EWMA profile
        negative_vec: user's negative EWMA profile
        qdrant_scores: per-candidate Qdrant cosine scores
        cluster_importance: importance weight of serving cluster
        cluster_medoid: cluster medoid embedding vector
        suppressed_categories: user's suppressed categories
        onboarding_categories: user's onboarding categories
        user_total_saves: total papers saved by user
        user_total_dismissals: total papers dismissed by user
        user_days_since_last_save: days since last save
        user_session_save_count: saves this session

    Returns:
        (sorted_ids, sorted_scores, sorted_embeddings)
        all in descending score order
    """
    features = compute_features(
        candidate_embeddings,
        candidate_metadata,
        long_term_vec,
        short_term_vec,
        negative_vec,
        qdrant_scores=qdrant_scores,
        cluster_importance=cluster_importance,
        cluster_medoid=cluster_medoid,
        is_suppressed_category=is_suppressed_category,
        onboarding_category_match=onboarding_category_match,
        suppressed_categories=suppressed_categories,
        onboarding_categories=onboarding_categories,
        user_total_saves=user_total_saves,
        user_total_dismissals=user_total_dismissals,
        user_days_since_last_save=user_days_since_last_save,
        user_session_save_count=user_session_save_count,
    )

    # Phase 6.3: Log per-request feature activation
    if features.shape[0] > 0:
        nonzero_rate = (features != 0).mean(axis=0)
        active_count = int((nonzero_rate > 0).sum())
        print(
            f"[reranker] features: {active_count}/37 active, "
            f"n_candidates={features.shape[0]}, "
            f"model={'lgb' if _USE_LGB else 'heuristic'}"
        )

    # ── Score: LightGBM or heuristic ─────────────────────────────────────
    if _USE_LGB and _lgb_model is not None:
        try:
            scores = _lgb_model.predict(features)
        except Exception as e:
            print(f"[reranker] ⚠️ LightGBM prediction failed: {e} — using heuristic")
            scores = heuristic_score(features)
    else:
        scores = heuristic_score(features)

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_ids = [candidate_ids[i] for i in order]
    sorted_scores = scores[order].tolist()
    sorted_embs = candidate_embeddings[order]

    return sorted_ids, sorted_scores, sorted_embs
