"""
EWMA-based user profile embeddings.

Maintains three vector profiles per user:
  - long_term   (α=0.03): enduring research interests (~66-interaction window)
  - short_term  (α=0.40): current session context (~3-5 interactions)
  - negative    (α=0.15): topics the user dislikes

Each profile is a 1024-dim L2-normalised vector (BGE-M3 cosine space).
Storage: binary numpy blobs in SQLite (4096 bytes per vector).

Reference: Research-MultiInterest_Recommender_Architecture.md §3
  "EWMA updates user embeddings with the formula
   embedding_t = α × item_embedding_t + (1−α) × embedding_{t-1}"

Correction (Doc 06): PinnerSage tested λ=0.1 and explicitly rejected it as
  too recent-biased.  Their optimal was λ=0.01.  α=0.03 is our compromise.
"""
from __future__ import annotations

import numpy as np
from app import db

EMBEDDING_DIM = 1024  # BGE-M3 dense dimension

# EWMA smoothing factors
# Doc 06 correction: α_long was 0.10, but PinnerSage (KDD 2020) found λ=0.1
# too recent-biased and selected λ=0.01.  α=0.03 gives ~66-interaction
# effective window — a compromise that preserves minority interests.
ALPHA_LONG_TERM = 0.03   # effective window ~66 interactions (PinnerSage optimal)
ALPHA_SHORT_TERM = 0.40  # effective window ~3-5 interactions
ALPHA_NEGATIVE = 0.15    # moderate responsiveness for negatives


def _normalise(v: np.ndarray) -> np.ndarray | None:
    """L2-normalise a vector.  Returns None if the vector is near-zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return None
    return (v / norm).astype(np.float32)


def ewma_update(
    current: np.ndarray | None,
    new_embedding: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Exponentially Weighted Moving Average update.

    If current is None (first interaction), the new embedding IS the profile.
    Otherwise: profile = (1 - α) × current  +  α × new_embedding

    The result is always L2-normalised (BGE-M3 operates in cosine space).
    """
    new_embedding = new_embedding.astype(np.float64)

    if current is None:
        result = new_embedding
    else:
        current = current.astype(np.float64)
        result = (1.0 - alpha) * current + alpha * new_embedding

    normalised = _normalise(result)
    if normalised is None:
        # Edge case: vectors cancel out → keep old profile
        return current.astype(np.float32) if current is not None else np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return normalised


# ── Storage helpers ───────────────────────────────────────────────────────────

def _to_bytes(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def _from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


async def load_profile(user_id: str, profile_type: str) -> np.ndarray | None:
    """Load a profile vector from SQLite.  Returns None if not found."""
    row = await db.get_user_profile(user_id, profile_type)
    if row is None:
        return None
    return _from_bytes(row["vector"])


async def save_profile(
    user_id: str,
    profile_type: str,
    vector: np.ndarray,
    interaction_count: int,
) -> None:
    """Persist a profile vector to SQLite."""
    await db.upsert_user_profile(
        user_id=user_id,
        profile_type=profile_type,
        vector=_to_bytes(vector),
        interaction_count=interaction_count,
    )


async def get_interaction_count(user_id: str, profile_type: str) -> int:
    """Get the current interaction count for a profile."""
    row = await db.get_user_profile(user_id, profile_type)
    if row is None:
        return 0
    return row["interaction_count"]


# ── High-level update API ────────────────────────────────────────────────────

async def update_on_save(user_id: str, paper_embedding: np.ndarray) -> None:
    """
    Called when a user saves a paper.
    Updates both long-term and short-term profiles.
    """
    # Long-term
    lt_current = await load_profile(user_id, "long_term")
    lt_count = await get_interaction_count(user_id, "long_term")
    lt_updated = ewma_update(lt_current, paper_embedding, ALPHA_LONG_TERM)
    await save_profile(user_id, "long_term", lt_updated, lt_count + 1)

    # Short-term
    st_current = await load_profile(user_id, "short_term")
    st_count = await get_interaction_count(user_id, "short_term")
    st_updated = ewma_update(st_current, paper_embedding, ALPHA_SHORT_TERM)
    await save_profile(user_id, "short_term", st_updated, st_count + 1)


async def update_on_dismiss(user_id: str, paper_embedding: np.ndarray) -> None:
    """
    Called when a user dismisses a paper.
    Updates the negative profile.
    """
    neg_current = await load_profile(user_id, "negative")
    neg_count = await get_interaction_count(user_id, "negative")
    neg_updated = ewma_update(neg_current, paper_embedding, ALPHA_NEGATIVE)
    await save_profile(user_id, "negative", neg_updated, neg_count + 1)
