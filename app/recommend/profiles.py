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

import asyncio
import weakref

import numpy as np
from app import db

# ── Per-user update serialisation ────────────────────────────────────────────
#
# Every profile update is a read-modify-write: load the vector, blend the new
# paper in, store it back. Nothing serialised those, and each step awaits, so
# concurrent saves interleave and all but one write is lost.
#
# This is not theoretical. Measured with ten concurrent saves for one user, the
# stored interaction_count came back as 1 — nine updates gone — and the live
# database shows exactly that signature: users with 10 saves carrying profile
# counts of 2, 4 and 6. Saving several papers quickly is the NORMAL way to use
# this product, so most of what a new user tells the system was being discarded.
#
# A WeakValueDictionary rather than a plain dict so the table cannot grow
# without bound: while a lock is held or awaited, the coroutine's frame keeps it
# alive, and once nobody holds it there is by definition no contention to lose.
_user_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = (
    weakref.WeakValueDictionary())


def _lock_for(user_id: str) -> asyncio.Lock:
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock

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


def effective_alpha(alpha: float, count: int | None) -> float:
    """The update rate to actually use, given how much history exists.

    THE STARTUP PROBLEM
    -------------------
    A plain EWMA seeded with its first observation never escapes it. Because
    this profile is L2-normalised after every step, it always has unit
    magnitude, so each new paper gets a fixed α pull against a full-strength
    incumbent — and at α=0.03 that pull is 3%. Measured with the real function
    over near-orthogonal saves:

        saves   cos(profile, first save)   cos(profile, library centroid)
           5              0.998                       0.492
          10              0.996                       0.446
          20              0.991                       0.346
          40              0.982                       0.291

    After FORTY saves the profile is still 98% the first paper the user ever
    clicked. This vector is the largest term in heuristic_score (0.40), the
    relevance axis MMR selects against, and the entirety of Tier 2 — so "the
    overall profile built from your saved papers", which is what the UI tells
    the reader it is, was in practice one arbitrary paper.

    THE FIX, AND WHY IT IS NOT A PARAMETER CHANGE
    ---------------------------------------------
    α is not the problem. α is the STEADY-STATE adaptation rate — how fast an
    established profile drifts as interests change — and 0.03 (~23-save
    half-life) is a doc 06 non-negotiable that PinnerSage evidence supports.
    The problem is purely the TRANSIENT: there is no sensible reason for save
    #2 to move a one-paper profile by only 3%.

    So warm up with a running mean and hand over to the EWMA once the EWMA is
    the slower of the two:

        α_effective = max(α, 1 / (count + 1))

    At count=1 that is 1/2, so the second save gives an even blend of two
    papers. At count=2 it is 1/3, and so on — a true running mean — until
    1/(count+1) falls below α, at which point α takes over permanently. For
    α=0.03 the handover is at 33 saves, and it is continuous: the two rates are
    equal exactly where they cross, so there is no discontinuity in behaviour.

    The steady state is therefore bit-for-bit the documented EWMA. Only the
    first few dozen interactions differ, and only in the direction of being
    representative rather than being one paper.

    This also self-heals existing users without a migration: their stored
    interaction_count feeds straight into the formula, so a profile that is
    currently stuck on its first paper adapts quickly until it reaches the
    handover and then settles into normal behaviour.

    Passing count=None keeps the raw α, for callers that genuinely want the
    unwarmed rate.
    """
    if count is None:
        return alpha
    return max(alpha, 1.0 / (max(0, count) + 1))


def ewma_update(
    current: np.ndarray | None,
    new_embedding: np.ndarray,
    alpha: float,
    count: int | None = None,
    normalise: bool = True,
) -> np.ndarray:
    """
    Exponentially Weighted Moving Average update, with a running-mean warmup.

    If current is None (first interaction), the new embedding IS the profile.
    Otherwise: profile = (1 - α_eff) × current  +  α_eff × new_embedding

    `count` is how many interactions are ALREADY in `current`. It selects the
    warmup rate — see effective_alpha() for why that matters. Omitting it
    preserves the original un-warmed behaviour, so existing callers and tests
    are unaffected.

    `normalise` controls whether the RESULT is scaled back to unit length.

    Normalising every step is lossy, and not in a harmless way. The magnitude
    of a running mean of unit vectors is the signal for how much those vectors
    AGREE: near-identical saves keep it close to 1, saves spread across several
    interests shrink it toward 0. Rescaling to 1 after every update throws that
    away and re-inflates the incumbent to full strength, so early papers keep a
    weight they have not earned. Measured against the true mean, over a user
    with three distinct interests and twenty saves, cos(profile, centroid) was
    0.726 normalising each step versus 1.000 accumulating — and three distinct
    interests is precisely the case this product exists to serve.

    So the persistence path accumulates raw (normalise=False) and
    `load_profile` normalises on read. Every consumer still receives a unit
    vector and none of their contracts change — they all normalise defensively
    anyway (reranker._cosine_sim_batch, mmr_rerank, and Qdrant's Cosine
    metric). Only the stored accumulator is different.
    """
    new_embedding = new_embedding.astype(np.float64)
    alpha = effective_alpha(alpha, count)

    if current is None:
        result = new_embedding
    else:
        current = current.astype(np.float64)
        result = (1.0 - alpha) * current + alpha * new_embedding

    if not normalise:
        return result.astype(np.float32)

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
    """Load a profile vector from SQLite, L2-normalised.  None if not found.

    The stored form is an un-normalised accumulator — its magnitude carries how
    much the user's saves agree (see ewma_update). Consumers want a direction,
    so they get one here, and their contract is exactly what it was before the
    accumulator changed.
    """
    row = await db.get_user_profile(user_id, profile_type)
    if row is None:
        return None
    return _normalise(_from_bytes(row["vector"]))


async def load_profile_raw(user_id: str, profile_type: str) -> np.ndarray | None:
    """The stored accumulator, magnitude intact. For the update path only."""
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

    Serialised per user: this is a read-modify-write and saving several papers
    in quick succession is the normal way to use the product. See _lock_for.
    """
    async with _lock_for(user_id):
        await _update_on_save_locked(user_id, paper_embedding)


async def _update_on_save_locked(user_id: str, paper_embedding: np.ndarray) -> None:
    # Long-term
    lt_current = await load_profile_raw(user_id, "long_term")
    lt_count = await get_interaction_count(user_id, "long_term")
    lt_updated = ewma_update(lt_current, paper_embedding, ALPHA_LONG_TERM,
                             count=lt_count, normalise=False)
    await save_profile(user_id, "long_term", lt_updated, lt_count + 1)

    # Short-term
    st_current = await load_profile_raw(user_id, "short_term")
    st_count = await get_interaction_count(user_id, "short_term")
    st_updated = ewma_update(st_current, paper_embedding, ALPHA_SHORT_TERM,
                             count=st_count, normalise=False)
    await save_profile(user_id, "short_term", st_updated, st_count + 1)


async def update_on_dismiss(user_id: str, paper_embedding: np.ndarray) -> None:
    """
    Called when a user dismisses a paper.
    Updates the negative profile.

    Serialised per user for the same reason as update_on_save.
    """
    async with _lock_for(user_id):
        await _update_on_dismiss_locked(user_id, paper_embedding)


async def _update_on_dismiss_locked(user_id: str, paper_embedding: np.ndarray) -> None:
    neg_current = await load_profile_raw(user_id, "negative")
    neg_count = await get_interaction_count(user_id, "negative")
    neg_updated = ewma_update(neg_current, paper_embedding, ALPHA_NEGATIVE,
                              count=neg_count, normalise=False)
    await save_profile(user_id, "negative", neg_updated, neg_count + 1)
