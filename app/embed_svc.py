"""
BGE-M3 embedding model singleton — Phase 3.

Responsibilities:
  - Load BAAI/bge-m3 once (lazily on first call or eagerly via get_model())
  - encode_query(text) → (dense: np.ndarray[1024], sparse: dict[int, float])
  - LRU cache on query text to avoid re-encoding repeats
  - CPU float32, no GPU dependency
  - Thread-safe (model is read-only after load)
"""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from app import config

# ── Module-level singleton ────────────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()


def get_model():
    """
    Return the BGE-M3 model singleton.  Thread-safe, loads once.

    Called eagerly in main.py lifespan so the first request doesn't pay
    the ~15 s model-download cost.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model

        from FlagEmbedding import BGEM3FlagModel

        print(f"[embed_svc] Loading {config.BGE_M3_MODEL} on {config.BGE_M3_DEVICE}…")

        # use_fp16=False on CPU (fp16 requires CUDA)
        use_fp16 = config.BGE_M3_DEVICE != "cpu"
        _model = BGEM3FlagModel(
            config.BGE_M3_MODEL,
            use_fp16=use_fp16,
            device=config.BGE_M3_DEVICE,
        )
        print("[embed_svc] Model loaded successfully")
        return _model


# ── Cached query encoding ────────────────────────────────────────────────────

@lru_cache(maxsize=config.ENCODE_CACHE_SIZE)
def _encode_cached(text: str) -> tuple:
    """
    Encode a single query string.  Returns (dense_vec, sparse_dict).

    The LRU cache key is the raw text string.  Cached results avoid
    re-running BGE-M3 inference for repeated queries.

    Returns a tuple so it's hashable for the cache decorator.
    The caller unpacks it.
    """
    model = get_model()
    out = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=512,
    )
    dense = out["dense_vecs"][0]          # shape (1024,) float32
    sparse = out["lexical_weights"][0]    # dict {token_id_int: float}

    # Ensure dense is a numpy array (model may return tensor)
    if not isinstance(dense, np.ndarray):
        dense = np.array(dense, dtype=np.float32)

    # Ensure sparse values are plain floats (not tensors)
    sparse_clean = {int(k): float(v) for k, v in sparse.items()}

    return (dense, sparse_clean)


def encode_query(text: str) -> tuple[np.ndarray, dict[int, float]]:
    """
    Encode a query string into dense + sparse representations.

    Args:
        text: User's search query (raw or rewritten).

    Returns:
        (dense_vec, sparse_dict) where:
          dense_vec:   np.ndarray of shape (1024,), float32
          sparse_dict: {int_token_id: float_weight}
    """
    text = text.strip()
    if not text:
        return np.zeros(1024, dtype=np.float32), {}
    return _encode_cached(text)
