"""
Centralised settings for the arxiv recommender app.
All credentials live in .env locally; override with env vars in production.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present (won't override existing env vars)

# ── Qdrant (BGE-M3 dense, 1 024-dim) ─────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "arxiv_bgem3_dense")

# ── SQLite ────────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "interactions.db")

# ── arXiv API ─────────────────────────────────────────────────────────────────
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = 10          # results per search page
METADATA_CACHE_TTL_DAYS = 30    # re-fetch metadata after this many days

# ── Turso (libSQL) — arXiv metadata DB — Phase 3.5 ───────────────────────────
TURSO_URL = os.getenv("TURSO_URL", "")
TURSO_DB_TOKEN = os.getenv("TURSO_DB_TOKEN", "")

# ── Recommendation settings ───────────────────────────────────────────────────
REC_LIMIT = 10                  # how many recommendations to show
REC_POSITIVE_LIMIT = 20         # max positive examples sent to Qdrant
REC_MIN_POSITIVES = 1           # minimum saves needed before showing recs

# ── Zilliz Cloud (BGE-M3 sparse vectors) — Phase 3 ────────────────────────────
ZILLIZ_URI = os.getenv("ZILLIZ_URI", "")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN", "")
ZILLIZ_COLLECTION = os.getenv("ZILLIZ_COLLECTION", "arxiv_bgem3_sparse")

# Zilliz schema (confirmed from notebooks/01-bme-upload.ipynb):
#   id            INT64  (auto_id, primary key)
#   arxiv_id      VARCHAR
#   sparse_vector SPARSE_FLOAT_VECTOR  (BGE-M3 lexical weights, int token IDs)
#   Index: SPARSE_INVERTED_INDEX, metric_type="IP"

# ── Groq (LLM query rewriter) — Phase 3 ──────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── BGE-M3 (embedding model) — Phase 3 ───────────────────────────────────────
BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
BGE_M3_DEVICE = os.getenv("BGE_M3_DEVICE", "cpu")
ENCODE_CACHE_SIZE = 128  # LRU cache for encoded queries

# ── Hybrid search tuning — Phase 3 ───────────────────────────────────────────
SEARCH_RRF_K = 60                  # RRF denominator
SEARCH_FETCH_K_MULTIPLIER = 6     # candidates = top_k × 6 before rerank
SEARCH_SEMANTIC_WEIGHT = 0.80     # RRF contribution to final score
SEARCH_RECENCY_WEIGHT = 0.20      # recency contribution to final score

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE = "ArXiv Recommender"
COOKIE_NAME = "arxiv_user_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year
APP_PORT = int(os.getenv("PORT", "7860"))  # HF Spaces requires 7860
