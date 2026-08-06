"""
Centralised settings for the arxiv recommender app.
All credentials live in .env locally; override with env vars in production.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present (won't override existing env vars)

# ── Qdrant (BGE-M3 dense, 1 024-dim) ─────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "arxiv_bgem3_dense")

# ── SQLite ────────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "interactions.db")

# ── arXiv API ─────────────────────────────────────────────────────────────────
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = 10          # results per search page
METADATA_CACHE_TTL_DAYS = 30    # re-fetch metadata after this many days

# ── Turso (libSQL) — arXiv metadata DB — Phase 3.5 ───────────────────────────
TURSO_URL = os.getenv("TURSO_URL", "").strip()
TURSO_DB_TOKEN = os.getenv("TURSO_DB_TOKEN", "").strip()



# ── Qdrant vector encoding ────────────────────────────────────────────────────
# Set QDRANT_UINT8_SCALE > 0 when the collection stores vectors as `uint8`.
#
# A uint8 collection is NOT transparent. Qdrant stores and returns the raw
# integers, so both directions need converting:
#
#   * queries sent as raw floats match nothing. Measured against a uint8
#     collection, an un-quantised query returned 0 of the correct top-5 — the
#     results were unrelated papers with near-consecutive ids, i.e. the query
#     landed in an arbitrary region of the space.
#   * vectors read back arrive as 0..255 with an L2 norm around 4176.
#     cos(raw, original) = 0.21, versus 0.996 once dequantised. Those vectors
#     feed Ward clustering, MMR, the EWMA similarity features, and — worst —
#     profile updates, where storing a raw uint8 vector would permanently
#     corrupt a user's profile.
#
# Leave SCALE at 0 for a float16/float32 collection and both conversions are
# skipped, so the same code serves either cluster during a migration.
#
# Values come from scripts/build_metadata_sidecar-era quantisation:
#   quantised = round((x - LO) / SCALE), clipped to 0..255, LO = -4*sigma
QDRANT_UINT8_LO = float(os.getenv("QDRANT_UINT8_LO", "0") or 0)
QDRANT_UINT8_SCALE = float(os.getenv("QDRANT_UINT8_SCALE", "0") or 0)


def qdrant_is_uint8() -> bool:
    return QDRANT_UINT8_SCALE > 0


# ── Backend B: online A/B comparison ──────────────────────────────────────────
# Optional second Qdrant cluster, used only by GET /healthz/ab. Leave unset and
# the endpoint reports "not configured" and nothing else changes.
#
# Comparing two clusters by standing up a second Space would confound the
# result: different container, different host, cold caches. Routing both from
# one process holds every other variable fixed, so the cluster is the only
# difference being measured.
#
# When B wins, promote it by copying these values into the primary QDRANT_*
# variables — no code change, and rollback is the same move reversed.
QDRANT_B_URL = os.getenv("QDRANT_B_URL", "").strip()
QDRANT_B_API_KEY = os.getenv("QDRANT_B_API_KEY", "").strip()
QDRANT_B_COLLECTION = os.getenv("QDRANT_B_COLLECTION", QDRANT_COLLECTION)
QDRANT_B_UINT8_LO = float(os.getenv("QDRANT_B_UINT8_LO", "0") or 0)
QDRANT_B_UINT8_SCALE = float(os.getenv("QDRANT_B_UINT8_SCALE", "0") or 0)


def qdrant_b_configured() -> bool:
    return bool(QDRANT_B_URL and QDRANT_B_API_KEY)


# Backend B was originally an A/B comparison slot. It is now the second half of
# the split old corpus: shard A on the primary cluster, shard B here, and the
# recent-papers shard alongside. Inert until QDRANT_B_URL is set, so this can be
# uploaded and verified before anything routes to it.
SEARCH_FANOUT_B = os.getenv("SEARCH_FANOUT_B", "1").strip().lower() in (
    "1", "true", "yes")


def fanout_b_enabled() -> bool:
    return SEARCH_FANOUT_B and qdrant_b_configured()


# ── Recent-papers shard ───────────────────────────────────────────────────────
# The main collection is a static snapshot whose newest paper is from 2025-05.
# The ~202k papers published since then live in a separate small collection on a
# second cluster, and dense retrieval queries both and merges.
#
# Split rather than one big collection because the primary is full: float16 for
# 1.8M papers needs ~4.5 GB against a 4 GB limit, and it is already degraded.
# Re-encoding everything into uint8 to fit was measured and rejected — dropping
# binary quantization to gain recall also removes the in-RAM traversal path, and
# cold queries went from 927 ms to 3093 ms.
#
# The shard mirrors the primary's config (float16, binary quantization, Cosine,
# same m/ef_construct) for one specific reason: scores are then directly
# comparable, so merging is a plain sort. Fan-out is concurrent, so the added
# latency is the slower of the two rather than their sum, and the shard is small
# enough to keep its whole HNSW graph in RAM.
QDRANT_RECENT_URL = os.getenv("QDRANT_RECENT_URL", "").strip()
QDRANT_RECENT_API_KEY = os.getenv("QDRANT_RECENT_API_KEY", "").strip()
QDRANT_RECENT_COLLECTION = os.getenv("QDRANT_RECENT_COLLECTION", "arxiv_recent")

# Off by default: the shard is inert until this is set, so it can be uploaded,
# indexed and verified in production without affecting a single user request.
SEARCH_FANOUT_RECENT = os.getenv("SEARCH_FANOUT_RECENT", "0").strip().lower() in (
    "1", "true", "yes")


def qdrant_recent_configured() -> bool:
    return bool(QDRANT_RECENT_URL and QDRANT_RECENT_API_KEY)


def fanout_enabled() -> bool:
    return SEARCH_FANOUT_RECENT and qdrant_recent_configured()


# ── Sparse retrieval backend ──────────────────────────────────────────────────
# "fts"    -- BM25 over the local metadata sidecar (covers all 1.8M papers)
# "zilliz" -- BGE-M3 lexical weights on Zilliz (covers only the 1.6M snapshot)
#
# Coverage is the reason to prefer "fts". RRF sums 1/(k+rank) across lists, so a
# paper missing from one list scores half of one present in both. With Zilliz
# indexing only the pre-2025-06 snapshot, every newer paper was penalised for
# being absent from a list that could never have contained it — at k=60, a new
# paper at dense rank 1 lost to any old paper above rank 62 in both lists.
#
# Falls back to Zilliz automatically when the sidecar has no FTS index, so an
# older sidecar keeps working.
SPARSE_BACKEND = os.getenv("SPARSE_BACKEND", "fts").strip().lower()


# ── Recommendation reranker selection ─────────────────────────────────────────
# "heuristic" | "lightgbm" | "auto"
#
# Default is "heuristic", and that is a deliberate downgrade on paper.
#
# Parsing models/reranker-phase6/production_model/reranker_v1.txt shows features
# 20-30 have ZERO splits across all 141 trees — every EWMA similarity, both
# cluster features, the suppression and onboarding flags, and all four
# interaction counts. A gradient-boosted tree only reads features it splits on,
# so changing a user's entire profile provably cannot change the model's output.
# It is a citation-and-recency ranker wearing a personalisation schema.
# Compounding that, candidate_num_cited_by carries 65.2% of total importance and
# is hardcoded to 0 at serving time (docs/PHASE6-HANDOFF.md:173).
#
# heuristic_score() in app/recommend/reranker.py does read features 20-22, so
# the "fallback" responds to the user where the model cannot.
#
# This is a reasoned choice, not a measured one — there is no engagement data to
# A/B against yet, which is itself a consequence of the ephemeral database. Flip
# to "lightgbm" to compare once real interactions accumulate. "auto" restores
# the previous behaviour (model when loaded, heuristic otherwise).
RERANKER_MODE = os.getenv("RERANKER_MODE", "heuristic").strip().lower()

# ── Tier 0 trending (cold start) ──────────────────────────────────────────────
# How far back "trending" looks, measured from the newest paper in the corpus
# rather than from today (the corpus is a static snapshot ending 2025-05-30).
# Ranking by all-time citations instead returns the same canonical papers to
# every user forever — Adam, scikit-learn, BatchNorm — which is a poor first
# impression for a feed that claims to surface current research.
TRENDING_RECENCY_MONTHS = int(os.getenv("TRENDING_RECENCY_MONTHS", "24"))

# ── Recommendation settings ───────────────────────────────────────────────────
REC_LIMIT = 10                  # how many recommendations to show
REC_POSITIVE_LIMIT = 20         # max positive examples sent to Qdrant
REC_MIN_POSITIVES = 1           # minimum saves needed before showing recs

# ── Zilliz Cloud (BGE-M3 sparse vectors) — Phase 3 ────────────────────────────
ZILLIZ_URI = os.getenv("ZILLIZ_URI", "").strip()
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN", "").strip()
ZILLIZ_COLLECTION = os.getenv("ZILLIZ_COLLECTION", "arxiv_bgem3_sparse")

# Zilliz schema (confirmed from notebooks/01-bme-upload.ipynb):
#   id            INT64  (auto_id, primary key)
#   arxiv_id      VARCHAR
#   sparse_vector SPARSE_FLOAT_VECTOR  (BGE-M3 lexical weights, int token IDs)
#   Index: SPARSE_INVERTED_INDEX, metric_type="IP"

# ── Groq (LLM query rewriter) — Phase 3 ──────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ── BGE-M3 (embedding model) — Phase 3 ───────────────────────────────────────
BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
BGE_M3_DEVICE = os.getenv("BGE_M3_DEVICE", "cpu")
ENCODE_CACHE_SIZE = 128  # LRU cache for encoded queries

# ── Cross-Encoder Reranker (search reranking) ─────────────────────────────────
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# How many fused candidates get cross-encoded.
#
# This MUST exceed the number of results returned or the stage is pointless:
# at 10 it re-ordered exactly the set that was already going to be shown, so it
# could never pull a better paper up from the rest of the retrieved pool.
# Retrieval fetches limit * SEARCH_FETCH_K_MULTIPLIER (= 60 at limit 10), so 50
# covers the overwhelming majority of the pool.
#
# Latency: cross-encoding is CPU-bound and roughly linear in this number.
# Dial down to ~30 if p95 needs it.
# How deep the cross-encoder reranks. Measured 2026-08-02 against a 60-query
# known-item eval set on the live Space:
#
#   depth   recall@10   MRR@10   rank-1   rerank latency
#      50       66.7%    0.352    20.0%        2,279 ms
#      25       73.3%    0.401    25.0%          993 ms
#      10       85.0%    0.476    30.0%          ~400 ms
#
# Quality improves MONOTONICALLY as the window narrows, and it is not noise:
# comparing 50 vs 25 on targets found in both runs, 14 moved up and 0 moved down
# (sign test p = 0.000122).
#
# The reason is that first-stage retrieval already reaches 99.7% recall@60, so
# candidates below ~10 are near-misses. Letting an imperfect cross-encoder reach
# into them only creates chances to promote a wrong paper over the right one.
# Note the code drops the un-scored tail, so this ALSO caps how many results can
# be returned -- values below ARXIV_MAX_RESULTS would shrink the result page.
SEARCH_RERANK_TOP_N = int(os.getenv("SEARCH_RERANK_TOP_N", "10"))

# Whether to run the cross-encoder at all.
#
# It costs ~58% of end-to-end latency, and the depth sweep above shows its
# influence is negatively correlated with quality. This flag exists so that
# "is it earning its cost?" is an experiment rather than an argument: set it to
# 0 and results become the fusion ordering, with the title boost still applied.
SEARCH_BGE_RERANK = os.getenv("SEARCH_BGE_RERANK", "1").strip().lower() in (
    "1", "true", "yes")

# How many candidates the binary-quantized index proposes per requested result
# before rescoring against the full vectors. See qdrant_svc._SEARCH_PARAMS for
# the measurements. Env-tunable because the right value depends on where the
# HNSW graph lives: free on a RAM-resident graph, not free on an on-disk one.
# Set SEARCH_OVERSAMPLING=1.0 to restore the previous behaviour without a deploy.
SEARCH_OVERSAMPLING = float(os.getenv("SEARCH_OVERSAMPLING", "8.0"))

# ── Hybrid search tuning — Phase 3 ───────────────────────────────────────────
SEARCH_RRF_K = 60                  # RRF denominator
SEARCH_FETCH_K_MULTIPLIER = 6     # candidates = top_k × 6 before rerank
SEARCH_SEMANTIC_WEIGHT = 0.80     # RRF contribution to final score
SEARCH_RECENCY_WEIGHT = 0.20      # recency contribution to final score

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE = "ResearchIT"
COOKIE_NAME = "arxiv_user_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year
APP_PORT = int(os.getenv("PORT", "7860"))  # HF Spaces requires 7860

# ── Phase 5: Onboarding category taxonomy ─────────────────────────────────────
# Each group maps a user-friendly label to real arXiv primary_topic codes.
# Used by the onboarding wizard AND as pool filters / LightGBM features later.
CATEGORY_GROUPS: dict[str, dict] = {
    "nlp": {
        "name": "Natural Language Processing",
        "icon": "💬",
        "arxiv": ["cs.CL", "cs.IR"],
        "desc": "Language models, text generation, information retrieval",
    },
    "cv": {
        "name": "Computer Vision",
        "icon": "👁️",
        "arxiv": ["cs.CV"],
        "desc": "Image recognition, object detection, video understanding",
    },
    "ml": {
        "name": "Machine Learning",
        "icon": "🧠",
        "arxiv": ["cs.LG", "stat.ML"],
        "desc": "Learning theory, optimization, generalization",
    },
    "ai": {
        "name": "Artificial Intelligence",
        "icon": "🤖",
        "arxiv": ["cs.AI"],
        "desc": "Reasoning, planning, knowledge representation",
    },
    "robotics": {
        "name": "Robotics",
        "icon": "🦾",
        "arxiv": ["cs.RO"],
        "desc": "Control, manipulation, autonomous systems",
    },
    "hep": {
        "name": "High Energy Physics",
        "icon": "⚛️",
        "arxiv": ["hep-ph", "hep-th", "hep-ex", "hep-lat"],
        "desc": "Particle physics, quantum field theory, colliders",
    },
    "astro": {
        "name": "Astrophysics",
        "icon": "🔭",
        "arxiv": ["astro-ph.GA", "astro-ph.CO", "astro-ph.SR", "astro-ph.HE"],
        "desc": "Galaxies, cosmology, stellar physics",
    },
    "quant_ph": {
        "name": "Quantum Computing",
        "icon": "💠",
        "arxiv": ["quant-ph"],
        "desc": "Quantum algorithms, error correction, quantum info",
    },
    "math": {
        "name": "Mathematics",
        "icon": "📐",
        "arxiv": ["math.CO", "math.AG", "math.NT", "math.PR", "math.AP"],
        "desc": "Pure and applied mathematics",
    },
    "bio": {
        "name": "Computational Biology",
        "icon": "🧬",
        "arxiv": ["q-bio.BM", "q-bio.GN", "q-bio.QM"],
        "desc": "Bioinformatics, genomics, protein structure",
    },
    "neuro": {
        "name": "Neuroscience",
        "icon": "🧪",
        "arxiv": ["q-bio.NC"],
        "desc": "Computational neuroscience, brain modeling",
    },
    "econ": {
        "name": "Economics & Game Theory",
        "icon": "📊",
        "arxiv": ["econ.TH", "cs.GT"],
        "desc": "Mechanism design, auctions, market models",
    },
    "crypto": {
        "name": "Cryptography & Security",
        "icon": "🔐",
        "arxiv": ["cs.CR"],
        "desc": "Encryption, protocols, privacy",
    },
    "systems": {
        "name": "Systems & Networking",
        "icon": "🌐",
        "arxiv": ["cs.DC", "cs.NI"],
        "desc": "Distributed systems, networks, cloud",
    },
    "hci": {
        "name": "Human-Computer Interaction",
        "icon": "🖱️",
        "arxiv": ["cs.HC"],
        "desc": "Interface design, accessibility, user studies",
    },
    "audio": {
        "name": "Speech & Audio",
        "icon": "🎵",
        "arxiv": ["cs.SD", "eess.AS"],
        "desc": "Speech recognition, audio generation, music AI",
    },
    "pde": {
        "name": "Physics — General",
        "icon": "🌊",
        "arxiv": ["physics.flu-dyn", "physics.comp-ph", "physics.optics"],
        "desc": "Fluid dynamics, computational physics, optics",
    },
    "cond_mat": {
        "name": "Condensed Matter",
        "icon": "🧊",
        "arxiv": ["cond-mat.mes-hall", "cond-mat.mtrl-sci", "cond-mat.str-el"],
        "desc": "Materials science, superconductivity",
    },
    "se": {
        "name": "Software Engineering",
        "icon": "💻",
        "arxiv": ["cs.SE", "cs.PL"],
        "desc": "Testing, verification, compilers",
    },
    "acl_acl": {
        "name": "Signal Processing",
        "icon": "📡",
        "arxiv": ["eess.SP", "eess.IV"],
        "desc": "Image & signal processing, medical imaging",
    },
}


def expand_category_groups(group_keys: list[str]) -> set[str]:
    """Convert a list of group keys (e.g. ['nlp', 'cv']) into a flat set of arXiv categories."""
    cats: set[str] = set()
    for key in group_keys:
        grp = CATEGORY_GROUPS.get(key)
        if grp:
            cats.update(grp["arxiv"])
    return cats

# Optional bearer token for the researchit-space JSON API (app/routers/space.py).
# Unset means the router is open, matching the rest of the app today.
SPACE_SERVICE_TOKEN = os.getenv("SPACE_SERVICE_TOKEN", "").strip()
