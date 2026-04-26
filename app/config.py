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
