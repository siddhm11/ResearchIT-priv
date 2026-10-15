"""Shared paths, constants and text recipes for the embedding-trustworthiness eval.

Read-only against production: nothing in this package writes to Qdrant, Zilliz,
Turso or the sidecar. Large inputs (sidecar, original CSV, venvs, intermediate
encodings) live in CACHE, outside the repo; only the report and results.json
land in reports/embedding_eval/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "embedding_eval"
CACHE = Path(os.getenv("EMBEVAL_CACHE", Path.home() / ".cache" / "researchit-embedding-eval"))
WORK = CACHE / "work"
SIDECAR = CACHE / "metadata.sqlite"
CSV = CACHE / "csv" / "arxiv_comprehensive_papers.csv"

SEED = 42
N_BASELINE_PAIRS = 2000
VARIANTS = ("title", "abstract", "title_abstract")

# The focal pair and the named controls. Trap and unrelated are chosen from the
# corpus by select_pairs.py and appended to this set at run time.
A = "1706.03762"
B = "1905.09856"
NAMED = {
    A: "focal_A",
    B: "focal_B",
    "1409.0473": "strong_positive",
    "1508.04025": "strong_positive",
    "1810.04805": "strong_positive",
    "1406.2661": "same_field_diff_topic",
}

# ── Production text recipes (Step 0) ─────────────────────────────────────────
#
# ORIGINAL: siddhm11/bme-upload (Kaggle), encode_batch(), which produced every
#   vector in arxiv_dense_a / arxiv_dense_b and every sparse vector in Zilliz:
#       title = str(p["title"]).strip()[:256]
#       abstract = str(p.get("abstract", "")).strip()[:1024]
#       title + "\n\n" + abstract if abstract else title
#   Text is the raw arxiv_comprehensive_papers.csv cell (no whitespace collapse).
#
# INGEST: scripts/ingest_arxiv.py:516/525 (and the Kaggle backfill notebook that
#   filled arxiv_recent): f"{title[:256]} {abstract[:1024]}" on clean_text()-ed
#   fields (whitespace collapsed; ingest also NFKC-normalises).


def bge_text(title: str, abstract: str, variant: str, recipe: str = "original") -> str:
    """BGE-M3 input text. title_abstract reproduces production exactly; the
    single-field variants apply the same per-field truncation."""
    if recipe == "original":
        t, a = str(title).strip()[:256], str(abstract or "").strip()[:1024]
        joined = t + "\n\n" + a if a else t
    elif recipe == "ingest":
        t, a = str(title)[:256], str(abstract or "")[:1024]
        joined = f"{t} {a}"
    else:
        raise ValueError(recipe)
    return {"title": t, "abstract": a, "title_abstract": joined}[variant]


def plain_text(title: str, abstract: str, variant: str, sep: str = " ") -> str:
    """Untruncated text for the reference models; each truncates at its own
    tokenizer limit."""
    t, a = str(title).strip(), str(abstract or "").strip()
    return {"title": t, "abstract": a, "title_abstract": t + sep + a}[variant]


def load_json(name: str):
    with open(WORK / name) as fh:
        return json.load(fh)


def save_json(name: str, obj) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / name
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    tmp.replace(path)
    return path


def log(stage: str, msg: str) -> None:
    print(f"[embedding_eval:{stage}] {msg}", flush=True)


def app_import_path() -> None:
    """Make `app.*` and `scripts/*` importable (production code paths).

    app/config.py calls load_dotenv(".env.local") relative to the CWD, so the
    process must sit at the repo root or every credential silently reads empty.
    """
    os.chdir(ROOT)
    for p in (ROOT, ROOT / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
