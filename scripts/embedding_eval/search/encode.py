"""Document vectors for the search library, one model at a time, reusing cached vectors.

    python encode.py <model>        # venv-main: bert-family models
    python encode.py <model>        # venv-specter: qwen3_06b, gte_modernbert, nomic_v15

Every model is run with its own documented document prefix. BGE-M3 uses the
production vectors (bake-off/AI-ML caches, Qdrant for anything missing).
Writes CACHE/search/vec_<model>.npy (float16, unit length) + ids_<model>.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, log  # noqa: E402

S = CACHE / "search"
DEV = os.getenv("EMBEVAL_BASELINE_DEVICE", "mps")

# hf id, document prefix, query prefix, extra kwargs
MODELS = {
    "minilm":         ("sentence-transformers/all-MiniLM-L6-v2", "", "", {}),
    "bge_small":      ("BAAI/bge-small-en-v1.5", "", "Represent this sentence for searching relevant passages: ", {}),
    "bge_base":       ("BAAI/bge-base-en-v1.5", "", "Represent this sentence for searching relevant passages: ", {}),
    "e5_small":       ("intfloat/e5-small-v2", "passage: ", "query: ", {}),
    "gte_small":      ("thenlper/gte-small", "", "", {}),
    "arctic_s":       ("Snowflake/snowflake-arctic-embed-s", "", "Represent this sentence for searching relevant passages: ", {}),
    "arctic_m":       ("Snowflake/snowflake-arctic-embed-m-v1.5", "", "Represent this sentence for searching relevant passages: ", {}),
    "gte_large":      ("thenlper/gte-large", "", "", {}),
    "nomic_v15":      ("nomic-ai/nomic-embed-text-v1.5", "search_document: ", "search_query: ", {"trust_remote_code": True}),
    "gte_modernbert": ("Alibaba-NLP/gte-modernbert-base", "", "", {}),
    "qwen3_06b":      ("Qwen/Qwen3-Embedding-0.6B", "", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", {}),
    "bge_m3":         ("BAAI/bge-m3", "", "", {}),
}
# Bake-off caches whose document text matches this recipe (same prefix, title + ". " + abstract).
REUSE = {"minilm", "gte_large", "nomic_v15", "gte_modernbert", "qwen3_06b", "bge_m3"}


def cached(name: str) -> dict[str, np.ndarray]:
    got = {}
    if name not in REUSE:
        return got
    for d in (CACHE / "bakeoff", CACHE / "aiml"):
        p = d / f"vec_{name}.npy"
        if p.exists():
            ids = json.loads((d / f"ids_{name}.json").read_text())
            got.update(zip(ids, np.load(p, mmap_mode="r")))
    return got


def main(name: str) -> None:
    hf, dpre, _q, kw = MODELS[name]
    lib = json.loads((S / "library.json").read_text())
    ids = lib["ids"]
    have = cached(name)
    got = {i: np.asarray(have[i], np.float32) for i in ids if i in have}
    todo = [i for i in ids if i not in got]
    log("search_enc", f"{name}: {len(got):,} reused, {len(todo):,} to encode")
    if todo and name == "bge_m3":
        from dotenv import load_dotenv
        load_dotenv("/Users/siddh/Projects/researchit/.env.local")
        root = Path(__file__).resolve().parents[3]
        os.chdir(root)
        sys.path.insert(0, str(root))
        from app import qdrant_svc

        async def run():
            for s in range(0, len(todo), 200):
                got.update(await qdrant_svc.get_paper_vectors(todo[s:s + 200]))
        asyncio.run(run())
    elif todo:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(hf, device=DEV, **kw)
        m.max_seq_length = 512
        texts = [dpre + lib["titles"][i] + ". " + lib["abstracts"][i] for i in todo]
        t0 = time.time()
        for s in range(0, len(texts), 4000):
            M = m.encode(texts[s:s + 4000], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
            got.update(zip(todo[s:s + 4000], M))
            log("search_enc", f"{name}: {s + len(M):,}/{len(texts):,} ({time.time() - t0:.0f}s)")
    out = [i for i in ids if i in got]
    M = np.stack([np.asarray(got[i], np.float32) for i in out])
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    np.save(S / f"vec_{name}.npy", M.astype(np.float16))
    (S / f"ids_{name}.json").write_text(json.dumps(out))
    log("search_enc", f"{name}: saved {len(out):,}/{len(ids):,}")


if __name__ == "__main__":
    main(sys.argv[1])
