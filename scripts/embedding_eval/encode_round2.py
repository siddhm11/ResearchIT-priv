"""Bake-off round 2: newer general-purpose embedding models (venv-specter).

Runs in the environment with transformers 4.57 (the production pin, 4.44, is too
old for Qwen3 and ModernBERT). Encodes a SUBSET of the round-1 universe so every
round-1 model can be re-scored on exactly the same papers:

  * all Test A papers (citation triplets, lookalike traps)
  * every paper rated by the first 300 of round 1's 600 sampled readers
  * 10,000 of round 1's 30,000 random background papers (seed 42)

Models
  qwen3_06b       Qwen/Qwen3-Embedding-0.6B (Apache-2.0, 2025). Documents are
                  encoded without an instruction, as its model card specifies.
  gte_modernbert  Alibaba-NLP/gte-modernbert-base (Apache-2.0)
  nomic_v15       nomic-ai/nomic-embed-text-v1.5, "search_document: " prefix
Not run: google/embeddinggemma-300m is gated and the HF token has no access.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

from common import CACHE, SEED, log

BK = CACHE / "bakeoff"
MODELS = {
    "qwen3_06b": {"hf": "Qwen/Qwen3-Embedding-0.6B", "prefix": "", "kw": {}},
    "gte_modernbert": {"hf": "Alibaba-NLP/gte-modernbert-base", "prefix": "", "kw": {}},
    "nomic_v15": {"hf": "nomic-ai/nomic-embed-text-v1.5", "prefix": "search_document: ",
                  "kw": {"trust_remote_code": True}},
}


def subset() -> list[str]:
    path = BK / "round2_ids.json"
    if path.exists():
        return json.loads(path.read_text())
    u = json.loads((BK / "universe.json").read_text())
    users = u["real_users"][:300]
    r = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    real = set(r[r.user_id.isin(users)].arxiv_id) & set(u["ids"])
    bg = sorted(random.Random(SEED).sample(u["background"], 10000))
    ids = sorted(set(u["test_a"]) | real | set(bg))
    path.write_text(json.dumps(ids))
    (BK / "round2_meta.json").write_text(json.dumps({"users": users, "background": bg}))
    log("round2", f"subset: {len(ids):,} papers ({len(real):,} rated by 300 readers, {len(bg):,} background)")
    return ids


def main(name: str) -> None:
    from sentence_transformers import SentenceTransformer
    cfg = MODELS[name]
    ids = subset()
    u = json.loads((BK / "universe.json").read_text())
    texts = [cfg["prefix"] + u["titles"].get(i, "") + ". " + u["abstracts"].get(i, "") for i in ids]
    m = SentenceTransformer(cfg["hf"], device=os.getenv("EMBEVAL_BASELINE_DEVICE", "mps"), **cfg["kw"])
    m.max_seq_length = 512          # same budget as the other models
    t0 = time.time()
    parts = []
    for s in range(0, len(texts), 4000):
        parts.append(m.encode(texts[s:s + 4000], batch_size=16, normalize_embeddings=True, convert_to_numpy=True))
        log("round2", f"{name}: {s + len(parts[-1]):,}/{len(texts):,} ({time.time() - t0:.0f}s)")
    np.save(BK / f"vec_{name}.npy", np.concatenate(parts).astype(np.float16))
    (BK / f"ids_{name}.json").write_text(json.dumps(ids))
    # CPU speed on 2 threads, the Space's vCPU count.
    import torch
    torch.set_num_threads(2)
    mc = SentenceTransformer(cfg["hf"], device="cpu", **cfg["kw"])
    mc.max_seq_length = 512
    mc.encode(texts[:8])
    t1 = time.time()
    mc.encode(texts[8:72], batch_size=16)
    speed = 64 / (time.time() - t1)
    dims = int(parts[0].shape[1])
    (BK / f"cost_{name}.json").write_text(json.dumps({"dims": dims, "cpu_docs_per_s_2threads": speed,
                                                       "gb_float16_1.85M": 1.853777e6 * dims * 2 / 1e9,
                                                       "gb_int8_1.85M": 1.853777e6 * dims / 1e9}))
    log("round2", f"{name}: done; {dims} dims, {speed:.1f} papers/s on 2 CPU threads")


if __name__ == "__main__":
    main(sys.argv[1])
