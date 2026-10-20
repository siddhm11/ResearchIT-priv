"""Stage 2a: BGE-M3 (dense / sparse / ColBERT) and all-MiniLM-L6-v2 scores.

Runs in the production-pinned venv (FlagEmbedding 1.2.10, transformers 4.44.2).

BGE-M3 is loaded through app.embed_svc.get_model() -- the app's own singleton,
CPU float32 exactly as served -- and encoded with max_length=512, the value both
app/embed_svc.py and the original bme-upload notebook pass. ColBERT vectors are
requested in the same forward pass; that does not change dense or sparse output.

Test-pair texts are encoded on CPU (production device). The 8,000-paper random
baselines may use MPS for speed (EMBEVAL_BASELINE_DEVICE); stage output records
the device, and integrity_retrieval.py measures CPU-vs-MPS agreement.
"""
from __future__ import annotations

import os
import time

import numpy as np

from common import (VARIANTS, WORK, app_import_path, bge_text, load_json, log,
                    plain_text, save_json)

app_import_path()
os.environ.setdefault("BGE_M3_DEVICE", "cpu")

STAGE = "bge"
BATCH = 16


def bge_model(device: str):
    from app import config, embed_svc
    if device == config.BGE_M3_DEVICE:
        return embed_svc.get_model()                # the app's own singleton
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(config.BGE_M3_MODEL, use_fp16=False, device=device)


def bge_encode(model, texts: list[str]) -> dict:
    out = model.encode(texts, batch_size=BATCH, max_length=512,
                       return_dense=True, return_sparse=True, return_colbert_vecs=True)
    return {
        "dense": np.asarray(out["dense_vecs"], dtype=np.float32),
        "sparse": [{int(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]],
        "colbert": [np.asarray(c, dtype=np.float32) for c in out["colbert_vecs"]],
    }


def lexical(a: dict, b: dict) -> float:
    """FlagEmbedding's compute_lexical_matching_score: sum of w_a*w_b over shared tokens."""
    if len(a) > len(b):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))


def colbert(q: np.ndarray, p: np.ndarray) -> float:
    """FlagEmbedding's colbert_score: mean over query tokens of max dot product."""
    return float((q @ p.T).max(axis=1).sum() / q.shape[0])


def bge_pair_scores(L: dict, R: dict, i: int, j: int) -> dict:
    return {
        "dense": float(L["dense"][i] @ R["dense"][j]),
        "sparse": lexical(L["sparse"][i], R["sparse"][j]),
        "colbert": colbert(L["colbert"][i], R["colbert"][j]),
    }


def pair_texts(paper: dict, variant: str, family: str) -> str:
    if family == "bge":
        return bge_text(paper["title"], paper["abstract"], variant, paper["recipe"])
    return plain_text(paper["title"], paper["abstract"], variant, sep=" ")


def run_pairset(model_bge, model_mini) -> dict:
    ps = load_json("pairset.json")
    ids = list(ps["papers"])
    out = {"ids": ids, "bge": {}, "minilm": {}}
    for variant in VARIANTS:
        texts = [pair_texts(ps["papers"][i], variant, "bge") for i in ids]
        enc = bge_encode(model_bge, texts)
        np.save(WORK / f"pairset_bge_dense_{variant}.npy", enc["dense"])
        save_json(f"pairset_bge_sparse_{variant}.json",
                  {i: {str(k): v for k, v in s.items()} for i, s in zip(ids, enc["sparse"])})
        mini = model_mini.encode(
            [pair_texts(ps["papers"][i], variant, "plain") for i in ids],
            normalize_embeddings=True, convert_to_numpy=True)
        scores_b, scores_m = {}, {}
        for x, ax in enumerate(ids):
            for y, cy in enumerate(ids):
                if x == y:
                    continue
                key = f"{ax}|{cy}"
                scores_b[key] = bge_pair_scores(enc, enc, x, y)
                scores_m[key] = float(mini[x] @ mini[y])
        out["bge"][variant] = scores_b
        out["minilm"][variant] = scores_m
        log(STAGE, f"pair set encoded: {variant}")
    return out


def run_baseline(model_bge, model_mini, device: str) -> dict:
    bl = load_json("baseline.json")
    texts = bl["texts"]
    out = {"device": device, "bge": {}, "minilm": {}}
    for set_name in ("random", "cscl_lg"):
        pairs = bl[set_name]
        for variant in VARIANTS:
            t0 = time.time()
            rows_b, rows_m = [], []
            CH = 128
            for s in range(0, len(pairs), CH):
                chunk = pairs[s:s + CH]
                lt = [pair_texts(texts[l], variant, "bge") for l, _ in chunk]
                rt = [pair_texts(texts[r], variant, "bge") for _, r in chunk]
                L, R = bge_encode(model_bge, lt), bge_encode(model_bge, rt)
                rows_b += [bge_pair_scores(L, R, k, k) for k in range(len(chunk))]
                lm = model_mini.encode([pair_texts(texts[l], variant, "plain") for l, _ in chunk],
                                       normalize_embeddings=True, convert_to_numpy=True)
                rm = model_mini.encode([pair_texts(texts[r], variant, "plain") for _, r in chunk],
                                       normalize_embeddings=True, convert_to_numpy=True)
                rows_m += [float(a @ b) for a, b in zip(lm, rm)]
            out["bge"][f"{set_name}/{variant}"] = {
                k: [r[k] for r in rows_b] for k in ("dense", "sparse", "colbert")}
            out["minilm"][f"{set_name}/{variant}"] = rows_m
            log(STAGE, f"baseline {set_name}/{variant}: {len(pairs)} pairs "
                       f"in {time.time() - t0:.0f}s")
    return out


def main() -> None:
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    cpu_bge = bge_model("cpu")
    mini = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    log(STAGE, f"MiniLM max_seq_length={mini.max_seq_length}")
    save_json("scores_pairset_bge_minilm.json",
              {**run_pairset(cpu_bge, mini), "device": "cpu",
               "minilm_max_seq_length": mini.max_seq_length})

    if os.getenv("EMBEVAL_SKIP_BASELINE") == "1":
        return
    device = os.getenv("EMBEVAL_BASELINE_DEVICE", "cpu")
    base_bge = cpu_bge if device == "cpu" else bge_model(device)
    base_mini = mini if device == "cpu" else SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device)
    save_json("scores_baseline_bge_minilm.json", run_baseline(base_bge, base_mini, device))
    log(STAGE, f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
