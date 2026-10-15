"""Stage 2b: SPECTER2 scores (allenai/specter2_base + allenai/specter2 proximity adapter).

Runs in its own venv: the `adapters` library needs transformers 4.57, while
production pins 4.44.2 for FlagEmbedding. Input follows the model card exactly:
title + tokenizer.sep_token + abstract, max_length=512, [CLS] of the last
hidden state. Similarity is cosine.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from common import VARIANTS, load_json, log, save_json

STAGE = "specter2"
BATCH = 16


def load():
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    model.eval()
    return tok, model


def text(tok, paper: dict, variant: str) -> str:
    t, a = str(paper["title"]).strip(), str(paper["abstract"] or "").strip()
    return {"title": t, "abstract": a, "title_abstract": t + tok.sep_token + a}[variant]


@torch.no_grad()
def embed(tok, model, texts: list[str], device: str) -> np.ndarray:
    out = []
    for s in range(0, len(texts), BATCH):
        inputs = tok(texts[s:s + BATCH], padding=True, truncation=True,
                     return_tensors="pt", return_token_type_ids=False, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        cls = model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
        out.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
    return np.concatenate(out)


def main() -> None:
    t0 = time.time()
    tok, model = load()
    log(STAGE, f"adapter active: {model.active_adapters}")

    ps = load_json("pairset.json")
    ids = list(ps["papers"])
    pair_out = {"ids": ids, "specter2": {}, "device": "cpu"}
    for variant in VARIANTS:
        e = embed(tok, model, [text(tok, ps["papers"][i], variant) for i in ids], "cpu")
        pair_out["specter2"][variant] = {
            f"{a}|{b}": float(e[x] @ e[y])
            for x, a in enumerate(ids) for y, b in enumerate(ids) if x != y}
    save_json("scores_pairset_specter2.json", pair_out)
    log(STAGE, "pair set done")

    if os.getenv("EMBEVAL_SKIP_BASELINE") == "1":
        return
    device = os.getenv("EMBEVAL_BASELINE_DEVICE", "cpu")
    model.to(device)
    bl = load_json("baseline.json")
    base_out = {"device": device, "specter2": {}}
    for set_name in ("random", "cscl_lg"):
        pairs = bl[set_name]
        for variant in VARIANTS:
            L = embed(tok, model, [text(tok, bl["texts"][l], variant) for l, _ in pairs], device)
            R = embed(tok, model, [text(tok, bl["texts"][r], variant) for _, r in pairs], device)
            base_out["specter2"][f"{set_name}/{variant}"] = [float(x) for x in (L * R).sum(1)]
            log(STAGE, f"baseline {set_name}/{variant} done")
    save_json("scores_baseline_specter2.json", base_out)
    log(STAGE, f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
