"""Encode the bake-off universe with SPECTER2 locally (venv-specter: `adapters`).

Same model as encode_specter2.py (allenai/specter2_base + proximity adapter,
title + [SEP] + abstract, [CLS]). Local encoding uses the same text every other
bake-off model sees, and avoids the Semantic Scholar API's rate limit. The API
vectors already fetched are used only to check that the two agree.
"""
from __future__ import annotations

import json
import sqlite3
import time

import numpy as np

from common import CACHE, log
from encode_specter2 import embed, load

BK = CACHE / "bakeoff"


def main() -> None:
    u = json.loads((BK / "universe.json").read_text())
    ids = u["ids"]
    tok, model = load()
    model.to("mps")
    texts = [u["titles"].get(i, "") + tok.sep_token + u["abstracts"].get(i, "") for i in ids]
    t0 = time.time()
    parts = []
    for s in range(0, len(texts), 4000):
        parts.append(embed(tok, model, texts[s:s + 4000], "mps"))
        log("specter2_local", f"{s + len(parts[-1]):,}/{len(texts):,} ({time.time() - t0:.0f}s)")
    M = np.concatenate(parts)
    np.save(BK / "vec_specter2.npy", M.astype(np.float16))
    (BK / "ids_specter2.json").write_text(json.dumps(ids))
    # Agreement with the Semantic Scholar API vectors (their text may differ slightly).
    db = sqlite3.connect(BK / "s2_specter2.sqlite")
    pos = {a: k for k, a in enumerate(ids)}
    cos = []
    for aid, blob in db.execute("SELECT id, v FROM e"):
        if blob and aid in pos:
            api = np.frombuffer(blob, np.float16).astype(np.float32)
            loc = M[pos[aid]]
            cos.append(float(api @ loc / (np.linalg.norm(api) * np.linalg.norm(loc))))
    c = np.asarray(cos)
    res = {"n": int(len(c)), "mean": float(c.mean()), "p5": float(np.percentile(c, 5)),
           "p50": float(np.percentile(c, 50)), "share_ge_0.99": float((c >= 0.99).mean()),
           "share_ge_0.95": float((c >= 0.95).mean())}
    (BK / "specter2_api_agreement.json").write_text(json.dumps(res, indent=1))
    log("specter2_local", f"API agreement {res}")


if __name__ == "__main__":
    main()
