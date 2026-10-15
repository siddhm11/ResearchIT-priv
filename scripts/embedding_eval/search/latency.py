"""CPU latency on 2 threads (the Space's vCPUs): query encoding and cross-encoder scoring.

    python latency.py embed <model> [<model> ...]
    python latency.py ce <ce> [<ce> ...]

Measures median and p90 over real queries, batch of one query (embedders) and
10 / 30 / 50 query-document pairs (cross-encoders). Appends to CACHE/search/latency.json.
Production calibration point: BGE-M3 encode 258 ms, MiniLM-L6 x10 pairs 341 ms
(median over 14 live searches, 2026-10-11).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, log  # noqa: E402

S = CACHE / "search"


def _save(key: str, val: dict) -> None:
    p = S / "latency.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[key] = val
    p.write_text(json.dumps(cur, indent=1))


def main() -> None:
    import torch
    torch.set_num_threads(2)
    from search_eval import queries
    Q = [q["query"] for q in queries()][::20][:40]
    lib = json.loads((S / "library.json").read_text())
    docs = [f"Title: {lib['titles'][a]}\nAbstract: {lib['abstracts'][a]}" for a in lib["ids"][:50]]
    kind, names = sys.argv[1], sys.argv[2:]
    for name in names:
        if kind == "embed":
            from encode import MODELS
            hf, _d, qpre, kw = MODELS[name]
            if name == "bge_m3":
                from FlagEmbedding import BGEM3FlagModel
                m = BGEM3FlagModel(hf, use_fp16=False, device="cpu")
                f = lambda t: m.encode([t], batch_size=1, max_length=512, return_sparse=True)  # noqa: E731  production encodes dense + sparse
            else:
                from sentence_transformers import SentenceTransformer
                m = SentenceTransformer(hf, device="cpu", **kw)
                f = lambda t: m.encode([qpre + t], batch_size=1)  # noqa: E731
            for t in Q[:3]:
                f(t)
            ts = []
            for t in Q:
                t0 = time.perf_counter()
                f(t)
                ts.append((time.perf_counter() - t0) * 1000)
            val = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}
        else:
            from rerank import CES
            from sentence_transformers import CrossEncoder
            base, _, L = name.partition("_L")
            hf, kw = CES[base]
            m = CrossEncoder(hf, max_length=int(L) if L else 512, device="cpu", **kw)
            m.predict([[Q[0], docs[0]]])
            val = {}
            for n in (10, 30, 50):
                ts = []
                for t in Q[:6]:
                    t0 = time.perf_counter()
                    m.predict([[t, d] for d in docs[:n]], batch_size=n)
                    ts.append((time.perf_counter() - t0) * 1000)
                val[f"{n}_pairs_ms"] = float(np.median(ts))
        _save(f"{kind}:{name}", val)
        log("latency", f"{kind}:{name} {val}")


if __name__ == "__main__":
    main()
