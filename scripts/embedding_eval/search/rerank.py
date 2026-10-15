"""Search quality, stage 2: cross-encoder reranking of the hybrid candidates.

    python rerank.py score <ce>      # cross-encoder scores for every (query, candidate) pair needed
    python rerank.py eval            # metrics for each retriever x cross-encoder x depth

Replicates production: the cross-encoder scores the top-N merged candidates
(Title/Abstract format), scores go through a sigmoid, the window is kept and the
title + citation boost is applied on it, and the first 10 are shown. Production
today is hybrid:bge_m3 + ms-marco-MiniLM-L-6 at depth 10.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, SEED, SIDECAR, log  # noqa: E402

S = CACHE / "search"
CES = {
    "minilm_l6":   ("cross-encoder/ms-marco-MiniLM-L-6-v2", {}),
    "minilm_l12":  ("cross-encoder/ms-marco-MiniLM-L-12-v2", {}),
    "mxbai_xs":    ("mixedbread-ai/mxbai-rerank-xsmall-v1", {}),
    "mxbai_base":  ("mixedbread-ai/mxbai-rerank-base-v1", {}),
    "bge_rr_base": ("BAAI/bge-reranker-base", {}),
    "bge_rr_m3":   ("BAAI/bge-reranker-v2-m3", {}),
    "gte_rr_mb":   ("Alibaba-NLP/gte-reranker-modernbert-base", {}),
}
RETRIEVERS = os.getenv("RERANK_RETRIEVERS", "bge_m3,gte_large").split(",")
DEPTHS = tuple(int(x) for x in os.getenv("RERANK_DEPTHS", "10,20,30,50").split(","))


def score(name: str) -> None:
    """name may carry a length suffix, e.g. mxbai_xs_L256 (max input tokens), and a
    trailing _q8 for int8 dynamic quantization of the Linear layers (CPU only)."""
    from sentence_transformers import CrossEncoder
    # _sc: production text. The reranker in production reads the sidecar's
    # abstract_preview (cut at ~500 chars for most papers), not the full CSV
    # abstract the library holds, so this variant scores exactly what it sees.
    prod_text = name.endswith("_sc")
    q8 = name.removesuffix("_sc").endswith("_q8")
    base, _, L = name.removesuffix("_sc").removesuffix("_q8").partition("_L")
    hf, kw = CES[base]
    maxlen = int(L) if L else 512
    lib = json.loads((S / "library.json").read_text())
    cand = json.loads((S / "candidates.json").read_text())
    from search_eval import queries
    step = int(os.getenv("RERANK_QUERY_STEP", "1"))       # every step-th query (subset for heavy models)
    Q = {q["qid"]: q for k, q in enumerate(queries()) if k % step == 0}
    prev = json.loads((S / f"ce_{name}.json").read_text()) if (S / f"ce_{name}.json").exists() else {}
    abstracts = lib["abstracts"]
    if prod_text:
        sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
        ids_ = lib["ids"]
        abstracts = {}
        for s0 in range(0, len(ids_), 900):
            part = ids_[s0:s0 + 900]
            for a, ab in sc.execute(f"SELECT arxiv_id, abstract_preview FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})", part):
                abstracts[a] = (ab or "").replace("\n", " ")
    pairs, keys = [], []
    for qid, q in Q.items():
        docs = set()
        for r in RETRIEVERS:
            docs |= set(cand.get(r, {}).get(qid, [])[:max(DEPTHS)])
        for a in sorted(docs):
            if f"{qid}|{a}" in prev:
                continue
            keys.append(f"{qid}|{a}")
            pairs.append([q["query"], f"Title: {lib['titles'][a]}\nAbstract: {abstracts.get(a, '')}"])
    dev = os.getenv("RERANK_DEVICE", "cpu")
    if dev == "cpu":
        import torch
        torch.set_num_threads(int(os.getenv("RERANK_THREADS", "10")))
    m = CrossEncoder(hf, max_length=maxlen, device=dev, **kw)
    if q8:
        import torch
        if torch.backends.quantized.engine == "none":       # Apple silicon: qnnpack; the Space (x86) uses fbgemm
            torch.backends.quantized.engine = torch.backends.quantized.supported_engines[-1]
        torch.ao.quantization.quantize_dynamic(m.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    log("rerank", f"{name}: {len(pairs):,} pairs")
    out = dict(prev)          # keep pairs scored on earlier runs
    for s in range(0, len(pairs), 4000):
        sc = m.predict(pairs[s:s + 4000], batch_size=32, show_progress_bar=False)
        out.update(zip(keys[s:s + 4000], [float(x) for x in np.ravel(sc)]))
        log("rerank", f"{name}: {s + 4000:,}")
    (S / f"ce_{name}.json").write_text(json.dumps(out))


def evaluate() -> None:
    from search_eval import boost_order, metrics_for, queries
    lib = json.loads((S / "library.json").read_text())
    ids = lib["ids"]
    titles = lib["titles"]
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    cites = {}
    for s in range(0, len(ids), 900):
        part = ids[s:s + 900]
        for a, c in sc.execute(f"SELECT arxiv_id, citation_count FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})", part):
            cites[a] = c or 0
    cand = json.loads((S / "candidates.json").read_text())
    Q = queries()
    ces = {p.stem[3:]: json.loads(p.read_text()) for p in sorted(S.glob("ce_*.json"))}
    scored_q = {n: {k.split("|", 1)[0] for k in CE} for n, CE in ces.items()}
    # Some cross-encoders already return probabilities. Production applies a sigmoid to
    # logits, so map probabilities back to logits first, or the scale collapses.
    for n, CE in ces.items():
        v = list(CE.values())
        if v and min(v) >= 0.0 and max(v) <= 1.0:
            ces[n] = {k: math.log(max(p, 1e-6) / max(1 - p, 1e-6)) for k, p in CE.items()}
            log("rerank", f"{n}: probability outputs, converted to logits")
    per: dict[str, dict[str, list]] = {}
    qidx: dict[str, list[int]] = {}
    for r in RETRIEVERS:
        if r not in cand:
            continue
        for qi, q in enumerate(Q):
            pool = cand[r].get(q["qid"], [])
            for n, CE in ces.items():
                if q["qid"] not in scored_q[n]:
                    continue
                for N in DEPTHS:
                    win = pool[:N]
                    fused = [(a, 1 / (1 + math.exp(-CE.get(f"{q['qid']}|{a}", -20.0)))) for a in win]
                    fused.sort(key=lambda x: -x[1])
                    shown = boost_order(fused, q["query"], titles, cites)
                    for k, v in metrics_for(shown, q["gold"]).items():
                        per.setdefault(f"{r}+{n}@{N}", {}).setdefault(k, []).append(v)
                    if len(per[f"{r}+{n}@{N}"]["rr10"]) > len(qidx.get(f"{r}+{n}@{N}", [])):
                        qidx.setdefault(f"{r}+{n}@{N}", []).append(qi)
    rng = np.random.default_rng(SEED)
    base = per.get("bge_m3+minilm_l6@10")

    def boot(x):
        x = np.asarray(x, float)
        mm = x[rng.integers(0, len(x), (3000, len(x)))].mean(axis=1)
        return [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]

    sets = sorted({q["set"] for q in Q})
    res = {"systems": {}, "production": "bge_m3+minilm_l6@10"}
    for sysname, mets in per.items():
        row = {}
        for k, v in mets.items():
            v = np.asarray(v, float)
            qs = [Q[i] for i in qidx[sysname]]
            e = {"mean": float(v.mean()), "ci": boot(v), "n": len(v),
                 "by_set": {s: float(np.mean([x for x, q in zip(v, qs) if q["set"] == s] or [np.nan])) for s in sets}}
            if base:
                bpos = {i: j for j, i in enumerate(qidx["bge_m3+minilm_l6@10"])}
                b = np.asarray([base[k][bpos[i]] for i in qidx[sysname]], float)
                d = v - b
                e["d_prod"], e["d_prod_ci"] = float(d.mean()), boot(d)
                e["prod_same_queries"] = float(b.mean())
            row[k] = e
        res["systems"][sysname] = row
    res["per_query"] = {s: {k: [float(x) for x in v] for k, v in m.items() if k in ("rr10", "r10")} for s, m in per.items()}
    (S / "stage2.json").write_text(json.dumps(res))
    for sysname in sorted(res["systems"], key=lambda s: -res["systems"][s]["rr10"]["mean"])[:40]:
        r = res["systems"][sysname]
        log("rerank", f"{sysname:30} n={r['rr10']['n']:3} MRR@10 {r['rr10']['mean']:.3f} (prod same q {r['rr10'].get('prod_same_queries', 0):.3f})  R@10 {r['r10']['mean']:.3f}  Δprod {r['rr10'].get('d_prod', 0):+.3f}")


if __name__ == "__main__":
    {"score": lambda: score(sys.argv[2]), "eval": evaluate}[sys.argv[1]]()
