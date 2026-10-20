"""Search quality, stage 1: retrieval for every query model, plus production's hybrid fusion.

    python search_eval.py qenc <model>   # encode the queries (run in the model's venv)
    python search_eval.py fts            # word-match lists (venv-main, needs the sidecar)
    python search_eval.py eval           # metrics + candidate lists for the reranker stage

Queries: LitSearch (real literature-search questions, gold = cited paper) and
LLM-written keyword / question queries (gold = source paper). Retrieval runs over
the 98k-paper library; full-corpus ranks for the dense arm are estimated from the
30k random corpus papers inside it.

Systems per query model m:
  dense:m    cosine top-k of the query vector
  hybrid:m   production fusion: RRF(dense top 60, word top 60, k=60), then
             title + citation boost over the top 50 (hybrid_search_svc), top 10 shown
fts          word match alone (SQLite FTS5, production query builder)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, SEED, log  # noqa: E402
from encode import MODELS  # noqa: E402

S = CACHE / "search"
STOPWORDS = set("""a an the of for and or in on to with by is are was were be been being what which who whom how why
when where do does did there any some that this these those it its as at from into about than such can could would
should may might will their they them we our you your i me my papers paper studies study research works work
literature exist exists existing available explore explores exploring looking find finding examples example
show shows give gives use used using approach approaches method methods technique techniques""".split())
FETCH_K, RRF_K, BOOST_N = 60, 60, 50
N_CORPUS = 1_853_777


def queries() -> list[dict]:
    L = json.loads((S / "litsearch_queries.json").read_text())
    for q in L:
        q["set"] = "LitSearch " + ("specific" if q["specificity"] == 1 else "broad")
        q["group"] = "AI/ML (LitSearch)"
    return L + json.loads((S / "synthetic_queries.json").read_text())


def qenc(name: str) -> None:
    hf, _d, qpre, kw = MODELS[name]
    Q = queries()
    texts = [q["query"] for q in Q]
    dev = os.getenv("EMBEVAL_BASELINE_DEVICE", "mps")
    if name == "bge_m3":
        from FlagEmbedding import BGEM3FlagModel
        m = BGEM3FlagModel(hf, use_fp16=False, device=dev)
        V = np.asarray(m.encode(texts, batch_size=32, max_length=512)["dense_vecs"])
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(hf, device=dev, **kw)
        V = m.encode([qpre + t for t in texts], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    np.save(S / f"q_{name}.npy", V.astype(np.float32))
    log("search", f"queries encoded with {name}: {V.shape}")


def fts() -> None:
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    sys.path.insert(0, str(root))
    from app import fts_svc
    lib = set(json.loads((S / "library.json").read_text())["ids"])
    i, n = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0/1").split("/"))
    out = {}
    for k, q in enumerate(queries()):
        if k % n != i:
            continue
        text = q["query"]
        if os.getenv("FTS_STOP") == "1":       # variant: drop filler words before matching
            text = " ".join(w for w in re.findall(r"[A-Za-z0-9]+", text) if w.lower() not in STOPWORDS)
        rows = fts_svc._search_sync(text, 5000) if text.strip() else []
        out[q["qid"]] = [r["arxiv_id"] for r in rows if r["arxiv_id"] in lib][:200]
    tag = "fts_stop" if os.getenv("FTS_STOP") == "1" else "fts"
    (S / f"{tag}_lists_{i}of{n}.json").write_text(json.dumps(out))
    parts = sorted(S.glob(f"{tag}_lists_*of{n}.json"))
    if len(parts) == n:      # last shard to finish merges
        merged = {}
        for p_ in parts:
            merged.update(json.loads(p_.read_text()))
        (S / f"{tag}_lists.json").write_text(json.dumps(merged))
    log("search", f"fts shard {i}/{n}: {len(out)} queries")


def boost_order(fused: list[tuple[str, float]], query: str, titles, cites) -> list[str]:
    """Production's title + citation boost over the top BOOST_N (hybrid_search_svc._title_match_rerank)."""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.hybrid_search_svc import _compute_citation_boost, _compute_title_boost, _normalize_for_match
    qn = _normalize_for_match(query)
    if not fused:
        return []
    mx = max(s for _, s in fused)
    head = fused[:BOOST_N]
    scored = []
    for aid, s in fused:
        tb = _compute_title_boost(qn, titles.get(aid, "")) if qn and (aid, s) in head else 0.0
        cb = _compute_citation_boost(cites.get(aid, 0)) if (aid, s) in head else 0.0
        scored.append((aid, s + mx * (tb + cb)))
    scored.sort(key=lambda x: -x[1])
    return [a for a, _ in scored]


def rrf(lists: list[list[str]]) -> list[tuple[str, float]]:
    sc: dict[str, float] = {}
    for L in lists:
        for r, a in enumerate(L, start=1):
            sc[a] = sc.get(a, 0.0) + 1.0 / (RRF_K + r)
    return sorted(sc.items(), key=lambda x: -x[1])


def metrics_for(ranked: list[str], gold: list[str]) -> dict:
    pos = {a: k for k, a in enumerate(ranked)}
    ranks = [pos[g] + 1 for g in gold if g in pos]
    first = min(ranks) if ranks else None
    return {"rr10": (1 / first) if first and first <= 10 else 0.0,
            "r1": float(bool(first and first == 1)),
            "r10": np.mean([g in pos and pos[g] < 10 for g in gold]),
            "r50": np.mean([g in pos and pos[g] < 50 for g in gold]),
            "r100": np.mean([g in pos and pos[g] < 100 for g in gold])}


def evaluate() -> None:
    import sqlite3
    from scipy.stats import beta
    from common import SIDECAR
    lib = json.loads((S / "library.json").read_text())
    ids = lib["ids"]
    pos = {a: k for k, a in enumerate(ids)}
    titles = lib["titles"]
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    cites = {}
    for s in range(0, len(ids), 900):
        part = ids[s:s + 900]
        for a, c in sc.execute(f"SELECT arxiv_id, citation_count FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})", part):
            cites[a] = c or 0
    Q = queries()
    F = json.loads((S / os.getenv("FTS_FILE", "fts_lists.json")).read_text())
    bg = np.array([pos[a] for a in lib["background"] if a in pos])
    per: dict[str, dict[str, list]] = {}
    cand: dict[str, dict[str, list]] = {}
    models = [m for m in MODELS if (S / f"vec_{m}.npy").exists() and (S / f"q_{m}.npy").exists()]
    log("search", f"models with library + query vectors: {models}")
    for q in Q:
        m_ = metrics_for(F.get(q["qid"], []), q["gold"])
        for k, v in m_.items():
            per.setdefault("fts", {}).setdefault(k, []).append(v)
    for m in models:
        mids = json.loads((S / f"ids_{m}.json").read_text())
        D = np.load(S / f"vec_{m}.npy").astype(np.float32)
        if mids != ids:      # align to the library order; missing papers score -inf
            full = np.zeros((len(ids), D.shape[1]), np.float32)
            have = np.zeros(len(ids), bool)
            idx = [pos[a] for a in mids]
            full[idx] = D
            have[idx] = True
            D = full
        else:
            have = np.ones(len(ids), bool)
        QV = np.load(S / f"q_{m}.npy")
        cand[m] = {}
        for s0 in range(0, len(Q), 64):
            S_ = QV[s0:s0 + 64] @ D.T
            S_[:, ~have] = -9
            top = np.argpartition(-S_, 300, axis=1)[:, :300]
            for j, q in enumerate(Q[s0:s0 + 64]):
                row = S_[j]
                t = top[j][np.argsort(-row[top[j]])]
                dense = [ids[x] for x in t]
                for k, v in metrics_for(dense, q["gold"]).items():
                    per.setdefault(f"dense:{m}", {}).setdefault(k, []).append(v)
                # full-corpus estimate for the dense arm (gold vs the random background)
                sb = np.sort(row[bg])
                pr10, pr100 = [], []
                for g in q["gold"]:
                    if g not in pos:
                        continue
                    b = len(sb) - np.searchsorted(sb, row[pos[g]], side="right")
                    for K, acc in ((10, pr10), (100, pr100)):
                        acc.append(float(beta.cdf((K - 1) / N_CORPUS, b + 0.5, len(sb) - b + 0.5)))
                per.setdefault(f"dense:{m}", {}).setdefault("corpus_r10", []).append(np.mean(pr10))
                per.setdefault(f"dense:{m}", {}).setdefault("corpus_r100", []).append(np.mean(pr100))
                fused = rrf([dense[:FETCH_K], F.get(q["qid"], [])[:FETCH_K]])
                shown = boost_order(fused, q["query"], titles, cites)
                for k, v in metrics_for(shown, q["gold"]).items():
                    per.setdefault(f"hybrid:{m}", {}).setdefault(k, []).append(v)
                rr = [a for a, _ in fused]
                for K in (10, 30, 50):
                    per.setdefault(f"hybrid:{m}", {}).setdefault(f"pool{K}", []).append(
                        np.mean([g in set(rr[:K]) for g in q["gold"]]))
                cand[m][q["qid"]] = rr[:50]
                cand.setdefault(f"dense:{m}", {})[q["qid"]] = dense[:50]
        log("search", f"{m}: hybrid MRR@10 {np.mean(per[f'hybrid:{m}']['rr10']):.3f}  dense {np.mean(per[f'dense:{m}']['rr10']):.3f}")
    # summaries with paired intervals vs current production arms
    rng = np.random.default_rng(SEED)

    def boot(x):
        x = np.asarray(x, float)
        mm = x[rng.integers(0, len(x), (3000, len(x)))].mean(axis=1)
        return [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]

    sets = sorted({q["set"] for q in Q})
    res = {"n_queries": len(Q), "sets": {s: sum(q["set"] == s for q in Q) for s in sets}, "systems": {}}
    for sysname, mets in per.items():
        base = per.get("hybrid:bge_m3" if sysname.startswith("hybrid") or sysname == "fts" else "dense:bge_m3", {})
        row = {}
        for k, v in mets.items():
            v = np.asarray(v, float)
            entry = {"mean": float(v.mean()), "ci": boot(v)}
            if k in base:
                d = v - np.asarray(base[k], float)
                entry["d_bge"] = float(d.mean())
                entry["d_bge_ci"] = boot(d)
            entry["by_set"] = {s: float(np.mean([x for x, q in zip(v, Q) if q["set"] == s])) for s in sets}
            row[k] = entry
        res["systems"][sysname] = row
    res["per_query"] = {s: {k: [float(x) for x in v] for k, v in m.items() if k in ("rr10", "r10")} for s, m in per.items()}
    res["qids"] = [q["qid"] for q in Q]
    sfx = os.getenv("OUT_SUFFIX", "")
    (S / f"stage1{sfx}.json").write_text(json.dumps(res))
    (S / f"candidates{sfx}.json").write_text(json.dumps(cand))
    for sysname in sorted(res["systems"], key=lambda s: -res["systems"][s]["rr10"]["mean"]):
        r = res["systems"][sysname]
        log("search", f"{sysname:22} MRR@10 {r['rr10']['mean']:.3f}  R@10 {r['r10']['mean']:.3f}  R@100 {r['r100']['mean']:.3f}"
                      + (f"  corpusR@100 {r['corpus_r100']['mean']:.3f}" if "corpus_r100" in r else ""))


if __name__ == "__main__":
    {"qenc": lambda: qenc(sys.argv[2]), "fts": fts, "eval": evaluate}[sys.argv[1]]()
