"""Embedding bake-off: which model (or mix) is best at each step of the feed?

Models (all on the SAME papers, readers and metrics):
  bge_m3     BAAI/bge-m3, production stored vectors (Qdrant)
  specter2   allenai SPECTER2 proximity, vectors from the Semantic Scholar API
  scincl     malteos/scincl, citation-neighbourhood contrastive SciBERT (local)
  gte_large  thenlper/gte-large, Scholar Inbox's production model (local)
  minilm     sentence-transformers/all-MiniLM-L6-v2, cheap general baseline (local)

Tests
  A  related vs random / keyword-lookalike   (Test A triplets from scaled_eval)
  R  "related papers" retrieval               recall@10/50/100 of a seed's citation
                                              links inside a 30k-paper random
                                              background + all linked papers
  C  feed ranking on real readers             Scholar Inbox votes, time split,
                                              profile scoring (app's ewma_update)
  C4 per-interest scoring (plan step 4)       production Ward clusters on BGE-M3,
                                              each model scores by nearest medoid
  M  mixes                                     BGE-M3 + another model; weight picked
                                              on half the readers / seeds and
                                              reported on the other half
  F  freshness: does Semantic Scholar have SPECTER2 vectors for brand-new papers?
  $  cost: dimensions, storage for 1.85M papers, CPU encoding speed on 2 threads

Usage (from scripts/embedding_eval, cached env):
  python model_bakeoff.py prepare | fetch_bge | fetch_specter2 | encode <model> | evaluate
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEED, app_import_path, log

app_import_path()
STAGE = "bakeoff"
BK = CACHE / "bakeoff"
N_REAL = int(os.getenv("BAKEOFF_REAL_USERS", "600"))
N_BACKGROUND = 30000
LOCAL = {
    "scincl": {"hf": "malteos/scincl", "kind": "cls"},
    "gte_large": {"hf": "thenlper/gte-large", "kind": "st"},
    "minilm": {"hf": "sentence-transformers/all-MiniLM-L6-v2", "kind": "st"},
    # Fairness check: production BGE-M3 vectors were built from abstract[:1024]
    # characters; this re-encodes the SAME model from the full text the other
    # models read (production separator and max_length=512 kept).
    "bge_m3_full": {"hf": "BAAI/bge-m3", "kind": "bge"},
}
MODELS = ["bge_m3", "specter2", "scincl", "gte_large", "minilm"]
if os.getenv("BAKEOFF_WITH_BGE_FULL") == "1":
    MODELS.append("bge_m3_full")


# ── prepare ──────────────────────────────────────────────────────────────────

def prepare() -> None:
    from scaled_eval import build_examples
    b = build_examples()
    meta = b["meta"]
    test_a = b["need"]
    ratings = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    g = ratings.groupby("user_id")["rating"].agg(pos=lambda r: (r > 0).sum(), neg=lambda r: (r <= 0).sum())
    eligible = sorted(g[(g.pos >= 20) & (g.neg >= 10)].index)
    users = sorted(np.random.default_rng(SEED).choice(eligible, N_REAL, replace=False).tolist())
    real = sorted(set(ratings[ratings.user_id.isin(users)].arxiv_id) & set(meta))
    pool = sorted(set(meta) - set(test_a))
    background = sorted(random.Random(SEED).sample(pool, N_BACKGROUND))
    universe = sorted(set(test_a) | set(real) | set(background))
    from select_pairs import csv_rows   # original CSV text for snapshot papers
    rows = csv_rows(universe)
    titles = {i: r["title"].strip() for i, r in rows.items()}
    abstracts = {i: (r["abstract"] or "").strip() for i, r in rows.items()}
    rest = [i for i in universe if i not in rows]
    for s0 in range(0, len(rest), 500):
        part = rest[s0:s0 + 500]
        q = f"SELECT arxiv_id, title, abstract_preview FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})"
        for a, t, ab in b["sc"].execute(q, part):
            titles[a], abstracts[a] = (t or "").strip(), (ab or "").strip()
    BK.mkdir(parents=True, exist_ok=True)
    (BK / "universe.json").write_text(json.dumps({
        "ids": universe, "test_a": test_a, "real_papers": real, "background": background,
        "real_users": users, "seeds": b["seeds"], "links": b["links"], "ex": b["ex"],
        "titles": titles, "abstracts": abstracts}))
    log(STAGE, f"universe {len(universe):,} papers (A {len(test_a):,}, real {len(real):,}, "
               f"background {len(background):,}); {len(users)} readers")


def load_universe() -> dict:
    return json.loads((BK / "universe.json").read_text())


def save_vecs(name: str, ids: list[str], M: np.ndarray) -> None:
    np.save(BK / f"vec_{name}.npy", M.astype(np.float16))
    (BK / f"ids_{name}.json").write_text(json.dumps(ids))


def load_vecs(name: str) -> dict[str, np.ndarray]:
    ids = json.loads((BK / f"ids_{name}.json").read_text())
    M = np.load(BK / f"vec_{name}.npy").astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return dict(zip(ids, M))


# ── fetch production vectors ─────────────────────────────────────────────────

def fetch_bge() -> None:
    from app import qdrant_svc
    u = load_universe()["ids"]
    got: dict[str, np.ndarray] = {}

    async def run():
        for s in range(0, len(u), 200):
            got.update(await qdrant_svc.get_paper_vectors(u[s:s + 200]))
            if s % 10000 == 0:
                log(STAGE, f"bge {s:,}/{len(u):,}")
    asyncio.run(run())
    ids = [i for i in u if i in got]
    save_vecs("bge_m3", ids, np.stack([got[i] for i in ids]))
    log(STAGE, f"bge_m3 vectors: {len(ids):,}/{len(u):,}")


def _s2_db() -> sqlite3.Connection:
    db = sqlite3.connect(BK / "s2_specter2.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS e (id TEXT PRIMARY KEY, v BLOB)")
    return db


def fetch_specter2(ids: list[str] | None = None) -> dict[str, np.ndarray | None]:
    """SPECTER2 vectors from the Semantic Scholar batch API, cached in SQLite.
    A row with an empty blob records 'S2 has no vector for this paper'."""
    import urllib.request
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    db = _s2_db()
    if (CACHE / "s2_cache.json").exists() and not db.execute("SELECT 1 FROM e LIMIT 1").fetchone():
        old = json.loads((CACHE / "s2_cache.json").read_text())
        rows = []
        for k, r in old.items():
            f, aid = k.split("|", 1)
            if f == "embedding.specter_v2":
                v = (r or {}).get("embedding") or {}
                vec = v.get("vector")
                rows.append((aid, np.asarray(vec, np.float16).tobytes() if vec else b""))
        db.executemany("INSERT OR REPLACE INTO e VALUES (?,?)", rows)
        db.commit()
        log(STAGE, f"seeded SPECTER2 cache with {len(rows):,} rows")
    want = ids if ids is not None else load_universe()["ids"]
    have = {r[0] for r in db.execute("SELECT id FROM e")}
    todo = [i for i in want if i not in have]
    log(STAGE, f"SPECTER2: {len(todo):,} to fetch, {len(want) - len(todo):,} cached")
    key = os.environ.get("S2_API_KEY", "")
    for n, s in enumerate(range(0, len(todo), 100)):
        part = todo[s:s + 100]
        req = urllib.request.Request(
            "https://api.semanticscholar.org/graph/v1/paper/batch?fields=embedding.specter_v2",
            data=json.dumps({"ids": [f"ARXIV:{i}" for i in part]}).encode(),
            headers={"x-api-key": key, "Content-Type": "application/json"}, method="POST")
        for attempt in range(14):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    rows = json.loads(r.read())
                break
            except Exception:  # noqa: BLE001 -- S2 throttles with 429
                if attempt == 13:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
        out = []
        for aid, r in zip(part, rows):
            vec = ((r or {}).get("embedding") or {}).get("vector")
            out.append((aid, np.asarray(vec, np.float16).tobytes() if vec else b""))
        db.executemany("INSERT OR REPLACE INTO e VALUES (?,?)", out)
        db.commit()
        if n % 50 == 0:
            log(STAGE, f"SPECTER2 {s:,}/{len(todo):,}")
        time.sleep(1.5)
    res = {}
    for aid, blob in db.execute("SELECT id, v FROM e"):
        res[aid] = np.frombuffer(blob, np.float16).astype(np.float32) if blob else None
    return {i: res.get(i) for i in want}


def fetch_specter2_universe() -> None:
    got = fetch_specter2()
    ids = [i for i, v in got.items() if v is not None]
    save_vecs("specter2", ids, np.stack([got[i] for i in ids]))
    log(STAGE, f"specter2 vectors: {len(ids):,}/{len(got):,} (missing {len(got) - len(ids):,})")


# ── local encoders ───────────────────────────────────────────────────────────

def _texts(u: dict, kind: str, sep: str) -> tuple[list[str], list[str]]:
    ids = u["ids"]
    out = []
    for i in ids:
        out.append(u["titles"].get(i, "") + sep + u["abstracts"].get(i, ""))
    return ids, out


def encoder(name: str, device: str):
    cfg = LOCAL[name]
    if cfg["kind"] == "bge":
        from FlagEmbedding import BGEM3FlagModel
        m = BGEM3FlagModel(cfg["hf"], use_fp16=False, device=device)
        return (lambda xs: np.asarray(m.encode(xs, batch_size=16, max_length=512)["dense_vecs"])), "\n\n"
    if cfg["kind"] == "st":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(cfg["hf"], device=device)
        return (lambda xs: m.encode(xs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)), ". "
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    mod = AutoModel.from_pretrained(cfg["hf"]).to(device).eval()

    @torch.no_grad()
    def enc(xs):
        res = []
        for s in range(0, len(xs), 32):
            b = tok(xs[s:s + 32], padding=True, truncation=True, max_length=512, return_tensors="pt")
            b = {k: v.to(device) for k, v in b.items()}
            res.append(mod(**b).last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(res)
    return enc, tok.sep_token


def encode(name: str) -> None:
    u = load_universe()
    enc, sep = encoder(name, os.getenv("EMBEVAL_BASELINE_DEVICE", "mps"))
    ids, texts = _texts(u, LOCAL[name]["kind"], sep)
    t0 = time.time()
    parts = []
    for s in range(0, len(texts), 4000):
        parts.append(enc(texts[s:s + 4000]))
        log(STAGE, f"{name}: {s + len(parts[-1]):,}/{len(texts):,} ({time.time() - t0:.0f}s)")
    save_vecs(name, ids, np.concatenate(parts))


def cpu_speed(name: str) -> float:
    """Docs/second on CPU with 2 threads, the Space's vCPU count."""
    import torch
    torch.set_num_threads(2)
    u = load_universe()
    enc, sep = encoder(name, "cpu")
    _, texts = _texts(u, LOCAL[name]["kind"], sep)
    sample = texts[:64]
    enc(sample[:8])
    t0 = time.time()
    enc(sample)
    return 64 / (time.time() - t0)


# ── evaluation ───────────────────────────────────────────────────────────────

def boot(x: np.ndarray, rng, n=4000):
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def unit(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def evaluate() -> None:
    from sklearn.metrics import roc_auc_score
    from app.recommend.clustering import compute_clusters
    from app.recommend.profiles import (ALPHA_LONG_TERM, ALPHA_NEGATIVE, ALPHA_SHORT_TERM,
                                        ewma_update)
    u = load_universe()
    V = {m: load_vecs(m) for m in MODELS}
    common_ids = set.intersection(*(set(v) for v in V.values()))
    rng = np.random.default_rng(SEED)
    res: dict = {"models": MODELS, "coverage": {m: len(V[m]) for m in MODELS},
                 "universe": len(u["ids"]), "common": len(common_ids)}
    log(STAGE, f"coverage {res['coverage']} common {len(common_ids):,}")

    # Per-model random-pair statistics on the background set (for z-score mixing).
    bg = [i for i in u["background"] if i in common_ids]
    r = random.Random(SEED)
    pairs = [(r.choice(bg), r.choice(bg)) for _ in range(4000)]
    zstat = {}
    for m in MODELS:
        c = np.array([V[m][a] @ V[m][b] for a, b in pairs if a != b])
        zstat[m] = (float(c.mean()), float(c.std()))
    res["random_pair_stats"] = zstat

    def zc(m, a, b):
        mu, sd = zstat[m]
        return (V[m][a] @ V[m][b] - mu) / sd

    mixes = [("bge_m3", "specter2"), ("bge_m3", "scincl"), ("bge_m3", "gte_large"), ("specter2", "scincl")]

    def scorers():
        out = {m: (lambda a, b, m=m: zc(m, a, b)) for m in MODELS}
        for x, y in mixes:
            out[f"{x}+{y}"] = (lambda a, b, x=x, y=y: 0.5 * zc(x, a, b) + 0.5 * zc(y, a, b))
        return out
    S = scorers()

    # ── Test A ──
    ex, seeds = u["ex"], u["seeds"]
    A = {}
    for name, f in S.items():
        trip, lure, trapc = [], [], []
        for s in seeds:
            e = ex[s]
            if s not in common_ids:
                continue
            for l in e["linked"][:5]:
                for n in e["same_cat"]:
                    if l in common_ids and n in common_ids:
                        trip.append(f(s, l) > f(s, n))
            t = (e.get("trap") or {}).get("id")
            c = e.get("control")
            if t in common_ids and c in common_ids:
                trapc.append(f(s, t) > f(s, c))
                for l in e["linked"][:5]:
                    if l in common_ids:
                        lure.append(f(s, t) > f(s, l))
        A[name] = {"related_beats_random": float(np.mean(trip)), "related_ci": boot(np.asarray(trip, float), rng),
                   "lookalike_error": float(np.mean(lure)), "lookalike_ci": boot(np.asarray(lure, float), rng),
                   "trap_beats_control": float(np.mean(trapc)), "n_trip": len(trip), "n_lure": len(lure)}
    res["A"] = A
    log(STAGE, "A " + json.dumps({k: (round(v["related_beats_random"], 3), round(v["lookalike_error"], 3)) for k, v in A.items()}))

    # ── Test R: related-paper retrieval in a fixed universe ──
    R_ids = sorted(common_ids & (set(u["background"]) | set(u["test_a"])))
    idx = {a: k for k, a in enumerate(R_ids)}
    Ms = {m: np.stack([V[m][a] for a in R_ids]) for m in MODELS}
    links = u["links"]
    q_seeds = [s for s in seeds if s in idx and len([x for x in links[s]["refs"] + links[s]["cits"] if x in idx]) >= 3]
    ks = (10, 50, 100)

    def zs(m, s):
        mu, sd = zstat[m]
        return (Ms[m] @ V[m][s] - mu) / sd

    def recall(score, s):
        rel = {idx[x] for x in links[s]["refs"] + links[s]["cits"] if x in idx}
        score = score.copy()
        score[idx[s]] = -1e9
        order = np.argsort(-score)[:max(ks)]
        return [len(rel & set(order[:k].tolist())) / len(rel) for k in ks]

    retr = {}
    names = list(MODELS) + [f"{x}+{y}" for x, y in mixes] + [f"rrf:{x}+{y}" for x, y in mixes]
    per_seed = {n: [] for n in names}
    for s in q_seeds:
        z = {m: zs(m, s) for m in MODELS}
        for m in MODELS:
            per_seed[m].append(recall(z[m], s))
        for x, y in mixes:
            per_seed[f"{x}+{y}"].append(recall(0.5 * z[x] + 0.5 * z[y], s))
            rx = np.empty(len(R_ids)); rx[np.argsort(-z[x])] = np.arange(len(R_ids))
            ry = np.empty(len(R_ids)); ry[np.argsort(-z[y])] = np.arange(len(R_ids))
            per_seed[f"rrf:{x}+{y}"].append(recall(1 / (60 + rx + 1) + 1 / (60 + ry + 1), s))
    for n, rows in per_seed.items():
        a = np.asarray(rows)
        retr[n] = {f"recall@{k}": float(a[:, j].mean()) for j, k in enumerate(ks)}
        retr[n]["recall@100_ci"] = boot(a[:, 2], rng)
        retr[n]["paired_recall@100_vs_bge"] = float((a[:, 2] - np.asarray(per_seed["bge_m3"])[:, 2]).mean())
        retr[n]["paired_recall@100_vs_bge_ci"] = boot(a[:, 2] - np.asarray(per_seed["bge_m3"])[:, 2], rng)
    res["R"] = {"n_seeds": len(q_seeds), "universe": len(R_ids), "by_scorer": retr}
    # Visual examples: the first three query seeds in seed order (not chosen by result).
    ex_r = []
    for s in q_seeds[:3]:
        rel = {x for x in links[s]["refs"] + links[s]["cits"] if x in idx}
        lists = {}
        for name, sc_ in (("bge_m3", zs("bge_m3", s)), ("specter2", zs("specter2", s)),
                          ("bge_m3+specter2", 0.5 * zs("bge_m3", s) + 0.5 * zs("specter2", s))):
            sc_ = sc_.copy()
            sc_[idx[s]] = -1e9
            top = [R_ids[j] for j in np.argsort(-sc_)[:6]]
            lists[name] = [{"id": t, "title": u["titles"].get(t, "")[:110], "linked": t in rel} for t in top]
        ex_r.append({"seed": s, "seed_title": u["titles"].get(s, ""), "n_linked_in_universe": len(rel), "lists": lists})
    res["R"]["examples"] = ex_r
    log(STAGE, "R " + json.dumps({k: round(v["recall@100"], 3) for k, v in retr.items()}))

    # Mix weight for bge+specter2 on retrieval: choose on odd seeds, report on even.
    def mix_recall(w, ss):
        return np.mean([recall(w * zs("bge_m3", s) + (1 - w) * zs("specter2", s), s)[2] for s in ss])
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    odd, even = q_seeds[1::2], q_seeds[0::2]
    tune = {w: float(mix_recall(w, odd)) for w in ws}
    wbest = max(tune, key=tune.get)
    res["R"]["weight_sweep_bge_share"] = {"tuned_on_odd": tune, "chosen": wbest,
                                          "held_out_even": {w: float(mix_recall(w, even)) for w in ws}}

    # ── Test C / C4: real readers ──
    ratings = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    ratings["time"] = pd.to_datetime(ratings["time"], format="mixed")
    sub = ratings[ratings.user_id.isin(set(u["real_users"]))].sort_values(["user_id", "time"])
    C = {n: [] for n in list(MODELS) + [f"{x}+{y}" for x, y in mixes]}
    C4 = {m: [] for m in MODELS}
    sweep = {w: [] for w in ws}
    used = []
    for usr in u["real_users"]:
        rr = sub[(sub.user_id == usr) & sub.arxiv_id.isin(common_ids)]
        cut = int(len(rr) * 0.7)
        tr, te = rr.iloc[:cut], rr.iloc[cut:]
        if tr.rating.gt(0).sum() < 5 or tr.rating.le(0).sum() < 3 or te.rating.nunique() < 2:
            continue
        used.append(usr)
        y = (te.rating > 0).astype(int).values
        prof = {}
        for m in MODELS:
            lt = st = ng = None
            npos = nneg = 0
            for aid, rating in zip(tr.arxiv_id, tr.rating):
                if rating > 0:
                    lt = ewma_update(lt, V[m][aid], ALPHA_LONG_TERM, count=npos, normalise=False)
                    st = ewma_update(st, V[m][aid], ALPHA_SHORT_TERM, count=npos, normalise=False)
                    npos += 1
                else:
                    ng = ewma_update(ng, V[m][aid], ALPHA_NEGATIVE, count=nneg, normalise=False)
                    nneg += 1
            X = np.stack([V[m][i] for i in te.arxiv_id])
            s = (0.40 * unit(X @ (lt / np.linalg.norm(lt))) + 0.25 * unit(X @ (st / np.linalg.norm(st)))
                 - 0.15 * np.clip(X @ (ng / np.linalg.norm(ng)), 0, None))
            prof[m] = s
            C[m].append(roc_auc_score(y, s))
        for x, yy in mixes:
            C[f"{x}+{yy}"].append(roc_auc_score(y, 0.5 * unit(prof[x]) + 0.5 * unit(prof[yy])))
        for w in ws:
            sweep[w].append(roc_auc_score(y, w * unit(prof["bge_m3"]) + (1 - w) * unit(prof["specter2"])))
        # C4: production Ward clusters on BGE-M3 saves, each model scores by nearest medoid.
        pos = list(tr.arxiv_id[tr.rating > 0])[::-1]          # most recent first, as the app passes
        clusters = compute_clusters(pos, np.stack([V["bge_m3"][i] for i in pos]))
        for m in MODELS:
            meds = np.stack([V[m][c.medoid_paper_id] for c in clusters])
            X = np.stack([V[m][i] for i in te.arxiv_id])
            C4[m].append(roc_auc_score(y, (X @ meds.T).max(axis=1)))
    base = np.asarray(C["bge_m3"])
    res["C"] = {"n_readers": len(used), "by_scorer": {}}
    for n, v in C.items():
        a = np.asarray(v)
        res["C"]["by_scorer"][n] = {"auc": float(a.mean()), "ci": boot(a, rng),
                                    "vs_bge": float((a - base).mean()), "vs_bge_ci": boot(a - base, rng),
                                    "readers_better": float((a > base).mean())}
    res["C4"] = {m: {"auc": float(np.mean(v)), "vs_profile_same_model": float(np.mean(np.asarray(v) - np.asarray(C[m]))),
                     "vs_profile_ci": boot(np.asarray(v) - np.asarray(C[m]), rng)} for m, v in C4.items()}
    half = np.arange(len(used)) % 2 == 1
    tune = {w: float(np.asarray(sweep[w])[half].mean()) for w in ws}
    wbest = max(tune, key=tune.get)
    res["C"]["weight_sweep_bge_share"] = {"tuned_on_odd": tune, "chosen": wbest,
                                          "held_out_even": {w: float(np.asarray(sweep[w])[~half].mean()) for w in ws}}
    log(STAGE, "C " + json.dumps({k: round(v["auc"], 4) for k, v in res["C"]["by_scorer"].items()}))
    log(STAGE, "C4 " + json.dumps({k: round(v["vs_profile_same_model"], 4) for k, v in res["C4"].items()}))

    # ── F: freshness of S2 SPECTER2 vectors ──
    sc = sqlite3.connect(f"file:{CACHE / 'metadata.sqlite'}?mode=ro", uri=True)
    fresh = {}
    for prefix in ("2609.", "2607.", "2604.", "2601.", "2509.", "2506.", "2406."):
        ids = [x[0] for x in sc.execute("SELECT arxiv_id FROM papers WHERE arxiv_id LIKE ? ORDER BY arxiv_id", (prefix + "%",))]
        ids = sorted(random.Random(SEED).sample(ids, min(200, len(ids))))
        got = fetch_specter2(ids)
        fresh["20" + prefix[:2] + "-" + prefix[2:4]] = {"n": len(ids), "has_vector": float(np.mean([v is not None for v in got.values()]))}
    res["F"] = fresh
    log(STAGE, f"F {fresh}")

    # ── $: cost ──
    dims = {m: int(len(next(iter(V[m].values())))) for m in MODELS}
    cost = {m: {"dims": d, "gb_float16_1.85M": 1.853777e6 * d * 2 / 1e9, "gb_int8_1.85M": 1.853777e6 * d / 1e9}
            for m, d in dims.items()}
    for m in LOCAL:
        if m in cost and m != "bge_m3_full":
            cost[m]["cpu_docs_per_s_2threads"] = cpu_speed(m)
    sp = BK / "specter2_cpu_speed.json"           # measured in venv-specter
    if sp.exists():
        cost["specter2"].update(json.loads(sp.read_text()))
    ag = BK / "specter2_api_agreement.json"
    if ag.exists():
        res["specter2_local_vs_api"] = json.loads(ag.read_text())
    res["cost"] = cost
    log(STAGE, f"cost {cost}")
    (OUT / "model_bakeoff.json").write_text(json.dumps(res, indent=1, default=float))
    log(STAGE, f"wrote {OUT / 'model_bakeoff.json'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prepare": prepare, "fetch_bge": fetch_bge, "fetch_specter2": fetch_specter2_universe,
     "evaluate": evaluate}.get(cmd, lambda: encode(sys.argv[2]))()
