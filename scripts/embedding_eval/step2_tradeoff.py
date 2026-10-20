"""Step 2 trade-off: relevance gained vs freshness lost, across variants.

validate_steps.py showed that rescaling the similarity terms (step 2 as first
written) raises relevance on both citation labels and real readers' votes, but
cuts the share of last-12-month papers in a reader's top 10 from ~43% to ~10%.
This sweeps variants that keep relevance on a common scale while giving
recency a defined say, so the choice is made on a measured frontier.

Every variant includes step 3 (publication-date recency). The heuristic's
weights are untouched in code here: a recency weight w is emulated by scaling
the recency feature by w / 0.15 before calling the app's heuristic_score.

Outputs per variant:
  simulated readers (V1, pool 100): hit@10 overall / older answers / newer
      answers, MRR, share of top 10 published in the last 12 months
  real readers (V2): liked-vs-disliked AUC with recency at vote time
  reference: share of each reader's candidate pool from the last 12 months
"""
from __future__ import annotations

import asyncio
import json
import pickle
import random
import sqlite3

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEED, SIDECAR, app_import_path, log
from scaled_eval import N_SEED_CANDIDATES, N_SEEDS, arxiv_ids, primary, s2_batch, stored

app_import_path()
STAGE = "tradeoff"
NOW = np.datetime64("2026-10-09")
POOL_CACHE = CACHE / "validate_pools.pkl"
# name: (rescale similarities, recency treatment, recency weight)
VARIANTS = {
    "today": (False, "raw", 0.15),
    "step3 only": (False, "raw", 0.15),
    "step2 as written": (True, "raw", 0.15),
    "balanced (both rescaled)": (True, "unit", 0.15),
    "rescaled + recency 0.30": (True, "raw", 0.30),
    "rescaled + recency 0.45": (True, "raw", 0.45),
    "rescaled + recency 0.60": (True, "raw", 0.60),
    "balanced + recency 0.30": (True, "unit", 0.30),
    "balanced + recency 0.45": (True, "unit", 0.45),
}


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    from app import local_meta, qdrant_svc
    from app.recommend.profiles import (ALPHA_LONG_TERM, ALPHA_NEGATIVE, ALPHA_SHORT_TERM,
                                        ewma_update)
    from app.recommend.reranker import _unit_spread, heuristic_score
    from sklearn.metrics import roc_auc_score

    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    meta = {r[0]: {"cat": primary(r[1]), "upd": r[2]}
            for r in sc.execute("SELECT arxiv_id, categories, update_date FROM papers")}

    def pub_date(aid):
        ym = local_meta.pub_year_month(aid)
        return np.datetime64(f"{ym[0]:04d}-{ym[1]:02d}-15") if ym else None

    def days(a, b):
        return float((a - b) / np.timedelta64(1, "D"))

    def recency(aid, now, source):
        if source == "pub" and pub_date(aid) is not None:
            return np.exp(-0.002 * max(0.0, days(now, pub_date(aid))))
        u = meta.get(aid, {}).get("upd")
        return np.exp(-0.002 * max(0.0, days(now, np.datetime64(u[:10])))) if u else np.exp(-0.73)

    def score(F, rec_u, rec_p, name):
        resc, rtreat, w = VARIANTS[name]
        G = F.copy()
        rec = rec_u if name == "today" else rec_p
        if rtreat == "unit":
            rec = _unit_spread(rec)
        G[:, 6] = rec * (w / 0.15)
        if resc:
            G[:, 20], G[:, 21] = _unit_spread(F[:, 20]), _unit_spread(F[:, 21])
        return heuristic_score(G)

    # ── simulated readers: build or load cached candidate pools ──
    if POOL_CACHE.exists():
        pools = pickle.loads(POOL_CACHE.read_bytes())
    else:
        all_ids = sorted(meta)
        cand = random.Random(SEED).sample(all_ids, N_SEED_CANDIDATES)
        raw = s2_batch(cand, "references.externalIds,citations.externalIds")
        seeds, links = [], {}
        for a in cand:
            row = raw.get(a) or {}
            refs = [x for x in arxiv_ids(row.get("references")) if x in meta and x != a]
            cits = [x for x in arxiv_ids(row.get("citations")) if x in meta and x != a]
            if len(set(refs) | set(cits)) >= 6:
                seeds.append(a)
                links[a] = {"refs": sorted(set(refs)), "cits": sorted(set(cits))}
            if len(seeds) == N_SEEDS:
                break
        bge_seed = asyncio.run(stored(seeds))
        rrnd = random.Random(SEED)
        pools = []
        for s in seeds:
            linked = links[s]["refs"] + links[s]["cits"]
            vecs = asyncio.run(stored(linked)) if s in bge_seed else {}
            linked = [x for x in linked if x in vecs]
            if s not in bge_seed or len(linked) < 6:
                continue
            lk = sorted(linked)
            rrnd.shuffle(lk)
            saves, held = [s] + lk[:4], set(lk[4:])
            allv = {**vecs, s: bge_seed[s]}
            lt = st = None
            for k, a in enumerate(saves):
                lt = ewma_update(lt, allv[a], ALPHA_LONG_TERM, count=k, normalise=False)
                st = ewma_update(st, allv[a], ALPHA_SHORT_TERM, count=k, normalise=False)
            lt_n, st_n = lt / np.linalg.norm(lt), st / np.linalg.norm(st)
            hits = asyncio.run(qdrant_svc.search_by_vector_with_scores(lt_n.tolist(), 100, exclude_ids=set(saves)))
            if not ({h["arxiv_id"] for h in hits} & held):
                continue
            cv = asyncio.run(stored([h["arxiv_id"] for h in hits]))
            sub = [h for h in hits if h["arxiv_id"] in cv]
            pools.append({"seed": s, "held": held, "refs": set(links[s]["refs"]), "cits": set(links[s]["cits"]),
                          "ids": [h["arxiv_id"] for h in sub], "scores": [h["score"] for h in sub],
                          "E": np.stack([cv[h["arxiv_id"]] for h in sub]).astype(np.float16),
                          "lt": lt_n, "st": st_n})
        POOL_CACHE.write_bytes(pickle.dumps(pools))
    log(STAGE, f"simulated readers: {len(pools)}")

    rows = {n: {"hit": [], "mrr": [], "cites": [], "cited_by": [], "fresh": []} for n in VARIANTS}
    pool_fresh = []
    for p in pools:
        ids = p["ids"]
        E = p["E"].astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        F = np.zeros((len(ids), 37), np.float32)
        F[:, 0], F[:, 1] = p["scores"], np.arange(len(ids))
        F[:, 20], F[:, 21] = E @ p["lt"], E @ p["st"]
        F[:, 35] = 1.0 / (F[:, 1] + 1)
        ru = np.array([recency(i, NOW, "upd") for i in ids])
        rp = np.array([recency(i, NOW, "pub") for i in ids])
        is_fresh = np.array([(pub_date(i) is not None and days(NOW, pub_date(i)) <= 365) for i in ids])
        pool_fresh.append(is_fresh.mean())
        target = np.array([i in p["held"] for i in ids])
        for n in VARIANTS:
            order = np.argsort(-score(F, ru, rp, n))
            r = np.where(target[order])[0][0]
            rows[n]["hit"].append(float(r < 10))
            rows[n]["mrr"].append(1.0 / (r + 1))
            rows[n]["fresh"].append(float(is_fresh[order[:10]].mean()))
            for kind, members in (("cites", p["refs"]), ("cited_by", p["cits"])):
                kt = np.array([i in p["held"] and i in members for i in ids])
                if kt.any():
                    rows[n][kind].append(float(np.where(kt[order])[0][0] < 10))

    # ── real readers (same 600 sampled as validate_steps / model_bakeoff) ──
    ratings = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    ratings["time"] = pd.to_datetime(ratings["time"], format="mixed")
    g = ratings.groupby("user_id")["rating"].agg(pos=lambda r: (r > 0).sum(), neg=lambda r: (r <= 0).sum())
    eligible = sorted(g[(g.pos >= 20) & (g.neg >= 10)].index)
    users = list(np.random.default_rng(SEED).choice(eligible, 600, replace=False))
    sub = ratings[ratings.user_id.isin(users)].sort_values(["user_id", "time"])
    bk = CACHE / "bakeoff"
    ids_b = json.loads((bk / "ids_bge_m3.json").read_text())
    Mb = np.load(bk / "vec_bge_m3.npy").astype(np.float32)
    vec = dict(zip(ids_b, Mb))
    auc = {n: [] for n in VARIANTS}
    for u in users:
        r = sub[(sub.user_id == u) & sub.arxiv_id.isin(vec)]
        cut = int(len(r) * 0.7)
        tr, te = r.iloc[:cut], r.iloc[cut:]
        if tr.rating.gt(0).sum() < 5 or tr.rating.le(0).sum() < 3 or te.rating.nunique() < 2:
            continue
        lt = st = ng = None
        npos = nneg = 0
        for aid, rating in zip(tr.arxiv_id, tr.rating):
            if rating > 0:
                lt = ewma_update(lt, vec[aid], ALPHA_LONG_TERM, count=npos, normalise=False)
                st = ewma_update(st, vec[aid], ALPHA_SHORT_TERM, count=npos, normalise=False)
                npos += 1
            else:
                ng = ewma_update(ng, vec[aid], ALPHA_NEGATIVE, count=nneg, normalise=False)
                nneg += 1
        X = np.stack([vec[i] / np.linalg.norm(vec[i]) for i in te.arxiv_id])
        F = np.zeros((len(te), 37), np.float32)
        F[:, 20], F[:, 21], F[:, 22] = X @ (lt / np.linalg.norm(lt)), X @ (st / np.linalg.norm(st)), X @ (ng / np.linalg.norm(ng))
        F[:, 35] = 1.0
        tnow = [np.datetime64(t.to_datetime64(), "D") for t in te.time]
        ru = np.array([recency(i, t, "upd") for i, t in zip(te.arxiv_id, tnow)])
        rp = np.array([recency(i, t, "pub") for i, t in zip(te.arxiv_id, tnow)])
        y = (te.rating > 0).astype(int).values
        for n in VARIANTS:
            auc[n].append(roc_auc_score(y, score(F, ru, rp, n)))

    rng = np.random.default_rng(SEED)

    def ci(x):
        x = np.asarray(x)
        m = x[rng.integers(0, len(x), (4000, len(x)))].mean(axis=1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    base_hit, base_auc = np.asarray(rows["today"]["hit"]), np.asarray(auc["today"])
    out = {"n_simulated": len(pools), "n_real": len(auc["today"]),
           "pool_fresh_share": float(np.mean(pool_fresh)), "variants": {}}
    for n, v in rows.items():
        out["variants"][n] = {
            "spec": {"rescale_similarity": VARIANTS[n][0], "recency": VARIANTS[n][1], "recency_weight": VARIANTS[n][2],
                     "recency_date": "update_date" if n == "today" else "arXiv ID"},
            "hit10": float(np.mean(v["hit"])), "hit10_vs_today_ci": ci(np.asarray(v["hit"]) - base_hit),
            "MRR": float(np.mean(v["mrr"])),
            "hit10_older_answers": float(np.mean(v["cites"])), "hit10_newer_answers": float(np.mean(v["cited_by"])),
            "fresh_share_top10": float(np.mean(v["fresh"])),
            "real_auc": float(np.mean(auc[n])), "real_auc_vs_today_ci": ci(np.asarray(auc[n]) - base_auc)}
        log(STAGE, f"{n:28} hit10 {out['variants'][n]['hit10']:.3f}  newer {out['variants'][n]['hit10_newer_answers']:.3f}  "
                   f"fresh {out['variants'][n]['fresh_share_top10']:.3f}  realAUC {out['variants'][n]['real_auc']:.4f}")
    (OUT / "step2_tradeoff.json").write_text(json.dumps(out, indent=1, default=float))
    log(STAGE, f"pool fresh share {out['pool_fresh_share']:.3f}; wrote step2_tradeoff.json")


if __name__ == "__main__":
    main()
