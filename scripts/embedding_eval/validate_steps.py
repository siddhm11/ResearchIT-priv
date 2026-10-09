"""Validate plan steps 2 and 3 before (and after) implementing them.

  step 2  rescale the long/short-term similarities across the candidate set
          before weighting them in heuristic_score (as the cold-start branch does)
  step 3  compute paper age from the arXiv ID (publication) instead of
          update_date (last revision)

V1  simulated readers (Test B replay, same seeds and RNG as scaled_eval.py)
    * hit@10 / MRR for: today, step 2, step 3, steps 2+3
    * split by the kind of right answer: papers the seed CITES (older) vs papers
      that CITE the seed (newer), so a gain from simply ignoring recency shows up
    * freshness cost: share of the top 10 published in the last 12 months
    * three candidate-pool sizes (30, 60, 100); production pools span this range
    * paired bootstrap intervals on every difference
V2  real readers (Scholar Inbox votes): liked vs disliked AUC with the full
    heuristic, recency measured at the moment of each vote
    * today: recency from update_date (what the code computes now)
    * step 3: recency from the arXiv ID
    * with and without step 2

MODE=reference (default) evaluates the variants by re-implementing them on the
feature matrix. MODE=installed scores "today" with the app's heuristic_score and
build path as currently installed, so after the code change it must reproduce
the reference "steps 2+3" numbers exactly.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEED, SIDECAR, app_import_path, log
from scaled_eval import primary, stored

app_import_path()
STAGE = "validate"
NOW = np.datetime64("2026-10-09")
MODE = os.getenv("VALIDATE_MODE", "reference")
# In installed mode: which reference variant the installed code must reproduce.
EXPECT = os.getenv("VALIDATE_EXPECT", "steps2+3")
POOLS = (30, 60, 100)


def paired(a: np.ndarray, b: np.ndarray, rng, n=4000) -> dict:
    d = b - a
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "n": int(len(d))}


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    from app import local_meta
    from app.recommend.profiles import (ALPHA_LONG_TERM, ALPHA_NEGATIVE, ALPHA_SHORT_TERM,
                                        ewma_update)
    from app.recommend import reranker
    from app.recommend.reranker import _unit_spread, heuristic_score

    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    rows = sc.execute("SELECT arxiv_id, categories, update_date FROM papers").fetchall()
    meta = {r[0]: {"cat": primary(r[1]), "upd": r[2]} for r in rows}

    def pub_date(aid):
        ym = local_meta.pub_year_month(aid)
        return np.datetime64(f"{ym[0]:04d}-{ym[1]:02d}-15") if ym else None

    def days(a, b):
        return float((a - b) / np.timedelta64(1, "D"))

    def rec_upd(aid, now):
        u = meta.get(aid, {}).get("upd")
        if not u:
            return np.exp(-0.002 * 365)
        return np.exp(-0.002 * max(0.0, days(now, np.datetime64(u[:10]))))

    def rec_pub(aid, now):
        p = pub_date(aid)
        if p is None:
            return rec_upd(aid, now)
        return np.exp(-0.002 * max(0.0, days(now, p)))

    def frozen_heuristic(F, rescale, rec_weight):
        """heuristic_score as it was on origin/main (b2aea4b), with the two knobs
        under test exposed. Frozen here so the reference never depends on the
        code being checked, even in installed mode."""
        lt, st, neg, q = F[:, 20], F[:, 21], F[:, 22], F[:, 0]
        if np.any(lt != 0):
            relevance = (0.40 * _unit_spread(lt) + 0.25 * _unit_spread(st)) if rescale else (0.40 * lt + 0.25 * st)
        else:
            relevance = 0.65 * _unit_spread(q)
        return relevance + rec_weight * F[:, 6] + 0.10 * F[:, 35] - 0.15 * np.clip(neg, 0.0, None)

    def variants(F, rec_u, rec_p):
        """Reference scores for today / step2 / step3 / steps 2+3 / option C."""
        out = {}
        for name, resc, rec, w in (("today", False, rec_u, 0.15), ("step2", True, rec_u, 0.15),
                                   ("step3", False, rec_p, 0.15), ("steps2+3", True, rec_p, 0.15),
                                   ("optionC", True, rec_p, 0.45)):
            G = F.copy()
            G[:, 6] = rec
            out[name] = frozen_heuristic(G, resc, w)
        return out

    res = {"mode": MODE, "reranker_file": reranker.__file__}

    # ── V1: simulated readers ────────────────────────────────────────────────
    # Candidate pools are cached by step2_tradeoff.py (same seeds, RNG and
    # search); the top-100 list is cached, so pools 30 and 60 are its prefixes.
    import pickle
    pools = pickle.loads((CACHE / "validate_pools.pkl").read_bytes())
    VARS = ("today", "step2", "step3", "steps2+3", "optionC")
    per = {pool: {v: [] for v in VARS} for pool in POOLS}
    by_kind = {k: {v: [] for v in VARS} for k in ("cites", "cited_by")}
    fresh = {v: [] for v in VARS + (("installed",) if MODE == "installed" else ())}
    agree_top10: list[float] = []
    users = len(pools)
    for p in pools:
        held = p["held"]
        for pool in POOLS:
            ids = p["ids"][:pool]
            target = np.array([i in held for i in ids])
            names = list(per[pool]) + (["installed"] if MODE == "installed" else [])
            if not target.any():
                for v in dict.fromkeys(names):
                    per[pool].setdefault(v, []).append((0.0, 0.0))
                continue
            E = p["E"][:pool].astype(np.float32)
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            F = np.zeros((len(ids), 37), np.float32)
            F[:, 0] = p["scores"][:pool]
            F[:, 1] = np.arange(len(ids))
            F[:, 20], F[:, 21] = E @ p["lt"], E @ p["st"]
            F[:, 35] = 1.0 / (F[:, 1] + 1)
            ru = np.array([rec_upd(i, NOW) for i in ids])
            rp = np.array([rec_pub(i, NOW) for i in ids])
            scores = variants(F, ru, rp)
            if MODE == "installed":
                # The app's own feature builder and scorer, as installed. Metadata
                # mirrors turso_svc._to_paper_dict: "published" is update_date.
                metas = [{"arxiv_id": i, "published": meta[i]["upd"] or ""} for i in ids]
                Fi = reranker.compute_features(E, metas, p["lt"], p["st"], None,
                                               qdrant_scores=np.asarray(p["scores"][:pool]))
                scores["installed"] = heuristic_score(Fi)
                ref = scores[EXPECT]
                agree_top10.append(float([ids[j] for j in np.argsort(-scores["installed"])[:10]]
                                         == [ids[j] for j in np.argsort(-ref)[:10]]))
            for v, sc_ in scores.items():
                order = np.argsort(-sc_)
                r = np.where(target[order])[0][0]
                per[pool].setdefault(v, []).append((1.0 / (r + 1), float(r < 10)))
                if pool == 100 and v in fresh:
                    top = [ids[j] for j in order[:10]]
                    fresh[v].append(np.mean([(pub_date(i) is not None and days(NOW, pub_date(i)) <= 365) for i in top]))
                    for kind, members in (("cites", p["refs"]), ("cited_by", p["cits"])):
                        kt = np.array([i in held and i in members for i in ids])
                        if kt.any():
                            rk = np.where(kt[order])[0][0]
                            by_kind[kind].setdefault(v, []).append(float(rk < 10))
    lengths = {pool: {v: len(x) for v, x in d.items()} for pool, d in per.items()}
    assert all(len(set(d.values())) == 1 for d in lengths.values()), lengths
    rng = np.random.default_rng(SEED)
    v1 = {"n_readers": users, "pools": {}, "by_kind_hit10": {}, "fresh_share_top10": {}}
    for pool in POOLS:
        A = {v: np.asarray(x) for v, x in per[pool].items()}
        v1["pools"][pool] = {v: {"MRR": float(a[:, 0].mean()), "hit10": float(a[:, 1].mean())} for v, a in A.items()}
        v1["pools"][pool]["paired_hit10_vs_today"] = {v: paired(A["today"][:, 1], A[v][:, 1], rng) for v in A if v != "today"}
        if "installed" in A:
            v1["pools"][pool][f"installed_minus_reference_{EXPECT}"] = {
                "hit10": float(A["installed"][:, 1].mean() - A[EXPECT][:, 1].mean()),
                "MRR": float(A["installed"][:, 0].mean() - A[EXPECT][:, 0].mean())}
        v1["pools"][pool]["paired_mrr_vs_today"] = {v: paired(A["today"][:, 0], A[v][:, 0], rng) for v in A if v != "today"}
    for kind, d in by_kind.items():
        A = {v: np.asarray(x) for v, x in d.items()}
        v1["by_kind_hit10"][kind] = {"n": int(len(A["today"])), **{v: float(a.mean()) for v, a in A.items()},
                                     "paired_steps2+3": paired(A["today"], A["steps2+3"], rng)}
    v1["fresh_share_top10"] = {v: float(np.mean(x)) for v, x in fresh.items() if x}
    if agree_top10:
        v1[f"installed_top10_identical_to_reference_{EXPECT}"] = float(np.mean(agree_top10))
    res["V1_simulated"] = v1
    log(STAGE, f"V1 pool100: {json.dumps(v1['pools'][100], default=float)[:600]}")
    log(STAGE, f"V1 by kind: {v1['by_kind_hit10']}")
    log(STAGE, f"V1 fresh: {v1['fresh_share_top10']}")

    if MODE == "installed":
        (OUT / f"validate_steps_installed_{EXPECT}.json").write_text(json.dumps(res, indent=1, default=float))
        log(STAGE, f"installed check written; agreement {v1.get(f'installed_top10_identical_to_reference_{EXPECT}')}")
        return

    # ── V2: real readers, full heuristic, recency at vote time ───────────────
    ratings = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    ratings["time"] = pd.to_datetime(ratings["time"], format="mixed")
    g = ratings.groupby("user_id")["rating"].agg(pos=lambda r: (r > 0).sum(), neg=lambda r: (r <= 0).sum())
    eligible = sorted(g[(g.pos >= 20) & (g.neg >= 10)].index)
    n_users = int(os.getenv("VALIDATE_REAL_USERS", "600"))
    users_r = list(np.random.default_rng(SEED).choice(eligible, n_users, replace=False))
    sub = ratings[ratings.user_id.isin(users_r)].sort_values(["user_id", "time"])
    vec = asyncio.run(stored(sorted(sub.arxiv_id.unique())))
    from sklearn.metrics import roc_auc_score
    auc = {v: [] for v in ("today", "step2", "step3", "steps2+3", "optionC")}
    for u in users_r:
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
        F[:, 20] = X @ (lt / np.linalg.norm(lt))
        F[:, 21] = X @ (st / np.linalg.norm(st))
        F[:, 22] = X @ (ng / np.linalg.norm(ng))
        F[:, 35] = 1.0     # no positions: identical for every item
        tnow = [np.datetime64(t.to_datetime64(), "D") for t in te.time]
        ru = np.array([rec_upd(i, t) for i, t in zip(te.arxiv_id, tnow)])
        rp = np.array([rec_pub(i, t) for i, t in zip(te.arxiv_id, tnow)])
        y = (te.rating > 0).astype(int).values
        for v, sc_ in variants(F, ru, rp).items():
            auc[v].append(roc_auc_score(y, sc_))
    A = {v: np.asarray(x) for v, x in auc.items()}
    res["V2_real"] = {"n_readers": int(len(A["today"])), **{v: float(a.mean()) for v, a in A.items()},
                      "paired_vs_today": {v: paired(A["today"], A[v], rng) for v in A if v != "today"}}
    log(STAGE, f"V2 {json.dumps(res['V2_real'], default=float)[:700]}")
    (OUT / f"validate_steps_{MODE}.json").write_text(json.dumps(res, indent=1, default=float))
    log(STAGE, f"wrote validate_steps_{MODE}.json")


if __name__ == "__main__":
    main()
