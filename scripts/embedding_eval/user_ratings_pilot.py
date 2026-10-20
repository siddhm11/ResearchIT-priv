"""Stage 7 (optional): pilot on real users' ratings (Scholar Inbox dataset).

Question: given a reader's earlier up/down votes, which way of scoring separates
the papers they later liked from the ones they later disliked?

Data: github.com/avg-dev/scholar_inbox_datasets, rated_papers.csv
(774k explicit thumbs up/down by arXiv id, user, time). Licence CC BY-NC-ND 4.0:
used here for non-commercial evaluation only, kept in CACHE, never committed or
redistributed. Downvotes are on papers Scholar Inbox already chose to show, so
they are hard negatives: papers that look relevant but were not wanted.

Protocol (seed 42): 200 users with >= 20 upvotes and >= 10 downvotes. Each
user's ratings are split by time, first 70% to learn from, last 30% to test.
Test metric: per-user ROC AUC of liked vs disliked test papers, averaged.

Scorers
  ewma_profile   ResearchIT's profile score: 0.40*cos(lt) + 0.25*cos(st)
                 - 0.15*max(cos(neg), 0), profiles built with the app's own
                 ewma_update in time order (recency and position left out, since
                 they are not content signals).
  lt_only        cosine to the long-term profile alone.
  max_sim        best cosine to any single earlier upvote (a multi-interest proxy).
  logreg_bge     per-user logistic regression on BGE-M3 stored vectors:
                 upvotes vs downvotes + 2,000 random corpus papers (Scholar Inbox style).
  logreg_specter2  the same on SPECTER2 vectors from the Semantic Scholar API.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEED, WORK, app_import_path, log

app_import_path()
STAGE = "pilot"
RATINGS = CACHE / "scholar_inbox" / "rated_papers.csv"
N_USERS = 200
N_RANDOM_NEG = 2000


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from app.recommend.profiles import (ALPHA_LONG_TERM, ALPHA_NEGATIVE, ALPHA_SHORT_TERM,
                                        ewma_update)
    from scaled_eval import boot_ci, s2_batch, stored

    d = pd.read_csv(RATINGS, dtype={"arxiv_id": str, "user_id": str})
    d["time"] = pd.to_datetime(d["time"], format="mixed")
    g = d.groupby("user_id")["rating"].agg(pos=lambda r: (r > 0).sum(), neg=lambda r: (r <= 0).sum())
    eligible = sorted(g[(g.pos >= 20) & (g.neg >= 10)].index)
    users = list(np.random.default_rng(SEED).choice(eligible, N_USERS, replace=False))
    sub = d[d.user_id.isin(users)].sort_values(["user_id", "time"])
    papers = sorted(sub.arxiv_id.unique())
    log(STAGE, f"{len(eligible)} eligible users; sampled {len(users)}; {len(papers):,} rated papers")

    bge = asyncio.run(stored(papers))
    bl = json.loads((WORK / "baseline.json").read_text())
    rand_ids = sorted({x for p in bl["random"] for x in p})[:N_RANDOM_NEG]
    bge_rand = asyncio.run(stored(rand_ids))
    s2 = s2_batch(papers + rand_ids, "embedding.specter_v2")
    spec = {i: np.asarray(r["embedding"]["vector"], np.float32) for i, r in s2.items()
            if r and r.get("embedding") and r["embedding"].get("vector")}
    log(STAGE, f"coverage: BGE {len(bge)}/{len(papers)}, SPECTER2 {sum(i in spec for i in papers)}/{len(papers)}")

    def unit(M):
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    R_bge = unit(np.stack([bge_rand[i] for i in rand_ids if i in bge_rand]))
    R_spec = unit(np.stack([spec[i] for i in rand_ids if i in spec]))

    per = {k: [] for k in ("ewma_profile", "lt_only", "max_sim", "logreg_bge",
                           "ewma_profile_specter2", "max_sim_specter2", "logreg_specter2")}
    used = 0
    for u in users:
        r = sub[sub.user_id == u]
        r = r[r.arxiv_id.isin(bge) & r.arxiv_id.isin(spec)]
        cut = int(len(r) * 0.7)
        tr, te = r.iloc[:cut], r.iloc[cut:]
        if tr.rating.gt(0).sum() < 5 or tr.rating.le(0).sum() < 3 or te.rating.nunique() < 2:
            continue
        used += 1
        y = (te.rating > 0).astype(int).values
        Xte = unit(np.stack([bge[i] for i in te.arxiv_id]))
        # ResearchIT profiles, built exactly as the app does, in time order.
        def profile_scores(V, Xt):
            lt = st = ng = None
            n_pos = n_neg = 0
            for aid, rating in zip(tr.arxiv_id, tr.rating):
                v = V[aid]
                if rating > 0:
                    lt = ewma_update(lt, v, ALPHA_LONG_TERM, count=n_pos, normalise=False)
                    st = ewma_update(st, v, ALPHA_SHORT_TERM, count=n_pos, normalise=False)
                    n_pos += 1
                else:
                    ng = ewma_update(ng, v, ALPHA_NEGATIVE, count=n_neg, normalise=False)
                    n_neg += 1
            lt, st, ng = (x / np.linalg.norm(x) for x in (lt, st, ng))
            return Xt @ lt, Xt @ st, Xt @ ng

        s_lt, s_st, s_ng = profile_scores(bge, Xte)
        per["ewma_profile"].append(roc_auc_score(y, 0.40 * s_lt + 0.25 * s_st - 0.15 * np.clip(s_ng, 0, None)))
        per["lt_only"].append(roc_auc_score(y, s_lt))
        P = unit(np.stack([bge[i] for i in tr.arxiv_id[tr.rating > 0]]))
        per["max_sim"].append(roc_auc_score(y, (Xte @ P.T).max(axis=1)))
        Xte_s = unit(np.stack([spec[i] for i in te.arxiv_id]))
        p_lt, p_st, p_ng = profile_scores(spec, Xte_s)
        per["ewma_profile_specter2"].append(
            roc_auc_score(y, 0.40 * p_lt + 0.25 * p_st - 0.15 * np.clip(p_ng, 0, None)))
        Ps = unit(np.stack([spec[i] for i in tr.arxiv_id[tr.rating > 0]]))
        per["max_sim_specter2"].append(roc_auc_score(y, (Xte_s @ Ps.T).max(axis=1)))
        for name, V, R in (("logreg_bge", bge, R_bge), ("logreg_specter2", spec, R_spec)):
            Xtr = unit(np.stack([V[i] for i in tr.arxiv_id]))
            ytr = (tr.rating > 0).astype(int).values
            X = np.vstack([Xtr, R])
            Y = np.concatenate([ytr, np.zeros(len(R), int)])
            # Scholar Inbox weighting: positives balanced against all negatives,
            # explicit downvotes weighted above random negatives.
            w = np.where(Y == 1, 1.0 / max(1, ytr.sum()), 0.0)
            n_dn, n_rd = (ytr == 0).sum(), len(R)
            w[: len(ytr)][ytr == 0] = 0.5 / max(1, n_dn)
            w[len(ytr):] = 0.5 / n_rd
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(X, Y, sample_weight=w * len(Y))
            Xt = unit(np.stack([V[i] for i in te.arxiv_id]))
            per[name].append(roc_auc_score(y, clf.decision_function(Xt)))
    rng = np.random.default_rng(SEED)
    res = {"n_users_sampled": len(users), "n_users_scored": used, "eligible_users": len(eligible),
           "protocol": "time split 70/30 per user; AUC liked vs disliked test papers", "scorers": {}}
    base = np.asarray(per["ewma_profile"])
    for k, v in per.items():
        a = np.asarray(v)
        diff = a - base
        res["scorers"][k] = {"mean_auc": float(a.mean()), "ci95": list(boot_ci(a, rng)),
                             "minus_ewma_profile": float(diff.mean()),
                             "minus_ewma_profile_ci95": list(boot_ci(diff, rng)),
                             "users_better_than_ewma_profile": float((diff > 0).mean())}
        log(STAGE, f"{k}: {res['scorers'][k]}")
    (OUT / "user_ratings_pilot.json").write_text(json.dumps(res, indent=1, default=float))
    log(STAGE, f"wrote {OUT / 'user_ratings_pilot.json'}")


if __name__ == "__main__":
    main()
