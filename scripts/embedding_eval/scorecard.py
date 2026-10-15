"""One scorecard: every model, every test, same papers and readers, paired stats.

All models are scored on the SAME units (seed papers, real readers, up-next
moments), so any two can be compared paired, unit by unit. For every metric:
  * per-model mean with a 95% bootstrap interval
  * every pairwise difference with a 95% bootstrap interval and a two-sided
    sign-flip permutation p-value, Holm-corrected across all pairs of that metric
  * pass/fail against user expectations stated up front (EXPECTATIONS below)

Universe: the round-2 subset (Test A papers, 300 real readers' rated papers,
10,000 random background papers), so round-1 and round-2 models share it.

Metrics (unit in brackets)
  related      cited paper beats a same-category random paper        [seed]
  lookalike    different-field keyword lookalike beats a cited paper   [seed]  lower is better
  reach        recall@100 of a seed's citation links                   [seed]
  taste        liked-vs-disliked AUC, feed profile score, time split   [reader]
  up_next      next liked paper in the top 10 of 50 same-week papers   [moment]
  plain_first  newcomer "start here": plainly written share of top 10  [seed]
  start_kept   newcomer "start here": citation-linked share of top 10  [seed]
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
from itertools import combinations

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEED, SIDECAR, app_import_path, log

app_import_path()
BK = CACHE / "bakeoff"
STAGE = "scorecard"
ALL = ["bge_m3", "bge_m3_full", "specter2", "scincl", "gte_large", "minilm",
       "gte_modernbert", "nomic_v15", "qwen3_06b"]
SURVEY = re.compile(r"\b(survey|review|tutorial|overview|introduction to|primer)\b", re.I)

# What users expect, as proposed product targets for discussion, not facts;
# each maps to one metric. NOT blind: they were set after the round-1
# bake-off and persona results had been seen, so treat a pass or fail as a
# conversation starter, and re-set them on purpose before relying on them.
EXPECTATIONS = {
    "related":     (">=", 0.88, "Researcher: papers my field cites outrank random ones in my area"),
    "lookalike":   ("<=", 0.07, "Everyone: a shared buzzword alone doesn't pull in a wrong-field paper"),
    "reach":       (">=", 0.65, "Researcher: 'related papers' finds most of the papers my field links to this one"),
    "taste":       (">=", 0.75, "Returning reader: what I liked ranks above what I passed on"),
    "up_next":     (">=", 0.92, "Returning reader: what I want next is on the first screen"),
    "plain_first": (">=", 0.50, "Newcomer: at least half of 'start here' is plainly written"),
    "start_kept":  (">=", 0.60, "Newcomer: 'start here' is still about the paper I came from"),
}


def load(name, ids):
    p = BK / f"vec_{name}.npy"
    if not p.exists():
        return None
    got = json.loads((BK / f"ids_{name}.json").read_text())
    M = np.load(p).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    d = dict(zip(got, M))
    # A model counts only if it covers nearly all of the subset (in-progress
    # encodes have no file yet); papers any model lacks are dropped for all.
    return d if sum(i in d for i in ids) >= 0.999 * len(ids) else None


def unit(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def main() -> None:
    from sklearn.metrics import roc_auc_score
    from app import readability
    from app.recommend.profiles import (ALPHA_LONG_TERM, ALPHA_NEGATIVE, ALPHA_SHORT_TERM,
                                        ewma_update)
    rng = np.random.default_rng(SEED)
    u = json.loads((BK / "universe.json").read_text())
    ids = json.loads((BK / "round2_ids.json").read_text())
    meta2 = json.loads((BK / "round2_meta.json").read_text())
    idset = set(ids)
    wanted = os.getenv("SCORECARD_MODELS", ",".join(ALL)).split(",")
    V = {m: v for m in wanted if (v := load(m, ids)) is not None}
    ids = [i for i in ids if all(i in v for v in V.values())]
    idset = set(ids)
    log(STAGE, f"models: {list(V)}; papers every model has: {len(ids):,}")
    per = {}                          # metric -> model -> list of per-unit values

    def put(metric, model, value):
        per.setdefault(metric, {}).setdefault(model, []).append(float(value))

    def sim(m, a, b):
        return float(V[m][a] @ V[m][b])

    # ── seed-level tests ──
    seeds = [s for s in u["seeds"] if s in idset]
    pool = sorted(idset & (set(meta2["background"]) | set(u["test_a"])))
    pidx = {a: k for k, a in enumerate(pool)}
    Ms = {m: np.stack([V[m][a] for a in pool]) for m in V}
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    cit = {a: c or 0 for a, c in sc.execute("SELECT arxiv_id, citation_count FROM papers")}
    plain = np.array([readability.level(u["abstracts"].get(a, "")) == "accessible" for a in pool], float)
    surv = np.array([bool(SURVEY.search(u["titles"].get(a, ""))) for a in pool], float)
    logc = np.log1p(np.array([cit.get(a, 0) for a in pool], float))
    s_units = []
    for s in seeds:
        e = u["ex"][s]
        linked = {x for x in u["links"][s]["refs"] + u["links"][s]["cits"] if x in pidx}
        if len(linked) < 3:
            continue
        s_units.append(s)
        for m in V:
            trip = [sim(m, s, l) > sim(m, s, n) for l in e["linked"][:5] if l in idset for n in e["same_cat"] if n in idset]
            put("related", m, np.mean(trip) if trip else np.nan)
            t = (e.get("trap") or {}).get("id")
            lure = [sim(m, s, t) > sim(m, s, l) for l in e["linked"][:5] if l in idset] if t in idset else []
            put("lookalike", m, np.mean(lure) if lure else np.nan)
            sc_ = Ms[m] @ V[m][s]
            sc_[pidx[s]] = -9
            top = np.argsort(-sc_)[:100]
            put("reach", m, len(linked & {pool[j] for j in top}) / len(linked))
            r = unit(sc_[top])
            order = top[np.argsort(-(0.55 * r + 0.15 * plain[top] + 0.15 * surv[top] + 0.15 * unit(logc[top])))[:10]]
            put("plain_first", m, plain[order].mean())
            put("start_kept", m, np.mean([pool[j] in linked for j in order]))

    # ── reader-level tests ──
    ratings = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    ratings["time"] = pd.to_datetime(ratings["time"], format="mixed")
    r_all = ratings[ratings.arxiv_id.isin(idset)]
    users = meta2["users"]
    sub = r_all[r_all.user_id.isin(users)].sort_values(["user_id", "time"])
    rr = random.Random(SEED)
    for usr in users:
        mine = sub[sub.user_id == usr]
        cut = int(len(mine) * 0.7)
        tr, te = mine.iloc[:cut], mine.iloc[cut:]
        if tr.rating.gt(0).sum() >= 5 and tr.rating.le(0).sum() >= 3 and te.rating.nunique() == 2:
            y = (te.rating > 0).astype(int).values
            for m in V:
                lt = st = ng = None
                np_ = nn = 0
                for a, rating in zip(tr.arxiv_id, tr.rating):
                    if rating > 0:
                        lt = ewma_update(lt, V[m][a], ALPHA_LONG_TERM, count=np_, normalise=False)
                        st = ewma_update(st, V[m][a], ALPHA_SHORT_TERM, count=np_, normalise=False)
                        np_ += 1
                    else:
                        ng = ewma_update(ng, V[m][a], ALPHA_NEGATIVE, count=nn, normalise=False)
                        nn += 1
                X = np.stack([V[m][a] for a in te.arxiv_id])
                score = (0.40 * unit(X @ (lt / np.linalg.norm(lt))) + 0.25 * unit(X @ (st / np.linalg.norm(st)))
                         - 0.15 * np.clip(X @ (ng / np.linalg.norm(ng)), 0, None))
                put("taste", m, roc_auc_score(y, score))
        # up next
        ups = mine[mine.rating > 0]
        if len(ups) < 8:
            continue
        rated = set(mine.arxiv_id)
        steps = list(range(5, len(ups)))
        rr.shuffle(steps)
        for t in sorted(steps[:5]):
            hist, target, when = list(ups.arxiv_id.iloc[:t]), ups.arxiv_id.iloc[t], ups.time.iloc[t]
            win = r_all[(r_all.time - when).abs() <= pd.Timedelta(days=14)]
            cand_pool = sorted(set(win.arxiv_id) - rated - {target})
            if len(cand_pool) < 49:
                continue
            cands = [target] + rr.sample(cand_pool, 49)
            for m in V:
                X = np.stack([V[m][c] for c in cands])
                lt = st = None
                for k, a in enumerate(hist):
                    lt = ewma_update(lt, V[m][a], ALPHA_LONG_TERM, count=k, normalise=False)
                    st = ewma_update(st, V[m][a], ALPHA_SHORT_TERM, count=k, normalise=False)
                s_ = 0.40 * unit(X @ (lt / np.linalg.norm(lt))) + 0.25 * unit(X @ (st / np.linalg.norm(st)))
                put("up_next", m, float((s_ > s_[0]).sum() < 10))

    # ── statistics ──
    def boot(x):
        m = x[rng.integers(0, len(x), (3000, len(x)))].mean(axis=1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    def perm_p(d):
        signs = rng.choice([-1.0, 1.0], (5000, len(d)))
        null = np.abs((signs * d).mean(axis=1))
        return float((np.sum(null >= abs(d.mean())) + 1) / (len(null) + 1))

    card = {"models": list(V), "n_papers": len(ids), "expectations": EXPECTATIONS, "metrics": {}}
    for metric, by in per.items():
        A = np.array([by[m] for m in V])                 # models x units
        keep = ~np.isnan(A).any(axis=0)
        A = A[:, keep]
        rows = {m: {"mean": float(A[k].mean()), "ci95": boot(A[k])} for k, m in enumerate(V)}
        op, target, _ = EXPECTATIONS[metric]
        for m in rows:
            rows[m]["meets_expectation"] = rows[m]["mean"] >= target if op == ">=" else rows[m]["mean"] <= target
        pairs = []
        for (i, a), (j, b) in combinations(enumerate(V), 2):
            d = A[j] - A[i]
            pairs.append({"a": a, "b": b, "b_minus_a": float(d.mean()), "ci95": boot(d), "p": perm_p(d)})
        # Holm correction across this metric's pairs
        order = np.argsort([p["p"] for p in pairs])
        running = 0.0
        for rank, k in enumerate(order):
            running = max(running, min(1.0, pairs[k]["p"] * (len(pairs) - rank)))
            pairs[k]["p_holm"] = running
            pairs[k]["significant"] = running < 0.05
        card["metrics"][metric] = {"n_units": int(A.shape[1]), "by_model": rows, "pairs": pairs,
                                   "lower_is_better": op == "<="}
        log(STAGE, f"{metric:11} n={A.shape[1]:5} " + "  ".join(f"{m}:{rows[m]['mean']:.3f}" for m in V))
    (OUT / "scorecard.json").write_text(json.dumps(card, indent=1, default=float))
    log(STAGE, f"wrote {OUT / 'scorecard.json'}")


if __name__ == "__main__":
    main()
