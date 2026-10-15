"""User-centred test suite: does the recommender serve real kinds of readers?

T1  UP NEXT (the "liked X, Y, Z -> what next?" chain), real readers.
    Each reader's upvotes in time order. At up to 5 steps per reader (after at
    least 5 upvotes), the target is their NEXT upvote; candidates are the target
    plus 49 papers other readers rated within +/-14 days (a realistic same-week
    pool, so recency cannot give the answer away). Strategies x models:
      last_item  similarity to the paper just liked
      short      short-term EWMA profile (app's ewma_update, alpha 0.40)
      long       long-term EWMA profile (alpha 0.03)
      blend      0.40*unit(long) + 0.25*unit(short)   (feed formula, no recency)
      recent3    best similarity to any of the last 3 likes
    Metric: hit@10 and MRR of the true next paper.
T2  DENSE + LEXICAL (Semantic Scholar's recipe), real readers.
    Liked-vs-disliked AUC: dense profile, TF-IDF profile, and their average.
T3  NEWCOMER PATHWAY. A newcomer arrives on a paper (each Test A seed) and
    asks "what should I read?". Compare plain relevance with a "start here"
    re-rank that also rewards plainly written abstracts (app/readability.py),
    surveys/tutorials and foundational papers (citations), and report what each
    costs in relevance (citation-linked papers in the top 10).

Vectors come from the bake-off cache (round 1 and round 2 where present).
"""
from __future__ import annotations

import json
import random
import re
import sqlite3

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEED, SIDECAR, app_import_path, log

app_import_path()
BK = CACHE / "bakeoff"
STAGE = "persona"
import os
# Round 1 by default; add round-2 models with PERSONA_MODELS=... once encoded.
CANDIDATE_MODELS = os.getenv("PERSONA_MODELS", "bge_m3,specter2,scincl,gte_large,minilm").split(",")
SURVEY = re.compile(r"\b(survey|review|tutorial|overview|introduction to|primer|a gentle)\b", re.I)


def load_vecs(name):
    p = BK / f"vec_{name}.npy"
    if not p.exists():
        return None
    ids = json.loads((BK / f"ids_{name}.json").read_text())
    M = np.load(p).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return dict(zip(ids, M))


def unit(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def ci(x, rng, n=3000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_auc_score
    from app import readability
    from app.recommend.profiles import ALPHA_LONG_TERM, ALPHA_SHORT_TERM, ewma_update
    rng = np.random.default_rng(SEED)
    u = json.loads((BK / "universe.json").read_text())
    V = {m: v for m in CANDIDATE_MODELS if (v := load_vecs(m)) is not None}
    common = set.intersection(*(set(v) for v in V.values()))
    log(STAGE, f"models: {list(V)}; papers with every model: {len(common):,}")
    res = {"models": list(V), "common_papers": len(common)}

    ratings = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    ratings["time"] = pd.to_datetime(ratings["time"], format="mixed")
    users = [x for x in u["real_users"]]
    r_all = ratings[ratings.arxiv_id.isin(common)]
    sub = r_all[r_all.user_id.isin(users)].sort_values(["user_id", "time"])

    # ── T1 up next ──
    strategies = ("last_item", "short", "long", "blend", "recent3")
    hits = {(m, s): [] for m in V for s in strategies}
    mrr = {(m, s): [] for m in V for s in strategies}
    rr = random.Random(SEED)
    others_by_day = r_all.assign(day=r_all.time.dt.floor("D"))
    n_cases = 0
    for usr in users:
        mine = sub[sub.user_id == usr]
        ups = mine[mine.rating > 0]
        if len(ups) < 8:
            continue
        rated = set(mine.arxiv_id)
        steps = list(range(5, len(ups)))
        rr.shuffle(steps)
        for t in sorted(steps[:5]):
            hist = list(ups.arxiv_id.iloc[:t])
            target, when = ups.arxiv_id.iloc[t], ups.time.iloc[t]
            win = others_by_day[(others_by_day.time - when).abs() <= pd.Timedelta(days=14)]
            pool = sorted(set(win.arxiv_id) - rated - {target})
            if len(pool) < 49:
                continue
            cands = [target] + rr.sample(pool, 49)
            n_cases += 1
            for m, vec in V.items():
                X = np.stack([vec[c] for c in cands])
                lt = st = None
                for k, a in enumerate(hist):
                    lt = ewma_update(lt, vec[a], ALPHA_LONG_TERM, count=k, normalise=False)
                    st = ewma_update(st, vec[a], ALPHA_SHORT_TERM, count=k, normalise=False)
                sl, ss = X @ (lt / np.linalg.norm(lt)), X @ (st / np.linalg.norm(st))
                scores = {"last_item": X @ vec[hist[-1]], "short": ss, "long": sl,
                          "blend": 0.40 * unit(sl) + 0.25 * unit(ss),
                          "recent3": (X @ np.stack([vec[a] for a in hist[-3:]]).T).max(axis=1)}
                for s, sc in scores.items():
                    rank = int((sc > sc[0]).sum())
                    hits[(m, s)].append(float(rank < 10))
                    mrr[(m, s)].append(1.0 / (rank + 1))
    t1 = {f"{m}|{s}": {"hit@10": float(np.mean(hits[(m, s)])), "hit@10_ci": ci(hits[(m, s)], rng),
                       "MRR": float(np.mean(mrr[(m, s)]))} for (m, s) in hits}
    res["T1_up_next"] = {"n_cases": n_cases, "random_hit@10": 10 / 50, "by": t1}
    best = max(t1, key=lambda k: t1[k]["hit@10"])
    log(STAGE, f"T1 cases {n_cases}; best {best} {t1[best]['hit@10']:.3f}; "
               + " ".join(f"{m}:{t1[m + '|blend']['hit@10']:.3f}" for m in V))

    # ── T2 dense + lexical ──
    texts = {i: u["titles"].get(i, "") + " " + u["abstracts"].get(i, "") for i in common}
    ids_all = sorted(texts)
    tf = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200000, stop_words="english")
    T = tf.fit_transform([texts[i] for i in ids_all])
    row = {a: k for k, a in enumerate(ids_all)}
    best_dense = max(V, key=lambda m: t1[f"{m}|blend"]["hit@10"])
    auc = {"dense:" + best_dense: [], "dense:bge_m3": [], "tfidf": [], "avg:" + best_dense + "+tfidf": [],
           "avg:bge_m3+tfidf": []}
    for usr in users:
        mine = sub[sub.user_id == usr]
        cut = int(len(mine) * 0.7)
        tr, te = mine.iloc[:cut], mine.iloc[cut:]
        if tr.rating.gt(0).sum() < 5 or te.rating.nunique() < 2:
            continue
        y = (te.rating > 0).astype(int).values
        pos = list(tr.arxiv_id[tr.rating > 0])
        tcen = np.asarray(T[[row[a] for a in pos]].mean(axis=0)).ravel()
        ttest = T[[row[a] for a in te.arxiv_id]] @ tcen
        out = {"tfidf": np.asarray(ttest).ravel()}
        for m in {best_dense, "bge_m3"}:
            cen = np.mean([V[m][a] for a in pos], axis=0)
            out["dense:" + m] = np.stack([V[m][a] for a in te.arxiv_id]) @ cen
        out["avg:" + best_dense + "+tfidf"] = 0.5 * unit(out["dense:" + best_dense]) + 0.5 * unit(out["tfidf"])
        out["avg:bge_m3+tfidf"] = 0.5 * unit(out["dense:bge_m3"]) + 0.5 * unit(out["tfidf"])
        for k in auc:
            auc[k].append(roc_auc_score(y, out[k]))
    res["T2_dense_lexical"] = {k: {"auc": float(np.mean(v)), "ci": ci(v, rng)} for k, v in auc.items()}
    res["T2_dense_lexical"]["n_readers"] = len(auc["tfidf"])
    log(STAGE, "T2 " + json.dumps({k: round(v["auc"], 4) for k, v in res["T2_dense_lexical"].items() if k != "n_readers"}))

    # ── T3 newcomer pathway ──
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    cit = {a: c or 0 for a, c in sc.execute("SELECT arxiv_id, citation_count FROM papers")}
    pool_ids = sorted(common & (set(u["background"]) | set(u["test_a"])))
    lvl = {a: readability.level(u["abstracts"].get(a, "")) for a in pool_ids}
    is_survey = {a: bool(SURVEY.search(u["titles"].get(a, ""))) for a in pool_ids}
    M = np.stack([V[best_dense][a] for a in pool_ids])
    idx = {a: k for k, a in enumerate(pool_ids)}
    logc = np.log1p(np.array([cit.get(a, 0) for a in pool_ids], float))
    acc = np.array([1.0 if lvl[a] == "accessible" else 0.0 for a in pool_ids])
    surv = np.array([1.0 if is_survey[a] else 0.0 for a in pool_ids])
    rows = {"relevance": [], "start_here": []}
    examples = []
    for s in u["seeds"]:
        if s not in idx:
            continue
        linked = {x for x in u["links"][s]["refs"] + u["links"][s]["cits"] if x in idx}
        if len(linked) < 3:
            continue
        rel = M @ V[best_dense][s]
        rel[idx[s]] = -9
        top100 = np.argsort(-rel)[:100]          # newcomer re-rank only reorders relevant candidates
        r_unit = unit(rel[top100])
        newcomer = 0.55 * r_unit + 0.15 * acc[top100] + 0.15 * surv[top100] + 0.15 * unit(logc[top100])
        for name, order in (("relevance", top100[:10]), ("start_here", top100[np.argsort(-newcomer)[:10]])):
            ids10 = [pool_ids[j] for j in order]
            rows[name].append({"linked": np.mean([i in linked for i in ids10]),
                               "accessible": np.mean([lvl[i] == "accessible" for i in ids10]),
                               "survey": np.mean([is_survey[i] for i in ids10]),
                               "median_citations": float(np.median([cit.get(i, 0) for i in ids10]))})
        if len(examples) < 3:
            examples.append({"seed": u["titles"].get(s, ""), "relevance": [u["titles"].get(pool_ids[j], "")[:100] for j in top100[:6]],
                             "start_here": [{"title": u["titles"].get(pool_ids[j], "")[:100], "accessible": lvl[pool_ids[j]] == "accessible",
                                             "survey": is_survey[pool_ids[j]], "citations": cit.get(pool_ids[j], 0)}
                                            for j in top100[np.argsort(-newcomer)[:6]]]})
    res["T3_newcomer"] = {"model": best_dense, "n_seeds": len(rows["relevance"]),
                          "weights": "0.55 relevance + 0.15 plainly written + 0.15 survey/tutorial + 0.15 log citations",
                          **{n: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for n, v in rows.items()},
                          "examples": examples}
    log(STAGE, "T3 " + json.dumps({n: res["T3_newcomer"][n] for n in rows}))
    (OUT / "persona_eval.json").write_text(json.dumps(res, indent=1, default=float))
    log(STAGE, f"wrote {OUT / 'persona_eval.json'}")


if __name__ == "__main__":
    main()
