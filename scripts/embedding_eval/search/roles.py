"""Every library model on the three jobs: related papers, feed taste, and (from stage 1) search.

Related papers: the 700 citation-test seeds (300 general + 400 AI/ML) whose cited
papers are in the library; whole-corpus recall@100 estimated from the 30k random
background, and cited share of the 10 nearest in the library.
Feed taste: Scholar Inbox readers (round 1), profile from earlier votes with the
app's own profile code, AUC on votes cast on a later day.
Writes CACHE/search/roles.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, SEED, app_import_path, log  # noqa: E402
from encode import MODELS  # noqa: E402

app_import_path()
from app.recommend.profiles import ALPHA_LONG_TERM, ALPHA_NEGATIVE, ALPHA_SHORT_TERM, ewma_update  # noqa: E402

S = CACHE / "search"
N_CORPUS = 1_853_777
rng = np.random.default_rng(SEED)


def unit(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def boot(x):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (3000, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> None:
    lib = json.loads((S / "library.json").read_text())
    ids = lib["ids"]
    pos = {a: k for k, a in enumerate(ids)}
    U = json.loads((CACHE / "bakeoff" / "universe.json").read_text())
    A = json.loads((CACHE / "aiml" / "aiml_universe.json").read_text())
    links = {**{s: U["links"][s] for s in U["seeds"]}, **A["links"]}
    seeds = [s for s in links if s in pos]
    bg = np.array([pos[a] for a in lib["background"] if a in pos])
    r = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    r["time"] = pd.to_datetime(r["time"], format="mixed")
    r = r[r.arxiv_id.isin(pos)].sort_values(["user_id", "time"])
    readers = []
    for usr in U["real_users"]:
        mine = r[r.user_id == usr]
        cut = int(len(mine) * 0.7)
        tr, te = mine.iloc[:cut], mine.iloc[cut:]
        if tr.rating.gt(0).sum() < 5 or tr.rating.le(0).sum() < 3:
            continue
        te = te[te.time.dt.date > tr.time.iloc[-1].date()] if len(tr) else te
        if len(te) >= 10 and te.rating.nunique() == 2:
            readers.append((list(tr.arxiv_id[tr.rating > 0]), list(tr.arxiv_id[tr.rating <= 0]),
                            list(te.arxiv_id), (te.rating > 0).values))
    log("roles", f"seeds {len(seeds)}, readers {len(readers)}")
    per = {}
    for m in MODELS:
        if not (S / f"vec_{m}.npy").exists():
            continue
        mids = json.loads((S / f"ids_{m}.json").read_text())
        D = np.zeros((len(ids), 0), np.float32)
        raw = np.load(S / f"vec_{m}.npy").astype(np.float32)
        D = np.zeros((len(ids), raw.shape[1]), np.float32)
        D[[pos[a] for a in mids]] = raw
        rec, panel, taste = [], [], []
        Dbg = D[bg]
        for s in seeds:
            L = [x for x in links[s]["refs"] + links[s]["cits"] if x in pos and x != s]
            if len(L) < 3:
                continue
            q = D[pos[s]]
            sl = D[[pos[x] for x in L]] @ q
            sb = np.sort(Dbg @ q)
            b = len(sb) - np.searchsorted(sb, sl, side="right")
            o = np.array([(sl > x).sum() for x in sl])
            rec.append(np.mean([float(beta.cdf(max(0, 99 - oi) / N_CORPUS, bi + 0.5, len(sb) - bi + 0.5)) if oi < 99 else 0.0
                                for bi, oi in zip(b, o)]))
            sc = D @ q
            sc[pos[s]] = -9
            top = np.argpartition(-sc, 10)[:10]
            Ls = set(L)
            panel.append(np.mean([ids[j] in Ls for j in top]))
        for P, N, T, y in readers:
            lt = st = ng = None
            for k, a in enumerate(P):
                lt = ewma_update(lt, D[pos[a]], ALPHA_LONG_TERM, count=k, normalise=False)
                st = ewma_update(st, D[pos[a]], ALPHA_SHORT_TERM, count=k, normalise=False)
            for k, a in enumerate(N):
                ng = ewma_update(ng, D[pos[a]], ALPHA_NEGATIVE, count=k, normalise=False)
            X = D[[pos[a] for a in T]]
            s_ = 0.40 * unit(X @ (lt / np.linalg.norm(lt))) + 0.25 * unit(X @ (st / np.linalg.norm(st))) \
                - 0.15 * np.clip(X @ (ng / np.linalg.norm(ng)), 0, None)
            taste.append(roc_auc_score(y, s_))
        per[m] = {"related_corpus_r100": rec, "related_panel_p10": panel, "taste_later_auc": taste}
        log("roles", f"{m:15} related corpus R@100 {np.mean(rec):.3f}  panel P@10 {np.mean(panel):.3f}  taste AUC {np.mean(taste):.3f}")
    out = {"n_seeds": len(per[next(iter(per))]["related_corpus_r100"]), "n_readers": len(readers), "models": {}}
    for m, v in per.items():
        out["models"][m] = {}
        for k, x in v.items():
            d = np.asarray(x) - np.asarray(per["bge_m3"][k])
            out["models"][m][k] = {"mean": float(np.mean(x)), "ci": boot(x), "d_bge": float(d.mean()), "d_bge_ci": boot(d)}
    (S / "roles.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
