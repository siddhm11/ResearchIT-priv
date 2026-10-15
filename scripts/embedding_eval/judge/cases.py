"""Build the AI-judged test cases: what each model would show on two surfaces.

Surface "upnext"  (like YouTube's Up next): a reader just finished paper S; each
                  model lists the 10 papers it would suggest next.
Surface "home"    (like YouTube's Home): a real reader's recent saves; each model
                  lists the 10 papers it would put in their feed.

Lists are made with each model's own vectors over the same library: every paper
that all compared models have vectors for (the bake-off round-2 set plus the AI/ML
set, ~61k papers). "planned" is the cheap deployment: BGE-M3 retrieves, GTE-large
reorders. Home lists use the app's own profile code and feed similarity formula.

Writes CACHE/judge/cases.json. Deterministic (seed 42). Read-only.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, SEED, SIDECAR, app_import_path, log  # noqa: E402

app_import_path()
from app.recommend.profiles import (ALPHA_LONG_TERM, ALPHA_NEGATIVE,  # noqa: E402
                                    ALPHA_SHORT_TERM, ewma_update)

BK, AI, OUT = CACHE / "bakeoff", CACHE / "aiml", CACHE / "judge"
MODELS = ["bge_m3", "gte_large", "qwen3_06b", "specter2"]
AIML = {"cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML", "cs.NE", "cs.IR"}
N_AIML_PER_STRATUM, N_GENERAL, N_READERS, K = 30, 60, 60, 10


def unitv(M):
    M = np.asarray(M, np.float32)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def vectors(model: str) -> dict[str, np.ndarray]:
    out = {}
    for d in (BK, AI):
        p = d / f"vec_{model}.npy"
        if p.exists():
            ids = json.loads((d / f"ids_{model}.json").read_text())
            out.update(zip(ids, unitv(np.load(p))))
    return out


def unit(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def year(aid: str) -> float | None:
    m = re.match(r"(\d{2})(\d{2})\.", aid)
    if m:
        return 2000 + int(m.group(1)) + (int(m.group(2)) - 0.5) / 12
    m = re.search(r"/(\d{2})(\d{2})", aid)
    if m:
        y = int(m.group(1))
        return (1900 + y if y > 90 else 2000 + y) + (int(m.group(2)) - 0.5) / 12
    return None


def main() -> None:
    U = json.loads((BK / "universe.json").read_text())
    A = json.loads((AI / "aiml_universe.json").read_text())
    r2 = json.loads((BK / "round2_ids.json").read_text())
    m2 = json.loads((BK / "round2_meta.json").read_text())
    V = {m: vectors(m) for m in MODELS}
    lib = sorted((set(r2) | set(A["ids"])).intersection(*[set(v) for v in V.values()]))
    pos = {a: k for k, a in enumerate(lib)}
    M = {m: np.stack([V[m][a] for a in lib]) for m in MODELS}
    log("judge", f"library: {len(lib):,} papers every model has")
    titles = {**U["titles"], **A["titles"]}
    abstracts = {**U["abstracts"], **A["abstracts"]}
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    cat = {a: (c or "").split()[0] for a, c in sc.execute("SELECT arxiv_id, categories FROM papers")}
    used: set[str] = set()

    def lists_for(scores: dict[str, np.ndarray], exclude: set[int], retrieve_n: int) -> dict[str, list[str]]:
        out = {}
        for m in MODELS:
            s = scores[m].copy()
            s[list(exclude)] = -9
            out[m] = [lib[j] for j in np.argsort(-s)[:K]]
        s_b = scores["bge_m3"].copy()
        s_b[list(exclude)] = -9
        top = np.argsort(-s_b)[:retrieve_n]
        out["planned"] = [lib[j] for j in top[np.argsort(-scores["gte_large"][top])[:K]]]
        return out

    # ── Up next ──
    cases = []
    picks = []
    for st in ("established", "recent"):
        picks += [s for s in A["seeds"] if A["stratum"][s] == st and s in pos][:N_AIML_PER_STRATUM]
    picks += [s for s in U["seeds"] if s in pos and cat.get(s, "") not in AIML][:N_GENERAL]
    links = {**U["links"], **A["links"]}
    for s in picks:
        q = {m: M[m] @ M[m][pos[s]] for m in MODELS}
        lists = lists_for(q, {pos[s]}, 100)
        linked = sorted(set(links[s]["refs"]) | set(links[s]["cits"]))
        group = ("AI/ML " + A["stratum"][s]) if s in A["stratum"] else ("physics" if cat.get(s, "").split(".")[0] not in ("cs", "math", "stat", "eess", "q-bio", "q-fin", "econ") else cat.get(s, "").split(".")[0])
        cases.append({"surface": "upnext", "qid": f"u:{s}", "seed": s, "group": group, "lists": lists,
                      "linked": [x for x in linked if x in set(sum(lists.values(), []))]})
        used |= {s} | set(sum(lists.values(), []))
    log("judge", f"up-next cases: {len(cases)}")

    # ── Home ──
    ratings = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    ratings["time"] = pd.to_datetime(ratings["time"], format="mixed")
    r_all = ratings[ratings.arxiv_id.isin(pos)].sort_values(["user_id", "time"])
    rng = random.Random(SEED)
    n_home = 0
    for usr in m2["users"]:
        if n_home == N_READERS:
            break
        mine = r_all[r_all.user_id == usr]
        cut = int(len(mine) * 0.7)
        tr, te = mine.iloc[:cut], mine.iloc[cut:]
        if tr.rating.gt(0).sum() < 5 or tr.rating.le(0).sum() < 3:
            continue
        te = te[te.time.dt.date > tr.time.iloc[-1].date()]
        if len(te) < 6 or te.rating.nunique() < 2:
            continue
        when = te.time.iloc[0]
        win = r_all[(r_all.time - when).abs() <= pd.Timedelta(days=14)]
        wpool = sorted(set(win.arxiv_id) - set(mine.arxiv_id))
        if len(wpool) < 200:
            continue
        idx = np.array([pos[a] for a in wpool])
        P = list(tr.arxiv_id[tr.rating > 0])
        N = list(tr.arxiv_id[tr.rating <= 0])
        scores = {}
        for m in MODELS:
            lt = st = ng = None
            for k, a in enumerate(P):
                lt = ewma_update(lt, M[m][pos[a]], ALPHA_LONG_TERM, count=k, normalise=False)
                st = ewma_update(st, M[m][pos[a]], ALPHA_SHORT_TERM, count=k, normalise=False)
            for k, a in enumerate(N):
                ng = ewma_update(ng, M[m][pos[a]], ALPHA_NEGATIVE, count=k, normalise=False)
            X = M[m][idx]
            s = (0.40 * unit(X @ (lt / np.linalg.norm(lt))) + 0.25 * unit(X @ (st / np.linalg.norm(st)))
                 - 0.15 * np.clip(X @ (ng / np.linalg.norm(ng)), 0, None))
            full = np.full(len(lib), -9.0)
            full[idx] = s
            scores[m] = full
        lists = lists_for(scores, set(), 50)
        liked = [a for a in te.arxiv_id[te.rating > 0]]
        disliked = [a for a in te.arxiv_id[te.rating <= 0]]
        probe = rng.sample(liked, min(3, len(liked))) + rng.sample(disliked, min(3, len(disliked)))
        cases.append({"surface": "home", "qid": f"h:{n_home:03d}", "reader": f"Reader {n_home + 1}",
                      "group": "readers", "context_liked": P[-8:][::-1], "context_disliked": N[-3:][::-1],
                      "lists": lists, "probe": {a: (a in liked) for a in probe}, "pool": len(wpool),
                      "date": str(when.date())})
        used |= set(P[-8:]) | set(N[-3:]) | set(sum(lists.values(), [])) | set(probe)
        n_home += 1
    log("judge", f"home cases: {n_home}")
    papers = {a: {"title": " ".join((titles.get(a) or "").split()), "abstract": " ".join((abstracts.get(a) or "").split()),
                  "cat": cat.get(a, ""), "year": year(a)} for a in sorted(used)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps({"models": MODELS + ["planned"], "cases": cases, "papers": papers,
                                                "library_size": len(lib)}))
    n_items = sum(len(set(sum(c["lists"].values(), [])) | set(c.get("probe", {}))) for c in cases)
    log("judge", f"wrote cases.json: {len(cases)} cases, {n_items:,} (case, paper) items to judge")


if __name__ == "__main__":
    main()
