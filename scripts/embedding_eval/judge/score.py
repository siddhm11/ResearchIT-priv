"""Score every model's first screen with the LLM grades, check the judges, write results.json.

    python score.py          # first screen: each list's top 5 (default)
    python score.py 10       # top 10, once ranks 6-10 are graded

Per case and model (top K):
  ndcg          satisfaction-weighted ranking quality, gain 2^g - 1, ideal = best K pooled grades
  first         mean grade of the K papers
  good          papers graded >= 2 ("relevant" or better)
  great         papers graded >= 2.5
  off           papers graded <= 0.5 (off-topic)
  cited         citation-linked papers (up next only)
  cats          distinct primary categories (variety)
  fresh         published within a year before the case date (home only)

A paper's grade is the mean of the judges that graded EVERY paper of its case, so
all models within a case are graded by the same panel. Judge checks: cited vs
uncited AUC, real later likes vs dislikes AUC, pairwise judge agreement, retest.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, SEED, log  # noqa: E402

OUT = CACHE / "judge"
# A "judge" is a (model, prompt version) pair: the short prompt is kept apart from the full one.
JUDGES = [("openai/gpt-oss-120b", "v1"), ("qwen/qwen3.8-27b", "v1"), ("openai/gpt-oss-20b", "v1"),
          ("openai/gpt-oss-20b", "v1-short")]
JNAME = {k: k[0] + ("" if k[1] == "v1" else f" ({k[1]})") for k in JUDGES}
# The reporting panel: two strong judges, full prompt. Others are fallbacks and cross-checks.
PRIMARY = [("openai/gpt-oss-120b", "v1"), ("qwen/qwen3.8-27b", "v1")]
rng = np.random.default_rng(SEED)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def agreement(pairs):
    if len(pairs) < 10:
        return None
    a, b = np.array(pairs, float).T
    O = np.zeros((4, 4))
    for x, y in zip(a.astype(int), b.astype(int)):
        O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(4)] for i in range(4)]) / 9
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    from scipy.stats import spearmanr
    return {"n": int(len(a)), "exact": float((a == b).mean()), "within1": float((abs(a - b) <= 1).mean()),
            "weighted_kappa": float(1 - (W * O).sum() / (W * E).sum()), "spearman": float(spearmanr(a, b).correlation),
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def boot(x, n=3000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def perm_p(d, n=5000):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    null = np.abs((rng.choice([-1.0, 1.0], (n, len(d))) * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean())) + 1) / (n + 1))


def holm(ps):
    o = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for r, k in enumerate(o):
        run = max(run, min(1.0, ps[k] * (len(ps) - r)))
        out[k] = run
    return out


def dcg(g):
    return sum((2 ** x - 1) / np.log2(i + 2) for i, x in enumerate(g))


def main(K: int, cases_file: str = "cases.json", base: str = "bge_m3", tag: str = "") -> None:
    data = json.loads((OUT / cases_file).read_text())
    papers, cases, models = data["papers"], data["cases"], data["models"]
    con = sqlite3.connect(OUT / "judgments.sqlite")
    G: dict[tuple, dict] = {}
    for qid, paper, judge, version, grade, why in con.execute("SELECT qid, paper, judge, version, grade, why FROM j"):
        G.setdefault((qid, paper), {})[(judge, version)] = (grade, why)

    per: dict[str, dict[str, dict[str, list]]] = {}
    per_judge: dict[tuple, dict[str, list]] = {j: {m: [] for m in models} for j in JUDGES}
    case_out, panels = [], {}
    for c in cases:
        pool = sorted({a for L in c["lists"].values() for a in L[:K]})
        complete = [j for j in JUDGES if all(j in G.get((c["qid"], a), {}) for a in pool)]
        panel = [j for j in PRIMARY if j in complete] or complete
        if not panel:
            continue
        panels[c["qid"]] = panel
        gr = {a: float(np.mean([G[(c["qid"], a)][j][0] for j in panel])) for a in pool}
        idcg = dcg(sorted(gr.values(), reverse=True)[:K]) or 1.0
        linked = set(c.get("linked", []))
        scopes = [c["surface"], f"{c['surface']}:{c['group']}"]
        for m in models:
            L = c["lists"][m][:K]
            g = [gr[a] for a in L]
            vals = {"ndcg": dcg(g) / idcg, "first": float(np.mean(g)), "good": float(sum(x >= 2 for x in g)),
                    "great": float(sum(x >= 2.5 for x in g)), "off": float(sum(x <= 0.5 for x in g)),
                    "cats": float(len({papers[a]["cat"] for a in L}))}
            if c["surface"] == "upnext":
                vals["cited"] = float(sum(a in linked for a in L))
            if c.get("date"):
                d0 = date.fromisoformat(c["date"])
                t0 = d0.year + (d0.timetuple().tm_yday - 0.5) / 366
                vals["fresh"] = float(np.mean([(papers[a]["year"] or 0) >= t0 - 1 for a in L]))
            for sc in scopes:
                for k, v in vals.items():
                    per.setdefault(sc, {}).setdefault(k, {}).setdefault(m, []).append(float(v))
        for j in panel:
            gj = {a: G[(c["qid"], a)][j][0] for a in pool}
            idj = dcg(sorted(gj.values(), reverse=True)[:K]) or 1.0
            for m in models:
                per_judge[j][m].append(dcg([gj[a] for a in c["lists"][m][:K]]) / idj)
        why = {a: {JNAME[j]: G.get((c["qid"], a), {}).get(j, (None, ""))[1] for j in panel} for a in pool}
        lists10 = {m: L[:10] for m, L in c["lists"].items()}
        case_out.append({**{k: v for k, v in c.items() if k != "lists"}, "lists": {m: L[:K] for m, L in lists10.items()},
                         "grades": gr, "why": why, "panel": [JNAME[j] for j in panel],
                         "probe_grades": {a: (float(np.mean([G[(c["qid"], a)][j][0] for j in panel
                                                             if j in G.get((c["qid"], a), {})]))
                                              if any(j in G.get((c["qid"], a), {}) for j in panel) else None)
                                          for a in c.get("probe", {})}})

    def summarise(by):
        ms = [m for m in models if by.get(m)]
        A = np.array([by[m] for m in ms])
        means = A.mean(axis=1)
        best = int(np.argmax(means))
        rows, ps = {}, []
        for k, m in enumerate(ms):
            d = A[k] - A[ms.index(base)]
            rows[m] = {"mean": float(means[k]), "ci": boot(A[k]), "d_bge": float(d.mean()), "d_bge_ci": boot(d),
                       "p_vs_bge": perm_p(d), "win_share_vs_bge": float((d > 1e-9).mean()),
                       "loss_share_vs_bge": float((d < -1e-9).mean())}
            ps.append(perm_p(A[k] - A[best]))
        for (k, m), ph in zip(enumerate(ms), holm(ps)):
            rows[m]["top_tier"] = bool(k == best or ph >= 0.05)
        return {"n": int(A.shape[1]), "best": ms[best], "by": rows}

    res = {"K": K, "complete_cases": len(case_out), "total_cases": len(cases), "models": models,
           "library_size": data.get("library_size"),
           "panel_sizes": {str(n): sum(len(p) == n for p in panels.values()) for n in (1, 2, 3)},
           "metrics": {sc: {k: summarise(v) for k, v in mets.items()} for sc, mets in per.items()},
           "per_judge_ndcg": {JNAME[j]: summarise(v) for j, v in per_judge.items() if all(len(x) >= 10 for x in v.values())}}

    # ── judge checks (every graded item, not only complete cases) ──
    checks = {}
    for j in JUDGES + ["panel"]:
        cit, unc, lik, dis = [], [], [], []
        for c in cases:
            def g_of(a):
                d = G.get((c["qid"], a), {})
                if j == "panel":
                    v = [d[x][0] for x in JUDGES if x in d]
                    return float(np.mean(v)) if v else None
                return d[j][0] if j in d else None
            if c["surface"] == "upnext":
                for a in {a for L in c["lists"].values() for a in L}:
                    g = g_of(a)
                    if g is not None:
                        (cit if a in set(c["linked"]) else unc).append(g)
            for a, liked in c.get("probe", {}).items():
                g = g_of(a)
                if g is not None:
                    (lik if liked else dis).append(g)
        checks["panel" if j == "panel" else JNAME[j]] = {"cited_vs_uncited_auc": auc(cit, unc), "cited_mean": float(np.mean(cit)) if cit else None,
                     "uncited_mean": float(np.mean(unc)) if unc else None, "n_cited": len(cit), "n_uncited": len(unc),
                     "liked_vs_disliked_auc": auc(lik, dis), "liked_mean": float(np.mean(lik)) if lik else None,
                     "disliked_mean": float(np.mean(dis)) if dis else None, "n_liked": len(lik), "n_disliked": len(dis)}
    checks["pairs"] = {f"{JNAME[a]}|{JNAME[b]}": agreement([(v[a][0], v[b][0]) for v in G.values() if a in v and b in v])
                       for a, b in combinations(JUDGES, 2)}
    checks["retest"] = {JNAME[j]: agreement([(v[j][0], v[(j[0], j[1] + "-retest")][0]) for v in G.values()
                                             if j in v and (j[0], j[1] + "-retest") in v]) for j in JUDGES}
    res["checks"] = checks
    res["overlap"] = {f"{a}|{b}": float(np.mean([len(set(c["lists"][a][:K]) & set(c["lists"][b][:K])) / K for c in cases]))
                      for a, b in combinations(models, 2)}
    res["cases"] = case_out
    used = {a for c in case_out for L in c["lists"].values() for a in L} | {c["seed"] for c in case_out if c.get("seed")} \
        | {a for c in case_out for a in c.get("context_liked", []) + c.get("context_disliked", []) + list(c.get("probe", {}))}
    res["papers"] = {a: {**papers[a], "abstract": papers[a]["abstract"][:420]} for a in used}
    (OUT / f"results{tag}_k{K}.json").write_text(json.dumps(res))
    log("judge", f"K={K}: scored {len(case_out)}/{len(cases)} cases; panels {res['panel_sizes']}")
    for sc in ("upnext", "home"):
        if sc in res["metrics"]:
            for k in ("ndcg", "good", "off"):
                v = res["metrics"][sc][k]
                log("judge", f"{sc:7} {k:5} n={v['n']:3} " + "  ".join(f"{m}:{x['mean']:.3f}{'*' if x['top_tier'] else ''}" for m, x in v["by"].items()))
    for j in [JNAME[x] for x in JUDGES] + ["panel"]:
        cj = checks[j]
        log("judge", f"check {j:22} cited AUC {cj['cited_vs_uncited_auc']} | liked AUC {cj['liked_vs_disliked_auc']} (n {cj['n_liked']}/{cj['n_disliked']})")
    log("judge", f"pairs {json.dumps(checks['pairs'])[:600]}")


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 and a[1].isdigit() else 5,
         a[a.index("--cases") + 1] if "--cases" in a else "cases.json",
         a[a.index("--base") + 1] if "--base" in a else "bge_m3",
         a[a.index("--tag") + 1] if "--tag" in a else "")
