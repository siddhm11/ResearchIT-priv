"""Real before/after examples for the upgrade plan (read-only, deterministic).

For each proposed step, show what a reader would actually see before and after,
using production vectors and the app's own scoring functions. Examples are
chosen by a fixed rule (the first case in seed order that meets it), and every
example is reported next to the all-readers counts so it can't be cherry-picked.

  step2  one simulated reader's top 10, today's formula vs similarities rescaled;
         plus how many of all readers improve / worsen / stay the same.
  step3  ages and recency scores for revised classics, update_date vs arXiv ID.
  step4  a two-interest reader: the minority interest's top 5 when scored against
         the overall profile (today) vs against that interest's own representative.
  step5  Attention Is All You Need: its nearest text neighbours vs the papers that
         cite it, with the latters' ranks in text search.
  step6  seeds where BGE-M3 prefers a keyword lookalike over a cited paper, with the
         same comparison under SPECTER2.
"""
from __future__ import annotations

import asyncio
import json
import random
import sqlite3

import numpy as np

from common import OUT, SEED, SIDECAR, app_import_path, log
from scaled_eval import (N_SEED_CANDIDATES, N_SEEDS, arxiv_ids, domain, primary, s2_batch,
                         stored)

app_import_path()
STAGE = "examples"
NOW = np.datetime64("2026-10-09")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    from app import local_meta, qdrant_svc
    from app.recommend.profiles import ALPHA_LONG_TERM, ALPHA_SHORT_TERM, ewma_update
    from app.recommend.reranker import _unit_spread, heuristic_score

    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    rows = sc.execute("SELECT arxiv_id, title, categories, update_date FROM papers").fetchall()
    meta = {r[0]: {"title": r[1] or "", "cat": primary(r[2]), "upd": r[3]} for r in rows}
    all_ids = sorted(meta)

    def pub(aid):
        ym = local_meta.pub_year_month(aid)
        return f"{ym[0]:04d}-{ym[1]:02d}" if ym else None

    def age(d):
        return float((NOW - np.datetime64(d[:10] if len(d) > 7 else d + "-15")) / np.timedelta64(1, "D"))

    # Same seeds and links as scaled_eval.py (all cached).
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
    out: dict = {}

    def feat(ids, hits, cv, lt_n, st_n, recency_from):
        E = np.stack([cv[i] / np.linalg.norm(cv[i]) for i in ids])
        F = np.zeros((len(ids), 37), np.float32)
        score = {h["arxiv_id"]: h["score"] for h in hits}
        F[:, 0] = [score[i] for i in ids]
        F[:, 1] = np.arange(len(ids))
        F[:, 20], F[:, 21] = E @ lt_n, E @ st_n
        if recency_from == "update_date":
            F[:, 6] = [np.exp(-0.002 * age(meta[i]["upd"] or "2016-01-01")) for i in ids]
        else:
            F[:, 6] = [np.exp(-0.002 * age(pub(i) or "2016-01")) for i in ids]
        F[:, 35] = 1.0 / (F[:, 1] + 1)
        return F

    def row(i, held, s=None):
        return {"id": i, "title": meta.get(i, {}).get("title", "")[:110], "pub": pub(i),
                "revised": (meta.get(i, {}).get("upd") or "")[:7], "held_out": i in held,
                "score": None if s is None else float(s)}

    # ── step 2 (replays scaled_eval's E4 readers in the same order) ──────────
    bge_seed = asyncio.run(stored(seeds))
    rrnd = random.Random(SEED)
    tally = {"improved": 0, "worsened": 0, "same": 0}
    example2 = None
    for s in seeds:
        linked = [x for x in links[s]["refs"] + links[s]["cits"]]
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
        ids = [h["arxiv_id"] for h in hits]
        if not (set(ids) & held):
            continue
        cv = asyncio.run(stored(ids))
        ids = [i for i in ids if i in cv]
        F = feat(ids, hits, cv, lt_n, st_n, "update_date")
        prod = heuristic_score(F)
        G = F.copy()
        G[:, 20], G[:, 21] = _unit_spread(F[:, 20]), _unit_spread(F[:, 21])
        resc = heuristic_score(G)
        top = lambda sc_: [ids[j] for j in np.argsort(-sc_)[:10]]  # noqa: E731
        hb, ha = bool(set(top(prod)) & held), bool(set(top(resc)) & held)
        tally["improved" if ha and not hb else "worsened" if hb and not ha else "same"] += 1
        if example2 is None and ha and not hb:
            ob, oa = np.argsort(-prod), np.argsort(-resc)
            first_b = next(k for k, j in enumerate(ob) if ids[j] in held) + 1
            first_a = next(k for k, j in enumerate(oa) if ids[j] in held) + 1
            example2 = {"seed": row(s, held), "saves": [row(x, held) for x in saves],
                        "before": [row(ids[j], held, prod[j]) | {"recency": float(F[j, 6]), "sim": float(F[j, 20])} for j in ob[:10]],
                        "after": [row(ids[j], held, resc[j]) | {"recency": float(F[j, 6]), "sim": float(F[j, 20])} for j in oa[:10]],
                        "first_held_out_rank_before": first_b, "first_held_out_rank_after": first_a,
                        "n_held_out_in_pool": int(len(set(ids) & held))}
    out["step2"] = {"example": example2, "tally_hit_at_10": tally}
    log(STAGE, f"step2 tally {tally}")

    # ── step 3: revised classics ─────────────────────────────────────────────
    s3 = []
    for aid in ["1706.03762", "1409.0473", "1512.03385", "1412.6980", "1810.04805"]:
        if aid in meta:
            u, p = meta[aid]["upd"], pub(aid)
            s3.append({"id": aid, "title": meta[aid]["title"], "revised": u, "published": p,
                       "recency_today": float(np.exp(-0.002 * age(u))),
                       "recency_fixed": float(np.exp(-0.002 * age(p)))})
    out["step3"] = s3

    # ── step 4: a two-interest reader ────────────────────────────────────────
    comp = [s for s in seeds if domain(meta[s]["cat"]) == "computing" and s in bge_seed]
    other = [s for s in seeds if domain(meta[s]["cat"]) != "computing" and s in bge_seed]
    example4 = None
    for x in comp:
        lx = [i for i in sorted(links[x]["refs"] + links[x]["cits"])]
        vx = asyncio.run(stored(lx))
        lx = [i for i in lx if i in vx][:4]
        if len(lx) < 4:
            continue
        for y in other:
            ly = sorted(links[y]["refs"] + links[y]["cits"])
            vy = asyncio.run(stored(ly))
            ly = [i for i in ly if i in vy]
            if len(ly) < 5:
                continue
            saves_major, saves_minor = [x] + lx, [y, ly[0]]
            held = set(ly[1:])
            allv = {**vx, **vy, x: bge_seed[x], y: bge_seed[y]}
            lt = st = None
            for k, a in enumerate(saves_major + saves_minor):
                lt = ewma_update(lt, allv[a], ALPHA_LONG_TERM, count=k, normalise=False)
                st = ewma_update(st, allv[a], ALPHA_SHORT_TERM, count=k, normalise=False)
            lt_n, st_n = lt / np.linalg.norm(lt), st / np.linalg.norm(st)
            rep = allv[y] + allv[ly[0]]
            rep = rep / np.linalg.norm(rep)
            hits = asyncio.run(qdrant_svc.search_by_vector_with_scores(
                rep.tolist(), 30, exclude_ids=set(saves_major + saves_minor)))
            ids = [h["arxiv_id"] for h in hits]
            cv = asyncio.run(stored(ids))
            ids = [i for i in ids if i in cv]
            if not (set(ids) & held):
                continue
            F = feat(ids, hits, cv, lt_n, st_n, "pub")
            G = F.copy()
            G[:, 20], G[:, 21] = _unit_spread(F[:, 20]), _unit_spread(F[:, 21])
            today = heuristic_score(G)                       # steps 2+3 applied
            H = G.copy()
            H[:, 20] = _unit_spread(F[:, 0])                 # relevance to this interest,
            H[:, 21] = H[:, 20]                              # same 0.65 relevance budget
            per_interest = heuristic_score(H)
            ob, oa = np.argsort(-today)[:5], np.argsort(-per_interest)[:5]
            if [ids[j] for j in ob] == [ids[j] for j in oa]:
                continue
            example4 = {"major": [row(i, held) for i in saves_major], "minor": [row(i, held) for i in saves_minor],
                        "before": [row(ids[j], held, today[j]) | {"cat": meta[ids[j]]["cat"]} for j in ob],
                        "after": [row(ids[j], held, per_interest[j]) | {"cat": meta[ids[j]]["cat"]} for j in oa],
                        "major_cat": meta[x]["cat"], "minor_cat": meta[y]["cat"]}
            break
        if example4:
            break
    out["step4"] = example4
    log(STAGE, f"step4 found: {bool(example4)}")

    # ── step 5: Attention's text neighbours vs its citing papers ─────────────
    att = "1706.03762"
    va = asyncio.run(stored([att]))[att]
    nn = asyncio.run(qdrant_svc.search_by_vector_with_scores(va.tolist(), 6))
    ir = json.loads((OUT / "results.json").read_text())["retrieval"][att]
    out["step5"] = {"text_neighbours": [row(h["arxiv_id"], set(), h["score"]) for h in nn if h["arxiv_id"] != att][:5],
                    "citation_neighbours": [
                        {**row(c, set()), "dense_rank": ir["dense"][c]["rank"],
                         "rank_at_least": (ir.get("dense_rank_lower_bounds", {}).get(c) or {}).get("rank_at_least")}
                        for c in ["1810.04805", "1409.0473", "1905.09856"]]}

    # ── step 6: keyword lookalikes, BGE-M3 vs SPECTER2 ──────────────────────
    sres = json.loads((OUT / "scaled_results.json").read_text())
    traps = sres["trap_examples"]
    ids6 = sorted({t["seed"] for t in traps} | {t["trap"] for t in traps}
                  | {l for t in traps for l in (links[t["seed"]]["refs"] + links[t["seed"]]["cits"])[:5]})
    b6 = asyncio.run(stored(ids6))
    s6 = s2_batch(ids6, "embedding.specter_v2")
    sp6 = {i: np.asarray(r["embedding"]["vector"], np.float32) for i, r in s6.items()
           if r and r.get("embedding") and r["embedding"].get("vector")}

    def cos(V, a, b):
        x, y = V[a], V[b]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    ex6 = []
    for t in traps:
        s, tr = t["seed"], t["trap"]
        for l in (links[s]["refs"] + links[s]["cits"])[:5]:
            if not all(k in b6 and k in sp6 for k in (s, tr, l)):
                continue
            bt, bl = cos(b6, s, tr), cos(b6, s, l)
            st_, sl = cos(sp6, s, tr), cos(sp6, s, l)
            if bt > bl and sl > st_:
                ex6.append({"seed": row(s, set()), "cited": row(l, set()), "lookalike": row(tr, set()),
                            "word": t["word"], "lookalike_cat": t["trap_cat"], "seed_cat": meta[s]["cat"],
                            "bge": {"cited": bl, "lookalike": bt}, "specter2": {"cited": sl, "lookalike": st_}})
                break
        if len(ex6) == 3:
            break
    out["step6"] = ex6
    (OUT / "before_after_examples.json").write_text(json.dumps(out, indent=1, default=float))
    log(STAGE, f"wrote {OUT / 'before_after_examples.json'}")


if __name__ == "__main__":
    main()
