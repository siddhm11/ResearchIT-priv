"""Live-index check of the planned split: BGE-M3 finds on the real 1.85M index, GTE-large orders.

    python live_cases.py fetch      # production Qdrant candidates (read-only), 180 cases
    python live_cases.py encode     # GTE-large vectors for candidates not cached (MPS)
    python live_cases.py lists      # "today" (BGE-M3 order) vs "planned" (GTE-large order), top 10
Up next: the seed paper's own vector, top 300. Home: the reader's long- and
short-term BGE-M3 profiles (app profile code, training votes), top 300 + top 150.
Writes CACHE/judge/live_cases.json in the cases.json format, judged by judge.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, SIDECAR, app_import_path, log  # noqa: E402

J = CACHE / "judge"


def unit(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def readers():
    """Same reader selection and training split as cases.py, keeping user ids."""
    m2 = json.loads((CACHE / "bakeoff" / "round2_meta.json").read_text())
    data = json.loads((J / "cases.json").read_text())
    want = sum(c["surface"] == "home" for c in data["cases"])
    r = pd.read_csv(CACHE / "scholar_inbox" / "rated_papers.csv", dtype={"arxiv_id": str, "user_id": str})
    r["time"] = pd.to_datetime(r["time"], format="mixed")
    # cases.py built histories from its library only: (round-2 set | AI/ML set) with all four model vectors
    lib = set(json.loads((CACHE / "bakeoff" / "round2_ids.json").read_text())) | \
        set(json.loads((CACHE / "aiml" / "aiml_universe.json").read_text())["ids"])
    for m in ("bge_m3", "gte_large", "qwen3_06b", "specter2"):
        have = set()
        for d in (CACHE / "bakeoff", CACHE / "aiml"):
            f = d / f"ids_{m}.json"
            if f.exists():
                have |= set(json.loads(f.read_text()))
        lib &= have
    r = r[r.arxiv_id.isin(lib)]
    home = [c for c in data["cases"] if c["surface"] == "home"]
    out = []
    for usr in m2["users"]:
        if len(out) == want:
            break
        mine = r[r.user_id == usr].sort_values("time")
        cut = int(len(mine) * 0.7)
        tr = mine.iloc[:cut]
        P = list(tr.arxiv_id[tr.rating > 0])
        if P[-8:][::-1] == home[len(out)]["context_liked"]:
            out.append({"user": usr, "P": P, "N": list(tr.arxiv_id[tr.rating <= 0]), "rated": sorted(set(mine.arxiv_id))})
    return out


def fetch() -> None:
    from dotenv import load_dotenv
    load_dotenv("/Users/siddh/Projects/researchit/.env.local")
    app_import_path()
    from app import qdrant_svc
    from app.recommend.profiles import ALPHA_LONG_TERM, ALPHA_SHORT_TERM, ewma_update
    data = json.loads((J / "cases.json").read_text())
    R = readers()
    log("live", f"matched {len(R)} readers")

    async def run():
        out = {}
        hi = iter(R)
        for c in data["cases"]:
            if c["surface"] == "upnext":
                v = (await qdrant_svc.get_paper_vectors([c["seed"]])).get(c["seed"])
                hits = await qdrant_svc.search_dense_merged(v.tolist(), limit=301)
                out[c["qid"]] = {"cands": [h["arxiv_id"] for h in hits if h["arxiv_id"] != c["seed"]][:300]}
            else:
                rd = next(hi)
                vecs = await qdrant_svc.get_paper_vectors(rd["P"] + rd["N"])
                lt = st = None
                for k, a in enumerate([a for a in rd["P"] if vecs.get(a) is not None]):
                    lt = ewma_update(lt, vecs[a], ALPHA_LONG_TERM, count=k, normalise=False)
                    st = ewma_update(st, vecs[a], ALPHA_SHORT_TERM, count=k, normalise=False)
                h1 = await qdrant_svc.search_dense_merged((lt / np.linalg.norm(lt)).tolist(), limit=300)
                h2 = await qdrant_svc.search_dense_merged((st / np.linalg.norm(st)).tolist(), limit=150)
                rated = set(rd["rated"])
                cands = list(dict.fromkeys(h["arxiv_id"] for h in h1 + h2 if h["arxiv_id"] not in rated))
                out[c["qid"]] = {"cands": cands, "user": rd["user"], "P": rd["P"], "N": rd["N"]}
            if len(out) % 20 == 0:
                log("live", f"fetched {len(out)}")
        return out
    t0 = time.time()
    out = asyncio.run(run())
    (J / "live_candidates.json").write_text(json.dumps(out))
    log("live", f"candidates for {len(out)} cases, {len({a for v in out.values() for a in v['cands']}):,} distinct papers, {time.time() - t0:.0f}s")


def vectors(name: str, need: set[str]) -> dict[str, np.ndarray]:
    got = {}
    for d in (CACHE / "bakeoff", CACHE / "aiml", CACHE / "search", J):
        p = d / f"vec_{name}.npy"
        if p.exists():
            ids = json.loads((d / f"ids_{name}.json").read_text())
            M = np.load(p, mmap_mode="r")
            for k, a in enumerate(ids):
                if a in need and a not in got:
                    got[a] = np.asarray(M[k], np.float32)
    return got


def encode() -> None:
    import sqlite3
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from select_pairs import csv_rows
    C = json.loads((J / "live_candidates.json").read_text())
    data = json.loads((J / "cases.json").read_text())
    need = {a for v in C.values() for a in v["cands"] + v.get("P", []) + v.get("N", [])} | \
        {c["seed"] for c in data["cases"] if c.get("seed")}
    have = vectors("gte_large", need)
    todo = sorted(need - set(have))
    log("live", f"gte_large: {len(have):,} cached, {len(todo):,} to encode")
    rows = csv_rows(todo)
    text = {i: f"{r['title'].strip()}. {(r['abstract'] or '').strip()}" for i, r in rows.items()}
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    rest = [i for i in todo if i not in text]
    for s in range(0, len(rest), 500):
        part = rest[s:s + 500]
        for a, t, ab in sc.execute(f"SELECT arxiv_id, title, abstract_preview FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})", part):
            text[a] = f"{(t or '').strip()}. {(ab or '').strip()}"
    todo = [i for i in todo if i in text]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("thenlper/gte-large", device=os.getenv("EMBEVAL_BASELINE_DEVICE", "mps"))
    M = []
    t0 = time.time()
    for s in range(0, len(todo), 2000):
        M.append(m.encode([" ".join(text[i].split()) for i in todo[s:s + 2000]], batch_size=32, normalize_embeddings=True))
        log("live", f"gte_large {s + len(M[-1]):,}/{len(todo):,} ({time.time() - t0:.0f}s)")
    if todo:
        np.save(J / "vec_gte_large.npy", np.concatenate(M).astype(np.float16))
        (J / "ids_gte_large.json").write_text(json.dumps(todo))


def lists() -> None:
    from dotenv import load_dotenv
    load_dotenv("/Users/siddh/Projects/researchit/.env.local")
    app_import_path()
    from app import qdrant_svc
    from app.recommend.profiles import ALPHA_LONG_TERM, ALPHA_NEGATIVE, ALPHA_SHORT_TERM, ewma_update
    C = json.loads((J / "live_candidates.json").read_text())
    data = json.loads((J / "cases.json").read_text())
    need = {a for v in C.values() for a in v["cands"] + v.get("P", []) + v.get("N", [])} | \
        {c["seed"] for c in data["cases"] if c.get("seed")}
    G = vectors("gte_large", need)
    B = asyncio.run(qdrant_svc.get_paper_vectors(sorted(need)))
    papers = dict(data["papers"])
    lib = json.loads((CACHE / "search" / "library.json").read_text())
    import sqlite3
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    cases = []
    for c in data["cases"]:
        v = C[c["qid"]]
        cand = [a for a in v["cands"] if a in G and B.get(a) is not None]
        lists_ = {}
        for name, V in (("today", B), ("planned", G)):
            X = np.stack([np.asarray(V[a], np.float32) / np.linalg.norm(V[a]) for a in cand])
            if c["surface"] == "upnext":
                q = np.asarray(V[c["seed"]], np.float32)
                s = X @ (q / np.linalg.norm(q))
            else:
                lt = st = ng = None
                P = [a for a in v["P"] if V.get(a) is not None]
                N = [a for a in v["N"] if V.get(a) is not None]
                for k, a in enumerate(P):
                    lt = ewma_update(lt, np.asarray(V[a], np.float32), ALPHA_LONG_TERM, count=k, normalise=False)
                    st = ewma_update(st, np.asarray(V[a], np.float32), ALPHA_SHORT_TERM, count=k, normalise=False)
                for k, a in enumerate(N):
                    ng = ewma_update(ng, np.asarray(V[a], np.float32), ALPHA_NEGATIVE, count=k, normalise=False)
                s = 0.40 * unit(X @ (lt / np.linalg.norm(lt))) + 0.25 * unit(X @ (st / np.linalg.norm(st)))
                if ng is not None:
                    s = s - 0.15 * np.clip(X @ (ng / np.linalg.norm(ng)), 0, None)
            lists_[name] = [cand[j] for j in np.argsort(-s)[:10]]
        cases.append({**{k: c[k] for k in c if k not in ("lists", "linked")}, "lists": lists_,
                      "linked": [a for a in c.get("linked", [])]})
        for a in set(sum(lists_.values(), [])) - set(papers):
            papers[a] = None
    missing = [a for a, p in papers.items() if p is None]
    for s in range(0, len(missing), 500):
        part = missing[s:s + 500]
        for a, t, ab, cats in sc.execute(f"SELECT arxiv_id, title, abstract_preview, categories FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})", part):
            papers[a] = {"title": " ".join((lib["titles"].get(a) or t or "").split()),
                         "abstract": " ".join((lib["abstracts"].get(a) or ab or "").split()),
                         "cat": (cats or "").split()[0] if cats else "", "year": None}
    papers = {a: p for a, p in papers.items() if p}
    (J / "live_cases.json").write_text(json.dumps({"models": ["today", "planned"], "cases": cases, "papers": papers}))
    overlap = np.mean([len(set(c["lists"]["today"][:5]) & set(c["lists"]["planned"][:5])) / 5 for c in cases])
    log("live", f"wrote live_cases.json: {len(cases)} cases; top-5 overlap today vs planned {overlap:.2f}")


if __name__ == "__main__":
    {"fetch": fetch, "encode": encode, "lists": lists}[sys.argv[1]]()
