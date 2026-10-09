"""Stage 6 (optional): the audit at scale, with citation links as ground truth.

The one-pair audit showed *that* BGE-M3 can be pulled by a shared keyword and
that a crowded neighbourhood hides relatives. This stage measures *how often*,
over a few hundred seed papers, and checks the reranker's score mix on real
candidate pools. Read-only against production; Semantic Scholar is read with
the S2_API_KEY from .env.local at <= 1 request/second and cached in CACHE.

Experiments
  E1  citation triplets   P(cos(seed, linked) > cos(seed, random same-category))
                          linked = a paper the seed cites or is cited by.
  E2  lexical traps       P(cos(seed, trap) > cos(seed, in-domain control))
                          trap    = shares a distinctive title word with the seed
                                    but sits in a different scientific domain;
                          control = same archive as the seed, different primary
                                    category, no shared title word.
  E3  crowding            share of a seed's linked papers that appear among its
                          100 nearest neighbours in the production index.
  E4  reranker mix        synthetic users (seed + 4 linked papers as saves);
                          candidates = top-100 around the long-term profile; the
                          held-out linked papers are the "right answers". Compare
                          the production heuristic with variants.

Scorers for E1/E2: BGE-M3 stored vectors (production), BGE-M3 mean-centred,
SPECTER2 vectors served by the Semantic Scholar API, all-MiniLM-L6-v2.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sqlite3
import time
import urllib.request

import numpy as np

from common import CACHE, OUT, SEED, SIDECAR, WORK, app_import_path, log

app_import_path()
os.environ.setdefault("METADATA_SIDECAR_PATH", str(SIDECAR))
STAGE = "scaled"
N_SEED_CANDIDATES = 1200
N_SEEDS = 300
S2_CACHE = CACHE / "s2_cache.json"
DOMAIN = {"cs": "computing", "stat": "computing", "eess": "computing", "math": "math",
          "q-bio": "life", "q-fin": "econ", "econ": "econ"}
STOP = set("""about above across after against along among another approach approaches
based being between beyond method methods model models modeling modelling network networks
learning analysis results result system systems theory theoretical towards toward using study
studies problem problems general generalized efficient effective application applications new
novel improved improving framework structure structures dynamics properties large small high
low first second order case cases data dimensional function functions""".split())


def archive(cat: str) -> str:
    return cat.split(".")[0] if cat else ""


def domain(cat: str) -> str:
    return DOMAIN.get(archive(cat), "physics")


def primary(categories: str | None) -> str:
    return (categories or "").split()[0] if categories else ""


# ── Semantic Scholar (read-only, cached, throttled) ──────────────────────────

def s2_batch(ids: list[str], fields: str) -> dict[str, dict]:
    cache = json.loads(S2_CACHE.read_text()) if S2_CACHE.exists() else {}
    key = os.environ.get("S2_API_KEY", "")
    out, todo = {}, []
    for i in ids:
        k = f"{fields}|{i}"
        if k in cache:
            out[i] = cache[k]
        else:
            todo.append(i)
    if todo:
        log(STAGE, f"S2 {fields}: {len(todo):,} to fetch, {len(ids) - len(todo):,} cached")
    try:
        for n, s in enumerate(range(0, len(todo), 100)):
            part = todo[s:s + 100]
            req = urllib.request.Request(
                f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={fields}",
                data=json.dumps({"ids": [f"ARXIV:{i}" for i in part]}).encode(),
                headers={"x-api-key": key, "Content-Type": "application/json"}, method="POST")
            for attempt in range(12):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        rows = json.loads(r.read())
                    break
                except Exception as e:  # noqa: BLE001 -- S2 throttles with 429
                    if attempt == 11:
                        raise
                    wait = min(60, 5 * 2 ** attempt)
                    log(STAGE, f"S2 retry {attempt + 1} in {wait}s: {e}")
                    time.sleep(wait)
            for i, row in zip(part, rows):
                out[i] = row
                cache[f"{fields}|{i}"] = row
            if n % 5 == 4:
                S2_CACHE.write_text(json.dumps(cache))
            time.sleep(1.5)
    finally:
        if todo:
            S2_CACHE.write_text(json.dumps(cache))
    return out


def arxiv_ids(lst) -> list[str]:
    return [x["externalIds"]["ArXiv"] for x in (lst or [])
            if x and x.get("externalIds") and x["externalIds"].get("ArXiv")]


# ── text for the local models ────────────────────────────────────────────────

def texts_for(ids: list[str], sc: sqlite3.Connection) -> dict[str, str]:
    from select_pairs import csv_rows  # original CSV text for snapshot papers
    rows = csv_rows(ids)
    out = {i: f"{r['title'].strip()} {r['abstract'].strip()}" for i, r in rows.items()}
    rest = [i for i in ids if i not in out]
    for s in range(0, len(rest), 500):
        part = rest[s:s + 500]
        q = f"SELECT arxiv_id, title, abstract_preview FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})"
        for a, t, ab in sc.execute(q, part):
            out[a] = f"{t} {ab or ''}"
    return out


async def stored(ids: list[str]) -> dict[str, np.ndarray]:
    from app import qdrant_svc
    vecs = {}
    for s in range(0, len(ids), 200):
        vecs.update(await qdrant_svc.get_paper_vectors(ids[s:s + 200]))
    return vecs


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def rate(x: list[bool], rng) -> dict:
    a = np.asarray(x, float)
    lo, hi = boot_ci(a, rng)
    return {"rate": float(a.mean()) if len(a) else None, "ci95": [lo, hi], "n": int(len(a))}


# ── main ─────────────────────────────────────────────────────────────────────

def build_examples() -> dict:
    """Seeds, citation links and per-seed test triplets (deterministic, seed 42)."""
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    rows = sc.execute("SELECT arxiv_id, title, categories, update_date FROM papers").fetchall()
    meta = {r[0]: {"title": r[1] or "", "cat": primary(r[2]), "upd": r[3]} for r in rows}
    by_cat: dict[str, list[str]] = {}
    for a, m in meta.items():
        by_cat.setdefault(m["cat"], []).append(a)
    all_ids = sorted(meta)

    # Seeds: random corpus papers with >= 6 linked papers (cited or citing) in the corpus.
    cand = random.Random(SEED).sample(all_ids, N_SEED_CANDIDATES)
    links_raw = s2_batch(cand, "references.externalIds,citations.externalIds")
    seeds, links = [], {}
    for a in cand:
        row = links_raw.get(a) or {}
        refs = [x for x in arxiv_ids(row.get("references")) if x in meta and x != a]
        cits = [x for x in arxiv_ids(row.get("citations")) if x in meta and x != a]
        if len(set(refs) | set(cits)) >= 6:
            seeds.append(a)
            links[a] = {"refs": sorted(set(refs)), "cits": sorted(set(cits))}
        if len(seeds) == N_SEEDS:
            break
    log(STAGE, f"seeds with >=6 in-corpus links: {len(seeds)} of {cand.index(seeds[-1]) + 1} drawn")

    # Per-seed examples.
    rnd = random.Random(SEED)
    word = re.compile(r"[a-z]{6,}")
    ex = {}
    for s in seeds:
        linked = set(links[s]["refs"]) | set(links[s]["cits"])
        cat = meta[s]["cat"]
        e = {"linked": sorted(linked)}
        pool = [x for x in by_cat.get(cat, []) if x not in linked and x != s]
        e["same_cat"] = rnd.sample(pool, min(3, len(pool)))
        # trap: distinctive title word, different domain
        toks = [t for t in word.findall(meta[s]["title"].lower()) if t not in STOP]
        trap = None
        for t in sorted(set(toks)):
            hits = [r[0] for r in sc.execute(
                "SELECT p.arxiv_id FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid "
                "WHERE papers_fts MATCH ? LIMIT 3000", (f'title:"{t}"',))]
            if not (20 <= len(hits) < 3000):
                continue
            far = [h for h in hits if h in meta and domain(meta[h]["cat"]) != domain(cat)
                   and h not in linked and h != s]
            if far:
                trap = {"id": rnd.choice(sorted(far)), "word": t}
                break
        # control: same archive, different primary category, no shared title token
        stoks = set(re.findall(r"[a-z]{4,}", meta[s]["title"].lower()))
        ctrl_pool = [c for c in by_cat if archive(c) == archive(cat) and c != cat and c]
        ctrl = None
        for _ in range(30):
            if not ctrl_pool:
                break
            c = rnd.choice(sorted(ctrl_pool))
            x = rnd.choice(by_cat[c])
            if x not in linked and not (stoks & set(re.findall(r"[a-z]{4,}", meta[x]["title"].lower()))):
                ctrl = x
                break
        e["trap"], e["control"] = trap, ctrl
        ex[s] = e

    need = sorted({s for s in seeds} | {x for e in ex.values() for x in e["linked"] + e["same_cat"]}
                  | {e["trap"]["id"] for e in ex.values() if e["trap"]}
                  | {e["control"] for e in ex.values() if e["control"]})
    log(STAGE, f"papers involved: {len(need):,}")

    return {"sc": sc, "meta": meta, "seeds": seeds, "links": links, "ex": ex, "need": need}


def main() -> None:
    b = build_examples()
    sc, meta, seeds, links, ex, need = b["sc"], b["meta"], b["seeds"], b["links"], b["ex"], b["need"]
    # Vectors: BGE-M3 stored, SPECTER2 from S2, MiniLM local.
    bge = asyncio.run(stored(need))
    s2e = s2_batch(need, "embedding.specter_v2")
    spec = {i: np.asarray(r["embedding"]["vector"], np.float32) for i, r in s2e.items()
            if r and r.get("embedding") and r["embedding"].get("vector")}
    from sentence_transformers import SentenceTransformer
    mini_m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                 device=os.getenv("EMBEVAL_BASELINE_DEVICE", "cpu"))
    txt = texts_for(need, sc)
    mids = [i for i in need if i in txt]
    mv = mini_m.encode([txt[i] for i in mids], normalize_embeddings=True, batch_size=64)
    mini = dict(zip(mids, mv))
    # Corpus mean for centring, from the random-pair baseline's stored vectors.
    bl = json.loads((WORK / "baseline.json").read_text())
    base_ids = sorted({x for p in bl["random"] for x in p})
    base_vecs = asyncio.run(stored(base_ids))
    mu = np.mean(np.stack([v / np.linalg.norm(v) for v in base_vecs.values()]), axis=0)
    centred = {i: v / np.linalg.norm(v) - mu for i, v in bge.items()}
    scorers = {"bge_m3_stored": bge, "bge_m3_centred": centred, "specter2_s2api": spec, "minilm": mini}
    cover = {k: len([i for i in need if i in v]) for k, v in scorers.items()}
    log(STAGE, f"vector coverage: {cover}")

    def cos(V, a, b):
        if a not in V or b not in V:
            return None
        x, y = V[a], V[b]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    res = {"n_seeds": len(seeds), "papers": len(need), "coverage": cover,
           "mean_norm_of_corpus_mean_vector": float(np.linalg.norm(mu)), "scorers": {}}
    rng = np.random.default_rng(SEED)
    for name, V in scorers.items():
        trip, trap_win, cited_vs_trap = [], [], []
        for s in seeds:
            e = ex[s]
            for l in e["linked"][:5]:
                for n in e["same_cat"]:
                    a, b = cos(V, s, l), cos(V, s, n)
                    if a is not None and b is not None:
                        trip.append(a > b)
            if e["trap"] and e["control"]:
                t, c = cos(V, s, e["trap"]["id"]), cos(V, s, e["control"])
                if t is not None and c is not None:
                    trap_win.append(t > c)
                    for l in e["linked"][:5]:
                        lc = cos(V, s, l)
                        if lc is not None:
                            cited_vs_trap.append(lc > t)
        res["scorers"][name] = {"E1_linked_beats_same_category": rate(trip, rng),
                                "E2_trap_beats_in_domain_control": rate(trap_win, rng),
                                "E2b_linked_beats_trap": rate(cited_vs_trap, rng)}
        log(STAGE, f"{name}: {res['scorers'][name]}")
    res["trap_examples"] = [{"seed": s, "seed_title": meta[s]["title"], "word": ex[s]["trap"]["word"],
                             "trap": ex[s]["trap"]["id"], "trap_title": meta[ex[s]["trap"]["id"]]["title"],
                             "trap_cat": meta[ex[s]["trap"]["id"]]["cat"], "control": ex[s]["control"],
                             "control_title": meta[ex[s]["control"]]["title"]}
                            for s in seeds if ex[s]["trap"] and ex[s]["control"]][:12]

    # E3/E4: production retrieval + reranker mix.
    res.update(retrieval_and_rerank(seeds, links, meta, bge, rng))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scaled_results.json").write_text(json.dumps(res, indent=1, default=float))
    log(STAGE, f"wrote {OUT / 'scaled_results.json'}")


def retrieval_and_rerank(seeds, links, meta, bge, rng) -> dict:
    from app import local_meta, qdrant_svc
    from app.recommend.profiles import ALPHA_LONG_TERM, ALPHA_SHORT_TERM, ewma_update
    from app.recommend.reranker import heuristic_score, _unit_spread
    now = np.datetime64("2026-10-09")

    def age_from_date(d):
        try:
            return float((now - np.datetime64(d[:10])) / np.timedelta64(1, "D"))
        except Exception:  # noqa: BLE001
            return 3650.0

    def age_from_id(aid):
        ym = local_meta.pub_year_month(aid)
        if not ym:
            return 3650.0
        return float((now - np.datetime64(f"{ym[0]:04d}-{ym[1]:02d}-15")) / np.timedelta64(1, "D"))

    rrnd = random.Random(SEED)
    e3_hits, e3_n = [], 0
    variants = {"production": [], "rescaled_cosines": [], "rescaled+pub_date": [], "relevance_only": []}
    spreads = {"0.40*lt": [], "0.25*st": [], "0.15*recency": [], "0.10*position": []}
    rho_lt, rho_rec = [], []
    users = 0
    for s in seeds:
        linked = links[s]["refs"] + links[s]["cits"]
        linked = [x for x in linked if x in bge]
        if s not in bge or len(linked) < 6:
            continue
        # E3: are the seed's linked papers among its 100 nearest neighbours?
        nn = asyncio.run(qdrant_svc.search_by_vector_with_scores(bge[s].tolist(), 101))
        nn_ids = {h["arxiv_id"] for h in nn if h["arxiv_id"] != s}
        e3_hits.append(len(nn_ids & set(linked)) / len(linked))
        e3_n += 1
        # E4: a user who saved the seed + 4 linked papers; the rest are held out.
        lk = sorted(linked)
        rrnd.shuffle(lk)
        saves, held = [s] + lk[:4], set(lk[4:])
        lt = st = None
        for k, a in enumerate(saves):
            lt = ewma_update(lt, bge[a], ALPHA_LONG_TERM, count=k, normalise=False)
            st = ewma_update(st, bge[a], ALPHA_SHORT_TERM, count=k, normalise=False)
        lt_n, st_n = lt / np.linalg.norm(lt), st / np.linalg.norm(st)
        hits = asyncio.run(qdrant_svc.search_by_vector_with_scores(lt_n.tolist(), 100, exclude_ids=set(saves)))
        ids = [h["arxiv_id"] for h in hits]
        if not (set(ids) & held):
            continue  # nothing to find in this pool; ordering cannot be judged
        cv = asyncio.run(stored(ids))
        ids = [i for i in ids if i in cv]
        E = np.stack([cv[i] / np.linalg.norm(cv[i]) for i in ids])
        F = np.zeros((len(ids), 37), np.float32)
        F[:, 0] = [h["score"] for h in hits if h["arxiv_id"] in cv]
        F[:, 1] = np.arange(len(ids))
        F[:, 20], F[:, 21] = E @ lt_n, E @ st_n
        F[:, 6] = [np.exp(-0.002 * age_from_date(meta.get(i, {}).get("upd") or "")) for i in ids]
        F[:, 35] = 1.0 / (F[:, 1] + 1)
        users += 1
        target = np.array([i in held for i in ids])

        def mrr_hit(score):
            order = np.argsort(-score)
            ranks = np.where(target[order])[0]
            return (1.0 / (ranks[0] + 1), float(ranks[0] < 10))

        prod = heuristic_score(F)
        G = F.copy()
        G[:, 20], G[:, 21] = _unit_spread(F[:, 20]), _unit_spread(F[:, 21])
        resc = heuristic_score(G)
        H = G.copy()
        H[:, 6] = [np.exp(-0.002 * age_from_id(i)) for i in ids]
        resc_pub = heuristic_score(H)
        variants["production"].append(mrr_hit(prod))
        variants["rescaled_cosines"].append(mrr_hit(resc))
        variants["rescaled+pub_date"].append(mrr_hit(resc_pub))
        variants["relevance_only"].append(mrr_hit(F[:, 20]))
        spreads["0.40*lt"].append(0.40 * float(np.std(F[:, 20])))
        spreads["0.25*st"].append(0.25 * float(np.std(F[:, 21])))
        spreads["0.15*recency"].append(0.15 * float(np.std(F[:, 6])))
        spreads["0.10*position"].append(0.10 * float(np.std(F[:, 35])))
        from scipy.stats import spearmanr
        rho_lt.append(float(spearmanr(prod, F[:, 20]).statistic))
        rho_rec.append(float(spearmanr(prod, F[:, 6]).statistic))
    out = {"E3_linked_in_top100": {"mean_share": float(np.mean(e3_hits)),
                                   "ci95": list(boot_ci(np.asarray(e3_hits), rng)), "n_seeds": e3_n,
                                   "share_of_seeds_with_none": float(np.mean(np.asarray(e3_hits) == 0))},
           "E4": {"n_users": users, "variants": {}, "term_spread_mean": {k: float(np.mean(v)) for k, v in spreads.items()},
                  "spearman_production_vs_lt_mean": float(np.mean(rho_lt)),
                  "spearman_production_vs_recency_mean": float(np.mean(rho_rec))}}
    for k, v in variants.items():
        a = np.asarray(v, float)
        out["E4"]["variants"][k] = {"MRR": float(a[:, 0].mean()), "MRR_ci95": list(boot_ci(a[:, 0], rng)),
                                    "hit@10": float(a[:, 1].mean()), "hit@10_ci95": list(boot_ci(a[:, 1], rng))}
    log(STAGE, f"E3 {out['E3_linked_in_top100']}")
    log(STAGE, f"E4 {json.dumps(out['E4'], default=float)[:800]}")
    return out


if __name__ == "__main__":
    main()
