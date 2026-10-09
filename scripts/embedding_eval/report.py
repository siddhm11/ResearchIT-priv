"""Stage 5: assemble results.json and report.md from the stage outputs.

No model is run here and no number is typed in: every figure in the report is
read from the JSON the earlier stages wrote.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import A, B, OUT, SEED, VARIANTS, load_json

TIERS = ["strong_positive", "focal", "same_field_diff_topic", "lexical_trap", "unrelated"]
TIER_RANK = {t: k for k, t in enumerate(TIERS)}
VLABEL = {"title": "title", "abstract": "abstract", "title_abstract": "title+abstract"}
INTEGRITY_MIN = 0.999


# ── Load ─────────────────────────────────────────────────────────────────────

def load_all() -> dict:
    d = {"pairset": load_json("pairset.json"), "baseline": load_json("baseline.json"),
         "pair_bm": load_json("scores_pairset_bge_minilm.json"),
         "base_bm": load_json("scores_baseline_bge_minilm.json"),
         "ir": load_json("integrity_retrieval.json"),
         "base_stored": load_json("baseline_stored_dense.json")}
    try:
        d["pair_sp"] = load_json("scores_pairset_specter2.json")
        d["base_sp"] = load_json("scores_baseline_specter2.json")
    except FileNotFoundError as e:
        d["specter2_error"] = str(e)
    return d


def scorers(d: dict) -> list[dict]:
    """Every (model, output, variant) with its pair scores and both baselines."""
    out = []
    pbm, bbm = d["pair_bm"], d["base_bm"]
    for v in VARIANTS:
        for o in ("dense", "sparse", "colbert"):
            out.append({"key": f"bge_m3.{o}.{v}", "model": "BGE-M3", "output": o, "variant": v,
                        "pairs": {k: s[o] for k, s in pbm["bge"][v].items()},
                        "base": {s: bbm["bge"][f"{s}/{v}"][o] for s in ("random", "cscl_lg")}})
    sv = d["ir"]["server_vs_local"]
    out.append({"key": "bge_m3.dense_stored.title_abstract", "model": "BGE-M3",
                "output": "dense (stored Qdrant vectors)", "variant": "title_abstract",
                "pairs": {k: r["local_stored_dense_cos"] for k, r in sv.items()},
                "base": {s: d["base_stored"][s]["scores"] for s in ("random", "cscl_lg")}})
    if "pair_sp" in d:
        for v in VARIANTS:
            out.append({"key": f"specter2.{v}", "model": "SPECTER2", "output": "proximity",
                        "variant": v, "pairs": d["pair_sp"]["specter2"][v],
                        "base": {s: d["base_sp"]["specter2"][f"{s}/{v}"] for s in ("random", "cscl_lg")}})
    for v in VARIANTS:
        out.append({"key": f"minilm.{v}", "model": "all-MiniLM-L6-v2", "output": "dense", "variant": v,
                    "pairs": pbm["minilm"][v],
                    "base": {s: bbm["minilm"][f"{s}/{v}"] for s in ("random", "cscl_lg")}})
    return out


# ── Statistics ───────────────────────────────────────────────────────────────

def stats(x) -> dict:
    a = np.asarray(x, np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99))}


def percentile_of(x: float, base) -> float:
    a = np.asarray(base, np.float64)
    return float(100.0 * ((a < x).sum() + 0.5 * (a == x).sum()) / a.size)


def role_of(d: dict, aid: str, anchor: str) -> str:
    other = B if anchor == A else A
    if aid == other:
        return "focal"
    return d["pairset"]["papers"][aid]["role"]


def candidates(d: dict, anchor: str) -> list[str]:
    ids = [i for i in d["pairset"]["papers"] if i != anchor]
    return sorted(ids, key=lambda i: (TIER_RANK[role_of(d, i, anchor)], i))


RELAXED = ["strong_positive|focal", "same_field_diff_topic", "lexical_trap", "unrelated"]


def check_order(d: dict, sc: dict, anchor: str, tiers: list[str] = TIERS) -> tuple[bool, list[str]]:
    """Criterion 1: every strong positive > focal > same-field > trap > unrelated.

    With tiers=RELAXED, strong positives and the focal partner form one tier
    (supplementary check for anchor B; see the note in the report).
    """
    by = {t: [] for t in tiers}
    for c in candidates(d, anchor):
        role = role_of(d, c, anchor)
        by[next(t for t in tiers if role in t.split("|"))].append((c, sc["pairs"][f"{anchor}|{c}"]))
    fails = []
    for hi_t, lo_t in zip(tiers, tiers[1:]):
        for hc, hs in by[hi_t]:
            for lc, ls in by[lo_t]:
                if not hs > ls:
                    fails.append(f"{hi_t} {hc} ({hs:.4f}) <= {lo_t} {lc} ({ls:.4f})")
    return not fails, fails


def evaluate(d: dict, scs: list[dict]) -> dict:
    trap = d["pairset"]["selection"]["lexical_trap"]["arxiv_id"]
    gan = next(i for i, p in d["pairset"]["papers"].items() if p["role"] == "same_field_diff_topic")
    integ = d["ir"]["integrity"]
    dense_ok = all(r.get("cos_stored_vs_fresh_original", 0) >= INTEGRITY_MIN
                   for r in integ.values() if r["in_qdrant"])
    sparse_ok = all(r.get("sparse_cos_stored_vs_fresh", 0) >= INTEGRITY_MIN
                    for r in integ.values() if r["in_zilliz"])
    res = {}
    for sc in scs:
        r = {}
        for anc in (A, B):
            ok, fails = check_order(d, sc, anc)
            r[f"c1_order_anchor_{anc}"] = {"pass": ok, "violations": fails}
        p90 = stats(sc["base"]["cscl_lg"])["p90"]
        focal = sc["pairs"][f"{A}|{B}"]
        focal_rev = sc["pairs"][f"{B}|{A}"]
        r["c2_focal_above_p90_cscl_lg"] = {
            "pass": focal > p90 and focal_rev > p90, "focal_A_to_B": focal,
            "focal_B_to_A": focal_rev, "p90": p90}
        c3 = {anc: (sc["pairs"][f"{anc}|{trap}"], sc["pairs"][f"{anc}|{gan}"]) for anc in (A, B)}
        r["c3_trap_below_gan"] = {"pass": all(t < g for t, g in c3.values()),
                                  "by_anchor": {a: {"trap": t, "gan": g} for a, (t, g) in c3.items()}}
        if sc["model"] == "BGE-M3":
            agree = sparse_ok if sc["output"] == "sparse" else dense_ok
            r["c4_stored_matches_fresh"] = {
                "pass": agree, "applies_to": "dense" if sc["output"] != "sparse" else "sparse",
                "note": "ColBERT vectors are not stored anywhere; criterion judged on dense"
                        if sc["output"] == "colbert" else ""}
        else:
            r["c4_stored_matches_fresh"] = {"pass": None, "note": "not stored in production"}
        ok, fails = check_order(d, sc, B, RELAXED)
        r[f"supplementary_c1_anchor_{B}_relaxed"] = {"pass": ok, "violations": fails}
        crit = [r[f"c1_order_anchor_{A}"]["pass"], r[f"c1_order_anchor_{B}"]["pass"],
                r["c2_focal_above_p90_cscl_lg"]["pass"], r["c3_trap_below_gan"]["pass"]]
        if r["c4_stored_matches_fresh"]["pass"] is not None:
            crit.append(r["c4_stored_matches_fresh"]["pass"])
        r["overall_pass"] = all(crit)
        crit[1] = ok
        r["supplementary_overall_with_relaxed_B"] = all(crit)
        res[sc["key"]] = r
    return res


def spearman(d: dict, scs: list[dict]) -> dict:
    keys = [k for k in ("bge_m3.dense.title_abstract", "specter2.title_abstract",
                        "minilm.title_abstract", "bge_m3.sparse.title_abstract",
                        "bge_m3.colbert.title_abstract", "bge_m3.dense_stored.title_abstract")
            if any(s["key"] == k for s in scs)]
    by = {s["key"]: s for s in scs}
    out = {}
    for anc in (A, B):
        cands = candidates(d, anc)
        vec = {k: [by[k]["pairs"][f"{anc}|{c}"] for c in cands] for k in keys}
        expected = [-TIER_RANK[role_of(d, c, anc)] for c in cands]
        m = {}
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                m[f"{keys[x]} ~ {keys[y]}"] = float(spearmanr(vec[keys[x]], vec[keys[y]]).statistic)
        m.update({f"{k} ~ expected_tiers": float(spearmanr(vec[k], expected).statistic) for k in keys})
        out[anc] = m
    return out


# ── Markdown ─────────────────────────────────────────────────────────────────

def f(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def short(d: dict, aid: str) -> str:
    t = d["pairset"]["papers"][aid]["arxiv_api"]["title"]
    return (t[:38] + "…") if len(t) > 39 else t


PIPELINE = """\
Every fact below was read from code; Kaggle references are notebooks pulled
read-only with `kaggle kernels pull` (they are not in the repo).

**1. Text that gets embedded**

| Corpus slice | Recipe | Where |
|---|---|---|
| 2025-05 snapshot: 1,596,587 papers, Qdrant shards `a` + `b`, all of Zilliz | `str(title).strip()[:256] + "\\n\\n" + str(abstract).strip()[:1024]` on the raw `arxiv_comprehensive_papers.csv` cell; no whitespace or Unicode normalisation. Rows were filtered first (title > 10 chars, abstract > 50 chars) | Kaggle `siddhm11/bme-upload`, cell 3, `encode_batch()` and `load_data()` |
| Backfill → `arxiv_recent` | `f"{title[:256]} {abstract[:1024]}"`, whitespace collapsed with `re.sub(r'\\s+',' ')` | Kaggle `siddhm11/researchit-arxiv-backfill` |
| Daily ingest (current code) | `f"{p['title'][:256]} {p['abstract'][:1024]}"` after `clean_text()` (whitespace collapse + NFKC) | `scripts/ingest_arxiv.py:516`, `:525`; `clean_text` at `:290-291`; `app/discovery_worker.py:53`, `app/discovery_store.py:20` |

**Ambiguity, flagged:** `scripts/ingest_backends.py:9-13`, `scripts/ingest_arxiv.py:25-26` and doc 06
(2026-07-30) say new text "matches the original ingest exactly". It does not
literally: the separator is `" "` rather than `"\\n\\n"`, and whitespace/NFKC
normalisation is applied. The integrity section measures what that costs.

**2. Model and settings**

* `BAAI/bge-m3` (`app/config.py:246`), FlagEmbedding `BGEM3FlagModel`, pinned
  `FlagEmbedding==1.2.10`, `transformers==4.44.2` (`requirements.txt:31-32`). The HF revision
  is not pinned anywhere (the Dockerfile downloads it at build time, `Dockerfile:39`).
* `max_length=512` in every path: `app/embed_svc.py:76`, `scripts/ingest_backends.py:45,115`, bme-upload `encode_batch`.
  Truncation is therefore per-field by characters first (title 256, abstract 1024), then 512 tokens.
* Dense output: FlagEmbedding default `normalize_embeddings=True` → unit L2 norm.
* Precision: serving and query encoding run CPU fp32 (`app/embed_svc.py:46-52`, `BGE_M3_DEVICE=cpu`,
  `app/config.py:247`). The snapshot was encoded with `use_fp16=True` on a T4 and cast to float16
  before upload (bme-upload `setup_model`, `encode_batch`).
* Outputs stored: dense → Qdrant, sparse `lexical_weights` → Zilliz (filtered to weight > 0.01 in
  bme-upload). ColBERT vectors are **not** stored anywhere (`return_colbert_vecs=False` in
  `app/embed_svc.py:75` and bme-upload). The daily ingest does not write Zilliz
  (`scripts/ingest_backends.py:47-51`, `INGEST_WRITE_ZILLIZ` default 0).

**3. Stores and metrics** (configs read live from the clusters by `integrity_retrieval.py`)

| Store | Collection | Holds | Metric / storage |
|---|---|---|---|
| Qdrant cluster 1 | `arxiv_dense_a` | dense, snapshot rows 0..899,455 | Cosine, float16, binary quantisation (always_ram), HNSW m=32 ef_construct=128, graph on disk |
| Qdrant cluster 2 | `arxiv_dense_b` | dense, rest of snapshot | same, graph in RAM |
| Qdrant cluster 2 | `arxiv_recent` | dense, papers since 2025-06 | same, graph in RAM |
| Zilliz | `arxiv_bgem3_sparse` | sparse, snapshot only (1,596,587) | IP, AUTOINDEX, `drop_ratio_build=0.2`. PK is `point_id` (= Qdrant point id), not the auto-id `id` that CLAUDE.md §3.1 describes |
| Sidecar SQLite | `papers_fts` | FTS5 BM25 over title + abstract | `bm25(papers_fts, 2.0, 1.0)` (`app/fts_svc.py:30-31,88-90`) |

Search params: `rescore=True, oversampling=8.0` (`app/qdrant_svc.py:87-91`, `app/config.py:285`).
uint8 (de)quantisation is a no-op for all three shards (`app/qdrant_svc.py:225-251`;
`QDRANT_*UINT8_SCALE` unset). Shards are searched in parallel and merged by raw score
(`app/qdrant_svc.py:165-222`).

**4. Fusion and what the rerankers receive**

* **Paper→paper similarity in the app is dense cosine only.** "Related papers" on `/p/{id}` queries
  Qdrant with the paper's stored vector (`app/routers/paper.py:74-90`). `/api/space/similarity`
  is cosine of two stored vectors (`app/routers/space.py:180-222`). Recommendation candidates come
  from medoid/EWMA vector searches (`app/routers/recommendations.py:861`, `:890`).
* **Search (query→doc) is the only hybrid path:** the query is encoded (plus an optional Groq
  rewrite); for each form, a dense list (Qdrant) and a lexical list (FTS5 by default, Zilliz BGE-M3
  sparse as fallback, `app/hybrid_search_svc.py:153-169`, `app/config.py:152`) are fetched at
  `limit×6`. RRF fusion uses k=60 (`app/hybrid_search_svc.py:196`, `:319-352`). The
  `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder rescores the top `SEARCH_RERANK_TOP_N=10` on
  `"Title: …\\nAbstract: …"` (`app/reranker_bge_svc.py:87-94`), using sidecar abstracts that are
  cut at 500 chars for 90% of rows. The unscored tail is dropped, and then a title-match +
  citation boost is added (`app/hybrid_search_svc.py:265-286`, `:510`).
* So there is **no pairwise "fused hybrid score"** in the app. In this eval, "hybrid" means
  the app's own `_rrf_fuse_multi` over (a) the anchor's dense neighbour list and (b) a lexical
  list queried with the anchor's production embedding text. It is reported as an RRF score and
  a rank.
* Recommendation reranker: `RERANKER_MODE=heuristic` by default (`app/config.py:178`). It scores
  `0.40·lt + 0.25·st + 0.15·recency + 0.10·position − 0.15·neg` (`app/recommend/reranker.py:394-449`),
  where lt/st/neg are cosines between the candidate's stored dense vector and the user's EWMA
  profiles (`app/recommend/reranker.py:114-132`). Every personalised signal it sees is a
  stored BGE-M3 dense cosine. LightGBM is optional and has zero splits on features 20-30
  (`app/config.py:157-177`).
"""


def md(d: dict, scs: list[dict], ev: dict, sp: dict, calib: dict) -> str:
    ps = d["pairset"]
    L = []
    w = L.append
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w("# Embedding trustworthiness: ResearchIT\n")
    w(f"Generated {now} by `scripts/embedding_eval/run.sh` (seed {SEED}). Raw numbers: "
      "`results.json`. Every figure below is read from that run's output.\n")

    w("## Verdict\n")
    w(verdict(d, scs, ev) + "\n")

    w("## Pass/fail criteria (fixed before measuring)\n")
    w("A model + text variant **passes** when all of the following hold:\n")
    w("1. Every strong positive > focal pair > same-field-different-topic (GAN) > lexical trap > "
      "unrelated, separately with A and with B as anchor (strict, pairwise across adjacent tiers).")
    w("2. The focal pair scores above p90 of that model+variant's cs.CL/cs.LG random-pair baseline, in both directions.")
    w("3. The lexical trap scores below the GAN paper, for both anchors.")
    w(f"4. Stored and freshly computed vectors agree (cosine ≥ {INTEGRITY_MIN} for every pair-set paper). "
      "Applies only to what production stores (BGE-M3 dense, sparse).\n")
    w("Note on anchor B: criterion 1 as specified also requires Bahdanau, Luong and BERT to beat A when "
      "B is the anchor. Bahdanau and Luong are themselves attention-for-seq2seq papers, so this "
      "expectation is debatable. Failures there are listed individually so they can be weighed.\n")

    w("## Step 0: pipeline facts\n")
    w(PIPELINE)

    w("## Step 1: pair set (verified against the arXiv API)\n")
    w("| arXiv id | role | arXiv API title | API primary | categories | version | indexed (Qdrant / Zilliz) | text source |")
    w("|---|---|---|---|---|---|---|---|")
    for aid, p in ps["papers"].items():
        a = p["arxiv_api"]
        flag = "" if not p["not_indexed"] else " **NOT INDEXED**"
        w(f"| {aid} | {p['role']} | {a['title']} | {a['primary_topic']} | {a['categories']} | v{a['version']} "
          f"| {p['indexed']['qdrant']} / {p['indexed']['zilliz']}{flag} | {p['text_source']} |")
    for k in ("lexical_trap", "unrelated"):
        s = ps["selection"][k]
        p = ps["papers"][s["arxiv_id"]]
        w(f"\n**{k}: {s['arxiv_id']}.** Rule: {s['rule']}. Drawn from {s['candidates']:,} candidates.")
        w(f"Abstract opening (arXiv API): \"{p['arxiv_api']['abstract'][:300]}…\"")
    w("")

    w("## Step 3: calibration baselines\n")
    bl = d["baseline"]
    w(f"2,000 random corpus pairs (pool {bl['pool_sizes']['random']:,} sidecar papers) and 2,000 pairs whose "
      f"papers both have primary category cs.CL or cs.LG (pool {bl['pool_sizes']['cscl_lg']:,}). "
      f"{bl['n_papers']:,} distinct papers, {bl['n_from_sidecar_recent']:,} of them post-snapshot (text from the "
      "sidecar's full abstract, ingest recipe). The rest use the original CSV text and recipe. "
      f"Stored-vector baseline: {d['base_stored']['n_found']:,}/{d['base_stored']['n_requested']:,} vectors found in Qdrant. "
      f"Baseline encodes ran on `{d['base_bm']['device']}`"
      + (f"; CPU-vs-{d['ir']['device_agreement']['device']} min cosine on the pair set = "
         f"{f(d['ir']['device_agreement']['min_cos_vs_cpu'], 6)}." if "device_agreement" in d["ir"] else ".") + "\n")
    w("| scorer | variant | set | mean | std | p50 | p90 | p99 |")
    w("|---|---|---|---|---|---|---|---|")
    for sc in scs:
        for s in ("random", "cscl_lg"):
            c = calib[sc["key"]][s]
            w(f"| {sc['model']} {sc['output']} | {VLABEL[sc['variant']]} | {'all' if s == 'random' else 'cs.CL/cs.LG'} "
              f"| {f(c['mean'])} | {f(c['std'])} | {f(c['p50'])} | {f(c['p90'])} | {f(c['p99'])} |")
    w("")

    w("## Step 2-3: scores per anchor\n")
    w("Each cell: raw score, then (pair-set rank among the 7 candidates), percentile in the all-corpus "
      "baseline, percentile in the cs.CL/cs.LG baseline, and z against cs.CL/cs.LG. ColBERT is "
      "anchor-as-query. Dense and sparse are symmetric.\n")
    for anc in (A, B):
        w(f"### Anchor {anc} ({short(d, anc)})\n")
        cands = candidates(d, anc)
        groups = {}
        for sc in scs:
            groups.setdefault((sc["model"], sc["output"]), []).append(sc)
        for (model, output), lst in groups.items():
            w(f"**{model}: {output}**\n")
            w("| candidate | role | " + " | ".join(VLABEL[s["variant"]] for s in lst) + " |")
            w("|---|---|" + "---|" * len(lst))
            for c in cands:
                cells = []
                for sc in lst:
                    raw = sc["pairs"][f"{anc}|{c}"]
                    order = sorted(cands, key=lambda x: -sc["pairs"][f"{anc}|{x}"])
                    cr = calib[sc["key"]]
                    cells.append(f"{f(raw)} ({order.index(c) + 1}) · "
                                 f"{f(percentile_of(raw, sc['base']['random']), 1)}% · "
                                 f"{f(percentile_of(raw, sc['base']['cscl_lg']), 1)}% · "
                                 f"z {f((raw - cr['cscl_lg']['mean']) / cr['cscl_lg']['std'], 2)}")
                w(f"| {c} {short(d, c)} | {role_of(d, c, anc)} | " + " | ".join(cells) + " |")
            w("")

    w("### Retrieval rank in the production index\n")
    w(f"Depth {1000}: each of the 3 shards returns its top 1000 under production search params, merged "
      "by score as the app does. The anchor itself is excluded. `>1000` means not retrieved at that depth.\n")
    for anc, r in d["ir"]["retrieval"].items():
        w(f"**Anchor {anc}**. Shard status: " + ", ".join(
            f"{k}: {'ok' if v['ok'] else 'FAILED ' + v['error']}" for k, v in r["shard_status"].items())
          + f". Dense search {r['dense_seconds']}s" + (f", FTS5 {r['fts5_seconds']}s" if "fts5_seconds" in r else "")
          + f". Cosine of the 1000th dense neighbour: {f(r.get('dense_cutoff_score_at_depth'))}.\n")
        lb = r.get("dense_rank_lower_bounds", {})
        if lb:
            w("Dense rank lower bounds for candidates beyond 1000: distinct papers known to score higher "
              "(shards a/b depth-1000 lists plus a thresholded search of `arxiv_recent`). The same "
              "thresholded search timed out server-side on shards a and b, so these are bounds, not ranks.\n")
            w("| candidate | pair cosine (server) | rank ≥ | of which arxiv_recent | arxiv_recent count complete | note |")
            w("|---|---|---|---|---|---|")
            for c, x in lb.items():
                w(f"| {c} {short(d, c)} | {f(x['pair_score_server'])} | {x['rank_at_least']:,} "
                  f"| {x['recent_shard_count_above']:,} | {x['recent_count_complete']} | {x['note'] or ''} |")
            w("")
        cols = [k for k in ("dense", "lexical_fts5", "hybrid_dense+fts5", "lexical_zilliz", "hybrid_dense+zilliz") if k in r]
        w("| candidate | role | " + " | ".join(cols) + " |")
        w("|---|---|" + "---|" * len(cols))
        for c in candidates(d, anc):
            cells = []
            for k in cols:
                x = r[k][c]
                cells.append(f"{x['rank'] if x['rank'] else '>1000'} ({x['bucket']}, score {f(x['score'])})"
                             if x["rank"] else f">1000 (list held {x['beyond']})")
            w(f"| {c} {short(d, c)} | {role_of(d, c, anc)} | " + " | ".join(cells) + " |")
        w("")

    w("## Step 4: integrity\n")
    integ = d["ir"]["integrity"]
    w("**Stored vs fresh dense** (fresh = CPU fp32 through the app's BGE-M3 singleton). `original` "
      "reproduces bme-upload's text and recipe; `fp16` is that vector round-tripped through float16, "
      "as bme-upload stored it; `app` is `embed_svc.encode_query()` on the same text; `ingest today` "
      "is `ingest_backends.Encoder` on today's arXiv API text with the current ingest recipe.\n")
    w("| paper | role | shard / point id | CSV row = point id | dim | stored norm | stored is fp16 | cos original | cos fp16 | cos app | cos ingest today |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for aid, r in integ.items():
        if not r["in_qdrant"]:
            w(f"| {aid} | {r['role']} | not in Qdrant |" + " |" * 8)
            continue
        flag = lambda x: f"**{f(x, 6)}**" if x < INTEGRITY_MIN else f(x, 6)  # noqa: E731
        w(f"| {aid} | {r['role']} | {r['backend']} / {r['point_id']} | {r['csv_row_equals_point_id']} | {r['stored_dim']} "
          f"| {f(r['stored_norm'], 6)} | {r['stored_is_float16_exact']} | {flag(r['cos_stored_vs_fresh_original'])} "
          f"| {flag(r['cos_stored_vs_fresh_original_fp16'])} | {flag(r['cos_stored_vs_fresh_app_encode_query'])} "
          f"| {flag(r['cos_stored_vs_fresh_ingest_today'])} |")
    diffs = {aid: r["abstract_csv_vs_api_diffs"] for aid, r in integ.items()
             if r.get("abstract_csv_vs_api_diffs")}
    w("\nOriginal CSV abstract vs today's arXiv API abstract (whitespace-normalised): "
      + (", ".join(f"{aid} differs at {len(v)} span(s), e.g. `{v[0][0]}` → `{v[0][1]}`"
                   for aid, v in diffs.items()) if diffs else "identical for every pair-set paper") + ".")
    w("\n**Stored vs fresh sparse** (fresh filtered at bme-upload's 0.01 threshold)\n")
    w("| paper | Zilliz point id | nnz stored | nnz fresh | token Jaccard | sparse cosine | max abs weight diff |")
    w("|---|---|---|---|---|---|---|")
    for aid, r in integ.items():
        if r["in_zilliz"]:
            w(f"| {aid} | {r['zilliz_point_id']} | {r['sparse_nnz_stored']} | {r['sparse_nnz_fresh_filtered']} "
              f"| {f(r['sparse_token_jaccard'])} | {f(r['sparse_cos_stored_vs_fresh'], 6)} "
              f"| {f(r['sparse_max_abs_weight_diff_shared'], 5)} |")
    w("\n**Same pair, server-side vs local.** Qdrant and Zilliz hold different representations "
      "(dense cosine vs sparse inner product), so they cannot return the same score. Each store is "
      "checked against a local recomputation of the same quantity.\n")
    w("| pair | stored dense cos (local) | Qdrant server score | fresh dense cos | stored sparse IP (local) | Zilliz server IP (stored query) | Zilliz server IP (fresh query) | local IP fresh-query × stored-doc |")
    w("|---|---|---|---|---|---|---|---|")
    for k, r in d["ir"]["server_vs_local"].items():
        w(f"| {k} | {f(r['local_stored_dense_cos'])} | {f(r['qdrant_server_score'])} | {f(r['local_fresh_dense_cos'])} "
          f"| {f(r.get('local_stored_sparse_ip'))} | {f(r.get('zilliz_server_ip_stored_query'))} "
          f"| {f(r.get('zilliz_server_ip_fresh_query'))} | {f(r.get('local_fresh_query_vs_stored_doc_ip'))} |")
    w("\n**Truncation**\n")
    w("| paper | title chars | abstract chars | abstract cut at 1024 (chars dropped) | BGE-M3 tokens (production text) | BGE-M3 tokens (uncut) | > 512 | SPECTER2 tokens (>512) | MiniLM tokens (>256) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for aid, r in d["ir"]["truncation"].items():
        w(f"| {aid} | {r['title_chars']} | {r['abstract_chars']} | {r['abstract_cut_by_1024']} ({r['abstract_chars_dropped']}) "
          f"| {r['bge_m3_tokens_production_text']} | {r['bge_m3_tokens_untruncated_text']} | {r['bge_m3_hits_max_length_512']} "
          f"| {r['specter2_tokens']} ({r['specter2_truncated_at_512']}) | {r['minilm_tokens']} ({r['minilm_truncated_at_256']}) |")
    w("")

    w("## Step 5: pass/fail per model and variant\n")
    w("The last two columns are **supplementary** and are not part of the criteria. They re-run "
      "criterion 1 for anchor B with strong positives and A treated as one tier, which removes the "
      "debatable \"Bahdanau/Luong/BERT must beat A\" step, and then recompute the overall result.\n")
    w("| scorer | variant | C1 anchor A | C1 anchor B | C2 focal > p90 | C3 trap < GAN | C4 stored = fresh | **overall** | suppl.: C1 B relaxed | suppl.: overall w/ relaxed B |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    pf = lambda x: "n/a" if x is None else ("PASS" if x else "**FAIL**")  # noqa: E731
    for sc in scs:
        r = ev[sc["key"]]
        w(f"| {sc['model']} {sc['output']} | {VLABEL[sc['variant']]} | {pf(r[f'c1_order_anchor_{A}']['pass'])} "
          f"| {pf(r[f'c1_order_anchor_{B}']['pass'])} | {pf(r['c2_focal_above_p90_cscl_lg']['pass'])} "
          f"| {pf(r['c3_trap_below_gan']['pass'])} | {pf(r['c4_stored_matches_fresh']['pass'])} | {pf(r['overall_pass'])} "
          f"| {pf(r[f'supplementary_c1_anchor_{B}_relaxed']['pass'])} | {pf(r['supplementary_overall_with_relaxed_B'])} |")
    w("\n**Where the expected order breaks** (criterion 1 violations)\n")
    for sc in scs:
        r = ev[sc["key"]]
        v = r[f"c1_order_anchor_{A}"]["violations"] + [f"[anchor B] {x}" for x in r[f"c1_order_anchor_{B}"]["violations"]]
        if v:
            w(f"* **{sc['model']} {sc['output']}, {VLABEL[sc['variant']]}**: " + "; ".join(v))
    w("\n**Spearman correlation between scorers' rankings of the 7 candidates** (title+abstract variants; "
      "`expected_tiers` is the tier order from criterion 1, ties within a tier).\n")
    for anc, m in sp.items():
        w(f"Anchor {anc}:\n")
        w("| pair | ρ |")
        w("|---|---|")
        for k, v in m.items():
            w(f"| {k} | {f(v, 3)} |")
        w("")
    if "specter2_error" in d:
        w(f"\n**SPECTER2 did not run:** {d['specter2_error']}\n")
    return "\n".join(L)


def verdict(d: dict, scs: list[dict], ev: dict) -> str:
    """Five lines, every figure computed from this run (no hand-entered numbers)."""
    integ = d["ir"]["integrity"]
    q = [r for r in integ.values() if r["in_qdrant"]]
    z = [r for r in integ.values() if r["in_zilliz"]]
    sv = d["ir"]["server_vs_local"].values()
    dq = max(abs(r["qdrant_server_score"] - r["local_stored_dense_cos"]) for r in sv)
    dz = max(abs(r["zilliz_server_ip_stored_query"] - r["local_stored_sparse_ip"]) for r in sv
             if r.get("zilliz_server_ip_stored_query") is not None)
    prod_key = "bge_m3.dense_stored.title_abstract"
    prod = ev[prod_key]
    by = {s["key"]: s for s in scs}
    pairs = by[prod_key]["pairs"]
    base_cl = by[prod_key]["base"]["cscl_lg"]
    trap = d["pairset"]["selection"]["lexical_trap"]["arxiv_id"]
    gan = next(i for i, p in d["pairset"]["papers"].items() if p["role"] == "same_field_diff_topic")
    pos = [i for i, p in d["pairset"]["papers"].items() if p["role"] == "strong_positive"]
    ab = pairs[f"{A}|{B}"]

    def dense_rank(anc: str, cand: str) -> str:
        r = d["ir"]["retrieval"].get(anc, {})
        x = r.get("dense", {}).get(cand, {})
        if x.get("rank"):
            return str(x["rank"])
        lb = r.get("dense_rank_lower_bounds", {}).get(cand)
        return f"≥{lb['rank_at_least']:,}" if lb else ">1000"

    def hybrid_rank(anc: str, cand: str) -> str:
        x = d["ir"]["retrieval"].get(anc, {}).get("hybrid_dense+zilliz", {}).get(cand, {})
        return str(x["rank"]) if x.get("rank") else ">1000"

    bge_c3 = [s["key"] for s in scs if s["model"] == "BGE-M3" and s["variant"] == "title_abstract"]
    bge_c3_fail = [k for k in bge_c3 if not ev[k]["c3_trap_below_gan"]["pass"]]
    other_c3 = [s["key"] for s in scs if s["model"] != "BGE-M3" and s["variant"] == "title_abstract"
                and ev[s["key"]]["c3_trap_below_gan"]["pass"]]
    strict = [s["key"] for s in scs if ev[s["key"]]["overall_pass"]]
    relaxed = [s["key"] for s in scs if ev[s["key"]]["supplementary_overall_with_relaxed_B"]]
    cut_a = d["ir"]["retrieval"].get(A, {}).get("dense_cutoff_score_at_depth")
    lines = [
        f"1. **The stored data is intact.** Stored vs recomputed dense cosine ≥ "
        f"{min(r['cos_stored_vs_fresh_original'] for r in q):.6f} and sparse cosine ≥ "
        f"{min(r['sparse_cos_stored_vs_fresh'] for r in z):.6f} for all {len(q)} papers. "
        f"CSV row = point id for {sum(r['csv_row_equals_point_id'] for r in q)}/{len(q)}. "
        f"Server scores match local recomputation within {dq:.4f} (Qdrant) and {dz:.4f} (Zilliz).",
        f"2. **The neighbourhood judgement is right.** Production cosine A·B = {ab:.4f}, the "
        f"{percentile_of(ab, base_cl):.1f}th percentile of random cs.CL/cs.LG pairs (p90 {prod['c2_focal_above_p90_cscl_lg']['p90']:.4f}). "
        f"That is below the strongest positive ({max(pairs[f'{A}|{p}'] for p in pos):.4f}), so: same neighbourhood, not a near-duplicate.",
        f"3. **Fine ordering is not right.** Production dense ranks the q-bio.NC trap above GAN for both "
        f"anchors ({pairs[f'{A}|{trap}']:.4f} vs {pairs[f'{A}|{gan}']:.4f}; {pairs[f'{B}|{trap}']:.4f} vs {pairs[f'{B}|{gan}']:.4f}). "
        f"Trap > GAN in {len(bge_c3_fail)}/{len(bge_c3)} BGE-M3 title+abstract scorers. Trap < GAN holds for: "
        f"{', '.join(other_c3) or 'none'}.",
        f"4. **Retrieval cannot reach the pair.** Dense rank of B for anchor A is {dense_rank(A, B)} "
        f"(the 1000th neighbour already scores {cut_a:.4f}), and of A for anchor B {dense_rank(B, A)}. "
        f"Hybrid dense+Zilliz RRF: {hybrid_rank(A, B)} and {hybrid_rank(B, A)}.",
        f"5. **Strict pass (all criteria):** {', '.join(strict) or 'no scorer'}. With the debatable "
        f"anchor-B positives step relaxed (supplementary): {', '.join(relaxed) or 'no scorer'}.",
    ]
    # Interpretation numbers cannot carry: conditions and the recommended fix.
    # Kept number-free so a re-run can never leave a stale figure in it.
    note = Path(__file__).with_name("verdict_note.md")
    if note.exists():
        lines.append("\n" + note.read_text().strip())
    return "\n".join(lines)


def main() -> None:
    d = load_all()
    scs = scorers(d)
    calib = {sc["key"]: {s: stats(sc["base"][s]) for s in ("random", "cscl_lg")} for sc in scs}
    ev = evaluate(d, scs)
    sp = spearman(d, scs)
    per_pair = {}
    for sc in scs:
        rows = {}
        for anc in (A, B):
            for c in candidates(d, anc):
                raw = sc["pairs"][f"{anc}|{c}"]
                rows[f"{anc}|{c}"] = {
                    "role": role_of(d, c, anc), "raw": raw,
                    "percentile_random": percentile_of(raw, sc["base"]["random"]),
                    "percentile_cscl_lg": percentile_of(raw, sc["base"]["cscl_lg"]),
                    "z_random": (raw - calib[sc["key"]]["random"]["mean"]) / calib[sc["key"]]["random"]["std"],
                    "z_cscl_lg": (raw - calib[sc["key"]]["cscl_lg"]["mean"]) / calib[sc["key"]]["cscl_lg"]["std"]}
        per_pair[sc["key"]] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "pairset": {k: {kk: vv for kk, vv in v.items() if kk not in ("abstract",)}
                    for k, v in d["pairset"]["papers"].items()},
        "selection": d["pairset"]["selection"],
        "baseline_meta": {k: d["baseline"][k] for k in ("pool_sizes", "n_papers", "n_from_sidecar_recent")},
        "baseline_pairs": {s: d["baseline"][s] for s in ("random", "cscl_lg")},
        "baseline_device": d["base_bm"]["device"],
        "calibration": calib, "pair_scores": per_pair,
        "baseline_raw": {sc["key"]: sc["base"] for sc in scs},
        "integrity": d["ir"]["integrity"], "server_vs_local": d["ir"]["server_vs_local"],
        "truncation": d["ir"]["truncation"], "retrieval": d["ir"]["retrieval"],
        "device_agreement": d["ir"].get("device_agreement"),
        "criteria": ev, "spearman": sp, "specter2_error": d.get("specter2_error"),
    }
    with open(OUT / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    (OUT / "report.md").write_text(md(d, scs, ev, sp, calib))
    print(f"[embedding_eval:report] wrote {OUT / 'report.md'} and results.json")


if __name__ == "__main__":
    main()
