"""Stage 4: integrity checks against the stored vectors, and retrieval ranks.

Read-only against production. Every Qdrant/Zilliz call is a scroll, query,
search or describe; nothing is upserted or deleted.

Integrity
  * stored dense (Qdrant) vs fresh dense, three fresh code paths:
      original  -- bme-upload recipe on the original CSV text, CPU fp32, via the
                   app's own BGE-M3 singleton (app.embed_svc.get_model)
      app_query -- app.embed_svc.encode_query() on that same text
      ingest    -- scripts/ingest_backends.Encoder on today's arXiv API text with
                   ingest_arxiv's recipe: what a re-ingest would write now
    plus a float16 round-trip of `original`, which is how bme-upload stored it.
  * stored sparse (Zilliz) vs fresh sparse filtered at bme-upload's 0.01.
  * server-side pair scores (Qdrant cosine, Zilliz IP) vs the same pair
    computed locally from the stored vectors.
  * CSV row == Qdrant point id (bme-upload assigned ids by row).
  * truncation at max_length for every model's tokenizer.

Retrieval
  * dense: anchor's stored vector through the app's shard search
    (qdrant_svc._run_vector_search per shard + _merge_by_score), limit 1000.
    This is what /p/{id} "related papers" does, at depth.
  * hybrid: the app's _rrf_fuse_multi (k=60) over that dense list and a lexical
    list queried with the anchor's production embedding text: FTS5 (the
    production default) and, separately, Zilliz (the fallback).
"""
from __future__ import annotations

import asyncio
import os
import time
from functools import lru_cache

import numpy as np

from common import (A, B, SIDECAR, WORK, app_import_path, bge_text,
                    load_json, log, save_json)

app_import_path()
os.environ.setdefault("METADATA_SIDECAR_PATH", str(SIDECAR))
os.environ.setdefault("BGE_M3_DEVICE", "cpu")

STAGE = "integrity"
DEPTH = 1000
SPARSE_THRESHOLD = 0.01        # bme-upload SPARSE_THRESHOLD


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def sparse_ip(a: dict, b: dict) -> float:
    return float(sum(w * b[t] for t, w in a.items() if t in b))


def sparse_cos(a: dict, b: dict) -> float:
    na = np.sqrt(sum(w * w for w in a.values()))
    nb = np.sqrt(sum(w * w for w in b.values()))
    return sparse_ip(a, b) / (na * nb) if na and nb else float("nan")


def patch_qdrant_timeout():
    """Depth-1000 searches on an on-disk HNSW graph can exceed the app's 30 s
    client timeout. Timeout is transport, not ranking, so raise it."""
    from qdrant_client import QdrantClient
    from app import qdrant_svc

    @lru_cache(maxsize=4)
    def _client_for(url: str, api_key: str):
        return QdrantClient(url=url, api_key=api_key, timeout=300, check_compatibility=False)
    qdrant_svc._client_for = _client_for


def stored_points(ids: list[str]) -> dict[str, dict]:
    """Per-shard scroll so each vector's shard and point id are known."""
    from app import qdrant_svc
    out = {}
    for backend in qdrant_svc._active_backends():
        with qdrant_svc.use_backend(backend):
            for p in qdrant_svc._scroll_vectors_by_arxiv_ids(ids):
                aid = p.payload["arxiv_id"]
                v = np.asarray(qdrant_svc._dequantize(p.vector), np.float32)
                out.setdefault(aid, {"backend": backend, "collection": qdrant_svc._collection(),
                                     "point_id": p.id, "vector": v,
                                     "dim": int(v.shape[0]), "norm": float(np.linalg.norm(v))})
    return out


def stored_sparse(ids: list[str]) -> dict[str, dict]:
    from app import zilliz_svc, config
    c = zilliz_svc._get_client()
    quoted = ",".join(f'"{i}"' for i in ids)
    rows = c.query(config.ZILLIZ_COLLECTION, filter=f"arxiv_id in [{quoted}]",
                   output_fields=["arxiv_id", "point_id", "sparse_vector"])
    return {r["arxiv_id"]: {"point_id": r["point_id"],
                            "sparse": {int(k): float(v) for k, v in r["sparse_vector"].items()}}
            for r in rows}


def qdrant_pair_score(anchor_vec, target: str, backend: str) -> float | None:
    """Server-side score of `target` for the anchor's vector, production search params."""
    from qdrant_client import models as m
    from app import qdrant_svc
    with qdrant_svc.use_backend(backend):
        res = qdrant_svc._client().query_points(
            collection_name=qdrant_svc._collection(),
            query=qdrant_svc._quantize_query(anchor_vec.tolist()),
            query_filter=m.Filter(must=[m.FieldCondition(key="arxiv_id", match=m.MatchValue(value=target))]),
            limit=1, with_payload=False, search_params=qdrant_svc._SEARCH_PARAMS)
    return float(res.points[0].score) if res.points else None


def zilliz_pair_score(query_sparse: dict, target: str) -> float | None:
    from app import zilliz_svc, config
    res = zilliz_svc._get_client().search(
        collection_name=config.ZILLIZ_COLLECTION, data=[query_sparse], anns_field="sparse_vector",
        search_params={"metric_type": "IP"}, limit=1, filter=f'arxiv_id == "{target}"',
        output_fields=["arxiv_id"])
    return float(res[0][0]["distance"]) if res and res[0] else None


def dense_neighbours(vec: np.ndarray) -> tuple[list[dict], dict, dict]:
    """The app's shard fan-out, per shard so a failing shard is reported, not hidden."""
    from app import qdrant_svc
    lists, status, per_shard = [], {}, {}
    for backend in qdrant_svc._active_backends():
        t0 = time.time()
        try:
            with qdrant_svc.use_backend(backend):
                pts = qdrant_svc._run_vector_search(vec.tolist(), DEPTH)
            lst = [{"arxiv_id": p.payload["arxiv_id"], "score": float(p.score)} for p in pts]
            lists.append(lst)
            per_shard[backend] = lst
            status[backend] = {"ok": True, "n": len(pts), "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001 -- recorded, not swallowed
            status[backend] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            log(STAGE, f"dense search on shard {backend} FAILED: {e}")
    return qdrant_svc._merge_by_score(lists, DEPTH + 1), status, per_shard


DEEP_LIMIT = 20000


def rank_lower_bounds(vec: np.ndarray, anchor: str, per_shard: dict,
                      missing: dict[str, float]) -> dict:
    """For candidates outside the top 1000: a provable lower bound on their rank.

    Counts DISTINCT papers known to score above the candidate: those in each
    shard's depth-1000 list, plus everything above the threshold in
    arxiv_recent (one thresholded search, limit 20,000; its graph is in RAM).
    The same thresholded search at depth 20,000 times out server-side on shards
    a and b (measured), so they contribute only their depth-1000 lists, which is
    why this is a bound and not a rank.
    """
    from app import qdrant_svc
    if not missing:
        return {}
    floor = min(missing.values())
    extra, note, limit_hit = [], None, False
    try:
        with qdrant_svc.use_backend("recent"):
            res = qdrant_svc._client().query_points(
                collection_name=qdrant_svc._collection(), query=vec.tolist(), limit=DEEP_LIMIT,
                score_threshold=floor, with_payload=["arxiv_id"],
                search_params=qdrant_svc._SEARCH_PARAMS)
        extra = [{"arxiv_id": p.payload["arxiv_id"], "score": float(p.score)} for p in res.points]
        limit_hit = len(extra) >= DEEP_LIMIT
    except Exception as e:  # noqa: BLE001 -- recorded
        note = f"recent thresholded search failed: {type(e).__name__}: {str(e)[:160]}"
    lowest = min((r["score"] for r in extra), default=None)
    out = {}
    for cand, s in missing.items():
        above = {r["arxiv_id"] for lst in list(per_shard.values()) + [extra]
                 for r in lst if r["score"] > s and r["arxiv_id"] not in (anchor, cand)}
        # Results come back best-first, so a limit-truncated result set still
        # holds EVERY point above s when its lowest score is at or above s.
        truncated = limit_hit and lowest is not None and s < lowest
        out[cand] = {"pair_score_server": s, "papers_scoring_higher_at_least": len(above),
                     "rank_at_least": len(above) + 1,
                     "recent_shard_count_above": sum(1 for r in extra if r["score"] > s),
                     "recent_count_complete": not truncated and note is None,
                     "note": note or (f"recent-shard count capped by the {DEEP_LIMIT:,} limit"
                                      if truncated else None)}
    return out


def ranks(lst: list[dict], anchor: str, ids: list[str], key: str = "score") -> dict:
    ordered = [r for r in lst if r["arxiv_id"] != anchor]
    pos = {r["arxiv_id"]: (k + 1, r[key]) for k, r in enumerate(ordered)}
    return {i: ({"rank": pos[i][0], "score": pos[i][1]} if i in pos else
                {"rank": None, "score": None, "beyond": len(ordered)})
            for i in ids if i != anchor}


def bucket(rank: int | None) -> str:
    if rank is None:
        return f">{DEPTH}"
    return "top10" if rank <= 10 else "top100" if rank <= 100 else "top1000" if rank <= 1000 else f">{DEPTH}"


def main() -> None:
    from app import embed_svc, fts_svc, hybrid_search_svc, zilliz_svc
    patch_qdrant_timeout()
    t_start = time.time()
    ps = load_json("pairset.json")
    papers = ps["papers"]
    ids = list(papers)
    out: dict = {"ids": ids}

    # ── fresh vectors ────────────────────────────────────────────────────────
    model = embed_svc.get_model()
    fresh_orig = dict(zip(load_json("scores_pairset_bge_minilm.json")["ids"],
                          np.load(WORK / "pairset_bge_dense_title_abstract.npy")))
    prod_text = {i: bge_text(papers[i]["title"], papers[i]["abstract"], "title_abstract",
                             papers[i]["recipe"]) for i in ids}
    fresh_app = {i: embed_svc.encode_query(prod_text[i])[0] for i in ids}
    from ingest_arxiv import clean_text
    from ingest_backends import Encoder
    enc = Encoder(device="cpu")
    ingest_text = {i: f"{clean_text(papers[i]['arxiv_api']['title'])[:256]} "
                      f"{clean_text(papers[i]['arxiv_api']['abstract'])[:1024]}" for i in ids}
    fresh_ingest = dict(zip(ids, (np.asarray(d, np.float32) for d, _ in
                                  enc.encode([ingest_text[i] for i in ids]))))
    fresh_sparse = {i: {int(k): v for k, v in s.items()}
                    for i, s in load_json("pairset_bge_sparse_title_abstract.json").items()}

    # Device agreement for the baseline encodes (if they ran off-CPU).
    dev = os.getenv("EMBEVAL_BASELINE_DEVICE", "cpu")
    if dev != "cpu":
        from FlagEmbedding import BGEM3FlagModel
        m2 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device=dev)
        d2 = m2.encode([prod_text[i] for i in ids], max_length=512)["dense_vecs"]
        out["device_agreement"] = {"device": dev, "min_cos_vs_cpu": min(
            cos(fresh_orig[i], d2[k]) for k, i in enumerate(ids))}
        del m2

    # ── stored vectors ───────────────────────────────────────────────────────
    sp = stored_points(ids)
    ss = stored_sparse(ids)
    integ = {}
    for i in ids:
        p, z = sp.get(i), ss.get(i)
        row = {"role": papers[i]["role"], "in_qdrant": p is not None, "in_zilliz": z is not None}
        if p:
            v = p["vector"]
            row.update({
                "backend": p["backend"], "collection": p["collection"], "point_id": p["point_id"],
                "csv_row": papers[i]["csv_row"],
                "csv_row_equals_point_id": papers[i]["csv_row"] == p["point_id"],
                "stored_dim": p["dim"], "stored_norm": p["norm"],
                "stored_is_float16_exact": bool(np.all(v.astype(np.float16).astype(np.float32) == v)),
                "fresh_norm": float(np.linalg.norm(fresh_orig[i])),
                "cos_stored_vs_fresh_original": cos(v, fresh_orig[i]),
                "cos_stored_vs_fresh_original_fp16": cos(v, fresh_orig[i].astype(np.float16)),
                "cos_stored_vs_fresh_app_encode_query": cos(v, fresh_app[i]),
                "cos_stored_vs_fresh_ingest_today": cos(v, fresh_ingest[i]),
                "cos_fresh_original_vs_ingest_today": cos(fresh_orig[i], fresh_ingest[i]),
            })
            # Explains any original-vs-ingest gap: does today's API text differ
            # from the CSV text beyond whitespace?
            import difflib
            csv_t = " ".join(str(papers[i]["abstract"]).split())
            api_t = clean_text(papers[i]["arxiv_api"]["abstract"])
            ops = [(csv_t[a1:a2][:60], api_t[b1:b2][:60]) for tag, a1, a2, b1, b2 in
                   difflib.SequenceMatcher(None, csv_t, api_t, autojunk=False).get_opcodes() if tag != "equal"]
            row.update({"abstract_csv_equals_api_modulo_whitespace": not ops,
                        "abstract_csv_vs_api_diffs": ops[:5]})
        if z:
            f = {t: w for t, w in fresh_sparse[i].items() if w > SPARSE_THRESHOLD}
            st, fr = set(z["sparse"]), set(f)
            row.update({
                "zilliz_point_id": z["point_id"],
                "sparse_nnz_stored": len(st), "sparse_nnz_fresh_filtered": len(fr),
                "sparse_token_jaccard": len(st & fr) / len(st | fr),
                "sparse_cos_stored_vs_fresh": sparse_cos(z["sparse"], f),
                "sparse_max_abs_weight_diff_shared": max(
                    (abs(z["sparse"][t] - f[t]) for t in st & fr), default=None),
            })
        integ[i] = row
        log(STAGE, f"{i} {row.get('cos_stored_vs_fresh_original', 'n/a')} "
                   f"sparse_cos={row.get('sparse_cos_stored_vs_fresh', 'n/a')}")
    out["integrity"] = integ

    # ── same pair, server vs local ───────────────────────────────────────────
    cross = {}
    for anc in (A, B):
        for t in ids:
            if t == anc or anc not in sp or t not in sp:
                continue
            r = {"local_stored_dense_cos": cos(sp[anc]["vector"], sp[t]["vector"]),
                 "local_fresh_dense_cos": cos(fresh_orig[anc], fresh_orig[t]),
                 "qdrant_server_score": qdrant_pair_score(sp[anc]["vector"], t, sp[t]["backend"])}
            if anc in ss and t in ss:
                r.update({
                    "local_stored_sparse_ip": sparse_ip(ss[anc]["sparse"], ss[t]["sparse"]),
                    "zilliz_server_ip_stored_query": zilliz_pair_score(ss[anc]["sparse"], t),
                    "zilliz_server_ip_fresh_query": zilliz_pair_score(fresh_sparse[anc], t),
                    "local_fresh_query_vs_stored_doc_ip": sparse_ip(fresh_sparse[anc], ss[t]["sparse"]),
                    "local_fresh_sparse_ip": sparse_ip(fresh_sparse[anc], fresh_sparse[t]),
                })
            cross[f"{anc}|{t}"] = r
    out["server_vs_local"] = cross

    # ── truncation ───────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    toks = {"bge_m3": (model.tokenizer, 512),
            "specter2": (AutoTokenizer.from_pretrained("allenai/specter2_base"), 512),
            "minilm": (AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"), 256)}
    trunc = {}
    for i in ids:
        t, a = str(papers[i]["title"]).strip(), str(papers[i]["abstract"]).strip()
        row = {"title_chars": len(t), "abstract_chars": len(a),
               "title_cut_by_256": len(t) > 256, "abstract_cut_by_1024": len(a) > 1024,
               "abstract_chars_dropped": max(0, len(a) - 1024)}
        for name, (tk, lim) in toks.items():
            if name == "bge_m3":
                full = bge_text(t, a, "title_abstract", papers[i]["recipe"])
                full_nocut = t + "\n\n" + a
                row["bge_m3_tokens_production_text"] = len(tk(full)["input_ids"])
                row["bge_m3_tokens_untruncated_text"] = len(tk(full_nocut)["input_ids"])
                row["bge_m3_hits_max_length_512"] = row["bge_m3_tokens_production_text"] > lim
            else:
                sep = tk.sep_token if name == "specter2" else " "
                n = len(tk(t + sep + a)["input_ids"])
                row[f"{name}_tokens"] = n
                row[f"{name}_truncated_at_{lim}"] = n > lim
        trunc[i] = row
    out["truncation"] = trunc

    # ── retrieval ranks ──────────────────────────────────────────────────────
    use_fts = fts_svc.is_available()
    retr = {}
    for anc in (A, B):
        if anc not in sp:
            continue
        t0 = time.time()
        dense, status, per_shard = dense_neighbours(sp[anc]["vector"])
        others = [x for x in dense if x["arxiv_id"] != anc]
        r = {"shard_status": status, "dense_seconds": round(time.time() - t0, 1),
             "dense": ranks(dense, anc, ids),
             "dense_cutoff_score_at_depth": others[DEPTH - 1]["score"] if len(others) >= DEPTH else None}
        missing = {c: cross[f"{anc}|{c}"]["qdrant_server_score"]
                   for c, v in r["dense"].items() if v["rank"] is None}
        r["dense_rank_lower_bounds"] = rank_lower_bounds(sp[anc]["vector"], anc, per_shard, missing)
        lex = {}
        if use_fts:
            t1 = time.time()
            lex["fts5"] = fts_svc._search_sync(prod_text[anc], DEPTH + 1)
            r["fts5_seconds"] = round(time.time() - t1, 1)
        else:
            r["fts5_error"] = "sidecar FTS index unavailable"
        lex["zilliz"] = zilliz_svc._run_sparse_search(fresh_sparse[anc], DEPTH + 1)
        for name, lst in lex.items():
            r[f"lexical_{name}"] = ranks(lst, anc, ids)
            fused = hybrid_search_svc._rrf_fuse_multi([dense, lst], k=60)
            r[f"hybrid_dense+{name}"] = ranks(fused, anc, ids, key="rrf_score")
        for k in [k for k in r if k == "dense" or k.startswith(("lexical_", "hybrid_"))]:
            for v in r[k].values():
                v["bucket"] = bucket(v["rank"])
        retr[anc] = r
        log(STAGE, f"retrieval anchor {anc}: dense rank of partner = "
                   f"{r['dense'][B if anc == A else A]}")
    out["retrieval"] = retr
    save_json("integrity_retrieval.json", out)

    # ── stored-vector dense baseline ─────────────────────────────────────────
    bl = load_json("baseline.json")
    from app import qdrant_svc
    need = sorted({x for s in ("random", "cscl_lg") for p in bl[s] for x in p})
    vecs: dict = {}
    for s in range(0, len(need), 200):
        vecs.update(asyncio.run(qdrant_svc.get_paper_vectors(need[s:s + 200])))
    base = {"n_requested": len(need), "n_found": len(vecs)}
    for s in ("random", "cscl_lg"):
        sc = [cos(vecs[l], vecs[r]) for l, r in bl[s] if l in vecs and r in vecs]
        base[s] = {"scores": sc, "n_pairs": len(sc), "n_missing_pairs": len(bl[s]) - len(sc)}
    save_json("baseline_stored_dense.json", base)
    log(STAGE, f"stored baseline vectors {len(vecs)}/{len(need)}; done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
