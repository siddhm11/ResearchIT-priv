"""
Health check routes.

Endpoints:
  /healthz/reranker  -- Phase 6.3: verify LightGBM model deployment status
  /healthz/deep      -- Deep health check: ping Qdrant, Zilliz, Turso (keepalive)
"""
import asyncio
import hashlib
import json
import time
from fastapi import APIRouter
from app.recommend import reranker as _rr
from app import config

router = APIRouter()


@router.get("/healthz/reranker")
async def healthz_reranker():
    """
    Report the live status of the LightGBM reranker.

    Used to verify deployment:
      curl https://siddhm11-researchit.hf.space/healthz/reranker

    Expected: model_loaded=true, n_trees=141, fallback_active=false
    """
    schema_hash = hashlib.sha256(
        json.dumps(_rr.FEATURE_NAMES).encode()
    ).hexdigest()[:12]

    # model_loaded and scoring_with are deliberately separate. The model can be
    # present and healthy while not being used, because RERANKER_MODE defaults
    # to "heuristic" — the trained model has zero splits on features 20-30 and
    # so cannot respond to the user. Without both fields a policy choice is
    # indistinguishable from a failed deployment.
    return {
        "model_loaded": _rr.is_model_loaded(),
        "model_path": _rr.get_loaded_model_path(),
        "model_version": "phase6.v1",
        "reranker_mode": config.RERANKER_MODE,
        "scoring_with": "lightgbm" if _rr.use_lightgbm() else "heuristic",
        "fallback_active": not _rr.use_lightgbm(),
        "feature_count": _rr.NUM_FEATURES,
        "feature_schema_hash": schema_hash,
        "n_trees": _rr.get_num_trees(),
    }


@router.get("/healthz/ab")
async def healthz_ab(q: str = "", limit: int = 10):
    """
    Online A/B: run one query against both Qdrant backends and diff the results.

        GET /healthz/ab?q=transformer+attention+mechanism

    Needs QDRANT_B_URL / QDRANT_B_API_KEY (plus QDRANT_B_UINT8_* if backend B
    stores uint8). Without them it reports "not configured" and does nothing.

    Deliberately one process rather than a second Space: standing up a duplicate
    deployment would compare containers, hosts and cache states as well as the
    clusters. Here the query is encoded ONCE and the only difference between the
    two runs is which cluster answered.
    """
    if not q.strip():
        return {"error": "pass ?q=<query>"}
    if not config.qdrant_b_configured():
        return {"error": "backend B not configured",
                "need": ["QDRANT_B_URL", "QDRANT_B_API_KEY",
                         "QDRANT_B_UINT8_LO", "QDRANT_B_UINT8_SCALE"]}

    import time as _t
    from app import embed_svc, qdrant_svc, turso_svc

    loop = asyncio.get_running_loop()
    t0 = _t.perf_counter()
    try:
        dense, _sparse = await loop.run_in_executor(
            None, embed_svc.encode_query, q.strip())
    except Exception as e:
        return {"error": f"encode failed: {e}"}
    encode_ms = int((_t.perf_counter() - t0) * 1000)
    qvec = dense.tolist()

    async def run(backend: str) -> dict:
        t = _t.perf_counter()
        try:
            with qdrant_svc.use_backend(backend):
                hits = await qdrant_svc.search_dense(qvec, limit=limit)
        except Exception as e:
            return {"error": str(e)[:200]}
        ms = int((_t.perf_counter() - t) * 1000)
        ids = [h["arxiv_id"] for h in hits]
        meta = await turso_svc.fetch_metadata_batch(ids)
        return {
            "ms": ms,
            "results": [
                {"arxiv_id": h["arxiv_id"],
                 "score": round(float(h["score"]), 5),
                 "title": (meta.get(h["arxiv_id"], {}).get("title") or "")[:90]}
                for h in hits
            ],
        }

    a = await run("a")
    b = await run("b")

    out = {
        "query": q.strip(),
        "encode_ms": encode_ms,
        "A": {"url": config.QDRANT_URL.split("//")[-1][:24],
              "uint8": config.qdrant_is_uint8(), **a},
        "B": {"url": config.QDRANT_B_URL.split("//")[-1][:24],
              "uint8": config.QDRANT_B_UINT8_SCALE > 0, **b},
    }

    ra = [r["arxiv_id"] for r in a.get("results", [])]
    rb = [r["arxiv_id"] for r in b.get("results", [])]
    if ra and rb:
        sa, sb = set(ra), set(rb)
        out["overlap"] = round(len(sa & sb) / max(1, len(sa | sb)), 3)
        out["same_top1"] = ra[0] == rb[0]
        out["only_in_A"] = sorted(sa - sb)[:5]
        out["only_in_B"] = sorted(sb - sa)[:5]
        # Scores from a uint8 collection sit in a much narrower band because of
        # the DC offset, so compare spreads rather than absolute values.
        for k, rs in (("A", a["results"]), ("B", b["results"])):
            if rs:
                out[k]["score_spread"] = round(rs[0]["score"] - rs[-1]["score"], 5)
    return out


@router.get("/healthz/deep")
async def healthz_deep():
    """
    Deep health check — pings all external services to keep them alive.

    Used by GitHub Actions cron to prevent free-tier sleep:
      curl https://siddhm11-researchit.hf.space/healthz/deep

    Returns JSON with status and timing for each service.
    The request itself keeps HuggingFace Spaces awake.
    """
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "services": {},
    }
    loop = asyncio.get_running_loop()

    # ── Ping Qdrant ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        from app.qdrant_svc import _client
        client = _client()
        # Run synchronous Qdrant call in executor to avoid blocking event loop
        info = await loop.run_in_executor(
            None, client.get_collection, config.QDRANT_COLLECTION
        )
        results["services"]["qdrant"] = {
            "status": "ok",
            "collection": config.QDRANT_COLLECTION,
            "points_count": info.points_count,
            "time_ms": int((time.perf_counter() - t0) * 1000),
        }
    except Exception as e:
        results["services"]["qdrant"] = {
            "status": "error",
            "error": str(e),
            "time_ms": int((time.perf_counter() - t0) * 1000),
        }

    # ── Ping Zilliz ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        from app.zilliz_svc import _get_client
        client = _get_client()
        # Run synchronous Zilliz call in executor to avoid blocking event loop
        collections = await loop.run_in_executor(
            None, client.list_collections
        )
        results["services"]["zilliz"] = {
            "status": "ok",
            "collections_count": len(collections),
            "time_ms": int((time.perf_counter() - t0) * 1000),
        }
    except Exception as e:
        results["services"]["zilliz"] = {
            "status": "error",
            "error": str(e),
            "time_ms": int((time.perf_counter() - t0) * 1000),
        }

    # ── Ping Turso ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        if config.TURSO_URL and config.TURSO_DB_TOKEN:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{config.TURSO_URL}/v2/pipeline",
                    headers={
                        "Authorization": f"Bearer {config.TURSO_DB_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "requests": [
                            {"type": "execute", "stmt": {"sql": "SELECT 1"}},
                            {"type": "close"},
                        ]
                    },
                )
                resp.raise_for_status()
            results["services"]["turso"] = {
                "status": "ok",
                "time_ms": int((time.perf_counter() - t0) * 1000),
            }
        else:
            results["services"]["turso"] = {
                "status": "skipped",
                "reason": "TURSO_URL or TURSO_DB_TOKEN not configured",
                "time_ms": 0,
            }
    except Exception as e:
        results["services"]["turso"] = {
            "status": "error",
            "error": str(e),
            "time_ms": int((time.perf_counter() - t0) * 1000),
        }

    # ── Local metadata sidecar ───────────────────────────────────────────
    # Not a network service, but worth surfacing: if the image built without
    # it, every metadata lookup silently reverts to a cross-region Turso call
    # and search gets ~1.2s slower with no other symptom.
    # "skipped" rather than "error" when absent: the sidecar is an accelerator,
    # and its absence must not mark the deployment unhealthy.
    try:
        from app import local_meta
        info = local_meta.stats()
        info["status"] = "ok" if info.get("available") else "skipped"
        results["services"]["metadata_sidecar"] = info
    except Exception as e:
        results["services"]["metadata_sidecar"] = {
            "status": "skipped", "available": False, "error": str(e),
        }

    # ── Overall status ───────────────────────────────────────────────────
    all_ok = all(
        s.get("status") == "ok"
        for s in results["services"].values()
        if s.get("status") != "skipped"
    )
    results["overall"] = "healthy" if all_ok else "degraded"

    return results
