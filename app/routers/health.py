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
