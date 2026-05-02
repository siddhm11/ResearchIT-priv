"""
Health check routes.

Phase 6.3: /healthz/reranker — verify LightGBM model deployment status.
"""
import hashlib
import json
from fastapi import APIRouter
from app.recommend import reranker as _rr

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

    return {
        "model_loaded": _rr.is_model_loaded(),
        "model_path": _rr.get_loaded_model_path(),
        "model_version": "phase6.v1",
        "fallback_active": not _rr.is_model_loaded(),
        "feature_count": _rr.NUM_FEATURES,
        "feature_schema_hash": schema_hash,
        "n_trees": _rr.get_num_trees(),
    }
