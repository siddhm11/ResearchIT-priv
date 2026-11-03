"""Inspect route configuration; opt in to bounded live access checks.

  .venv/bin/python scripts/check_groq_routes.py
  .venv/bin/python scripts/check_groq_routes.py --probe
  .venv/bin/python scripts/check_groq_routes.py --probe --complete

The last command makes one short completion per credential/model. It is an
access/parameter smoke test, not a quality benchmark. No secret values or model
output are printed. Uses existing environment/.env.local configuration.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, llm_router  # noqa: E402


async def inspect_routes(probe: bool = False, complete: bool = False) -> dict:
    """Return sanitized evidence; completions use isolated routers and no DB."""
    credentials = llm_router.credentials_from_config()
    orders = {task: llm_router.configured_models(task) for task in llm_router.TASK_BUDGETS}
    models = list(dict.fromkeys(model for order in orders.values() for model in order))
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "failover_enabled": config.GROQ_FAILOVER_ENABLED,
              "failover_tasks": sorted(config.GROQ_FAILOVER_TASKS),
              "mode": "completion_probe" if complete else "catalog_probe" if probe else "configuration_only",
              "model_order": orders, "routes": [], "quality_evaluated": False}
    for credential in credentials:
        row = {"credential": credential.alias,
               "quota_membership_known": credential.org != "unknown",
               "catalog": "not_checked", "models": {}}
        report["routes"].append(row)
        if not probe:
            continue
        try:
            from groq import AsyncGroq
            async with AsyncGroq(api_key=credential.secret, max_retries=0, timeout=5) as client:
                async with asyncio.timeout(5):
                    catalog = await client.models.list()
            active = {model.id for model in catalog.data}
            row["catalog"] = "success"
            row["models"] = {model: {"listed": model in active} for model in models}
        except Exception as exc:
            row["catalog"] = "failed"
            row["status_code"] = getattr(exc, "status_code", None)
            name = type(exc).__name__
            row["error_category"] = name if name in {
                "APIConnectionError", "APITimeoutError", "TimeoutError",
                "AuthenticationError", "PermissionDeniedError", "ImportError",
                "ModuleNotFoundError"} else "request_error"
            # No exception body, request, token, endpoint, or provider message.
            continue
        if complete:
            router = llm_router.Router((credential,))
            try:
                for model in models:
                    result = await router.complete(
                        "rewrite", [{"role": "user", "content": "Reply with exactly OK."}],
                        (model,), lambda text: text.strip() == "OK", budget=5)
                    row["models"][model].update(
                        outcome=result.outcome, elapsed_ms=result.elapsed_ms,
                        attempts=result.attempts, total_tokens=result.total_tokens)
            finally:
                await router.aclose()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Call the live model catalog")
    parser.add_argument("--complete", action="store_true", help="Also make short live completions")
    args = parser.parse_args()
    if args.complete and not args.probe:
        parser.error("--complete requires --probe")
    try:
        report = asyncio.run(inspect_routes(args.probe, args.complete))
    except ValueError:
        print(json.dumps({"error": "invalid_route_configuration"}))
        raise SystemExit(1) from None
    print(json.dumps(report, indent=2))
    if not report["routes"] or any(row["catalog"] == "failed" for row in report["routes"]):
        raise SystemExit(1)
    if args.complete and any(item.get("outcome") != "success"
                             for row in report["routes"] for item in row["models"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
