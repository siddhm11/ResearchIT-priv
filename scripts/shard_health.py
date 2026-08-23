#!/usr/bin/env python3
"""
Render /healthz/shard-latency as something you can read at a glance.

    # local dev server
    .venv/bin/python scripts/shard_health.py

    # the deployed Space
    .venv/bin/python scripts/shard_health.py --url https://siddhm11-researchit.hf.space

    # machine-readable, for a cron job or a notifier
    .venv/bin/python scripts/shard_health.py --json

Exit codes, so this can drive an alert without parsing anything:
    0  healthy
    1  a shard is slow or its spread is wide
    2  a shard is unreachable, or the endpoint is
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:7860"

# A shard slower than this in steady state is worth looking at. Not a hard SLO —
# the point of the number is to make "slow" a thing the script can decide rather
# than something a person has to eyeball every time.
SLOW_MS = 900

# A wide gap between fastest and slowest sample is the signature of a graph that
# only sometimes sits in page cache — which is exactly the on-disk question.
WIDE_SPREAD_MS = 600

_BAR = "█"


def _bar(ms: float, worst: float, width: int = 24) -> str:
    if worst <= 0:
        return ""
    return _BAR * max(1, round(ms / worst * width))


def fetch(base: str, samples: int, limit: int) -> dict:
    url = (f"{base.rstrip('/')}/healthz/shard-latency"
           f"?samples={samples}&limit={limit}")
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.load(r)


def render(data: dict) -> int:
    shards = data.get("shards", {})
    if not shards:
        print("no shards reported")
        return 2

    timed = {b: s for b, s in shards.items() if "median_ms" in s}
    broken = {b: s for b, s in shards.items() if "median_ms" not in s}
    worst = max((s["median_ms"] for s in timed.values()), default=1)

    print(f"\n  shard latency — {data.get('timestamp', '?')}")
    print(f"  probe: {data.get('probe', '?')}, limit {data.get('limit')}\n")
    print(f"  {'shard':<8}{'collection':<20}{'hnsw':<10}"
          f"{'median':>8}{'spread':>9}   ")
    print(f"  {'-' * 66}")

    for name, s in sorted(timed.items()):
        hnsw = "on disk" if s.get("hnsw_on_disk") else "in RAM"
        flag = ""
        if s["median_ms"] > SLOW_MS:
            flag = "  SLOW"
        elif s.get("spread_ms", 0) > WIDE_SPREAD_MS:
            flag = "  WIDE SPREAD"
        print(f"  {name:<8}{s.get('collection', '?'):<20}{hnsw:<10}"
              f"{s['median_ms']:>6}ms{s.get('spread_ms', 0):>7}ms"
              f"   {_bar(s['median_ms'], worst)}{flag}")

    for name, s in sorted(broken.items()):
        print(f"  {name:<8}{s.get('collection') or '?':<20}"
              f"{'—':<10}{'unreachable':>15}   {s.get('error', '')[:40]}")

    v = data.get("verdict")
    if v:
        print(f"\n  on disk {v['on_disk_shards']} median {v['on_disk_median_ms']}ms")
        print(f"  in RAM  {v['in_ram_shards']} median {v['in_ram_median_ms']}ms")
        print(f"  ratio   {v['ratio']}x")
        print(f"\n  {v['reading']}")
    elif len(timed) < 2:
        print("\n  only one shard reachable — no on-disk vs in-RAM comparison "
              "is possible from here.")
    else:
        print("\n  every reachable shard has the same hnsw setting, so there is "
              "nothing to compare.")

    print()
    if broken:
        return 2
    if any(s["median_ms"] > SLOW_MS or s.get("spread_ms", 0) > WIDE_SPREAD_MS
           for s in timed.values()):
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", action="store_true", help="raw JSON, no rendering")
    args = ap.parse_args()

    try:
        data = fetch(args.url, args.samples, args.limit)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} from {args.url} — is the endpoint deployed?",
              file=sys.stderr)
        return 2
    except Exception as e:
        print(f"could not reach {args.url}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    return render(data)


if __name__ == "__main__":
    sys.exit(main())
