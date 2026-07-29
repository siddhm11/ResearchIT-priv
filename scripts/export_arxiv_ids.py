"""
Export all arXiv IDs from Turso DB to arxiv_ids.txt.

Uses the same Turso HTTP pipeline API as turso_svc.py.
Paginates with LIMIT/OFFSET to handle 1.6M rows.

Usage:
  set TURSO_URL=libsql://...
  set TURSO_DB_TOKEN=...
  python scripts/export_arxiv_ids.py
"""
import os
import sys
import time
import httpx

BATCH_SIZE = 50_000  # rows per query (Turso handles this fine)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "arxiv_ids.txt")


def get_turso_config():
    url = os.getenv("TURSO_URL", "")
    token = os.getenv("TURSO_DB_TOKEN", "")
    if not url or not token:
        print("ERROR: Set TURSO_URL and TURSO_DB_TOKEN environment variables.")
        print("  Example:")
        print("    set TURSO_URL=libsql://your-db.turso.io")
        print("    set TURSO_DB_TOKEN=your-token")
        sys.exit(1)

    # Convert to HTTPS
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://"):]
    elif not url.startswith("https://"):
        url = "https://" + url

    return url.rstrip("/"), token


def turso_query(url: str, token: str, sql: str, args: list = None) -> list[list]:
    """Execute a query via Turso HTTP pipeline API. Returns list of rows."""
    stmt = {"sql": sql}
    if args:
        stmt["args"] = args

    payload = {
        "requests": [
            {"type": "execute", "stmt": stmt},
            {"type": "close"},
        ]
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    resp = httpx.post(
        f"{url}/v2/pipeline",
        json=payload,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Parse response
    result = data.get("results", [])
    if not result:
        return []

    execute_result = result[0]
    if execute_result.get("type") == "error":
        raise RuntimeError(f"Turso error: {execute_result.get('error')}")

    response = execute_result.get("response", {})
    result_data = response.get("result", {})
    rows = result_data.get("rows", [])

    # Each row is a list of {"type": "text", "value": "..."} dicts
    return [[col.get("value") for col in row] for row in rows]


def main():
    url, token = get_turso_config()

    # First, get total count
    print("[export] Counting papers in Turso...")
    count_rows = turso_query(url, token, "SELECT COUNT(*) FROM papers")
    total = int(count_rows[0][0]) if count_rows else 0
    print(f"[export] Found {total:,} papers")

    if total == 0:
        print("ERROR: No papers found. Check your Turso connection.")
        sys.exit(1)

    # Paginate and collect all IDs
    all_ids = []
    offset = 0
    t0 = time.perf_counter()

    while offset < total:
        batch_start = time.perf_counter()
        rows = turso_query(
            url, token,
            f"SELECT arxiv_id FROM papers LIMIT {BATCH_SIZE} OFFSET {offset}"
        )
        batch_ms = (time.perf_counter() - batch_start) * 1000

        batch_ids = [row[0] for row in rows if row[0]]
        all_ids.extend(batch_ids)
        offset += BATCH_SIZE

        pct = min(100, offset * 100 / total)
        print(f"[export] {len(all_ids):>10,} / {total:,} ({pct:.0f}%)  "
              f"batch: {len(batch_ids):,} in {batch_ms:.0f}ms")

        if len(rows) < BATCH_SIZE:
            break  # No more rows

    elapsed = time.perf_counter() - t0
    print(f"\n[export] Collected {len(all_ids):,} arXiv IDs in {elapsed:.1f}s")

    # Write to file
    output_path = os.path.abspath(OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        for aid in all_ids:
            f.write(aid + "\n")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[export] Written to: {output_path}")
    print(f"[export] File size: {file_size_mb:.1f} MB")
    print(f"[export] Lines: {len(all_ids):,}")
    print(f"\n✅ Done! Feed this file to the ML Intern's Script 1:")
    print(f"   python 01_fetch_citation_edges.py --corpus-file arxiv_ids.txt")


if __name__ == "__main__":
    main()
