"""
Turso (libSQL) metadata service — Phase 3.5.

Replaces arxiv_svc.fetch_metadata_batch() with direct Turso DB lookups.
Uses Turso's HTTP pipeline API — no additional Python dependencies needed
(just httpx, already installed).

The DB contains ~1.6M arXiv papers with metadata + citation counts from
Semantic Scholar, bulk-loaded from Kaggle.

Connection: TURSO_URL + TURSO_DB_TOKEN (env vars)
Table:      papers (arxiv_id UNIQUE INDEX)
"""
from __future__ import annotations

import json
import time

import httpx

from app import config


# ── Public API ───────────────────────────────────────────────────────────────

async def fetch_metadata(arxiv_id: str) -> dict | None:
    """Fetch metadata for a single paper from Turso."""
    result = await fetch_metadata_batch([arxiv_id])
    return result.get(arxiv_id)


async def fetch_metadata_batch(arxiv_ids: list[str]) -> dict[str, dict]:
    """
    Fetch metadata for multiple papers from Turso DB.

    Returns {arxiv_id: paper_dict} for all IDs found.
    Paper dict has keys: arxiv_id, title, abstract, authors, category,
    published, year, citation_count, influential_citations.

    Uses Turso HTTP pipeline API — single HTTP request for all IDs.
    """
    if not arxiv_ids:
        return {}

    url = config.TURSO_URL
    token = config.TURSO_DB_TOKEN

    if not url or not token:
        print("[turso] TURSO_URL or TURSO_DB_TOKEN not configured, skipping")
        return {}

    # Build parameterised query with placeholders
    placeholders = ", ".join(["?" for _ in arxiv_ids])
    sql = f"SELECT arxiv_id, title, authors, categories, primary_topic, update_date, abstract_preview, citation_count, influential_citations FROM papers WHERE arxiv_id IN ({placeholders})"

    args = [{"type": "text", "value": aid} for aid in arxiv_ids]

    # Turso HTTP pipeline API
    pipeline_url = url.rstrip("/")
    # Convert to HTTP API URL format
    if pipeline_url.startswith("libsql://"):
        pipeline_url = pipeline_url.replace("libsql://", "https://")
    if not pipeline_url.startswith("https://"):
        pipeline_url = "https://" + pipeline_url.lstrip("https://").lstrip("http://")

    payload = {
        "requests": [
            {
                "type": "execute",
                "stmt": {"sql": sql, "args": args},
            },
            {"type": "close"},
        ]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{pipeline_url}/v2/pipeline",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
    except Exception as e:
        print(f"[turso] HTTP request failed: {e}")
        return {}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[turso] Fetched metadata for {len(arxiv_ids)} IDs in {elapsed_ms:.0f}ms")

    try:
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return {}

        # First result is our execute response
        execute_result = results[0]
        if execute_result.get("type") == "error":
            print(f"[turso] Query error: {execute_result.get('error')}")
            return {}

        response = execute_result.get("response", {})
        result_data = response.get("result", {})
        cols = [c["name"] for c in result_data.get("cols", [])]
        rows = result_data.get("rows", [])

    except (KeyError, IndexError, TypeError) as e:
        print(f"[turso] Response parsing error: {e}")
        return {}

    # Convert rows to paper dicts matching the expected format
    output: dict[str, dict] = {}
    for row in rows:
        # Each row is a list of {"type": "text"|"integer"|"null", "value": ...}
        values = {}
        for i, col in enumerate(cols):
            cell = row[i]
            if cell.get("type") == "null":
                values[col] = None
            else:
                values[col] = cell.get("value", "")

        paper = _to_paper_dict(values)
        if paper:
            output[paper["arxiv_id"]] = paper

    return output


def _to_paper_dict(row: dict) -> dict | None:
    """
    Convert a Turso row into the paper dict format expected by templates.

    Template expects:
      arxiv_id, title, abstract, authors (JSON string), category, published, year
    Turso provides:
      arxiv_id, title, authors (comma-sep), categories, primary_topic,
      update_date, abstract_preview, citation_count, influential_citations
    """
    arxiv_id = row.get("arxiv_id")
    if not arxiv_id:
        return None

    # Convert authors from comma-separated to JSON array string
    authors_raw = row.get("authors") or ""
    if authors_raw.startswith("["):
        # Already JSON — leave as is
        authors_json = authors_raw
    else:
        # Comma-separated → JSON array (take first 5)
        author_list = [a.strip() for a in authors_raw.split(",") if a.strip()][:5]
        authors_json = json.dumps(author_list)

    # Use primary_topic as category, fall back to first in categories list
    category = row.get("primary_topic") or ""
    if not category:
        cats = row.get("categories") or ""
        category = cats.split()[0] if cats else ""

    # Extract year from update_date (YYYY-MM-DD format)
    update_date = row.get("update_date") or ""
    year = 0
    if len(update_date) >= 4:
        try:
            year = int(update_date[:4])
        except ValueError:
            pass

    # Citation count (bonus data from Semantic Scholar)
    citation_count = 0
    try:
        citation_count = int(row.get("citation_count") or 0)
    except (ValueError, TypeError):
        pass

    influential = 0
    try:
        influential = int(row.get("influential_citations") or 0)
    except (ValueError, TypeError):
        pass

    return {
        "arxiv_id": arxiv_id,
        "title": (row.get("title") or "").replace("\n", " "),
        "abstract": (row.get("abstract_preview") or "").replace("\n", " "),
        "authors": authors_json,
        "category": category,
        "published": update_date,
        "year": year,
        "citation_count": citation_count,
        "influential_citations": influential,
    }
