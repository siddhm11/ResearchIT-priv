"""
Step 1: Fetch citation edges from Semantic Scholar API.

Produces: citations.parquet → (citing_arxiv_id, cited_arxiv_id)
          where BOTH IDs exist in the ResearchIT Qdrant corpus.

Usage:
  # Option A: Batch API (no API key needed, slower, ~1-2 hours for 1.6M papers)
  python 01_fetch_citation_edges.py --corpus-file arxiv_ids.txt --output citations.parquet

  # Option B: Batch API with API key (faster, ~30-60 min)
  python 01_fetch_citation_edges.py --corpus-file arxiv_ids.txt --output citations.parquet --api-key YOUR_KEY

  # Option C: If you already have the S2 bulk datasets downloaded:
  python 01_fetch_citation_edges.py --bulk-papers paper-ids.jsonl.gz --bulk-citations citations.jsonl.gz \
      --corpus-file arxiv_ids.txt --output citations.parquet

Prerequisites:
  - arxiv_ids.txt: one arXiv ID per line (e.g. "2303.14957"), exported from Qdrant/Turso
  - pip install httpx pyarrow tqdm

Output schema:
  citing_arxiv_id  (string)  — the paper that contains the citation
  cited_arxiv_id   (string)  — the paper being cited
  is_influential   (bool)    — S2's influential citation flag (if available)

Author: ResearchIT ML Pipeline — Phase 6, Step 1
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ── Constants ────────────────────────────────────────────────────────────────

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_FIELDS = "externalIds,references.externalIds"
BATCH_SIZE = 500        # S2 hard limit
MAX_RETRIES = 5         # per batch
RETRY_BACKOFF_BASE = 2  # exponential backoff base (seconds)
CHECKPOINT_EVERY = 50   # save checkpoint every N batches


# ── Batch API Path ───────────────────────────────────────────────────────────

async def fetch_one_batch(
    client: httpx.AsyncClient,
    arxiv_ids: list[str],
    api_key: str | None,
    batch_idx: int,
) -> list[tuple[str, str, bool]]:
    """
    Fetch references for a batch of arXiv IDs via S2 batch endpoint.

    Returns list of (citing_arxiv_id, cited_arxiv_id, is_influential) tuples.
    Only returns edges where the cited paper has an arXiv ID.
    (In-corpus filtering happens later.)
    """
    # Format IDs for S2: "arXiv:2303.14957"
    s2_ids = [f"arXiv:{aid}" for aid in arxiv_ids]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BATCH_URL}?fields={S2_BATCH_FIELDS}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                url,
                json={"ids": s2_ids},
                headers=headers,
                timeout=30.0,
            )

            if resp.status_code == 200:
                results = resp.json()
                edges = []
                for i, paper in enumerate(results):
                    if paper is None:
                        continue
                    citing_id = arxiv_ids[i]
                    refs = paper.get("references") or []
                    for ref in refs:
                        ext_ids = ref.get("externalIds") or {}
                        cited_arxiv = ext_ids.get("ArXiv")
                        if cited_arxiv:
                            edges.append((citing_id, cited_arxiv, False))
                return edges

            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", RETRY_BACKOFF_BASE ** attempt))
                print(f"  [batch {batch_idx}] Rate limited. Waiting {retry_after}s (attempt {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(retry_after)

            elif resp.status_code == 400:
                print(f"  [batch {batch_idx}] Bad request (400). Skipping batch.")
                return []

            else:
                print(f"  [batch {batch_idx}] HTTP {resp.status_code}. Retrying (attempt {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
            print(f"  [batch {batch_idx}] Network error: {type(e).__name__}. Retrying (attempt {attempt+1}/{MAX_RETRIES})")
            await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)

    print(f"  [batch {batch_idx}] FAILED after {MAX_RETRIES} attempts. Skipping.")
    return []


async def fetch_all_batches(
    corpus_ids: list[str],
    api_key: str | None,
    checkpoint_dir: Path,
) -> list[tuple[str, str, bool]]:
    """
    Fetch citation edges for all corpus IDs using the S2 batch API.
    Supports checkpoint/resume.
    """
    # Check for existing checkpoint
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    all_edges: list[tuple[str, str, bool]] = []
    start_batch = 0

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        start_batch = ckpt["next_batch"]
        # Load previously saved edges
        edges_file = checkpoint_dir / "edges_partial.jsonl"
        if edges_file.exists():
            with open(edges_file) as f:
                for line in f:
                    row = json.loads(line)
                    all_edges.append((row["citing"], row["cited"], row["influential"]))
        print(f"Resuming from batch {start_batch} ({len(all_edges)} edges already collected)")

    # Split into batches
    batches = []
    for i in range(0, len(corpus_ids), BATCH_SIZE):
        batches.append(corpus_ids[i : i + BATCH_SIZE])

    total_batches = len(batches)
    print(f"Total: {len(corpus_ids)} papers → {total_batches} batches of {BATCH_SIZE}")
    print(f"Starting from batch {start_batch}")

    # Rate limiting: 1 req/s without key, slightly faster with key
    delay = 0.5 if api_key else 1.1

    edges_file = checkpoint_dir / "edges_partial.jsonl"

    async with httpx.AsyncClient() as client:
        pbar = tqdm(range(start_batch, total_batches), initial=start_batch, total=total_batches)
        for batch_idx in pbar:
            batch = batches[batch_idx]

            edges = await fetch_one_batch(client, batch, api_key, batch_idx)
            all_edges.extend(edges)

            # Append edges to partial file
            with open(edges_file, "a") as f:
                for citing, cited, influential in edges:
                    f.write(json.dumps({"citing": citing, "cited": cited, "influential": influential}) + "\n")

            pbar.set_postfix({"edges": len(all_edges), "batch_edges": len(edges)})

            # Checkpoint periodically
            if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
                with open(checkpoint_file, "w") as f:
                    json.dump({"next_batch": batch_idx + 1}, f)

            await asyncio.sleep(delay)

    # Final checkpoint
    with open(checkpoint_file, "w") as f:
        json.dump({"next_batch": total_batches, "status": "complete"}, f)

    return all_edges


# ── Bulk Download Path ───────────────────────────────────────────────────────

def process_bulk_downloads(
    papers_file: str,
    citations_file: str,
    corpus_set: set[str],
) -> list[tuple[str, str, bool]]:
    """
    Process S2 bulk dataset downloads to extract in-corpus citation edges.

    papers_file:    paper-ids.jsonl.gz (corpusid → externalIds mapping)
    citations_file: citations.jsonl.gz (citingcorpusid → citedcorpusid edges)
    """
    print("Step 1/2: Building corpusid → arxiv_id mapping from paper-ids...")
    corpusid_to_arxiv: dict[int, str] = {}
    with gzip.open(papers_file, "rt") as f:
        for line in tqdm(f, desc="Reading paper-ids"):
            try:
                rec = json.loads(line)
                ext_ids = rec.get("externalids") or rec.get("externalIds") or {}
                arxiv_id = ext_ids.get("ArXiv")
                corpus_id = rec.get("corpusid") or rec.get("corpusId")
                if arxiv_id and corpus_id and arxiv_id in corpus_set:
                    corpusid_to_arxiv[int(corpus_id)] = arxiv_id
            except (json.JSONDecodeError, ValueError):
                continue

    print(f"  Mapped {len(corpusid_to_arxiv)} corpus IDs to arXiv IDs in your corpus")

    print("Step 2/2: Filtering citation edges to in-corpus pairs...")
    edges: list[tuple[str, str, bool]] = []
    with gzip.open(citations_file, "rt") as f:
        for line in tqdm(f, desc="Reading citations"):
            try:
                rec = json.loads(line)
                citing_cid = rec.get("citingcorpusid") or rec.get("citingCorpusId")
                cited_cid = rec.get("citedcorpusid") or rec.get("citedCorpusId")
                is_influential = rec.get("isinfluential", False) or rec.get("isInfluential", False)

                citing_arxiv = corpusid_to_arxiv.get(int(citing_cid)) if citing_cid else None
                cited_arxiv = corpusid_to_arxiv.get(int(cited_cid)) if cited_cid else None

                if citing_arxiv and cited_arxiv:
                    edges.append((citing_arxiv, cited_arxiv, bool(is_influential)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    print(f"  Found {len(edges)} in-corpus citation edges")
    return edges


# ── Filter & Save ────────────────────────────────────────────────────────────

def filter_and_save(
    edges: list[tuple[str, str, bool]],
    corpus_set: set[str],
    output_path: str,
):
    """
    Filter edges to in-corpus pairs, deduplicate, and save as parquet.
    """
    print(f"Raw edges before filtering: {len(edges)}")

    # Filter: both citing and cited must be in corpus
    filtered = [
        (citing, cited, influential)
        for citing, cited, influential in edges
        if citing in corpus_set and cited in corpus_set and citing != cited
    ]
    print(f"In-corpus edges (both sides in corpus): {len(filtered)}")

    # Deduplicate
    seen = set()
    deduped = []
    for citing, cited, influential in filtered:
        key = (citing, cited)
        if key not in seen:
            seen.add(key)
            deduped.append((citing, cited, influential))

    print(f"After deduplication: {len(deduped)}")

    # Save as parquet
    table = pa.table({
        "citing_arxiv_id": pa.array([e[0] for e in deduped], type=pa.string()),
        "cited_arxiv_id": pa.array([e[1] for e in deduped], type=pa.string()),
        "is_influential": pa.array([e[2] for e in deduped], type=pa.bool_()),
    })

    pq.write_table(table, output_path, compression="snappy")
    print(f"Saved {len(deduped)} citation edges to {output_path}")

    # Print stats
    citing_papers = set(e[0] for e in deduped)
    cited_papers = set(e[1] for e in deduped)
    print(f"\nStats:")
    print(f"  Unique citing papers: {len(citing_papers)}")
    print(f"  Unique cited papers:  {len(cited_papers)}")
    print(f"  Unique papers total:  {len(citing_papers | cited_papers)}")
    print(f"  Avg references per citing paper: {len(deduped) / max(len(citing_papers), 1):.1f}")
    influential_count = sum(1 for e in deduped if e[2])
    print(f"  Influential citations: {influential_count} ({100*influential_count/max(len(deduped),1):.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def load_corpus_ids(path: str) -> list[str]:
    """Load arXiv IDs from a text file (one per line)."""
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle various formats: "2303.14957", "arXiv:2303.14957", etc.
                if line.startswith("arXiv:"):
                    line = line[6:]
                elif line.startswith("ARXIV:"):
                    line = line[6:]
                ids.append(line)
    print(f"Loaded {len(ids)} arXiv IDs from {path}")
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Fetch citation edges from Semantic Scholar for ResearchIT corpus"
    )
    parser.add_argument(
        "--corpus-file", required=True,
        help="Text file with one arXiv ID per line (e.g. arxiv_ids.txt)"
    )
    parser.add_argument(
        "--output", default="citations.parquet",
        help="Output parquet file path (default: citations.parquet)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Semantic Scholar API key (optional, speeds up rate limit)"
    )
    parser.add_argument(
        "--bulk-papers", default=None,
        help="Path to S2 bulk paper-ids.jsonl.gz (use bulk download path)"
    )
    parser.add_argument(
        "--bulk-citations", default=None,
        help="Path to S2 bulk citations.jsonl.gz (use bulk download path)"
    )
    parser.add_argument(
        "--checkpoint-dir", default="./citation_checkpoint",
        help="Directory for checkpoint files (batch API mode)"
    )
    parser.add_argument(
        "--max-papers", type=int, default=None,
        help="Limit to first N papers (for testing)"
    )

    args = parser.parse_args()

    # Load corpus
    corpus_ids = load_corpus_ids(args.corpus_file)
    if args.max_papers:
        corpus_ids = corpus_ids[:args.max_papers]
        print(f"  Limited to {len(corpus_ids)} papers (--max-papers)")

    corpus_set = set(corpus_ids)

    # Choose path
    if args.bulk_papers and args.bulk_citations:
        print("\n=== BULK DOWNLOAD PATH ===")
        edges = process_bulk_downloads(args.bulk_papers, args.bulk_citations, corpus_set)
    else:
        print("\n=== BATCH API PATH ===")
        if not args.api_key:
            # Check environment variable
            args.api_key = os.environ.get("S2_API_KEY")
        if args.api_key:
            print(f"Using API key: {args.api_key[:8]}...")
        else:
            print("No API key — using unauthenticated rate (1 req/s)")
            print("Get a free key at: https://www.semanticscholar.org/product/api#Partner-Form")

        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        edges = asyncio.run(fetch_all_batches(corpus_ids, args.api_key, checkpoint_dir))

    # Filter to in-corpus and save
    filter_and_save(edges, corpus_set, args.output)

    print(f"\n✅ Done! Citation edges saved to: {args.output}")


if __name__ == "__main__":
    main()
