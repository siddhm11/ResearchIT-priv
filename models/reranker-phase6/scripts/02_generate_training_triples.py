"""
Step 2: Generate LightGBM training triples from citation edges.

Produces: train.parquet + eval.parquet
  Each row = (query_arxiv_id, candidate_arxiv_id, label, feature_1, ..., feature_N)

Labels:
  2 = directly cited by query paper (strong positive)
  1 = co-cited with query paper (weak positive)
  0 = retrieved but not cited (negative)

Time-split:
  train: query papers published before 2023-01-01
  eval:  query papers published on or after 2023-01-01

Usage:
  python 02_generate_training_triples.py \
    --citations citations.parquet \
    --corpus-file arxiv_ids.txt \
    --qdrant-url https://YOUR_QDRANT_URL \
    --qdrant-api-key YOUR_KEY \
    --qdrant-collection arxiv_bgem3_dense \
    --turso-url https://YOUR_TURSO_URL \
    --turso-token YOUR_TOKEN \
    --output-dir ./ltr_dataset \
    --num-queries 100000 \
    --candidates-per-query 50

Prerequisites:
  - citations.parquet from Step 1
  - Qdrant Cloud access (ANN search + embedding retrieval)
  - Turso access (paper metadata)
  - pip install httpx pyarrow qdrant-client tqdm numpy

Feature Schema (37 features):
  See FEATURE_SCHEMA below for the full list.
  Features 1-20 are populated from citation graph + metadata.
  Features 21-27 are zero-filled (EWMA/cluster/suppression — need real users).
  All 37 feature columns are present so the model schema is stable.

Author: ResearchIT ML Pipeline — Phase 6, Step 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
except ImportError:
    print("ERROR: pip install qdrant-client")
    raise


# ── Feature Schema ───────────────────────────────────────────────────────────
# This defines ALL 37 features. Features 21-27 are zero-filled for pseudo-label
# training but will be populated when real user data is available.
#
# The schema is designed so that the LightGBM model trained on pseudo-labels
# can be retrained on real data without changing the feature layout.

FEATURE_SCHEMA = [
    # === Content/Retrieval features (populated during pseudo-label training) ===
    "qdrant_cosine_score",           # 0: ANN cosine similarity
    "candidate_position",            # 1: rank position in ANN results (0-indexed)
    "candidate_citation_count",      # 2: citation count of candidate paper
    "candidate_log_citations",       # 3: log(citation_count + 1)
    "candidate_influential_citations",  # 4: influential citation count
    "candidate_age_days",            # 5: days since candidate was published
    "candidate_recency_score",       # 6: exp(-0.002 * age_days) — matches heuristic
    "query_citation_count",          # 7: citation count of query/user paper
    "query_age_days",                # 8: days since query paper was published
    "year_diff",                     # 9: |query_year - candidate_year|
    "same_primary_category",         # 10: 1 if same primary arXiv category, else 0
    "co_citation_count",             # 11: papers that cite BOTH query and candidate
    "shared_author_count",           # 12: number of shared authors
    "candidate_is_newer",            # 13: 1 if candidate published after query, else 0
    "query_log_citations",           # 14: log(query_citation_count + 1)
    "citation_count_ratio",          # 15: candidate_citations / (query_citations + 1)
    "age_ratio",                     # 16: candidate_age / (query_age + 1)
    "candidate_citations_per_year",  # 17: citation_count / max(age_years, 0.5)
    "query_num_references",          # 18: how many papers the query paper cites (in-corpus)
    "candidate_num_cited_by",        # 19: how many corpus papers cite the candidate

    # === User behavior features (zero-filled for pseudo-labels, active for real users) ===
    "ewma_longterm_similarity",      # 20: cos(candidate, user long-term EWMA profile)
    "ewma_shortterm_similarity",     # 21: cos(candidate, user short-term EWMA profile)
    "ewma_negative_similarity",      # 22: cos(candidate, user negative EWMA profile)
    "cluster_importance",            # 23: importance weight of serving cluster
    "cluster_distance_to_medoid",    # 24: cos(candidate, cluster medoid)
    "is_suppressed_category",        # 25: 1 if candidate's category is suppressed
    "onboarding_category_match",     # 26: 1 if candidate matches user's onboarding categories

    # === Interaction features (zero-filled for pseudo-labels) ===
    "user_total_saves",              # 27: total papers user has saved
    "user_total_dismissals",         # 28: total papers user has dismissed
    "user_days_since_last_save",     # 29: days since user's last save
    "user_session_save_count",       # 30: saves in current session

    # === Cross features (computed from combinations) ===
    "cosine_x_recency",             # 31: qdrant_cosine_score × candidate_recency_score
    "cosine_x_citations",           # 32: qdrant_cosine_score × candidate_log_citations
    "category_x_recency",           # 33: same_primary_category × candidate_recency_score
    "cosine_x_cocitation",          # 34: qdrant_cosine_score × log(co_citation_count + 1)
    "position_inverse",             # 35: 1 / (candidate_position + 1)
    "citations_x_recency",          # 36: candidate_log_citations × candidate_recency_score
]

NUM_FEATURES = len(FEATURE_SCHEMA)  # 37
assert NUM_FEATURES == 37, f"Expected 37 features, got {NUM_FEATURES}"

# Time split cutoff
EVAL_CUTOFF = "2023-01-01"
EVAL_CUTOFF_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)


# ── Citation Graph Loading ───────────────────────────────────────────────────

def load_citation_graph(citations_path: str) -> tuple[dict, dict, dict]:
    """
    Load citation edges and build lookup structures.

    Returns:
        references: {citing_id: set(cited_ids)} — outgoing references
        cited_by:   {cited_id: set(citing_ids)} — incoming citations
        co_citation_counts: precomputed co-citation matrix (lazily computed per query)
    """
    table = pq.read_table(citations_path)
    citing_col = table.column("citing_arxiv_id").to_pylist()
    cited_col = table.column("cited_arxiv_id").to_pylist()

    references: dict[str, set[str]] = defaultdict(set)
    cited_by: dict[str, set[str]] = defaultdict(set)

    for citing, cited in zip(citing_col, cited_col):
        references[citing].add(cited)
        cited_by[cited].add(citing)

    print(f"Loaded citation graph:")
    print(f"  {len(references)} papers with outgoing references")
    print(f"  {len(cited_by)} papers with incoming citations")
    print(f"  {sum(len(v) for v in references.values())} total edges")

    return dict(references), dict(cited_by), {}


def compute_co_citation_count(
    query_id: str,
    candidate_id: str,
    cited_by: dict[str, set[str]],
) -> int:
    """Count papers that cite BOTH query and candidate."""
    citing_query = cited_by.get(query_id, set())
    citing_candidate = cited_by.get(candidate_id, set())
    return len(citing_query & citing_candidate)


# ── Turso Metadata Fetching ─────────────────────────────────────────────────

async def fetch_turso_metadata_batch(
    arxiv_ids: list[str],
    turso_url: str,
    turso_token: str,
) -> dict[str, dict]:
    """Fetch paper metadata from Turso DB."""
    if not arxiv_ids:
        return {}

    pipeline_url = turso_url.rstrip("/")
    if pipeline_url.startswith("libsql://"):
        pipeline_url = "https://" + pipeline_url[len("libsql://"):]
    elif not pipeline_url.startswith("https://"):
        pipeline_url = "https://" + pipeline_url

    placeholders = ", ".join(["?" for _ in arxiv_ids])
    sql = f"""SELECT arxiv_id, title, authors, primary_topic, update_date,
                     citation_count, influential_citations
              FROM papers WHERE arxiv_id IN ({placeholders})"""

    args = [{"type": "text", "value": aid} for aid in arxiv_ids]

    payload = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": args}},
            {"type": "close"},
        ]
    }

    headers = {
        "Authorization": f"Bearer {turso_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{pipeline_url}/v2/pipeline", json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return {}

    execute_result = results[0]
    if execute_result.get("type") == "error":
        print(f"[turso] Query error: {execute_result.get('error')}")
        return {}

    response = execute_result.get("response", {})
    result_data = response.get("result", {})
    cols = [c["name"] for c in result_data.get("cols", [])]
    rows = result_data.get("rows", [])

    output = {}
    for row in rows:
        values = {}
        for i, col in enumerate(cols):
            cell = row[i]
            values[col] = None if cell.get("type") == "null" else cell.get("value", "")

        arxiv_id = values.get("arxiv_id")
        if not arxiv_id:
            continue

        # Parse citation counts
        try:
            citation_count = int(values.get("citation_count") or 0)
        except (ValueError, TypeError):
            citation_count = 0
        try:
            influential = int(values.get("influential_citations") or 0)
        except (ValueError, TypeError):
            influential = 0

        # Parse authors
        authors_raw = values.get("authors") or ""
        if authors_raw.startswith("["):
            try:
                author_list = json.loads(authors_raw)
            except json.JSONDecodeError:
                author_list = [a.strip() for a in authors_raw.split(",") if a.strip()]
        else:
            author_list = [a.strip() for a in authors_raw.split(",") if a.strip()]

        output[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "primary_topic": values.get("primary_topic") or "",
            "update_date": values.get("update_date") or "",
            "citation_count": citation_count,
            "influential_citations": influential,
            "authors": author_list,
        }

    return output


# ── Feature Computation ──────────────────────────────────────────────────────

def compute_paper_age_days(published_str: str) -> int:
    """Compute age in days from a YYYY-MM-DD date string."""
    now = datetime.now(timezone.utc)
    try:
        pub_date = datetime.strptime(published_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return max(0, (now - pub_date).days)
    except (ValueError, TypeError):
        return 365  # default 1 year


def parse_year(published_str: str) -> int:
    """Extract year from YYYY-MM-DD string."""
    try:
        return int(published_str[:4])
    except (ValueError, TypeError, IndexError):
        return 2020  # default


def compute_shared_authors(authors_a: list[str], authors_b: list[str]) -> int:
    """Count shared authors between two papers (case-insensitive)."""
    set_a = {a.lower().strip() for a in authors_a if a.strip()}
    set_b = {b.lower().strip() for b in authors_b if b.strip()}
    return len(set_a & set_b)


def compute_features_for_pair(
    query_meta: dict,
    candidate_meta: dict,
    qdrant_score: float,
    candidate_position: int,
    co_citation_count: int,
    query_num_references: int,
    candidate_num_cited_by: int,
) -> np.ndarray:
    """
    Compute the full 37-feature vector for a (query, candidate) pair.

    Features 20-30 (user behavior) are zero-filled for pseudo-label training.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    # --- Content/Retrieval features (0-19) ---

    # 0: qdrant_cosine_score
    features[0] = qdrant_score

    # 1: candidate_position
    features[1] = float(candidate_position)

    # 2: candidate_citation_count
    cand_citations = candidate_meta.get("citation_count", 0)
    features[2] = float(cand_citations)

    # 3: candidate_log_citations
    features[3] = np.log(cand_citations + 1)

    # 4: candidate_influential_citations
    features[4] = float(candidate_meta.get("influential_citations", 0))

    # 5: candidate_age_days
    cand_age = compute_paper_age_days(candidate_meta.get("update_date", ""))
    features[5] = float(cand_age)

    # 6: candidate_recency_score (matches heuristic in reranker.py)
    features[6] = np.exp(-0.002 * cand_age)

    # 7: query_citation_count
    query_citations = query_meta.get("citation_count", 0)
    features[7] = float(query_citations)

    # 8: query_age_days
    query_age = compute_paper_age_days(query_meta.get("update_date", ""))
    features[8] = float(query_age)

    # 9: year_diff
    query_year = parse_year(query_meta.get("update_date", ""))
    cand_year = parse_year(candidate_meta.get("update_date", ""))
    features[9] = abs(query_year - cand_year)

    # 10: same_primary_category
    query_cat = query_meta.get("primary_topic", "")
    cand_cat = candidate_meta.get("primary_topic", "")
    features[10] = 1.0 if (query_cat and cand_cat and query_cat == cand_cat) else 0.0

    # 11: co_citation_count
    features[11] = float(co_citation_count)

    # 12: shared_author_count
    features[12] = float(compute_shared_authors(
        query_meta.get("authors", []),
        candidate_meta.get("authors", []),
    ))

    # 13: candidate_is_newer
    features[13] = 1.0 if cand_year > query_year else 0.0

    # 14: query_log_citations
    features[14] = np.log(query_citations + 1)

    # 15: citation_count_ratio
    features[15] = cand_citations / (query_citations + 1)

    # 16: age_ratio
    features[16] = cand_age / (query_age + 1)

    # 17: candidate_citations_per_year
    cand_age_years = max(cand_age / 365.0, 0.5)
    features[17] = cand_citations / cand_age_years

    # 18: query_num_references
    features[18] = float(query_num_references)

    # 19: candidate_num_cited_by
    features[19] = float(candidate_num_cited_by)

    # --- User behavior features (20-30): zero-filled for pseudo-labels ---
    # features[20] = ewma_longterm_similarity   → 0.0
    # features[21] = ewma_shortterm_similarity   → 0.0
    # features[22] = ewma_negative_similarity    → 0.0
    # features[23] = cluster_importance           → 0.0
    # features[24] = cluster_distance_to_medoid   → 0.0
    # features[25] = is_suppressed_category       → 0.0
    # features[26] = onboarding_category_match    → 0.0
    # features[27] = user_total_saves             → 0.0
    # features[28] = user_total_dismissals        → 0.0
    # features[29] = user_days_since_last_save    → 0.0
    # features[30] = user_session_save_count      → 0.0

    # --- Cross features (31-36) ---

    # 31: cosine_x_recency
    features[31] = features[0] * features[6]

    # 32: cosine_x_citations
    features[32] = features[0] * features[3]

    # 33: category_x_recency
    features[33] = features[10] * features[6]

    # 34: cosine_x_cocitation
    features[34] = features[0] * np.log(co_citation_count + 1)

    # 35: position_inverse
    features[35] = 1.0 / (candidate_position + 1)

    # 36: citations_x_recency
    features[36] = features[3] * features[6]

    return features


# ── Main Pipeline ────────────────────────────────────────────────────────────

async def generate_triples(
    citations_path: str,
    corpus_ids: list[str],
    qdrant_url: str,
    qdrant_api_key: str,
    qdrant_collection: str,
    turso_url: str,
    turso_token: str,
    output_dir: str,
    num_queries: int,
    candidates_per_query: int,
    seed: int = 42,
):
    """Main pipeline: load graph → sample queries → ANN search → compute features."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load citation graph ──────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading citation graph...")
    references, cited_by, _ = load_citation_graph(citations_path)

    corpus_set = set(corpus_ids)
    print(f"Corpus size: {len(corpus_set)}")

    # Pre-compute per-paper stats
    num_references = {pid: len(refs) for pid, refs in references.items()}
    num_cited_by = {pid: len(citers) for pid, citers in cited_by.items()}

    # ── Step 2: Connect to Qdrant ────────────────────────────────────────
    print("\nSTEP 2: Connecting to Qdrant...")
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
    collection_info = qdrant.get_collection(qdrant_collection)
    print(f"  Collection: {qdrant_collection}")
    print(f"  Points: {collection_info.points_count}")

    # ── Step 3: Sample query papers ──────────────────────────────────────
    print("\nSTEP 3: Sampling query papers...")

    # Only sample papers that have references (otherwise no positive labels)
    papers_with_refs = [pid for pid in corpus_ids if pid in references and len(references[pid]) >= 3]
    print(f"  Papers with ≥3 in-corpus references: {len(papers_with_refs)}")

    rng = random.Random(seed)
    if len(papers_with_refs) > num_queries:
        sampled_queries = rng.sample(papers_with_refs, num_queries)
    else:
        sampled_queries = papers_with_refs
        print(f"  Warning: only {len(sampled_queries)} papers have enough references")

    print(f"  Sampled {len(sampled_queries)} query papers")

    # ── Step 4: Fetch metadata for all relevant papers ───────────────────
    print("\nSTEP 4: Fetching metadata from Turso...")

    # Collect all paper IDs we'll need metadata for
    all_needed_ids = set(sampled_queries)
    for qid in sampled_queries:
        all_needed_ids.update(references.get(qid, set()))
    # We'll also need metadata for ANN candidates, but we fetch those per-batch

    # Fetch in batches of 500 (Turso limit)
    metadata_cache: dict[str, dict] = {}
    needed_list = list(all_needed_ids & corpus_set)
    batch_size = 500
    for i in tqdm(range(0, len(needed_list), batch_size), desc="Fetching metadata"):
        batch = needed_list[i:i + batch_size]
        try:
            meta = await fetch_turso_metadata_batch(batch, turso_url, turso_token)
            metadata_cache.update(meta)
        except Exception as e:
            print(f"  Warning: metadata batch failed: {e}")
    print(f"  Cached metadata for {len(metadata_cache)} papers")

    # ── Step 5: Time-split the queries ───────────────────────────────────
    print(f"\nSTEP 5: Applying time-split (eval cutoff: {EVAL_CUTOFF})...")

    train_queries = []
    eval_queries = []
    skipped = 0

    for qid in sampled_queries:
        meta = metadata_cache.get(qid)
        if not meta:
            skipped += 1
            continue
        pub_date = meta.get("update_date", "")
        year = parse_year(pub_date)
        if year < 2023:
            train_queries.append(qid)
        else:
            eval_queries.append(qid)

    print(f"  Train queries (pre-2023): {len(train_queries)}")
    print(f"  Eval queries (2023+):     {len(eval_queries)}")
    print(f"  Skipped (no metadata):    {skipped}")

    # Verify no temporal leakage
    if train_queries and eval_queries:
        max_train_year = max(parse_year(metadata_cache[q].get("update_date", "")) for q in train_queries if q in metadata_cache)
        min_eval_year = min(parse_year(metadata_cache[q].get("update_date", "")) for q in eval_queries if q in metadata_cache)
        print(f"  Max train year: {max_train_year}")
        print(f"  Min eval year:  {min_eval_year}")
        assert max_train_year < min_eval_year, "TEMPORAL LEAKAGE DETECTED!"
        print(f"  ✅ No temporal leakage")

    # ── Step 6: Generate triples ─────────────────────────────────────────
    print("\nSTEP 6: Generating training triples...")

    for split_name, query_ids in [("train", train_queries), ("eval", eval_queries)]:
        if not query_ids:
            print(f"  Skipping {split_name} — no queries")
            continue

        print(f"\n  Processing {split_name} split ({len(query_ids)} queries)...")

        all_query_ids = []
        all_candidate_ids = []
        all_labels = []
        all_features = []

        for qi, qid in enumerate(tqdm(query_ids, desc=f"  {split_name}")):
            query_meta = metadata_cache.get(qid, {})
            query_refs = references.get(qid, set())

            # Build co-cited set: papers that share references with query
            co_cited = set()
            for ref_id in query_refs:
                co_cited.update(references.get(ref_id, set()))
            co_cited -= query_refs  # exclude direct citations
            co_cited.discard(qid)    # exclude self

            # ANN search from Qdrant
            try:
                # Look up query paper by arxiv_id payload field
                # retrieve() takes point IDs (integers), not payload values.
                # Use scroll() with a FieldCondition filter to find by arxiv_id.
                scroll_results, _ = qdrant.scroll(
                    collection_name=qdrant_collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="arxiv_id", match=MatchValue(value=qid))]
                    ),
                    limit=1,
                    with_vectors=True,
                    with_payload=True,
                )
                if not scroll_results:
                    continue

                query_vector = scroll_results[0].vector
                if query_vector is None:
                    continue

                # ANN search using the query paper's embedding
                results = qdrant.query_points(
                    collection_name=qdrant_collection,
                    query=query_vector,
                    limit=candidates_per_query,
                    with_payload=True,
                )

                candidates = []
                for hit in results.points:
                    cand_id = hit.payload.get("arxiv_id") if hit.payload else None
                    if cand_id and cand_id != qid and cand_id in corpus_set:
                        candidates.append((cand_id, hit.score))

            except Exception as e:
                if qi < 3:  # Only print first few errors
                    print(f"    Warning: Qdrant query failed for {qid}: {e}")
                continue

            if not candidates:
                continue

            # Fetch metadata for candidates not yet cached
            uncached = [cid for cid, _ in candidates if cid not in metadata_cache]
            if uncached:
                try:
                    meta_batch = await fetch_turso_metadata_batch(
                        uncached[:500], turso_url, turso_token
                    )
                    metadata_cache.update(meta_batch)
                except Exception:
                    pass

            # Compute features and labels for each candidate
            for pos, (cand_id, qdrant_score) in enumerate(candidates):
                cand_meta = metadata_cache.get(cand_id, {})

                # Label assignment
                if cand_id in query_refs:
                    label = 2  # direct citation
                elif cand_id in co_cited:
                    label = 1  # co-cited
                else:
                    label = 0  # not cited

                # Co-citation count
                cocite_count = compute_co_citation_count(qid, cand_id, cited_by)

                # Feature vector
                feat = compute_features_for_pair(
                    query_meta=query_meta,
                    candidate_meta=cand_meta,
                    qdrant_score=qdrant_score,
                    candidate_position=pos,
                    co_citation_count=cocite_count,
                    query_num_references=num_references.get(qid, 0),
                    candidate_num_cited_by=num_cited_by.get(cand_id, 0),
                )

                all_query_ids.append(qid)
                all_candidate_ids.append(cand_id)
                all_labels.append(label)
                all_features.append(feat)

        # ── Save to parquet ──────────────────────────────────────────────
        if not all_features:
            print(f"  No data for {split_name} split!")
            continue

        feature_matrix = np.array(all_features, dtype=np.float32)

        # Build parquet table
        columns = {
            "query_arxiv_id": pa.array(all_query_ids, type=pa.string()),
            "candidate_arxiv_id": pa.array(all_candidate_ids, type=pa.string()),
            "label": pa.array(all_labels, type=pa.int32()),
        }

        # Add each feature as a named column
        for fi, fname in enumerate(FEATURE_SCHEMA):
            columns[fname] = pa.array(feature_matrix[:, fi].tolist(), type=pa.float32())

        # Add group_size info (candidates per query, needed for LightGBM)
        # We track this separately
        table = pa.table(columns)

        out_file = output_path / f"{split_name}.parquet"
        pq.write_table(table, str(out_file), compression="snappy")

        # Print stats
        label_counts = {0: 0, 1: 0, 2: 0}
        for l in all_labels:
            label_counts[l] = label_counts.get(l, 0) + 1

        num_queries_actual = len(set(all_query_ids))
        print(f"\n  {split_name} split saved to {out_file}")
        print(f"    Rows: {len(all_labels)}")
        print(f"    Queries: {num_queries_actual}")
        print(f"    Avg candidates/query: {len(all_labels) / max(num_queries_actual, 1):.1f}")
        print(f"    Labels: 0={label_counts[0]}, 1={label_counts[1]}, 2={label_counts[2]}")
        print(f"    Label 2 rate: {100*label_counts[2]/max(len(all_labels),1):.2f}%")
        print(f"    Label 1 rate: {100*label_counts[1]/max(len(all_labels),1):.2f}%")
        print(f"    Features: {NUM_FEATURES}")

    # ── Save feature schema ──────────────────────────────────────────────
    schema_file = output_path / "feature_schema.json"
    with open(schema_file, "w") as f:
        json.dump({
            "features": FEATURE_SCHEMA,
            "num_features": NUM_FEATURES,
            "pseudo_label_features": list(range(0, 20)) + list(range(31, 37)),
            "user_features_zero_filled": list(range(20, 31)),
            "eval_cutoff": EVAL_CUTOFF,
            "description": "37-feature schema for ResearchIT LightGBM reranker. "
                           "Features 20-30 are zero-filled during pseudo-label training "
                           "and will be populated when real user data is available.",
        }, f, indent=2)
    print(f"\nFeature schema saved to {schema_file}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LightGBM training triples from citation graph"
    )
    parser.add_argument("--citations", required=True, help="citations.parquet from Step 1")
    parser.add_argument("--corpus-file", required=True, help="Text file with arXiv IDs")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-api-key", required=True)
    parser.add_argument("--qdrant-collection", default="arxiv_bgem3_dense")
    parser.add_argument("--turso-url", required=True)
    parser.add_argument("--turso-token", required=True)
    parser.add_argument("--output-dir", default="./ltr_dataset")
    parser.add_argument("--num-queries", type=int, default=100000)
    parser.add_argument("--candidates-per-query", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load corpus IDs
    corpus_ids = []
    with open(args.corpus_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith("arXiv:"):
                    line = line[6:]
                corpus_ids.append(line)
    print(f"Loaded {len(corpus_ids)} corpus IDs")

    asyncio.run(generate_triples(
        citations_path=args.citations,
        corpus_ids=corpus_ids,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_collection=args.qdrant_collection,
        turso_url=args.turso_url,
        turso_token=args.turso_token,
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        candidates_per_query=args.candidates_per_query,
        seed=args.seed,
    ))

    print("\n✅ Done! Training triples generated.")


if __name__ == "__main__":
    main()
