# Hugging Face discovery: implemented shadow pipeline

Updated 2026-09-25. Local implementation, not deployed or connected to the
serving feed. No cloud-store mutations or user-history writes by the worker.

## Implemented

- `app/hf_papers_svc.py`: bounded, validated source adapter.
- `app/discovery_store.py`: separate durable SQLite for append-only source
  snapshots, collection failures, current candidate metadata and local vectors.
- `app/discovery_worker.py`: one collection cycle and metadata/embedding preparation.
- `app/discovery_shadow.py`: source-capped comparison against an exported baseline.
- `scripts/hf_discovery.py`: single-run/periodic worker, status, scratch-profile
  baseline capture, comparison HTML, and explicitly synthetic end-to-end demo.

“Ready” means **local shadow-ready**, not uploaded to Qdrant, Turso or the
production FTS index. Embeddings use BAAI/bge-m3, 1024 dimensions, the same
`title[:256] + ' ' + abstract[:1024]` construction and max_length=512 as ingestion.
Vectors must be finite, nonzero and approximately unit length. The contract
checks configured model and preprocessing; it does not attest to an immutable
remote model revision or certify the provenance of externally supplied vectors.

## Commands

From the project root, collect one observation and inspect persistent status:

```sh
.venv/bin/python scripts/hf_discovery.py collect
.venv/bin/python scripts/hf_discovery.py status
```

Default store: `data/discovery.sqlite`, excluded from Git. Use a durable volume
and absolute path when running elsewhere. Keep the database and WAL together
when backing up, or use SQLite's backup API. Never point it at interactions.db.

Prepare local candidates as part of each cycle (requires the existing model
runtime; weights may download). Missing abstracts are fetched from arXiv:

```sh
.venv/bin/python scripts/hf_discovery.py collect --prepare --limit 10
```

Run the scheduler every six hours, with no request-time HF fetches:

```sh
.venv/bin/python scripts/hf_discovery.py collect --prepare --watch --interval 21600
```

This command is suitable for a supervised worker. Alternatively invoke the
single-cycle command from cron. The scheduler was **not left running or installed
as a service** in this session. A same-host filesystem lock prevents overlapping
workers on the same store; this is not a distributed scheduler. Use a single
owner per database. No GitHub workflow with ephemeral-only storage is installed.

A cycle failure returns a nonzero exit for cron; watch mode logs the failure and
waits until the next scheduled cycle. `--report /path/to/status.json` atomically
writes current collection/readiness status. The minimum interval is 300 seconds.

## Recovery and eligibility

- A failed/malformed source response never deletes the previous good snapshot.
  An all-rejected nonempty response is a failure; a genuinely empty list is valid.
- Replaying a snapshot is idempotent. Delayed observations cannot overwrite newer
  metadata. Observation updates with unchanged encoding text keep cached vectors.
- Changed encoding text invalidates readiness. A slow encoder cannot overwrite
  a newer observation; it must retry against current content.
- Missing abstracts use a bounded arXiv request, spaced by at least 3.5 seconds.
- A candidate gets at most three failed preparation attempts, with 5/10/20-minute
  backoff eligibility. Scheduled invocation may retry later than these minima.
  One failure does not prevent other candidates being prepared. Exhausted items
  remain visible in status; inspect the cause before deliberately requeuing.
- Readiness excludes candidates unobserved for over 48 hours and future records.
  The comparison also limits paper publication age to 90 days by default; a
  recent mention cannot make a 2020 paper a new release. These are experimental
  source policies, not validated optimal settings.
- Snapshot age and failures are reported. No trend velocity is inferred from a
  single vote count. Storage retention and remote replication are not provided
  by this initial worker; monitor file growth and preserve evaluation snapshots.

## Capture an actual baseline for a scratch reader

With reachable configured metadata/vector services:

```sh
.venv/bin/python scripts/hf_discovery.py capture \
  --seeds 1706.03762 1810.04805 1907.11692 1910.10683 2201.11903 \
  --output /tmp/researchit-baseline.json
.venv/bin/python scripts/hf_discovery.py compare \
  --baseline /tmp/researchit-baseline.json
```

Capture invokes the existing feed builder and page builder, using a temporary
user database and those seed papers. It does not start the app lifespan or record
impressions. It requires at least five available seed vectors and complete
vectors for the returned baseline page. It restores local DB configuration on
exit. Run as a standalone CLI, not within a concurrent serving process.

The exported baseline includes paper metadata, aligned seed vectors, candidate
vectors, encoding contract and capture provenance. Other baseline exports can
use the same schema; include saved/dismissed/recent IDs in `excluded_ids` when
representing an existing reader. Capture's sample reader has no prior impressions.
Only public paper data and synthetic profile state are used by capture.

The comparison preserves the eligible baseline order, assigns candidates to
seed-derived medoids, and uses existing heuristic/MMR/quota helpers to select
HF candidates. It replaces up to three baseline slots within the **same assigned
interest**, retaining those interest counts. A .45 cosine eligibility cutoff
is an experiment, not calibrated confidence. The scorer uses a seed-mean profile
proxy, not the existing reader's reconstructed EWMA, and the assigned interest
IDs may differ from a production profile's IDs. This is explicitly not a full
counterfactual serving-pipeline replay. Blinded human comparisons are still needed.

HF-only output is a diagnostic arm. Empty, stale, incompatible, excluded or
off-topic candidates leave the baseline unchanged. Popularity does not override
topic eligibility. No experiment writes clicks, impressions or learned preferences.

## Reproducible local demonstration

```sh
.venv/bin/python scripts/hf_discovery.py demo
```

Open `reports/recommendations/shadow-demo/comparison.html`. This exercises the
real storage/worker/composition functions with a mock source and synthetic
vectors in temporary storage. It displays all three lists and every replacement.
It is labelled synthetic and does not link imaginary papers to arXiv.

## Validation and remaining gates

`tests/test_discovery_pipeline.py` covers persistent restart/idempotence,
concurrent-write protection, freshness, zero-vote ingestion, invalid embeddings,
bounded retries, missing abstracts, source outages, cap/interest preservation,
exclusions, scratch capture, worker locking and output escaping. The complete
suite is rerun by `scripts/run_recommendation_audit.py --live --encode`.

The live collection attempt in this session failed with ConnectError and wrote
a failure record, not invented candidates. Real encoding and service evaluation
remain blocked in this environment. Local HTML browser opening remains blocked
by the browser URL policy; static artifact checks are possible.

Before an opt-in pilot: deploy the worker on durable storage; confirm full-source
access and real-model compatibility; measure fresh coverage and relevance on
held-out reader profiles; validate browser flows; then implement production
candidate delivery and explicit experiment attribution. Neither source voting nor
synthetic test success qualifies as personalized quality ground truth.

## App-managed scheduler and the “not live yet” diagnosis

The FastAPI lifespan now supports the worker directly. On a host with working
outbound HTTPS and the existing BGE-M3 runtime, configure these environment
variables and restart the app:

```dotenv
HF_DISCOVERY_MODE=shadow
HF_DISCOVERY_INTERVAL_SECONDS=21600
HF_DISCOVERY_BATCH_SIZE=10
HF_DISCOVERY_DB_PATH=/data/discovery.sqlite
```

Use `/data` only if it is an actual writable persistent mount on your deployment;
otherwise choose an existing persistent volume. The default local path is
`data/discovery.sqlite`. This store is not replicated by Turso. Do not run the
CLI watcher alongside the app scheduler: both now share the same filesystem lock.
Multiple hosts must not independently own the same store. Disabled mode is the
default and creates no worker or source database. Invalid settings fail closed.

`GET /healthz/discovery` returns cached operational status without contacting
remote services or loading models. Check each field separately:

- `scheduler=running`: the loop is alive; it does not prove collection succeeded.
- `last_cycle.collection.status=ok`: a source response passed validation.
- `last_cycle.store.ready`: locally encoded, fresh candidates available.
- `last_cycle.preparation.blocked=true`: shared embedding runtime unavailable or
  incompatible. Candidate retries are preserved until the runtime is repaired.
- `serving_enabled=false`: always false in this milestone. The shadow worker does
  not insert papers into the personalized feed.

Collection failures preserve the previous snapshot, and the next scheduled cycle
retries automatically. Unexpected cycle errors are reported by exception class
only. Shutdown finishes the active batch before releasing its lock; allow for
metadata requests and model inference when configuring shutdown grace periods.

In this session, DNS resolution failed for both huggingface.co and pypi.org;
public source requests returned ConnectError before any HTTP response. The local
virtualenv also lacks torch and FlagEmbedding. This evidence describes the local
restricted execution environment, not the deployed Space. On the intended host,
install the project requirements and validate outbound access before enabling
shadow mode. The Docker image already installs the embedding runtime and model
weights. No production configuration or deployment was changed in this session.

To reach personalized serving, still complete real-model/source checks and
held-out relevance evaluation, then integrate candidate metadata and vectors into
reading/saving flows and the production interest quotas. Enabling the scheduler
alone does not complete these remaining serving changes.
