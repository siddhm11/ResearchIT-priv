# Recommendation audit

Open `index.html` directly in a browser. It is self-contained: filtering, section
navigation and evidence export need no server or internet connection.

The report is a dated snapshot. `results.json` holds machine-readable evidence;
`tests.xml` holds actual pytest outcomes. The small HF sample is a selected,
web-observed metadata excerpt, not a downloaded complete daily feed or benchmark.

## Reproduce

From the project root:

```sh
.venv/bin/python scripts/run_recommendation_audit.py --live --encode
```

Without flags, only local checks run. `--live` adds read-only HF/Qdrant probes,
stored-vector health, six seed-neighbor queries, two exact-vs-ANN queries per
active shard, and three scratch-user profiles with three HTTP refreshes each.
`--encode` runs six authored semantic triplets using real BGE-M3; the runtime
must be installed and may download weights. No fallback embeddings are used.

The runner creates temporary user storage before importing the app, disables
Turso replication, and never starts the app lifespan. Public paper/model reads
and local test writes are the intended side effects. Do not substitute the old
benchmark scripts: they use the ordinary configured user database.

Exact-vs-ANN agreement diagnoses index approximation in the stored vector space.
It does not establish that those vectors capture research relevance. The tiny
triplet suite is a semantic smoke test, not a validated benchmark. Cluster stress
runs compose the actual algorithm functions with synthetic 1024-dimensional
geometry; the HTTP refresh suite separately mocks retrieval.

## Independent relevance judgments

Pass `--judgments /path/to/judgments.json` to include pooled relevance metrics.
Example schema (illustrative IDs/labels only; do not use as measured evidence):

```json
[
  {
    "case_id": "evaluation-reader-day-1",
    "system": "baseline",
    "as_of": "2026-09-25T00:00:00+00:00",
    "latest_source_observed_at": "2026-09-24T22:00:00+00:00",
    "ranked": ["paper-a", "paper-b"],
    "judgments": {"paper-a": 3, "paper-b": 0, "paper-c": 2},
    "k": 10
  }
]
```

Grades: 0 irrelevant, 1 tangential, 2 useful, 3 strongly useful. Reviewers should
not know the system generating the ranking. Use the same pooled labels across
systems, preserve chronological splits, and include random eligible new papers.
Reported recall is over the judged pool only. Missing judgments are not assumed
irrelevant. Fully judged irrelevant pages get zero precision; NDCG/recall are
undefined when the pool has no corresponding positive grades. A short page does
not get a free precision boost. Timestamp checks rely on honest supplied provenance;
they cannot prove the underlying dataset was collected correctly.

## September 25 execution limitations

Direct HF and Qdrant runtime requests failed with ConnectError. The BGE-M3
runtime, torch, LightGBM and local metadata sidecar were absent. Therefore no
live embedding or recommendation-quality verdict is available. The dashboard
explicitly reports these gaps. Browser URL policy rejected automatic opening
of the file; visual/interactive browser verification was not performed. Static
HTML structure, evidence consistency, escaping and offline assets were checked.

## Structure and rollout

The new `app/hf_papers_svc.py` adapter is **not wired into the serving feed**.
It provides normalized candidate records and source provenance for a subsequent
scheduled ingestion/shadow-ranking experiment. Existing quota, Ward, EWMA, MMR
and serving constants are unchanged. Establish metadata/vector readiness before
including new IDs. Maintain an independent arXiv path and a no-HF baseline.
Do not treat community votes as user-specific labels. Remove a source only if
held-out source-removal tests preserve fresh coverage and useful recommendations;
a learned model does not discover future releases without external input.
