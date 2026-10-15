# Embedding and ranking evaluation harness

Read-only against production (Qdrant, Zilliz, arXiv, Semantic Scholar). Large
inputs and intermediate files live in `~/.cache/researchit-embedding-eval`;
results are written to `reports/embedding_eval/` and are not committed.

| Script | What it answers |
|---|---|
| `run.sh` | The original audit: are stored vectors exactly what BGE-M3 produces, and how does one pair (Attention / Copy this Sentence) score against controls. Builds the two virtualenvs and fetches the sidecar and the original Kaggle CSV. |
| `scaled_eval.py` | The audit at scale with citation links as ground truth: related vs random, keyword lookalikes, how much of a paper's citation neighbourhood dense search reaches, and how the feed formula orders real candidate pools. |
| `user_ratings_pilot.py` | Real readers' votes (Scholar Inbox ratings): which scoring separates liked from disliked papers. |
| `validate_steps.py` | Before/after for reranker changes. `VALIDATE_MODE=installed VALIDATE_EXPECT=<variant>` re-checks the code as installed against the tested reference. |
| `step2_tradeoff.py` | Relevance vs freshness for variants of the similarity rescaling (upgrade plan step 2). |
| `model_bakeoff.py` | BGE-M3, SPECTER2, SciNCL, GTE-large, MiniLM and mixes, per feed step: `prepare`, `fetch_bge`, `encode <model>`, `evaluate`. SPECTER2 is encoded by `bakeoff_specter2_local.py` in the `venv-specter` environment. |
| `before_after_examples.py` | Concrete before/after feed examples for each plan step. |

The Scholar Inbox dataset (github.com/avg-dev/scholar_inbox_datasets) is CC
BY-NC-ND 4.0: download it into the cache for non-commercial evaluation only,
and do not commit it or files derived from it.
