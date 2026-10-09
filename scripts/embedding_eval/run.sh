#!/usr/bin/env bash
# Embedding-trustworthiness eval. One command, re-runnable:
#
#     scripts/embedding_eval/run.sh
#
# Read-only against production (Qdrant, Zilliz, arXiv API, HF/Kaggle downloads).
# Writes only reports/embedding_eval/{report.md,results.json} plus a cache in
# $EMBEVAL_CACHE (default ~/.cache/researchit-embedding-eval): two venvs, the
# production sidecar, the original Kaggle CSV, and intermediate stage output.
#
# Needs: uv (~/.local/bin/uv), the kaggle CLI with credentials (for the original
# CSV), and .env.local with the Qdrant/Zilliz keys the app uses.
#
# Env knobs:
#   EMBEVAL_BASELINE_DEVICE=mps|cpu   device for the 8,000-paper baseline encodes
#                                     (default mps on Apple Silicon, else cpu; test
#                                     pairs and integrity always run on CPU)
#   EMBEVAL_FROM=<stage>              resume from pairs|bge|specter2|integrity|report
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CACHE="${EMBEVAL_CACHE:-$HOME/.cache/researchit-embedding-eval}"
UV="${UV:-$HOME/.local/bin/uv}"
# Pinned to the revision production's Dockerfile ships.
SIDECAR_URL="$(grep -oE 'https://huggingface.co/datasets/[^ ]+metadata.sqlite' "$ROOT/Dockerfile" | head -1)"
export EMBEVAL_CACHE="$CACHE" METADATA_SIDECAR_PATH="$CACHE/metadata.sqlite"
# Apple Silicon: MPS is ~4x faster for the baselines; stage 4 measures CPU-vs-MPS agreement.
default_dev=cpu
[ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] && default_dev=mps
export EMBEVAL_BASELINE_DEVICE="${EMBEVAL_BASELINE_DEVICE:-$default_dev}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$CACHE"

# ── environments ─────────────────────────────────────────────────────────────
# venv-main: production pins (constraints.txt, minus the Linux-only +cpu tag).
# venv-specter: `adapters` needs transformers 4.57, incompatible with the 4.44.2
# that FlagEmbedding 1.2.10 is pinned to in production.
if [ ! -x "$CACHE/venv-main/bin/python" ]; then
  grep -v -E '^\s*#|^\s*$' "$ROOT/constraints.txt" | sed 's/+cpu//' > "$CACHE/constraints-mac.txt"
  "$UV" venv -q --python 3.12 "$CACHE/venv-main"
  "$UV" pip install -q --python "$CACHE/venv-main/bin/python" -c "$CACHE/constraints-mac.txt" \
    torch -r "$ROOT/requirements.txt" pandas
fi
if [ ! -x "$CACHE/venv-specter/bin/python" ]; then
  "$UV" venv -q --python 3.12 "$CACHE/venv-specter"
  "$UV" pip install -q --python "$CACHE/venv-specter/bin/python" adapters torch numpy
fi

# ── data ─────────────────────────────────────────────────────────────────────
if [ ! -f "$CACHE/metadata.sqlite" ]; then
  curl -fSL --retry 3 -o "$CACHE/metadata.sqlite.part" "$SIDECAR_URL"
  mv "$CACHE/metadata.sqlite.part" "$CACHE/metadata.sqlite"
fi
if [ ! -f "$CACHE/csv/arxiv_comprehensive_papers.csv" ]; then
  kaggle datasets download siddhm11/arxivdata -f arxiv_comprehensive_papers.csv -p "$CACHE/csv"
  (cd "$CACHE/csv" && for z in *.zip; do [ -e "$z" ] && unzip -o -q "$z" && rm -f "$z"; done)
fi

# ── stages ───────────────────────────────────────────────────────────────────
MAIN="$CACHE/venv-main/bin/python"
SPEC="$CACHE/venv-specter/bin/python"
stages=(pairs bge specter2 integrity report)
from="${EMBEVAL_FROM:-pairs}"
run=0
cd "$HERE"
for s in "${stages[@]}"; do
  [ "$s" = "$from" ] && run=1
  [ "$run" = 1 ] || continue
  echo "== stage: $s =="
  case "$s" in
    pairs)     "$MAIN" select_pairs.py ;;
    bge)       "$MAIN" encode_bge_minilm.py ;;
    specter2)  rm -f "$CACHE"/work/scores_*_specter2.json   # never report a stale run
               "$SPEC" encode_specter2.py || echo "!! SPECTER2 stage failed; report will say so" ;;
    integrity) "$MAIN" integrity_retrieval.py ;;
    report)    "$MAIN" report.py ;;
  esac
done
echo "report: $ROOT/reports/embedding_eval/report.md"
