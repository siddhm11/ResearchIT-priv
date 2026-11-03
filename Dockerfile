# ── ResearchIT — HF Spaces Docker deployment ─────────────────────────────────
# Free tier: 16GB RAM, 2 vCPUs, ephemeral filesystem, port 7860 required
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user BEFORE model downloads so getpwuid(1000) works
# and model caches are written to a known, accessible location.
RUN useradd -m -u 1000 -s /bin/bash user

# Set HuggingFace cache to a shared path accessible by uid 1000
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers
# Owned by uid 1000 from the start: models and the sidecar are then written by the
# user that runs the app, so no `chmod -R` layer is needed afterwards. That layer
# re-copied every file under /app (the 2.8 GB sidecar and ~2.5 GB of models
# included), doubling the image, and on 2026-10-16 the HF builder began
# OOM-killing the image push (exit 137) at that size.
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence-transformers /app/data && \
    chown -R 1000:1000 /app

# Exact version of every package, generated from requirements.txt by `make lock`.
# Without it each rebuild installed whatever was newest that day, which is how a
# transformers release could break BGE-M3 encoding (see requirements.txt) with no
# code change at all. requirements.txt stays the human-edited list of intent.
COPY constraints.txt .

# Install torch CPU-only first (smaller than full CUDA build). The lock pins
# torch==<version>+cpu, which only the PyTorch index carries, so the CPU build is
# still what gets installed; PyPI stays primary because that index does not mirror
# the pinned versions of torch's own dependencies (filelock, sympy, ...).
RUN pip install --no-cache-dir torch -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Pre-download models into the image (baked in, no cold-start download), as the
# app's user and download-only: instantiating BGE-M3 here only to cache it
# peaked at ~4-5 GB of build memory for nothing.
USER 1000
# BGE-M3 for dense+sparse embeddings, pinned to the commit the corpus vectors
# were made with (config.BGE_M3_PINS; a test checks they agree). onnx/ (2.3 GB)
# is a second copy of the weights that nothing here uses.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181', ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'onnx/*'])"
# Search cross-encoders. mxbai-rerank-xsmall (~150MB on disk) is config.RERANKER_MODEL
# (tests/test_search_speed.py fails if the two disagree, so a model change cannot
# silently become a first-request download). MiniLM (~80MB) stays baked in so
# RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 rolls back without a rebuild.
# Revisions are pinned (config.RERANKER_PINS) so a rebuild bakes in exactly the evaluated weights;
# the ONNX/OpenVINO exports (371 MB for mxbai alone) are not used by sentence-transformers.
RUN python -c "from huggingface_hub import snapshot_download; ig = ['onnx/*', 'openvino/*', '*.msgpack', '*.h5', '*.ot']; snapshot_download('mixedbread-ai/mxbai-rerank-xsmall-v1', revision='b5c6e9da73abc3711f593f705371cdbe9e0fe422', ignore_patterns=ig); snapshot_download('cross-encoder/ms-marco-MiniLM-L-6-v2', revision='233902d25c440f23af6f7d6e94d2946bac0bee0a', ignore_patterns=ig)"

# ── Metadata sidecar (optional but strongly recommended) ─────────────────────
# A local SQLite mirror of the Turso `papers` table, built by
# scripts/build_metadata_sidecar.py and hosted as an HF dataset.
#
# Turso has a single index (arxiv_id), so category and recency queries are full
# scans of 1.6M rows — trending alone is ~15s cold and lands on new users during
# onboarding. The sidecar carries the missing indexes locally.
#
# Download failure is non-fatal: app/local_meta.py falls back to Turso, so the
# build still produces a working image. Set to "" to skip entirely.
# Pinned to a dataset commit rather than `main` for two reasons: the build is
# reproducible, and publishing a new sidecar changes this line, which
# invalidates the Docker layer below. Pointing at `main` meant a rebuild
# silently reused the cached copy and shipped stale metadata.
# 0d1449d = 1,972,571 papers (newest 2026-10-23), exactly Turso's on
# 2026-10-25 + FTS5 index (the sparse retrieval arm): 7585841 plus the 26,918
# papers the daily job stored after 2026-09-29 and the 91,876 re-harvested for
# June 2025 - September 2026 (missed by arXiv's 10,000-result cap; Fix Log item
# 9), with citation counts refreshed from Semantic Scholar for every paper since
# June 2025 on 2026-10-25. Built by scripts/append_to_sidecar.py and
# refresh_citations.py (PHASE7 §8). Previous: 7585841, 015c096, d241ff1.
ENV METADATA_SIDECAR_URL=https://huggingface.co/datasets/siddhm11/researchit-metadata/resolve/0d1449d3b9b0ca5376828d4ceb357f1233374cce/metadata.sqlite
ENV METADATA_SIDECAR_PATH=/app/data/metadata.sqlite
RUN mkdir -p /app/data && \
    if [ -n "$METADATA_SIDECAR_URL" ]; then \
      echo "Fetching metadata sidecar..." && \
      (curl -fSL --retry 3 "$METADATA_SIDECAR_URL" -o /app/data/metadata.sqlite \
        && ls -lh /app/data/metadata.sqlite) \
      || (echo "WARNING: sidecar download failed — falling back to Turso at runtime" \
          && rm -f /app/data/metadata.sqlite); \
    fi

# Copy application code, owned by the app's user (replaces the old chmod -R 777).
COPY --chown=1000:1000 . .

# HF Spaces requires port 7860 and non-root user
USER 1000
EXPOSE 7860

# SQLite must write to a writable path
ENV DB_PATH=/tmp/interactions.db
# Ensure HOME is set for the non-root user
ENV HOME=/home/user

CMD ["python", "run.py"]
