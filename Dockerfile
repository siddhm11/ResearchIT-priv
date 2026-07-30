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
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence-transformers

# Install torch CPU-only first (smaller than full CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models into the image (baked in, no cold-start download)
# BGE-M3 for dense+sparse embeddings (~2.2GB)
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)"
# MiniLM cross-encoder for search reranking (~80MB)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

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
ENV METADATA_SIDECAR_URL=https://huggingface.co/datasets/siddhm11/researchit-metadata/resolve/main/metadata.sqlite
ENV METADATA_SIDECAR_PATH=/app/data/metadata.sqlite
RUN mkdir -p /app/data && \
    if [ -n "$METADATA_SIDECAR_URL" ]; then \
      echo "Fetching metadata sidecar..." && \
      (curl -fSL --retry 3 "$METADATA_SIDECAR_URL" -o /app/data/metadata.sqlite \
        && ls -lh /app/data/metadata.sqlite) \
      || (echo "WARNING: sidecar download failed — falling back to Turso at runtime" \
          && rm -f /app/data/metadata.sqlite); \
    fi

# Copy application code
COPY . .

# Make app dir writable for non-root user (HF Spaces requires USER 1000)
RUN chmod -R 777 /app

# HF Spaces requires port 7860 and non-root user
USER 1000
EXPOSE 7860

# SQLite must write to a writable path
ENV DB_PATH=/tmp/interactions.db
# Ensure HOME is set for the non-root user
ENV HOME=/home/user

CMD ["python", "run.py"]
