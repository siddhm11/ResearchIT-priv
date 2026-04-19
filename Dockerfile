# ── ResearchIT — HF Spaces Docker deployment ─────────────────────────────────
# Free tier: 16GB RAM, 2 vCPUs, ephemeral filesystem, port 7860 required
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU-only first (smaller than full CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BGE-M3 model into the image (baked in, no cold-start download)
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)"

# Copy application code
COPY . .

# HF Spaces requires port 7860 and non-root user
USER 1000
EXPOSE 7860

CMD ["python", "run.py"]
