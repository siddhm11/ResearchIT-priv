# Notebooks

Kaggle/Jupyter notebooks used during data preparation and prototyping.
These are **reference only** — they ran on Kaggle with GPU.

## Files

| Notebook | Purpose | Where it ran |
|---|---|---|
| `01-bme-upload.ipynb` | BGE-M3 encoding of 1.6M arXiv papers → upload dense to Qdrant Cloud + sparse to Zilliz Cloud | Kaggle (2x Tesla T4) |
| `02-bme-arxiv-test.ipynb` | Testing Qdrant collection (BQ/PRM), search quality, and BGE-M3 encode+search prototype | Kaggle |
| `03-check-search-bq-prm.ipynb` | Search quality evaluation: BQ vs PRM quantization, latency benchmarks | Kaggle |

## Key Facts Extracted

### Credentials (from notebook configs)
```
QDRANT_URL        = "https://2fe1965b-c435-4e41-836b-8a4aa2cd8c42.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_COLLECTION = "arxiv_bgem3_dense"

ZILLIZ_URI        = "https://in03-0c01933b42a8df1.serverless.aws-eu-central-1.cloud.zilliz.com"
ZILLIZ_COLLECTION = "arxiv_bgem3_sparse"
```

### Data Pipeline (01-bme-upload)
- Source: `arxiv_comprehensive_papers.csv` (1,597,106 rows → 1,596,587 after cleaning)
- Model: `BAAI/bge-m3` on CUDA (fp16=True)
- Dense: 1024-dim float32 → Qdrant Cloud (BQ enabled, HNSW m=32)
- Sparse: BGE-M3 `lexical_weights` dict → Zilliz Cloud (SPARSE_FLOAT_VECTOR, IP metric)
- Batch size: 64 papers, GPU inference
- Total upload time: ~9 hours on 2x T4

### Zilliz Collection Schema
- Collection: `arxiv_bgem3_sparse`
- Fields:
  - `id` (INT64, auto_id, primary key)
  - `arxiv_id` (VARCHAR) — the paper identifier
  - `sparse_vector` (SPARSE_FLOAT_VECTOR) — BGE-M3 learned lexical weights
- Index: SPARSE_INVERTED_INDEX, metric_type="IP"
- Total: 1,596,587 vectors

### BGE-M3 Encoding (from 02-bme-arxiv-test)
```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # GPU
# For CPU: use_fp16=False

out = model.encode(
    ["query text"],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
    max_length=512,
)
dense  = out["dense_vecs"][0]       # shape (1024,)
sparse = out["lexical_weights"][0]  # dict {token_id_int: weight_float}
```

### Important: Sparse format
The sparse vectors use **integer token IDs** as keys (from BGE-M3's tokenizer),
not string words. Example: `{29: 0.0427, 6083: 0.1852, 73904: 0.3011, ...}`

When searching Zilliz, you pass the same format dict from `model.encode()`.
