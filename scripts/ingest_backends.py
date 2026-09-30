"""
Encode + upsert backends for scripts/ingest_arxiv.py.

Split out so the fetch/parse half of ingestion can be exercised (and its arXiv
API quirks debugged) without pulling in torch and a 2.2 GB model download.

Consistency requirements, in order of how badly getting them wrong hurts:

1. The embedding model and text construction must match the original ingest.
   New vectors share an ANN index with 1.6M existing ones; encoding different
   text, or with a different max_length, puts them in a subtly different
   region of the space and they rank incorrectly against their neighbours.
   Caller passes title[:256] + abstract[:1024]; max_length stays 512.

2. Qdrant point IDs are integers, and the payload carries `arxiv_id`. New
   points must continue past the existing maximum id rather than collide.

3. Zilliz uses auto_id, so only (arxiv_id, sparse_vector) is written.

4. Turso stores the FULL abstract. The original loader truncated at 500 chars,
   which is why 90% of existing rows are cut off mid-sentence; the mean arXiv
   abstract is ~1,244 chars, so that discarded roughly 60% of the text the
   cross-encoder needs. New rows keep all of it. The column is still called
   abstract_preview for schema compatibility.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BGE_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
# New papers go to the recent-papers shard, never the primary. The primary is
# at its free-tier disk limit and degraded (d985e35); that is why the shard
# exists. This used to read QDRANT_URL/QDRANT_COLLECTION, so with the app's own
# environment an ingest would have written straight into the full primary.
QDRANT_COLLECTION = os.getenv("QDRANT_RECENT_COLLECTION", "arxiv_recent")
ZILLIZ_COLLECTION = os.getenv("ZILLIZ_COLLECTION", "arxiv_bgem3_sparse")
MAX_LENGTH = 512  # must match the original ingest

# Zilliz holds only the 2025-05 snapshot (1,596,587 rows); the 202,251 papers
# ingested after it were never written there, and its schema carries a
# point_id field this writer does not set. Sparse retrieval is served by the
# sidecar's FTS5 index; Zilliz is only its fallback. Off unless asked for.
WRITE_ZILLIZ = os.getenv("INGEST_WRITE_ZILLIZ", "0") == "1"

_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topic_labels.json")


def topic_label(categories: str) -> str:
    """The friendly primary_topic existing rows use, from the primary code.

    Rows from the previous ingest carry labels ("NLP/Computational Linguistics"),
    not codes; writing "cs.CL" would give one column two taxonomies.
    """
    labels = _load_labels()
    first = (categories or "").split()[0] if categories else ""
    return labels.get(first, "Other Technical")


_labels_cache: dict | None = None


def _load_labels() -> dict:
    global _labels_cache
    if _labels_cache is None:
        with open(_LABELS_PATH) as fh:
            _labels_cache = json.load(fh)["labels"]
    return _labels_cache


class Encoder:
    """BGE-M3 dense + sparse, loaded once."""

    def __init__(self, device: str | None = None):
        from FlagEmbedding import BGEM3FlagModel
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ingest] loading {BGE_MODEL} on {dev} ...", flush=True)
        self.model = BGEM3FlagModel(
            BGE_MODEL, use_fp16=(dev != "cpu"), device=dev)
        print("[ingest] model ready", flush=True)

    def encode(self, texts: list[str]) -> list[tuple[list[float], dict]]:
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=MAX_LENGTH,
        )
        dense = out["dense_vecs"]
        sparse = out["lexical_weights"]
        results = []
        for i in range(len(texts)):
            d = dense[i]
            d = d.tolist() if hasattr(d, "tolist") else list(d)
            s = {int(k): float(v) for k, v in sparse[i].items()}
            results.append((d, s))
        return results


class Upserter:
    """Writes to Qdrant, Zilliz and Turso."""

    def __init__(self):
        self.qurl = os.environ.get("QDRANT_RECENT_URL", "").rstrip("/")
        self.qkey = os.environ.get("QDRANT_RECENT_API_KEY", "")
        if not self.qurl or not self.qkey:
            raise SystemExit(
                "QDRANT_RECENT_URL and QDRANT_RECENT_API_KEY must be set: ingest "
                "writes only to the recent-papers shard (see QDRANT_COLLECTION).")
        if self.qurl == os.environ.get("QDRANT_URL", "").rstrip("/"):
            raise SystemExit("QDRANT_RECENT_URL points at the primary cluster; refusing.")
        self.zuri = os.environ.get("ZILLIZ_URI", "").rstrip("/")
        self.ztok = os.environ.get("ZILLIZ_TOKEN", "")
        if WRITE_ZILLIZ and not (self.zuri and self.ztok):
            raise SystemExit("INGEST_WRITE_ZILLIZ=1 needs ZILLIZ_URI and ZILLIZ_TOKEN")
        self.turl = os.environ["TURSO_URL"].rstrip("/")
        self.ttok = os.environ["TURSO_DB_TOKEN"]
        self._next_id = self._max_qdrant_id() + 1
        print(f"[ingest] next Qdrant point id = {self._next_id:,}", flush=True)

    # ── Qdrant ───────────────────────────────────────────────────────────
    def _q(self, path, body=None, method="GET", timeout=180):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.qurl}{path}", data=data, method=method,
            headers={"api-key": self.qkey, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # A missing point id is a 404, which urllib raises rather than
            # returning. The id probe below depends on being able to ask "does
            # this point exist?" and get an answer instead of an exception.
            if e.code == 404:
                return {}
            raise

    def _max_qdrant_id(self) -> int:
        """Highest existing integer point id, so new points never collide."""
        info = self._q(f"/collections/{QDRANT_COLLECTION}")
        count = info["result"]["points_count"]
        # Point ids were assigned sequentially from 0 at bulk load, but scan the
        # tail rather than trusting that: an id collision silently overwrites a
        # paper, which is unrecoverable without a re-index.
        # Binary search the tail for the highest id that resolves.
        hi = count + 10_000
        lo = count - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            got = self._q(f"/collections/{QDRANT_COLLECTION}/points/{mid}")
            if got.get("result"):
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ── Zilliz ───────────────────────────────────────────────────────────
    def _z(self, path, body, timeout=180):
        req = urllib.request.Request(
            f"{self.zuri}{path}", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.ztok}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    # ── Turso ────────────────────────────────────────────────────────────
    def _t(self, stmts, timeout=180):
        payload = json.dumps({
            "requests": [{"type": "execute", "stmt": s} for s in stmts]
                        + [{"type": "close"}]}).encode()
        req = urllib.request.Request(
            f"{self.turl}/v2/pipeline", data=payload,
            headers={"Authorization": f"Bearer {self.ttok}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        for res in data.get("results", []):
            if res.get("type") == "error":
                raise RuntimeError(str(res.get("error"))[:200])

    # ── Public ───────────────────────────────────────────────────────────
    def qdrant_existing(self, ids: list[str]) -> set[str]:
        """arxiv_ids of `ids` that already have a point in the collection.

        Turso is written last, so it is the resume marker. A batch that died
        after its Qdrant write but before Turso would otherwise be re-upserted
        under fresh ids on the next run, duplicating those papers' vectors.
        """
        if not ids:
            return set()
        got = self._q(f"/collections/{QDRANT_COLLECTION}/points/scroll", {
            "filter": {"must": [{"key": "arxiv_id", "match": {"any": list(ids)}}]},
            "limit": len(ids) * 2, "with_payload": ["arxiv_id"], "with_vector": False,
        }, method="POST")
        return {p["payload"]["arxiv_id"] for p in got.get("result", {}).get("points", [])}

    def upsert(self, papers: list[dict], vecs: list[tuple[list[float], dict]]) -> None:
        assert len(papers) == len(vecs)

        # Qdrant: dense vectors keyed by fresh integer ids, skipping any paper a
        # failed earlier run already wrote.
        present = self.qdrant_existing([p["arxiv_id"] for p in papers])
        points = []
        for p, (dense, _sparse) in zip(papers, vecs):
            if p["arxiv_id"] in present:
                continue
            points.append({
                "id": self._next_id,
                "vector": dense,
                "payload": {"arxiv_id": p["arxiv_id"]},
            })
            self._next_id += 1
        if points:
            self._q(f"/collections/{QDRANT_COLLECTION}/points?wait=true",
                    {"points": points}, method="PUT")

        # Zilliz: sparse vectors, auto_id primary key. Off by default (above).
        if WRITE_ZILLIZ:
            rows = [{"arxiv_id": p["arxiv_id"],
                     "sparse_vector": {str(k): v for k, v in sparse.items()}}
                    for p, (_d, sparse) in zip(papers, vecs)]
            self._z("/v2/vectordb/entities/insert",
                    {"collectionName": ZILLIZ_COLLECTION, "data": rows})

        # Turso: full metadata, full abstract.
        stmts = []
        for p in papers:
            stmts.append({
                "sql": ("INSERT OR REPLACE INTO papers (arxiv_id, title, authors, "
                        "abstract_preview, categories, primary_topic, update_date, "
                        "citation_count, influential_citations) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)"),
                "args": [{"type": "text", "value": p["arxiv_id"]},
                         {"type": "text", "value": p["title"]},
                         {"type": "text", "value": p["authors"]},
                         {"type": "text", "value": p["abstract"]},
                         {"type": "text", "value": p["categories"]},
                         {"type": "text", "value": topic_label(p["categories"])},
                         {"type": "text", "value": p["update_date"]}],
            })
        self._t(stmts)
