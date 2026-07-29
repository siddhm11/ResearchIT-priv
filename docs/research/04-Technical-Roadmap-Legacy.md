> [!WARNING]
> **This document is LEGACY.** It was the initial technical roadmap and has been superseded by [03-MultiInterest-Recommender-Architecture.md](03-MultiInterest-Recommender-Architecture.md), which was the architecture we actually implemented. Key differences:
> - This doc recommended **BGE-reranker-v2** (~800ms on CPU) → We used **heuristic scorer / LightGBM** (~2ms)
> - This doc recommended **fixed K=3 K-Means** → We used **Ward hierarchical clustering** (auto K)
> - This doc recommended **Claude LLM reranking** → Deferred (too expensive per query at this stage)
> - This doc recommended **Redis** → We stuck with **SQLite** (simpler, sufficient at scale)
>
> Kept for historical reference and ideas that may be revisited later.

# Complete technical roadmap: arxiv recommender upgrade

**FastAPI + HTMX + SQLite, wired to Qdrant's Recommend API, with Claude-powered reranking layered in Week 3.** This is a phase-by-phase execution plan for adding personalized recommendations to your existing BGE-M3 hybrid search app, designed to ship an MVP in Week 1, sophistication in Week 2, and LLM augmentation in Week 3. Every architectural choice prioritizes a single Python developer moving fast with Claude Code.

The core insight driving this roadmap: **Qdrant's native Recommend API already does 80% of what you need**. Feed it positive and negative paper IDs from user interactions, and it returns personalized recommendations without you writing any ML code. Everything else — profile embeddings, clustering, LLM reranking — is progressive enhancement on top of that foundation.

---

## Architecture decisions: the full stack

**Backend: FastAPI.** Non-negotiable for this use case. Both `qdrant-client` and `pymilvus` offer async clients. FastAPI's ASGI foundation lets you query Qdrant Cloud and Zilliz Cloud concurrently — cutting hybrid search latency in half versus synchronous Flask:

```python
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
import asyncio

app = FastAPI()
qdrant = AsyncQdrantClient(url="https://your-cluster.qdrant.io", api_key=QDRANT_KEY)

@app.get("/api/search")
async def search(q: str):
    dense_results, sparse_results = await asyncio.gather(
        qdrant.query_points(collection_name="papers", query=query_vec, using="dense", limit=20),
        zilliz_sparse_search(query_sparse, limit=20)
    )
    return rrf_fusion(dense_results, sparse_results)
```

FastAPI benchmarks at **3,000+ RPS** for I/O-bound workloads versus Flask's ~500 RPS. Qdrant's own tutorials use FastAPI. Pydantic integration validates search payloads natively. Django is overkill — you don't need ORM, admin panels, or form handling.

**Frontend: HTMX + Jinja2 + TailwindCSS/DaisyUI.** Zero JavaScript build tooling. The interaction patterns you need — search, save, dismiss, paginate recommendations — are exactly what HTMX excels at: simple request-response cycles with HTML fragment swapping. Every button is a single `hx-post` attribute:

```html
<!-- Save button swaps itself to "Saved ✓" on click -->
<button hx-post="/api/papers/{paper_id}/save"
        hx-swap="outerHTML" hx-target="this">⭐ Save</button>

<!-- Not-interested removes the card with a fade -->
<button hx-post="/api/papers/{paper_id}/not-interested"
        hx-swap="delete" hx-target="closest .paper-card"
        class="transition-opacity">✕</button>
```

No React, no npm, no Node.js dependency. FastAPI + HTMX is well-established with dedicated libraries (`fasthx` on PyPI) and production tutorials. DaisyUI gives you pre-built card components for paper recommendations without custom CSS.

**Database: SQLite (now) → Supabase (later).** Start with `aiosqlite` — zero configuration, ships with Python, survives process restarts. SQLite in WAL mode handles **~462K SELECT QPS** concurrent with **~11K UPDATE QPS** — orders of magnitude beyond what you need. When you need authentication and multi-user support, migrate to Supabase (managed PostgreSQL + built-in auth, free tier: 500MB database, 50K MAUs).

**Deployment: Render.** Free tier gives 750 hours runtime/month, automatic TLS, git-push deploys, native FastAPI support. The FastAPI + HTMX TestDriven.io tutorial deploys to Render as its production target. Upgrade to $7/month Starter for always-on when past MVP (free tier sleeps after 15 minutes of inactivity).

| Layer | Choice | Monthly Cost |
|-------|--------|-------------|
| Backend | FastAPI + Uvicorn | $0 |
| Frontend | HTMX + Jinja2 + DaisyUI | $0 |
| User DB | SQLite → Supabase | $0 |
| Dense vectors | Qdrant Cloud (existing) | Free tier |
| Sparse vectors | Zilliz Cloud (existing) | Free tier |
| Deployment | Render | $0–7 |
| **Total** | | **$0–7/month** |

### Project structure after migration from Kaggle

```
arxiv-recommender/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings via pydantic-settings
│   ├── search/
│   │   ├── hybrid.py            # BGE-M3 + RRF fusion (port from notebook)
│   │   ├── qdrant_client.py     # AsyncQdrantClient wrapper
│   │   └── zilliz_client.py     # Zilliz sparse search wrapper
│   ├── recommend/
│   │   ├── engine.py            # Qdrant Recommend API integration
│   │   ├── profiles.py          # User profile vectors (Phase 2)
│   │   ├── clustering.py        # Multi-interest K-means (Phase 2)
│   │   ├── reranker.py          # BGE-reranker + Claude reranking (Phase 3)
│   │   └── exploration.py       # Epsilon-greedy + MMR (Phase 4)
│   ├── events/
│   │   ├── schema.py            # Pydantic event models
│   │   ├── logger.py            # SQLite event writer
│   │   └── state.py             # In-memory user state cache
│   ├── templates/               # Jinja2 + HTMX
│   │   ├── base.html
│   │   ├── index.html
│   │   └── partials/
│   │       ├── search_results.html
│   │       ├── paper_card.html
│   │       └── recommendations.html
│   └── evaluation/
│       └── metrics.py           # NDCG, Hit Rate, offline eval
├── CLAUDE.md                    # Claude Code project context
├── requirements.txt
├── render.yaml
└── .env
```

**Critical migration note**: Load CSV metadata into Qdrant payloads during indexing rather than reading CSV at query time. Your Qdrant collection becomes the metadata store:

```python
await qdrant.upsert(collection_name="papers", points=[
    models.PointStruct(
        id=paper_id, vector={"dense": embedding.tolist()},
        payload={"title": title, "abstract": abstract, "categories": cats, "year": year, "arxiv_id": arxid}
    ) for paper_id, embedding, title, abstract, cats, year, arxid in batch
])
```

For BGE-M3 query-time embedding on CPU (no GPU on Render free tier), use **`fastembed`** — Qdrant's ONNX-optimized library that runs BGE-M3 at ~100-300ms per query on CPU instead of full PyTorch.

---

## Phase 1 (Week 1): event logging + Qdrant Recommend API MVP

This week ships a working "Recommended for You" section powered by Qdrant's native recommendation engine. No custom ML code needed.

### Interaction event schema

Every user action becomes a structured event. The schema links impressions back to searches via `query_id`, enabling CTR computation later:

```python
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class EventType(str, Enum):
    SEARCH = "search"
    IMPRESSION = "impression"
    CLICK = "click"
    DWELL = "dwell"
    SAVE = "save"
    UNSAVE = "unsave"
    NOT_INTERESTED = "not_interested"
    RECOMMEND_CLICK = "recommend_click"

class InteractionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    session_id: str
    paper_id: str | None = None
    query_text: str | None = None
    query_id: str | None = None
    source: str | None = None          # "search", "recommendation", "similar"
    position: int | None = None        # rank position in results (0-indexed)
    dwell_time_ms: int | None = None

# Signal weights for converting implicit feedback to Qdrant positive/negative
SIGNAL_WEIGHTS = {
    EventType.SAVE: 1.0,
    EventType.CLICK: 0.3,
    EventType.NOT_INTERESTED: -1.0,
    EventType.IMPRESSION: 0.0,
}
```

### SQLite schema

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    event_type  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    paper_id    TEXT,
    query_text  TEXT,
    query_id    TEXT,
    source      TEXT,
    position    INTEGER,
    dwell_time_ms INTEGER
);

CREATE INDEX idx_events_user_ts ON events(user_id, timestamp DESC);
CREATE INDEX idx_events_paper ON events(paper_id, event_type) WHERE paper_id IS NOT NULL;

-- Materialized affinity: aggregated user-paper signal
CREATE TABLE IF NOT EXISTS user_paper_affinity (
    user_id    TEXT NOT NULL,
    paper_id   TEXT NOT NULL,
    affinity   REAL NOT NULL,
    last_event TEXT NOT NULL,
    PRIMARY KEY (user_id, paper_id)
);
```

### Wiring up Qdrant's Recommend API

**Critical**: As of `qdrant-client` v1.10+, the legacy `recommend()` method is deprecated in favor of `query_points()` with `RecommendQuery`. In v1.13+ it's removed entirely. Use the modern API:

```python
from qdrant_client import AsyncQdrantClient, models

async def get_recommendations(
    client: AsyncQdrantClient,
    positive_ids: list[int],
    negative_ids: list[int],
    limit: int = 20,
) -> list:
    """Core recommendation call using Qdrant's native Recommend API."""
    results = await client.query_points(
        collection_name="papers",
        query=models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=positive_ids,
                negative=negative_ids,
                strategy=models.RecommendStrategy.BEST_SCORE,
            ),
        ),
        using="dense",  # BGE-M3 dense vector name
        limit=limit,
        with_payload=True,
        score_threshold=0.5,
    )
    return results.points
```

**How `BEST_SCORE` works internally**: At each HNSW graph traversal step, Qdrant computes distance to every positive and every negative example separately. Points closer to any negative than any positive get penalized (score ≤ 0). This produces more diverse results than `AVERAGE_VECTOR` when you have multiple positive/negative examples.

For **hybrid recommendation** (dense retrieve → sparse rerank), use Qdrant's prefetch:

```python
results = await client.query_points(
    collection_name="papers",
    prefetch=[
        models.Prefetch(
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=positive_ids, negative=negative_ids,
                    strategy=models.RecommendStrategy.BEST_SCORE,
                ),
            ),
            using="dense",
            limit=100,
        ),
    ],
    query=models.RecommendQuery(
        recommend=models.RecommendInput(positive=positive_ids, negative=negative_ids),
    ),
    using="sparse",
    limit=20,
    with_payload=True,
)
```

### Hot-path user state: in-memory cache + SQLite write-behind

**No Redis.** A Python dict cache for the hot path (recent positive/negative IDs), with SQLite as the durable backend:

```python
from dataclasses import dataclass, field

@dataclass
class UserState:
    positive_ids: list[int] = field(default_factory=list)
    negative_ids: list[int] = field(default_factory=list)

    def add_positive(self, paper_id: int, max_size: int = 50):
        if paper_id in self.negative_ids:
            self.negative_ids.remove(paper_id)
        if paper_id not in self.positive_ids:
            self.positive_ids.insert(0, paper_id)
            self.positive_ids = self.positive_ids[:max_size]

    def add_negative(self, paper_id: int, max_size: int = 20):
        if paper_id in self.positive_ids:
            self.positive_ids.remove(paper_id)
        if paper_id not in self.negative_ids:
            self.negative_ids.insert(0, paper_id)
            self.negative_ids = self.negative_ids[:max_size]

class UserStateCache:
    def __init__(self, db_path: str = "interactions.db", max_users: int = 1000):
        self._cache: dict[str, UserState] = {}
        self._db_path = db_path

    def get(self, user_id: str) -> UserState:
        if user_id not in self._cache:
            state = UserState()
            # Load from SQLite user_paper_affinity table
            pos, neg = self._load_from_db(user_id)
            state.positive_ids = pos
            state.negative_ids = neg
            self._cache[user_id] = state
        return self._cache[user_id]
```

### FastAPI endpoint tying it all together

```python
@app.post("/api/papers/{paper_id}/save", response_class=HTMLResponse)
async def save_paper(paper_id: int, request: Request, user_id: str = Depends(get_user_id)):
    # 1. Update hot cache
    state = user_cache.get(user_id)
    state.add_positive(paper_id)

    # 2. Log event (async write-behind)
    await log_event(InteractionEvent(
        event_type=EventType.SAVE, user_id=user_id,
        session_id=get_session_id(request), paper_id=str(paper_id), source="search"
    ))

    # 3. Return swapped HTML (HTMX partial)
    return '<button class="btn btn-success btn-sm" disabled>✓ Saved</button>'

@app.get("/api/recommendations", response_class=HTMLResponse)
async def get_recs(request: Request, user_id: str = Depends(get_user_id)):
    state = user_cache.get(user_id)
    if not state.positive_ids:
        return templates.TemplateResponse("partials/empty_recs.html", {"request": request})

    papers = await get_recommendations(qdrant, state.positive_ids, state.negative_ids)
    return templates.TemplateResponse("partials/recommendations.html", {
        "request": request, "papers": papers
    })
```

### UI changes needed

Add three elements to your existing UI: a save button and not-interested button on every paper card, plus a "Recommended for You" section that loads via HTMX on page load:

```html
<!-- Recommendations section: loads on page load, refreshes after any save/dismiss -->
<div id="rec-section"
     hx-get="/api/recommendations"
     hx-trigger="load, paperInteraction from:body"
     hx-swap="innerHTML">
    Loading recommendations...
</div>
```

### Claude Code workflow for Phase 1

Start by initializing the project, then scaffold in sequence:

```
# 1. Init
/init
# Edit CLAUDE.md with the template from the "Claude Code workflows" section below

# 2. Scaffold FastAPI + event schema
"Create the FastAPI application skeleton following the project structure in CLAUDE.md.
Implement the Pydantic event models in app/events/schema.py with EventType enum
(search, impression, click, dwell, save, unsave, not_interested, recommend_click).
Set up SQLite with WAL mode in app/events/logger.py. Create the events table
and user_paper_affinity table with proper indexes."

# 3. Port search logic
"Read the existing search code [paste key function signatures].
Implement app/search/hybrid.py wrapping the existing BGE-M3 + RRF fusion logic.
Use AsyncQdrantClient for Qdrant and create async wrappers for Zilliz.
Keep the RRF fusion algorithm identical to the existing implementation."

# 4. Build recommendation endpoint
"Implement app/recommend/engine.py using Qdrant's query_points() with
RecommendQuery and BEST_SCORE strategy. Wire it to the UserStateCache.
Create GET /api/recommendations returning HTMX partial HTML."

# 5. HTMX templates
"Create Jinja2 templates with HTMX for: base layout with TailwindCSS/DaisyUI CDN,
search page with live results, paper card partial with save/not-interested buttons,
recommendations partial that loads on page load and refreshes after interactions."
```

**Estimated effort for Phase 1**: 3–4 days with Claude Code. Day 1: project scaffold + event logging. Day 2: port search logic. Day 3: Qdrant recommend wiring + HTMX UI. Day 4: testing + deploy to Render.

---

## Phase 2 (Week 2): user profile embeddings + multi-interest clustering

Phase 1 passes raw paper IDs to Qdrant's Recommend API. Phase 2 upgrades to computed user profile vectors with recency decay, enabling richer personalization and multi-interest modeling.

### Weighted user profile vector with recency decay

Three weighting dimensions combine multiplicatively: **interaction type** (save=1.0, click=0.5, not_interested=−0.8), **recency decay** (half-life of 7 days), and **L2 normalization** (BGE-M3 operates in cosine space):

```python
import numpy as np
from datetime import datetime

EMBEDDING_DIM = 1024  # BGE-M3

INTERACTION_WEIGHTS = {
    "save": 1.0, "click": 0.5, "search_click": 0.3,
    "view": 0.1, "not_interested": -0.8,
}

def compute_recency_weight(interaction_time: datetime, now: datetime, half_life_days: float = 7.0) -> float:
    """w = 2^(-Δt/half_life). At half_life=7 days, a week-old interaction has weight 0.5."""
    delta_days = (now - interaction_time).total_seconds() / 86400.0
    return 2.0 ** (-delta_days / half_life_days)

def compute_weighted_profile(interactions: list, now: datetime = None, half_life_days: float = 7.0):
    """Weighted average of paper embeddings. Final weight = type_weight × recency_weight."""
    if now is None:
        now = datetime.utcnow()
    interactions = sorted(interactions, key=lambda x: x.timestamp, reverse=True)[:200]

    weighted_sum = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    total_abs_weight = 0.0

    for ix in interactions:
        tw = INTERACTION_WEIGHTS.get(ix.interaction_type, 0.1)
        rw = compute_recency_weight(ix.timestamp, now, half_life_days)
        w = tw * rw
        weighted_sum += w * ix.embedding
        total_abs_weight += abs(w)

    if total_abs_weight < 1e-10:
        return None
    profile = weighted_sum / total_abs_weight
    norm = np.linalg.norm(profile)
    return (profile / norm).astype(np.float32) if norm > 1e-10 else None
```

This approach is validated at scale: **LinkedIn** uses geometrically-decaying averages of job embeddings for activity features and found it significantly outperformed unweighted averaging. **Pento** (Qdrant case study, 2025) uses exponential decay `S_c = Σ w_i * e^(-λ * Δt_i)` with λ=0.01.

### Multi-interest clustering with K-means

A single profile vector blurs distinct interests. If a user reads both NLP and computer vision papers, their centroid falls in a meaningless region between the two. **PinnerSage** (Pinterest, KDD 2020) demonstrated this problem — merging 3 unrelated pin embeddings yielded a centroid representing "energy boosting breakfast." The fix: cluster interaction embeddings into **K=3 interest vectors**, query Qdrant with each:

```python
from sklearn.cluster import KMeans
from dataclasses import dataclass

@dataclass
class InterestCluster:
    centroid: np.ndarray
    medoid_paper_id: str
    paper_ids: list[str]
    importance_score: float

def cluster_user_interests(interactions: list, k: int = 3, min_interactions: int = 5):
    positive = [i for i in interactions if INTERACTION_WEIGHTS.get(i.interaction_type, 0) > 0]
    if len(positive) < min_interactions:
        profile = compute_weighted_profile(interactions)
        return [InterestCluster(centroid=profile, medoid_paper_id=positive[0].paper_id,
                paper_ids=[i.paper_id for i in positive], importance_score=1.0)] if profile is not None else []

    effective_k = min(k, len(positive))
    embeddings = np.stack([i.embedding for i in positive])
    kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters = []
    now = datetime.utcnow()
    for cidx in range(effective_k):
        mask = labels == cidx
        cluster_ixs = [positive[i] for i in range(len(positive)) if mask[i]]
        cluster_embs = embeddings[mask]
        centroid = kmeans.cluster_centers_[cidx]
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

        # Medoid: actual paper closest to centroid (avoids meaningless centroids)
        midx = np.argmin(np.linalg.norm(cluster_embs - centroid, axis=1))

        importance = sum(
            INTERACTION_WEIGHTS.get(i.interaction_type, 0.1) * compute_recency_weight(i.timestamp, now)
            for i in cluster_ixs
        )
        clusters.append(InterestCluster(
            centroid=centroid_norm.astype(np.float32),
            medoid_paper_id=cluster_ixs[midx].paper_id,
            paper_ids=[i.paper_id for i in cluster_ixs],
            importance_score=importance,
        ))
    return sorted(clusters, key=lambda c: c.importance_score, reverse=True)
```

Query Qdrant with each cluster centroid, merge by weighted score:

```python
async def recommend_multi_interest(client, clusters, seen_ids, per_cluster=10, final_k=20):
    candidates = {}
    for cluster in clusters:
        results = await client.query_points(
            collection_name="papers",
            query=cluster.centroid.tolist(),
            using="dense",
            query_filter=models.Filter(must_not=[
                models.HasIdCondition(has_id=seen_ids)
            ]),
            limit=per_cluster, with_payload=True,
        )
        for r in results.points:
            ws = r.score * cluster.importance_score
            pid = r.id
            if pid not in candidates or ws > candidates[pid]["score"]:
                candidates[pid] = {"id": pid, "score": ws, "payload": r.payload}
    return sorted(candidates.values(), key=lambda x: x["score"], reverse=True)[:final_k]
```

### Incremental real-time profile updates

Full recomputation on every interaction is expensive. Use **EWMA (Exponentially Weighted Moving Average)** for real-time updates, with periodic full recomputation every 50 interactions to correct drift:

```python
def incremental_update(profile: np.ndarray, new_embedding: np.ndarray,
                       interaction_type: str, count: int, alpha_base: float = 0.1):
    """EWMA update: μ_new = (1-α)·μ_old + α·sign·x_new"""
    tw = INTERACTION_WEIGHTS.get(interaction_type, 0.1)
    experience_factor = min(1.0, 10.0 / (count + 10))  # stabilizes with experience
    alpha = np.clip(alpha_base * abs(tw) * experience_factor, 0.01, 0.5)
    sign = 1.0 if tw > 0 else -1.0

    updated = (1.0 - alpha) * profile + alpha * sign * new_embedding
    norm = np.linalg.norm(updated)
    return (updated / norm).astype(np.float32) if norm > 1e-10 else profile
```

### Storage: binary numpy in SQLite

Profile vectors are **4,096 bytes** (1024 × float32). Store as binary blobs in SQLite — no need for a separate vector DB for user profiles at this scale:

```python
def store_profile(conn, user_id: str, profile: np.ndarray):
    conn.execute(
        "INSERT OR REPLACE INTO user_profiles (user_id, vector, updated_at) VALUES (?, ?, ?)",
        (user_id, profile.astype(np.float32).tobytes(), datetime.utcnow().isoformat())
    )
    conn.commit()

def load_profile(conn, user_id: str) -> np.ndarray | None:
    row = conn.execute("SELECT vector FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
    return np.frombuffer(row[0], dtype=np.float32).copy() if row else None
```

### Claude Code workflow for Phase 2

```
# 1. Profile vector computation
"Implement app/recommend/profiles.py with compute_weighted_profile() using recency
decay (half_life=7 days) and interaction type weights. Include incremental_update()
using EWMA. Store profiles as binary numpy in SQLite. Add unit tests that verify
the profile vector is L2-normalized and that negative interactions push the vector
away from the paper embedding."

# 2. Multi-interest clustering (HUMAN REVIEW the math)
"Implement app/recommend/clustering.py with cluster_user_interests() using
sklearn KMeans. K=3 fixed, fallback to single centroid when <5 interactions.
Use medoids not centroids for Qdrant queries. Weight cluster importance by
recency-decayed interaction signals. I will review the clustering logic manually."

# 3. Wire multi-interest to recommendation endpoint
"Update GET /api/recommendations to: load user interactions from SQLite,
compute clusters, query Qdrant with each cluster centroid weighted by importance,
merge results, deduplicate against seen papers, return top-20."
```

**Estimated effort for Phase 2**: 3–4 days. Day 1: profile vector computation + storage. Day 2: clustering implementation. Day 3: wire to endpoint + session-based updates. Day 4: testing.

---

## Phase 3 (Week 3): LLM-augmented personalization with Claude

This phase adds two capabilities: Claude-generated interest summaries (semantic profile vectors) and Claude as a listwise reranker. Combined with BGE-reranker-v2 cross-encoder scoring, this creates a three-stage retrieval pipeline.

### Claude API for interest summary generation

Use `claude-sonnet-4-20250514` to analyze a user's reading history and produce a structured interest profile. Then embed that profile with BGE-M3 for vector search:

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY env var

INTEREST_PROMPT = """Analyze these academic papers a researcher recently read.
Generate a research interest profile.

## Papers
{papers}

## Instructions
Return JSON with:
- primary_areas: 2-4 dominant research themes
- methodological_interests: techniques/algorithms they favor
- application_domains: real-world areas of interest
- synthesis_statement: 2-3 sentence dense semantic description of this
  researcher's identity, suitable for embedding-based matching.

Respond with valid JSON only."""

def generate_interest_summary(reading_history: list[dict]) -> dict:
    papers_text = "\n".join(
        f"[{i}] {p['title']}\n    {p['abstract'][:200]}"
        for i, p in enumerate(reading_history, 1)
    )
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": INTEREST_PROMPT.format(papers=papers_text)}]
    )
    return json.loads(message.content[0].text)
```

Embed the synthesis statement with BGE-M3 to create a **semantic profile vector** — a single vector capturing the user's research identity in natural language:

```python
from FlagEmbedding import BGEM3FlagModel

embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def create_semantic_profile(summary: dict) -> np.ndarray:
    text = (f"Research interests: {', '.join(summary['primary_areas'])}. "
            f"Methods: {', '.join(summary['methodological_interests'])}. "
            f"{summary['synthesis_statement']}")
    output = embedding_model.encode([text], return_dense=True)
    return output['dense_vecs'][0]  # 1024-dim
```

**Cost at scale**: For 10,000 users updated weekly via Batch API (50% discount): ~$67.50/week. For a solo developer with <100 users, this costs pennies.

### BGE-reranker-v2 cross-encoder: the workhorse reranker

**`BAAI/bge-reranker-v2-m3`** (278M params, XLM-RoBERTa architecture) runs on CPU at ~130ms per 16-pair batch — perfectly acceptable for reranking 20–30 candidates per query. It scores **0.74 NDCG@10** on BEIR benchmark, deterministic, zero API cost:

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)  # CPU

def rerank_bge(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    pairs = [[query, f"{c['title']}. {c['abstract']}"] for c in candidates]
    scores = reranker.compute_score(pairs, batch_size=16, normalize=True)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [dict(**c, rerank_score=round(s, 4)) for s, c in scored[:top_k]]
```

### Claude as a listwise reranker: the precision layer

For the final top-10 candidates, Claude applies cross-document reasoning with custom scoring criteria. This is the highest-quality reranking step — **0.78 NDCG@10** (listwise) versus 0.74 for cross-encoders — but costs ~$0.02 per query and adds 200–500ms latency:

```python
RERANK_PROMPT = """You are an academic paper recommendation engine. Rank these
papers by relevance to the researcher's profile.

## Researcher Profile
{user_profile}

## Candidate Papers
{candidates}

## Scoring Criteria (0-10)
- Topical Relevance (40%): topic match to primary interests
- Methodological Fit (25%): methods alignment
- Novelty (20%): new ideas they'd find valuable
- Recency (15%): prefer cutting-edge work

Return JSON array sorted by score descending:
[{{"paper_id": "...", "score": 9.2, "reason": "brief justification"}}]
Only include papers scoring >= 5.0."""

async def claude_rerank(user_profile: str, candidates: list[dict]) -> list[dict]:
    cands_text = "\n".join(f"[{c['id']}] {c['title']}\n  {c['abstract'][:300]}" for c in candidates)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": RERANK_PROMPT.format(
            user_profile=user_profile, candidates=cands_text
        )}]
    )
    return json.loads(message.content[0].text)
```

### The complete three-stage pipeline

```
Qdrant multi-interest retrieve (top-50)  →  5ms
    ↓
BGE-reranker-v2 cross-encoder (top-10)   →  ~130ms CPU
    ↓
Claude listwise rerank (final order)     →  ~400ms  [optional, for high-value]
```

**Use Claude reranking selectively**: on the "Recommended for You" homepage (computed async, cached), not on every search query. For search, BGE-reranker-v2 alone is sufficient.

### Prompt caching for cost reduction

Cache the system prompt + user profile across requests. Cache writes cost 1.25× base, but hits cost only **0.1× base ($0.30/MTok for Sonnet 4)**. With 80% cache hit rate, reranking drops from ~$0.02/query to ~$0.005/query.

### MCP integration for development

```bash
# Qdrant MCP: semantic memory for Claude Code during development
claude mcp add qdrant -- uvx mcp-server-qdrant \
  --qdrant-url "https://your-cluster.qdrant.io" \
  --collection-name "papers"

# GitHub MCP: PR creation, code review
claude mcp add github -- npx -y @anthropic-ai/mcp-server-github

# SQLite MCP: query interaction database from Claude Code
claude mcp add sqlite -- npx -y @anthropic-ai/mcp-server-sqlite --db-path ./interactions.db
```

**Context budget warning**: Each MCP server adds 5–20K tokens of tool definitions. Load only 2–3 at a time. Use `/clear` between phases.

### Claude Code workflow for Phase 3

```
# 1. Interest summary generation (LET CLAUDE CODE WRITE THE BOILERPLATE)
"Implement app/recommend/llm_profiles.py: generate_interest_summary() calls
Claude claude-sonnet-4-20250514 with user reading history, parses JSON response,
embeds synthesis_statement with BGE-M3. Add create_semantic_profile() that
returns the 1024-dim vector. Include error handling for malformed JSON responses."

# 2. BGE-reranker integration
"Implement app/recommend/reranker.py with rerank_bge() using FlagReranker
with bge-reranker-v2-m3. Accept query string + candidate list, return top-k
with scores. Run on CPU (use_fp16=False). Add latency logging."

# 3. Claude reranker (HUMAN REVIEW THE PROMPTS)
"Add claude_rerank() to reranker.py. Accepts user profile text + top-10
candidates. Uses the reranking prompt I'll provide. I will review and tune
the prompt manually — just implement the API call, parsing, and error handling."

# 4. Pipeline orchestration
"Create app/recommend/pipeline.py that chains: multi-interest Qdrant retrieval
(top-50) → BGE-reranker-v2 (top-10) → optional Claude reranking.
Add a feature flag to enable/disable Claude reranking per request."
```

**Estimated effort for Phase 3**: 4–5 days. Day 1: interest summary generation. Day 2: BGE-reranker integration. Day 3: Claude reranking + pipeline orchestration. Day 4-5: prompt tuning (manual) + caching + testing.

---

## Phase 4 (ongoing): feedback loops, evaluation, and scaling

### Online metrics to track from Day 1

Instrument every recommendation response. **CTR target: 2–5%** for content recommendations. **Save rate** is the strongest signal — even 1% is meaningful:

```python
@dataclass
class OnlineMetrics:
    impressions: int = 0
    clicks: int = 0
    saves: int = 0
    total_dwell_ms: int = 0
    dwell_events: int = 0

    @property
    def ctr(self) -> float: return self.clicks / max(self.impressions, 1)
    @property
    def save_rate(self) -> float: return self.saves / max(self.impressions, 1)
    @property
    def avg_dwell_s(self) -> float:
        return (self.total_dwell_ms / max(self.dwell_events, 1)) / 1000
```

### Offline evaluation: NDCG@10 and Hit Rate@10 with time-split

Use **global temporal split** (GTS) at the 80th percentile timestamp. This avoids temporal leakage that plagues leave-one-out evaluation:

```python
def ndcg_at_k(relevances: list[float], k: int = 10) -> float:
    rels = np.array(relevances[:k], dtype=np.float64)
    if len(rels) == 0: return 0.0
    dcg = float(np.sum((2**rels - 1) / np.log2(np.arange(1, len(rels) + 1) + 1)))
    ideal = sorted(relevances, reverse=True)
    ideal_rels = np.array(ideal[:k], dtype=np.float64)
    idcg = float(np.sum((2**ideal_rels - 1) / np.log2(np.arange(1, len(ideal_rels) + 1) + 1)))
    return dcg / idcg if idcg > 0 else 0.0

def time_split_evaluate(interactions_df, predict_fn, split_q=0.8, k=10):
    split_time = interactions_df['timestamp'].quantile(split_q)
    train = interactions_df[interactions_df['timestamp'] <= split_time]
    test = interactions_df[interactions_df['timestamp'] > split_time]
    test_users = set(test['user_id'].unique()) & set(train['user_id'].unique())

    ndcg_scores, hit_rates = [], []
    for uid in test_users:
        true_items = set(test[test['user_id'] == uid]['paper_id'])
        predicted = predict_fn(uid)[:k]
        rels = [1.0 if p in true_items else 0.0 for p in predicted]
        ndcg_scores.append(ndcg_at_k(rels, k))
        hit_rates.append(1.0 if any(r > 0 for r in rels) else 0.0)

    return {"ndcg@10": np.mean(ndcg_scores), "hit_rate@10": np.mean(hit_rates),
            "n_users": len(test_users)}
```

### MMR reranking for diversity (anti-filter-bubble)

**Maximal Marginal Relevance** (Carbonell & Goldstein, 1998) balances relevance against redundancy. Start with **λ=0.7** (mostly relevance, slight diversity push):

```python
def mmr_rerank(query_emb: np.ndarray, candidates: list[dict],
               lambda_param: float = 0.7, top_k: int = 10) -> list[dict]:
    selected, remaining = [], list(range(len(candidates)))
    relevance = np.array([np.dot(query_emb, c['embedding']) /
        (np.linalg.norm(query_emb) * np.linalg.norm(c['embedding']) + 1e-9)
        for c in candidates])

    for _ in range(min(top_k, len(candidates))):
        best_score, best_idx = -float('inf'), -1
        for idx in remaining:
            diversity_penalty = max(
                (np.dot(candidates[idx]['embedding'], s['embedding']) /
                 (np.linalg.norm(candidates[idx]['embedding']) * np.linalg.norm(s['embedding']) + 1e-9))
                for s in selected
            ) if selected else 0.0
            score = lambda_param * relevance[idx] - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(candidates[best_idx])
        remaining.remove(best_idx)
    return selected
```

### Epsilon-greedy exploration

Replace 10% of recommendation slots with random/diverse papers to discover new user preferences. Decay epsilon over time as the system learns:

```python
class EpsilonGreedy:
    def __init__(self, epsilon=0.1, decay=0.999, min_epsilon=0.05):
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

    def select(self, ranked: list, exploration_pool: list, k: int = 10) -> list:
        results = []
        ranked_it, explore_it = iter(ranked), iter(exploration_pool)
        for _ in range(k):
            if random.random() < self.epsilon:
                item = next(explore_it, None)
                if item:
                    item['_source'] = 'exploration'
                    results.append(item)
                    continue
            item = next(ranked_it, None)
            if item:
                item['_source'] = 'exploitation'
                results.append(item)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        return results
```

### When to add collaborative filtering

**LightFM becomes worthwhile at ≥500–1,000 active users** with ≥10 interactions each. Below this threshold, the user-item matrix is too sparse for meaningful latent factors. LightFM's hybrid mode (with item content features from BGE-M3) can work with ~200–300 users, but won't outperform your content-based approach until interaction density is sufficient. Don't build it until you see diminishing returns from content-based recommendations.

### Component activation thresholds

- **Content-based (Qdrant recommend)**: Day 1 — works with zero users
- **MMR reranking**: When you have ≥50 papers — prevents near-duplicate recommendations
- **Epsilon-greedy**: At launch — even 10 users benefit from exploration
- **Offline eval (NDCG/HR)**: After 2 weeks of interaction data
- **LLM reranking**: When you want quality uplift and can afford ~$0.02/query
- **Collaborative filtering**: **≥500 active users** with ≥10 interactions each
- **A/B testing**: ≥1,000 DAU for statistical power. Below that, use epsilon-greedy as a lightweight auto-optimizing A/B test

---

## Claude Code specific workflows

### CLAUDE.md file

This is the single most important file for Claude Code productivity. Keep it lean (~200-400 lines), every line competes for context attention:

```markdown
# arxiv-recommender: Personalized Paper Discovery

FastAPI + HTMX + Qdrant + BGE-M3 hybrid search with personalized recommendations.

## Tech Stack
- Python 3.11+, FastAPI, Qdrant Cloud (dense), Zilliz Cloud (sparse)
- BGE-M3 embeddings (1024-dim dense), SQLite WAL for events
- HTMX + Jinja2 + TailwindCSS/DaisyUI frontend

## Commands
```bash
uvicorn app.main:app --reload --port 8000
pytest tests/ -v --cov=app
ruff check app/ && ruff format app/
```

## Code Conventions
- Type hints on ALL functions. Pydantic models for API schemas.
- Async handlers for all FastAPI endpoints. Use AsyncQdrantClient.
- Embeddings always np.ndarray shape (1024,), L2-normalized.
- Qdrant IDs are integers. User IDs are UUID strings.
- All DB writes go through service layer, never in routes.

## ⚠️ Human Review Required
- app/recommend/engine.py — scoring/ranking logic
- app/recommend/reranker.py — MMR lambda, prompt engineering
- app/recommend/exploration.py — epsilon values
- app/evaluation/metrics.py — NDCG implementation
- Any Qdrant collection schema changes
```

### What Claude Code writes autonomously vs. needs human oversight

**Let Claude Code write freely:**
- FastAPI route scaffolding, Pydantic models, CRUD endpoints
- SQLite schema, migration scripts, database helpers
- HTMX templates, TailwindCSS styling
- Test scaffolding (pytest fixtures, parameterized tests)
- Configuration files (pyproject.toml, render.yaml, Dockerfile)
- Event logging pipeline, error handling middleware
- CLI utilities and data loading scripts

**Require human review:**
- Recommendation scoring and ranking logic (subtle math bugs)
- Vector similarity calculations and normalization
- MMR lambda parameter values and diversity tuning
- Epsilon-greedy rates and decay schedules
- NDCG/Hit Rate implementations (edge cases with empty results)
- Qdrant collection HNSW parameters (m, ef_construct)
- All Claude API prompts (interest summary, reranking)
- Security: auth flows, API key handling

### Key Claude Code operating principles

Use **Plan Mode** (Shift+Tab) before implementing any complex feature. Claude Code reads files and proposes approaches without editing. Never exceed **60% context window** — use `/clear` between phases. Use the `#` key to inject one-off instructions during sessions.

**Phase-specific prompting pattern:**

```
# Phase 1 session
"Read app/search/hybrid.py. Understand the existing BGE-M3 + RRF fusion.
Now implement app/recommend/engine.py that wraps Qdrant's query_points()
with RecommendQuery using BEST_SCORE strategy. Wire positive/negative IDs
from UserStateCache. Return results as Pydantic models."

/clear  # Reset context between major features

# Phase 2 session
"Read app/recommend/engine.py and app/events/schema.py. Now implement
app/recommend/profiles.py following the weighted profile vector approach:
compute_weighted_profile() with half_life=7.0, interaction type weights,
L2 normalization. Include incremental_update() using EWMA."
```

### MCP server configuration

Load **at most 2–3 MCP servers** at a time to preserve context budget:

```bash
# Development: Qdrant + filesystem
claude mcp add qdrant -- uvx mcp-server-qdrant \
  --qdrant-url "https://your-cluster.qdrant.io" \
  --collection-name "papers"

claude mcp add filesystem -- npx -y @anthropic-ai/mcp-server-filesystem ./app

# Database debugging sessions
claude mcp add sqlite -- npx -y @anthropic-ai/mcp-server-sqlite \
  --db-path ./interactions.db

# Deployment/CI sessions
claude mcp add github -- npx -y @anthropic-ai/mcp-server-github
```

Create isolated slash commands in `.claude/commands/` for MCP-heavy tasks to avoid polluting the main context.

---

## What NOT to build (over-engineering traps)

**Do not build a custom embedding model.** BGE-M3 is SOTA for hybrid search. Fine-tuning it on arxiv data would take weeks and likely produce marginal gains over the foundation model.

**Do not build a separate frontend SPA.** React/Next.js adds a JavaScript build pipeline, CORS configuration, state management library, and doubles your deployment surface. HTMX does everything you need for this UI complexity.

**Do not use Redis until you have concurrent server processes.** SQLite WAL mode handles 462K reads/sec. Redis adds operational overhead for zero benefit at this scale.

**Do not implement collaborative filtering until ≥500 users.** The user-item matrix will be too sparse. Content-based with BGE-M3 vectors will outperform LightFM until you have interaction density.

**Do not build a real-time streaming pipeline.** Batch computation of user profiles (every N interactions or daily cron) is sufficient. Apache Kafka, Flink, or similar infrastructure is months of work for marginal latency improvement.

**Do not use Claude reranking on every search query.** Reserve it for cached recommendation feeds where the 400ms latency and ~$0.02/query cost are amortized. BGE-reranker-v2 handles search-time reranking at 130ms with zero API cost.

---

## Effort estimates and prioritization

| Phase | Scope | Days with Claude Code | Priority |
|-------|-------|-----------------------|----------|
| **Phase 1** | Event logging + Qdrant Recommend MVP | **3–4 days** | P0 — ships working recs |
| **Phase 2** | Profile vectors + multi-interest clustering | **3–4 days** | P1 — quality uplift |
| **Phase 3** | Claude interest summaries + BGE reranker | **4–5 days** | P2 — premium quality |
| **Phase 4** | Evaluation metrics + feedback loops | **2–3 days** | P1 — needed for iteration |
| Migration | Kaggle → Render deployment | **1–2 days** | P0 — prerequisite |

**Total: ~15–18 days of focused work.**

**If time runs short, cut in this order:**
1. ✅ Ship Phase 1 (Qdrant Recommend + event logging) — this alone is a functional recommender
2. ✅ Ship Phase 4 offline eval (NDCG/HR) — you need metrics before optimizing
3. ✅ Ship Phase 2 profile vectors — meaningful quality improvement
4. ⚠️ Phase 3 Claude reranking is a luxury — BGE-reranker-v2 alone gets you 90% of the quality
5. ⚠️ Phase 2 multi-interest clustering can be deferred — single weighted centroid works for most users with <50 interactions

The Qdrant Recommend API with `BEST_SCORE` strategy is doing heavy lifting for free. Everything else is progressive enhancement. Ship Phase 1, measure with Phase 4 metrics, then decide whether Phase 2 or Phase 3 delivers more value for your specific users.