# Code Tour — ArXiv Recommender (Phase 1)

A file-by-file walkthrough of every piece of the codebase: what it does, how it works, and why it was written the way it was.

---

## Entry Points

### `run.py`

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["app"],
    )
```

Nothing special here. Starts Uvicorn pointing at the FastAPI `app` object. `reload=True` watches the `app/` directory and hot-reloads on file changes. Run with `python run.py`.

---

### `app/main.py`

```python
from app.routers import search, events, recommendations, saved

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    yield

app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.include_router(search.router)
app.include_router(events.router)
app.include_router(recommendations.router)
app.include_router(saved.router)

@app.get("/", response_class=HTMLResponse)
async def home(request, user_id=Cookie(...)):
    user_id = user_id or str(uuid.uuid4())
    state = await us.ensure_loaded(user_id)
    resp = templates.TemplateResponse(request, "index.html", {
        "has_recs": state.has_enough_for_recs(),
        "save_count": len(state.positives),
    })
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365*24*3600, httponly=True)
    return resp
```

**`lifespan`** is a FastAPI context manager that runs `init_db()` once when the server starts — creates the three SQLite tables if they don't exist, then yields control to the app.

**The home route** is the only one that lives in `main.py`. Everything else is in routers. It reads the user's cookie, loads their state from memory/DB, and renders `index.html` with two flags: `has_recs` (enough saves to show recommendations?) and `save_count` (how many papers saved so far).

**Cookie pattern** — every route that might be a user's first visit creates a UUID4 if no cookie exists, and refreshes the cookie's max_age on every response. This way the cookie always stays 1 year from last visit.

---

## Configuration

### `app/config.py`

```python
import os

QDRANT_URL        = os.getenv("QDRANT_URL", "https://2fe1965b-...eu-west-2-0.aws...")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "eyJhbGci...")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "arxiv_bgem3_dense")

DB_PATH           = os.getenv("DB_PATH", "interactions.db")
ARXIV_API_URL     = "https://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = 10
METADATA_CACHE_TTL_DAYS = 30

REC_LIMIT          = 10
REC_POSITIVE_LIMIT = 20
REC_MIN_POSITIVES  = 1

APP_TITLE   = "ArXiv Recommender"
COOKIE_NAME = "arxiv_user_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365
```

Every credential and tunable lives here. All of them can be overridden with environment variables — `os.getenv("X", default)`. In production you'd set `QDRANT_API_KEY` as an env var and never commit it to git.

**`REC_POSITIVE_LIMIT = 20`** — controls how many saved papers are kept in the in-memory deque *and* how many are sent to Qdrant as positive examples. This is the only place you change it; `user_state.py` reads it directly.

---

## Database Layer

### `app/db.py`

Three tables. The schema runs once at startup via `init_db()`.

```python
_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS interactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT    NOT NULL,
    paper_id   TEXT    NOT NULL,
    event_type TEXT    NOT NULL,   -- save | not_interested
    source     TEXT,               -- search | recommendation | saved
    position   INTEGER,
    query_id   TEXT,
    timestamp  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ui_user_ts    ON interactions(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ui_user_paper ON interactions(user_id, paper_id);

CREATE TABLE IF NOT EXISTS paper_qdrant_map (
    arxiv_id        TEXT PRIMARY KEY,
    qdrant_point_id INTEGER NOT NULL,
    mapped_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_metadata (
    arxiv_id  TEXT PRIMARY KEY,
    title     TEXT,
    abstract  TEXT,
    authors   TEXT,   -- JSON array string e.g. '["Vaswani", "Shazeer"]'
    category  TEXT,
    published TEXT,
    cached_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""
```

**WAL mode** (`journal_mode=WAL`) allows one writer and multiple concurrent readers without blocking. Important because FastAPI handles requests concurrently and SQLite's default mode would serialize everything.

**`synchronous=NORMAL`** — safe against OS crashes but doesn't fsync on every write. Faster than `FULL` with acceptable durability for a research tool.

**Three tables, three jobs:**

| Table | Job |
|---|---|
| `interactions` | Append-only event log. Never updated, only inserted. Source of truth. |
| `paper_qdrant_map` | Cache translating arxiv_id strings → Qdrant integer point IDs |
| `paper_metadata` | Cache of arXiv API responses so we don't re-fetch titles/abstracts |

**Key functions:**

```python
# Write an event
await db.log_interaction(user_id, paper_id, "save", source="search", position=2)

# Read recent events for a user (used to hydrate the in-memory cache)
rows = await db.get_user_interactions(user_id, event_types=["save", "not_interested"], limit=70)

# Qdrant ID cache
await db.save_qdrant_id("1706.03762", 523419)
cached = await db.get_qdrant_ids_batch(["1706.03762", "0704.0002"])
# → {"1706.03762": 523419}  (only IDs that were in the cache)

# Metadata cache
await db.cache_metadata({"arxiv_id": "1706.03762", "title": "Attention...", ...})
batch = await db.get_cached_metadata_batch(["1706.03762", "0704.0002"])
# → {"1706.03762": {...}}
```

All functions use `async with aiosqlite.connect(DB_PATH)` — each call opens and closes its own connection. This is safe with WAL mode and avoids connection pool complexity.

---

## arXiv Service

### `app/arxiv_svc.py`

Handles all communication with the arXiv Atom XML API and the SQLite metadata cache.

#### ID Normalisation

arXiv IDs come in several formats from the API:

```python
_ID_RE = re.compile(r"(?:arxiv:|https?://arxiv\.org/abs/)?([^\s/v]+(?:v\d+)?)")

def _normalise_id(raw: str) -> str:
    m = _ID_RE.search(raw.strip())
    bare = m.group(1)
    return re.sub(r"v\d+$", "", bare)
```

| Input | Output |
|---|---|
| `http://arxiv.org/abs/1706.03762v5` | `1706.03762` |
| `https://arxiv.org/abs/1706.03762` | `1706.03762` |
| `arxiv:1706.03762v2` | `1706.03762` |
| `1706.03762v3` | `1706.03762` |
| `0704.0002` | `0704.0002` |

The bare ID is what we store everywhere — in SQLite, in the user state cache, and in the Qdrant `arxiv_id` payload field.

#### XML Parsing

The arXiv API returns Atom XML. One `<entry>` element per paper:

```python
_NS = {
    "atom":  "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

def _parse_entry(entry: ET.Element) -> dict:
    raw_id   = text("atom:id")
    arxiv_id = _normalise_id(raw_id)
    authors  = [a.findtext("atom:name", ...) for a in entry.findall("atom:author", _NS)]
    cat_el   = entry.find("arxiv:primary_category", _NS)
    category = cat_el.attrib.get("term", "")

    return {
        "arxiv_id": arxiv_id,
        "title":    text("atom:title").replace("\n", " "),
        "abstract": text("atom:summary").replace("\n", " "),
        "authors":  json.dumps(authors[:5]),   # stored as JSON string in SQLite
        "category": category,
        "published": text("atom:published")[:10],  # YYYY-MM-DD only
        "year":     int(published[:4]),
    }
```

Authors are stored as a JSON string (`'["Vaswani", "Shazeer"]'`) because SQLite has no array type. The `tojson_parse` filter in the template converts it back to a Python list for display.

#### Search and Fetch

```python
async def search(query: str, max_results=10) -> list[dict]:
    params = {"search_query": f"all:{query}", "start": 0,
              "max_results": max_results, "sortBy": "relevance"}
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        resp = await client.get(ARXIV_API_URL, params=params)
    papers = [_parse_entry(e) for e in ET.fromstring(resp.text).findall("atom:entry", _NS)]
    for paper in papers:
        await db.cache_metadata(paper)   # cache all results immediately
    return papers

async def fetch_metadata_batch(arxiv_ids: list[str]) -> dict[str, dict]:
    result  = await db.get_cached_metadata_batch(arxiv_ids)  # check SQLite first
    missing = [aid for aid in arxiv_ids if aid not in result]
    if missing:
        # Batch up to 20 IDs per request, 0.35s gap = ~3 req/s rate limit
        for i in range(0, len(missing), 20):
            chunk  = missing[i:i+20]
            params = {"id_list": ",".join(chunk), "max_results": len(chunk)}
            # ... fetch, parse, cache ...
            await asyncio.sleep(0.35)
    return result
```

`follow_redirects=True` is required — the arXiv API's HTTP URL redirects to HTTPS.

---

## User State

### `app/user_state.py`

The in-memory hot cache. Zero DB reads on the hot path.

```python
from app import db, config

MAX_POSITIVES = config.REC_POSITIVE_LIMIT   # = 20, kept in sync with config
MAX_NEGATIVES = 50

@dataclass
class UserState:
    positives: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_POSITIVES))
    negatives: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_NEGATIVES))
    loaded: bool = False

    def add_positive(self, paper_id: str) -> None:
        try:    self.negatives.remove(paper_id)   # mutual exclusion
        except ValueError: pass
        if paper_id not in self.positives:
            self.positives.appendleft(paper_id)   # most recent first

    def add_negative(self, paper_id: str) -> None:
        try:    self.positives.remove(paper_id)
        except ValueError: pass
        if paper_id not in self.negatives:
            self.negatives.appendleft(paper_id)

    def has_enough_for_recs(self) -> bool:
        return len(self.positives) >= config.REC_MIN_POSITIVES
```

**Mutual exclusion**: `add_positive` removes the paper from negatives before adding to positives, and vice versa. So a paper can never be in both lists simultaneously.

**`appendleft`**: deques are double-ended. `appendleft` inserts at index 0 (front). When `maxlen` is reached, the rightmost (oldest) element is silently dropped. So `positive_list[0]` is always the most recently saved paper.

```python
_cache: dict[str, UserState] = {}   # global in-process dict

async def ensure_loaded(user_id: str) -> UserState:
    state = get_user_state(user_id)
    if state.loaded:
        return state                 # O(1) — hot path

    # Cold path: first request from this user in this server process
    rows = await db.get_user_interactions(user_id,
               event_types=["save", "not_interested"], limit=70)
    for row in reversed(rows):      # oldest first so appendleft puts newest at front
        if row["event_type"] == "save":
            state.add_positive(row["paper_id"])
        else:
            state.add_negative(row["paper_id"])
    state.loaded = True
    return state
```

**Why `reversed(rows)`**: `get_user_interactions` returns rows newest-first (ORDER BY timestamp DESC). We want to replay them in chronological order so that `appendleft` in `add_positive` correctly ends up with the newest paper at `index 0`. If we replayed newest-first, the oldest save would end up at the front.

```python
def record_positive(user_id: str, paper_id: str) -> None:
    get_user_state(user_id).add_positive(paper_id)   # sync, no DB

def all_seen(user_id: str) -> set[str]:
    state = get_user_state(user_id)
    return set(state.positive_list) | set(state.negative_list)
```

`all_seen` feeds the recommendation engine — any paper the user has ever saved or dismissed is excluded from the results.

---

## Qdrant Service

### `app/qdrant_svc.py`

Two jobs: translate arxiv_ids → integer point IDs, and call the Recommend API.

#### Client Setup

```python
@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=30,
        check_compatibility=False,
    )
```

`@lru_cache(maxsize=1)` makes this a singleton. The client is created once, reused on every request. The sync `QdrantClient` is used (not the async one) because it runs inside `asyncio.run_in_executor` — this keeps the event loop free while the network call is in flight.

#### ID Lookup

```python
async def lookup_qdrant_ids(arxiv_ids: list[str]) -> dict[str, int]:
    cached  = await db.get_qdrant_ids_batch(arxiv_ids)
    missing = [aid for aid in arxiv_ids if aid not in cached]

    if missing:
        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _scroll_by_arxiv_ids, missing)
        for arxiv_id, point_id in results.items():
            await db.save_qdrant_id(arxiv_id, point_id)
            cached[arxiv_id] = point_id

    return cached

def _scroll_by_arxiv_ids(arxiv_ids: list[str]) -> dict[str, int]:
    pts, _ = _client().scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="arxiv_id", match=MatchAny(any=arxiv_ids))
        ]),
        limit=len(arxiv_ids),
        with_payload=True,
        with_vectors=False,
    )
    return {p.payload["arxiv_id"]: p.id for p in pts}
```

`MatchAny` is Qdrant's `IN (...)` — it filters points whose `arxiv_id` payload field matches any value in the list. Requires the keyword payload index created on the collection (created once, persists permanently).

The result is `{arxiv_id: integer_point_id}`. Any ID not found in the collection is simply absent from the dict — that paper hasn't been indexed yet.

#### Recommend

```python
async def recommend(positive_arxiv_ids, negative_arxiv_ids, seen_arxiv_ids, limit):
    all_ids = list(dict.fromkeys(positive_arxiv_ids + negative_arxiv_ids))
    id_map  = await lookup_qdrant_ids(all_ids)

    pos_ids = [id_map[aid] for aid in positive_arxiv_ids if aid in id_map]
    neg_ids = [id_map[aid] for aid in negative_arxiv_ids if aid in id_map]

    if not pos_ids:
        return []

    results = await loop.run_in_executor(None, _run_recommend, pos_ids, neg_ids, limit*2)

    filtered = [
        r.payload["arxiv_id"]
        for r in results
        if r.payload.get("arxiv_id") and r.payload["arxiv_id"] not in seen_arxiv_ids
    ]
    return filtered[:limit]

def _run_recommend(pos_ids, neg_ids, limit):
    result = _client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=RecommendQuery(
            recommend=RecommendInput(
                positive=pos_ids,
                negative=neg_ids if neg_ids else [],
                strategy=RecommendStrategy.BEST_SCORE,
            )
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return result.points
```

**`BEST_SCORE` strategy**: for each candidate paper, Qdrant computes its similarity to each positive example, takes the maximum score, then subtracts a penalty for similarity to negatives. Papers near your saves and far from your dismissals bubble to the top.

**`limit * 2` over-fetch**: we fetch double the target count so that after filtering out `seen_arxiv_ids` in Python, we still have enough results to return `limit` papers.

**`dict.fromkeys(...)` deduplication**: if a paper appears in both positive and negative lists (shouldn't happen due to mutual exclusion in `user_state`, but defensive), it's deduplicated before the lookup.

---

## Routers

### `app/routers/search.py`

```python
@router.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = "", user_id=Cookie(...)):
    papers = []
    if q.strip():
        papers = await arxiv_svc.search(q.strip())

    state      = await us.ensure_loaded(user_id)
    saved_ids  = set(state.positive_list)
    dismissed  = set(state.negative_list)

    for p in papers:
        p["saved"]     = p["arxiv_id"] in saved_ids
        p["dismissed"] = p["arxiv_id"] in dismissed_ids

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "partials/search_results.html",
                                          {"papers": papers, "query": q})
    else:
        return templates.TemplateResponse(request, "search.html",
                                          {"papers": papers, "query": q,
                                           "has_recs": state.has_enough_for_recs()})
```

**HTMX detection**: if the request has an `HX-Request` header (set automatically by HTMX), return only the `search_results.html` partial — just the list of cards, no `<html>` wrapper. This is what gets swapped into `#search-results` on the page without a full reload.

**Annotating papers**: after fetching from arXiv, each paper dict gets `saved` and `dismissed` booleans added. The template uses these to show the correct button state (e.g. already-saved papers show "✓ Saved" instead of "⭐ Save").

---

### `app/routers/events.py`

```python
@router.post("/{paper_id}/save", response_class=HTMLResponse)
async def save_paper(paper_id, request, source=Form("search"),
                     position=Form(0), query_id=Form(""), user_id=Cookie(...)):
    await db.log_interaction(user_id, paper_id, "save",
                             source=source, position=position or None)
    us.record_positive(user_id, paper_id)
    asyncio.create_task(qdrant_svc.lookup_qdrant_ids([paper_id]))  # background

    return templates.TemplateResponse(request, "partials/action_buttons.html",
        {"paper_id": paper_id, "saved": True, "dismissed": False, "source": source})


@router.post("/{paper_id}/not-interested", response_class=HTMLResponse)
async def not_interested(paper_id, request, source=Form("search"), ...):
    await db.log_interaction(user_id, paper_id, "not_interested", source=source)
    us.record_negative(user_id, paper_id)

    resp = HTMLResponse(content="")   # empty → HTMX removes the card
    resp.set_cookie(...)
    return resp
```

**Three things happen on save, in order:**
1. `db.log_interaction()` — durable write to SQLite (awaited, synchronous from caller's perspective)
2. `us.record_positive()` — in-memory update (synchronous, no I/O)
3. `asyncio.create_task(...)` — background task to look up the Qdrant point ID. Returns immediately; the lookup happens in the background. The response is sent before this finishes.

**Why background for Qdrant lookup?** The user doesn't need the Qdrant point ID for the save response. They only need it when recommendations are requested. The background task means the save response is fast (~5ms), and by the time the user navigates to the home page to see recommendations, the ID is likely already cached.

**Empty response for dismiss**: HTMX has a target set to `#paper-{id}` with `hx-swap="outerHTML swap:200ms"`. An empty response body tells HTMX to replace the entire card element with nothing — the card fades out and disappears.

**`source` is forwarded to the response template**: after a save, the rendered `action_buttons.html` partial receives the same `source` value that came in. So the "Remove" button on the now-saved card will log `source="recommendation"` if the save happened from the recs section, not `"search"`.

---

### `app/routers/recommendations.py`

```python
@router.get("/recommendations", response_class=HTMLResponse)
async def get_recommendations(request, user_id=Cookie(...)):
    state = await us.ensure_loaded(user_id)

    if not state.has_enough_for_recs():
        return _empty_resp()           # shows "Save 1 paper to unlock recs"

    rec_arxiv_ids = await qdrant_svc.recommend(
        positive_arxiv_ids=state.positive_list,
        negative_arxiv_ids=state.negative_list,
        seen_arxiv_ids=us.all_seen(user_id),
        limit=REC_LIMIT,
    )

    if not rec_arxiv_ids:
        return _empty_resp()

    meta   = await arxiv_svc.fetch_metadata_batch(rec_arxiv_ids)
    papers = [{**meta[aid], "saved": False, "dismissed": False}
              for aid in rec_arxiv_ids if aid in meta]

    return templates.TemplateResponse(request, "partials/recommendations.html",
                                      {"papers": papers})
```

Linear pipeline: load state → check threshold → Qdrant recommend → fetch metadata → render. If anything returns empty at any step, show the empty state partial.

---

### `app/routers/saved.py`

```python
@router.get("/saved", response_class=HTMLResponse)
async def saved_papers(request, user_id=Cookie(...)):
    state    = await us.ensure_loaded(user_id)
    saved_ids = state.positive_list   # most-recent first

    papers = []
    if saved_ids:
        meta   = await arxiv_svc.fetch_metadata_batch(saved_ids)
        papers = [{**meta[aid], "saved": True, "dismissed": False}
                  for aid in saved_ids if aid in meta]

    return templates.TemplateResponse(request, "saved.html",
                                      {"papers": papers, "count": len(papers)})
```

The simplest router. `positive_list` is already the source of truth for what's saved. Fetch metadata for all of them, render. `saved=True` is hardcoded because every paper on this page is by definition saved — the action button will show "✓ Saved" + "Remove".

---

## Templates

### `app/templates_env.py`

```python
from jinja2 import Environment
from fastapi.templating import Jinja2Templates

def _tojson_parse(value: str) -> list:
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except Exception:
        return []

templates = Jinja2Templates(directory="app/templates")
templates.env.filters["tojson_parse"] = _tojson_parse
```

One custom filter: `tojson_parse`. SQLite stores authors as a JSON string (`'["Vaswani", "Shazeer"]'`). In the template: `{{ paper.authors | tojson_parse | join(", ") }}`. The filter parses it back to a Python list. Returns `[]` on any error — never crashes the template.

All routers import `templates` from here. There is only one instance, shared everywhere.

---

### `app/templates/base.html`

```html
<head>
  <link href="https://cdn.jsdelivr.net/npm/daisyui@4.12.10/dist/full.min.css" rel="stylesheet"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <style>
    .htmx-swapping { opacity: 0; transition: opacity 200ms ease-out; }
    .htmx-request .htmx-indicator { display: inline-block !important; }
    .htmx-indicator { display: none; }
  </style>
</head>
<body>
  <div class="navbar bg-base-100 shadow-sm px-4">
    <a href="/" class="text-xl font-bold text-primary">📄 ArXiv Rec</a>
    <a href="/search" class="btn btn-ghost btn-sm">Search</a>
    <a href="/saved"  class="btn btn-ghost btn-sm">Saved</a>
  </div>
  <main class="container mx-auto px-4 py-6 max-w-4xl">
    {% block content %}{% endblock %}
  </main>
</body>
```

Zero build step. TailwindCSS + DaisyUI from CDN, HTMX from CDN.

**HTMX CSS hooks:**
- `.htmx-swapping` — HTMX adds this class to an element just before it's replaced. The `opacity: 0` + transition creates the fade-out animation on dismissed cards.
- `.htmx-indicator` — hidden by default. `.htmx-request .htmx-indicator` makes it visible while any HTMX request is in flight. Used for the loading spinners next to buttons.

---

### `app/templates/index.html`

```html
<!-- Search bar with live-search -->
<form hx-get="/search"
      hx-target="#search-results"
      hx-push-url="true"
      hx-indicator="#search-spinner">
  <input type="text" name="q" placeholder="e.g. transformer attention" />
  <button>Search <span id="search-spinner" class="htmx-indicator loading ..."></span></button>
</form>

<!-- Recommendations: loaded after page paint -->
<div id="rec-section"
     hx-get="/api/recommendations"
     hx-trigger="load"
     hx-swap="innerHTML">
  Loading...
</div>

<!-- Search results: swapped here by HTMX -->
<div id="search-results"></div>
```

**`hx-trigger="load"`**: the `#rec-section` div fires the HTMX request as soon as it loads. The page renders immediately with "Loading..." and the recs appear ~500ms later. This way the page never feels slow — you see content instantly, then recs fill in.

**`hx-push-url="true"`**: when a search fires, HTMX pushes `/search?q=...` to the browser history. So the back button works and the URL is shareable.

---

### `app/templates/partials/paper_card.html`

```html
<div class="card bg-base-100 shadow-sm border border-base-300 p-4 space-y-2"
     id="paper-{{ paper.arxiv_id }}">

  <div class="flex items-start justify-between gap-2">
    <a href="https://arxiv.org/abs/{{ paper.arxiv_id }}"
       target="_blank" class="font-semibold text-primary hover:underline">
      {{ paper.title }}
    </a>
    <span class="badge badge-outline badge-sm">{{ paper.category }}</span>
  </div>

  <div class="text-xs text-base-content/50">
    [{{ paper.arxiv_id }}]
    {% if paper.published %} · {{ paper.published[:4] }}{% endif %}
    {% if authors_list %} · {{ authors_list | join(", ") }}{% endif %}
  </div>

  <p class="text-sm line-clamp-3">{{ paper.abstract }}</p>

  <div id="actions-{{ paper.arxiv_id }}">
    {% include "partials/action_buttons.html" %}
  </div>
</div>
```

Two IDs per card: `#paper-{id}` on the outer div (target for dismiss — the whole card is removed) and `#actions-{id}` on the buttons div (target for save — only the buttons are swapped to "Saved" state).

`line-clamp-3` is a Tailwind utility that truncates the abstract to 3 lines with an ellipsis.

---

### `app/templates/partials/action_buttons.html`

```html
{% set pid     = paper_id if paper_id is defined else paper.arxiv_id %}
{% set is_saved = saved if saved is defined else (paper.saved | default(false)) %}
{% set _source  = source if source is defined else "search" %}

{% if is_saved %}
  <button class="btn btn-success btn-xs" disabled>✓ Saved</button>
  <button hx-post="/api/papers/{{ pid }}/not-interested"
          hx-target="#paper-{{ pid }}"
          hx-swap="outerHTML swap:200ms"
          hx-vals='{"source": "{{ _source }}"}'>Remove</button>
{% else %}
  <button hx-post="/api/papers/{{ pid }}/save"
          hx-target="#actions-{{ pid }}"
          hx-swap="innerHTML"
          hx-vals='{"source": "{{ _source }}", "position": "{{ position | default(0) }}"}'>
    ⭐ Save
  </button>
  <button hx-post="/api/papers/{{ pid }}/not-interested"
          hx-target="#paper-{{ pid }}"
          hx-swap="outerHTML swap:200ms"
          hx-vals='{"source": "{{ _source }}"}'>
    ✕ Not interested
  </button>
{% endif %}
```

This partial is used in two contexts:
1. **Inside `paper_card.html`** — `paper` is defined, `paper_id` is not
2. **As a direct response from `events.py/save_paper`** — `paper_id` is defined, `paper` is not

The `{% set pid = ... if ... is defined else ... %}` pattern handles both safely. `Jinja2`'s `default()` filter would crash here because it eagerly evaluates both branches regardless of which one is chosen.

**`hx-vals`** sends additional form fields with the HTMX request. The `source` and `position` values ride along with every button click to be logged in the DB.

---

### `app/templates/partials/recommendations.html`

```html
{% if papers %}
  <div class="space-y-3">
    {% for paper in papers %}
      {% set position = loop.index0 %}
      {% set source = "recommendation" %}
      {% include "partials/paper_card.html" %}
    {% endfor %}
  </div>
  <div class="text-center pt-3">
    <button hx-get="/api/recommendations"
            hx-target="#rec-section"
            hx-swap="innerHTML"
            hx-indicator="#rec-refresh-spinner">
      ↻ Show different recommendations
      <span id="rec-refresh-spinner" class="htmx-indicator loading ..."></span>
    </button>
  </div>
{% else %}
  {% include "partials/empty_recs.html" %}
{% endif %}
```

`{% set source = "recommendation" %}` before the include ensures that every action button rendered from this partial carries `source="recommendation"` in its `hx-vals`. The actions router will log that source to the DB.

The refresh button re-triggers the same `/api/recommendations` endpoint. Because the Qdrant Recommend API doesn't return deterministic results (it's an ANN search), re-requesting can surface different papers from the same vector neighborhood.

---

## Tests

### Test Isolation Pattern

Every test that touches the DB or in-memory cache uses this fixture:

```python
@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")

    # Point both the config and the db module at a fresh temp DB
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)

    # Clear in-memory caches so tests don't bleed into each other
    import app.user_state as us
    us._cache.clear()

    from app.qdrant_svc import _client
    _client.cache_clear()    # lru_cache singleton — need to clear between tests

    from app.main import app
    asyncio.get_event_loop().run_until_complete(db_mod.init_db())

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
```

`tmp_path` is a pytest built-in that gives each test its own temporary directory. Monkeypatching `DB_PATH` means every test gets a fresh, empty SQLite file. Clearing `us._cache` and `_client.cache_clear()` ensures no in-process state bleeds between tests.

### Mocking Pattern for Live Services

Tests that need recommendations mock both the Qdrant service and the arXiv metadata fetcher:

```python
def test_recommendations_after_save(client, monkeypatch):
    import app.qdrant_svc as qs
    import app.arxiv_svc as arxiv

    async def fake_recommend(positive_arxiv_ids, negative_arxiv_ids, seen_arxiv_ids, limit):
        return ["1706.03762"]
    monkeypatch.setattr(qs, "recommend", fake_recommend)

    async def fake_batch(ids):
        return {"1706.03762": {"arxiv_id": "1706.03762",
                               "title": "Attention Is All You Need", ...}}
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", fake_batch)

    client.get("/")
    client.post("/api/papers/0704.0002/save", data={"source": "search"})
    resp = client.get("/api/recommendations")
    assert "Attention Is All You Need" in resp.text
```

`monkeypatch.setattr` replaces the real function for the duration of the test, then automatically restores it. This lets integration tests run without network access.

---

## Data Flow Summary

```
User types "transformer attention" in search bar
  │
  │  HTMX: GET /search?q=transformer+attention  (HX-Request: true)
  ▼
search.py: arxiv_svc.search("transformer attention")
  │  → GET https://export.arxiv.org/api/query?search_query=all:transformer+attention
  │  ← Atom XML, 10 entries
  │  → parse → cache in paper_metadata table
  │  → annotate with saved/dismissed from user_state
  ▼
returns partials/search_results.html → HTMX swaps into #search-results
  │
User clicks ⭐ Save on paper 1706.03762
  │
  │  HTMX: POST /api/papers/1706.03762/save  {source: "search", position: 3}
  ▼
events.py:
  1. db.log_interaction(user_id, "1706.03762", "save", source="search", position=3)
  2. us.record_positive(user_id, "1706.03762")
  3. asyncio.create_task(qdrant_svc.lookup_qdrant_ids(["1706.03762"]))  ← background
  ▼
returns partials/action_buttons.html (saved=True) → HTMX swaps buttons in-place

  [Background task]
  qdrant_svc.lookup_qdrant_ids(["1706.03762"])
    → db.get_qdrant_ids_batch: miss
    → Qdrant scroll filter: arxiv_id = "1706.03762"
    ← point_id = 523419
    → db.save_qdrant_id("1706.03762", 523419)

User navigates to home page /
  │
  │  HTMX: GET /api/recommendations  (hx-trigger="load")
  ▼
recommendations.py:
  1. us.ensure_loaded(user_id) → positives = ["1706.03762"]
  2. qdrant_svc.recommend(positive=["1706.03762"], negative=[], seen={"1706.03762"})
       → db.get_qdrant_ids_batch(["1706.03762"]) → {523419}  (already cached)
       → Qdrant query_points with RecommendQuery(positive=[523419])
       ← [point_612003, point_88341, ...]
       → filter out seen papers in Python
       ← ["2302.13971", "2307.09288", ...]
  3. arxiv_svc.fetch_metadata_batch(["2302.13971", "2307.09288", ...])
       → check paper_metadata cache: some hits, some misses
       → arXiv API batch fetch for misses → cache results
  ▼
returns partials/recommendations.html → HTMX swaps into #rec-section
```

---

## File Count Summary

| File | Lines | Job |
|---|---|---|
| `app/config.py` | 36 | All settings |
| `app/db.py` | 185 | SQLite: 3 tables, 8 functions |
| `app/arxiv_svc.py` | 159 | arXiv API + metadata cache |
| `app/user_state.py` | 112 | In-memory deque cache per user |
| `app/qdrant_svc.py` | 166 | Qdrant ID lookup + Recommend |
| `app/templates_env.py` | ~20 | Shared Jinja2 env + tojson_parse |
| `app/main.py` | 54 | FastAPI app + home route |
| `app/routers/search.py` | 56 | GET /search |
| `app/routers/events.py` | 75 | POST save + not-interested |
| `app/routers/recommendations.py` | 62 | GET /api/recommendations |
| `app/routers/saved.py` | 47 | GET /saved |
| Templates | ~200 | All HTML |
| Tests | ~600 | 54 tests across 6 files |
