"""
SQLite database layer.

Tables
──────
interactions        – every user action (save, not_interested, click, view)
paper_qdrant_map    – arxiv_id → integer Qdrant point ID (cached lazily)
paper_metadata      – arXiv API response cache (title, abstract, …)

Phase 4.5 instrumentation columns (interactions table):
  ranker_version    – identifies which pipeline version served the paper
  candidate_source  – granular origin: 'cluster_0', 'exploration', 'ewma', etc.
  cluster_id        – which interest cluster served this paper (NULL if N/A)
"""
import aiosqlite
import hashlib
import json
import uuid as _uuid
from app.config import DB_PATH

# ── DDL ───────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS interactions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL,
    paper_id         TEXT    NOT NULL,
    event_type       TEXT    NOT NULL,   -- save | not_interested | click | view
    source           TEXT,               -- search | recommendation
    position         INTEGER,
    query_id         TEXT,
    ranker_version   TEXT,               -- Phase 4.5: pipeline version tag
    candidate_source TEXT,               -- Phase 4.5: 'cluster_0' | 'exploration' | 'ewma' | 'qdrant_recommend'
    cluster_id       INTEGER,            -- Phase 4.5: interest cluster index (NULL if N/A)
    timestamp        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ui_user_ts
    ON interactions(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ui_user_paper
    ON interactions(user_id, paper_id);

-- Maps arxiv_id -> Qdrant integer point ID (populated lazily on first save)
CREATE TABLE IF NOT EXISTS paper_qdrant_map (
    arxiv_id        TEXT PRIMARY KEY,
    qdrant_point_id INTEGER NOT NULL,
    mapped_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Cache of paper metadata fetched from the arXiv API
CREATE TABLE IF NOT EXISTS paper_metadata (
    arxiv_id    TEXT PRIMARY KEY,
    title       TEXT,
    abstract    TEXT,
    authors     TEXT,   -- JSON array string
    category    TEXT,
    published   TEXT,
    cached_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Phase 2a: EWMA user profile embeddings (long_term, short_term, negative)
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id           TEXT NOT NULL,
    profile_type      TEXT NOT NULL,  -- 'long_term' | 'short_term' | 'negative'
    vector            BLOB NOT NULL,  -- 4096 bytes (1024 × float32)
    interaction_count  INTEGER DEFAULT 0,
    updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, profile_type)
);

-- Phase 2b: Ward clustering results (medoid paper IDs per interest cluster)
CREATE TABLE IF NOT EXISTS user_clusters (
    user_id         TEXT NOT NULL,
    cluster_idx     INTEGER NOT NULL,
    medoid_paper_id TEXT NOT NULL,
    importance      REAL NOT NULL,
    paper_ids       TEXT NOT NULL,  -- JSON array of arxiv_ids
    medoid_embedding_blob BLOB,    -- Phase 6.3: persisted medoid for zero-vector fallback
    computed_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, cluster_idx)
);

-- Phase 5: Onboarding state and category selections
CREATE TABLE IF NOT EXISTS user_onboarding (
    user_id              TEXT PRIMARY KEY,
    selected_categories  TEXT,        -- JSON array of group keys: ["nlp", "cv", "ml"]
    onboarding_completed INTEGER DEFAULT 0,  -- 0 = in progress, 1 = done
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Phase 6.5 B3: Append-only cluster history (current-state still in user_clusters)
CREATE TABLE IF NOT EXISTS cluster_snapshots (
    user_id              TEXT NOT NULL,
    snapshot_id          TEXT NOT NULL,           -- UUID, one per recluster event
    cluster_idx          INTEGER NOT NULL,        -- stable index after Hungarian
    medoid_paper_id      TEXT NOT NULL,
    importance           REAL NOT NULL,
    paper_ids            TEXT NOT NULL,           -- JSON array
    medoid_embedding_blob BLOB,
    snapshot_date        TEXT NOT NULL DEFAULT (datetime('now')),
    paper_ids_hash       TEXT NOT NULL,           -- sha256(sorted(paper_ids))[:16]
    PRIMARY KEY (user_id, snapshot_id, cluster_idx)
);
CREATE INDEX IF NOT EXISTS idx_snap_user_date
    ON cluster_snapshots(user_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snap_hash
    ON cluster_snapshots(paper_ids_hash);

-- What the feed has SHOWN a user, as opposed to what they acted on.
--
-- `seen` (saves + dismissals) only remembers papers the user touched, so a
-- reader who refreshes without clicking anything was served the identical page
-- forever: the cold-start feed is a deterministic citation sort, and nothing
-- recorded that its ten papers had already been on screen.
--
-- The primary key is (user_id, paper_id) rather than an event id: this is a
-- "has this been on screen" set, not an event log, so re-showing a paper
-- refreshes shown_at instead of appending a row. That bounds the table by
-- distinct papers shown rather than by refresh count.
--
-- Deliberately NOT replicated to Turso by app/turso_sync.py. It is high-volume
-- and low-value-per-row, and losing it on restart degrades to exactly today's
-- behaviour (some repeats) rather than to anything broken.
CREATE TABLE IF NOT EXISTS feed_impressions (
    user_id   TEXT NOT NULL,
    paper_id  TEXT NOT NULL,
    shown_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_impr_user_time
    ON feed_impressions(user_id, shown_at DESC);

-- Which curated collections a user follows. The collections themselves are
-- repo content (data/collections/*.json); this is the user-data half.
CREATE TABLE IF NOT EXISTS collection_follows (
    user_id     TEXT NOT NULL,
    slug        TEXT NOT NULL,
    followed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, slug)
);
CREATE INDEX IF NOT EXISTS idx_follow_user
    ON collection_follows(user_id);
"""


# ── Phase 4.5: ALTER TABLE migration for existing DBs ─────────────────────────
# SQLite does not support IF NOT EXISTS for columns, so we try/except.
_MIGRATION_4_5 = [
    "ALTER TABLE interactions ADD COLUMN ranker_version TEXT",
    "ALTER TABLE interactions ADD COLUMN candidate_source TEXT",
    "ALTER TABLE interactions ADD COLUMN cluster_id INTEGER",
]

# ── Phase 6.3: Persist medoid embeddings for Bug B fallback ───────────────────
_MIGRATION_6_3 = [
    "ALTER TABLE user_clusters ADD COLUMN medoid_embedding_blob BLOB",
]

# ── Phase 6.5 B2: Propensity + policy_id for counterfactual evaluation ────────
_MIGRATION_6_5 = [
    "ALTER TABLE interactions ADD COLUMN propensity REAL",
    "ALTER TABLE interactions ADD COLUMN policy_id TEXT",
]


async def init_db() -> None:
    """Create tables if they don't exist. Called once at startup."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        # Phase 4.5: add instrumentation columns to existing DBs
        for stmt in _MIGRATION_4_5:
            try:
                await db.execute(stmt)
            except Exception:
                pass  # Column already exists — safe to ignore
        # Phase 6.3: add medoid embedding blob for Bug B fallback
        for stmt in _MIGRATION_6_3:
            try:
                await db.execute(stmt)
            except Exception:
                pass  # Column already exists — safe to ignore
        # Phase 6.5 B2: add propensity + policy_id for SNIPS evaluation
        for stmt in _MIGRATION_6_5:
            try:
                await db.execute(stmt)
            except Exception:
                pass  # Column already exists — safe to ignore
        await db.commit()


# ── Interaction helpers ───────────────────────────────────────────────────────

async def log_interaction(
    user_id: str,
    paper_id: str,
    event_type: str,
    source: str | None = None,
    position: int | None = None,
    query_id: str | None = None,
    ranker_version: str | None = None,
    candidate_source: str | None = None,
    cluster_id: int | None = None,
    propensity: float | None = None,
    policy_id: str | None = None,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO interactions
               (user_id, paper_id, event_type, source, position, query_id,
                ranker_version, candidate_source, cluster_id,
                propensity, policy_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, paper_id, event_type, source, position, query_id,
             ranker_version, candidate_source, cluster_id,
             propensity, policy_id),
        )
        await db.commit()


async def get_user_interactions(
    user_id: str, event_types: list[str] | None = None, limit: int = 50
) -> list[dict]:
    """Return recent interactions for a user, optionally filtered by event type."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            cur = await db.execute(
                f"""SELECT paper_id, event_type, timestamp
                    FROM interactions
                    WHERE user_id = ?
                      AND event_type IN ({placeholders})
                    ORDER BY timestamp DESC
                    LIMIT ?""",
                [user_id, *event_types, limit],
            )
        else:
            cur = await db.execute(
                """SELECT paper_id, event_type, timestamp
                   FROM interactions
                   WHERE user_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (user_id, limit),
            )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


# ── Qdrant map helpers ────────────────────────────────────────────────────────

async def get_qdrant_id(arxiv_id: str) -> int | None:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT qdrant_point_id FROM paper_qdrant_map WHERE arxiv_id = ?",
            (arxiv_id,),
        )
        row = await cur.fetchone()
        return row[0] if row else None


async def save_qdrant_id(arxiv_id: str, qdrant_point_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO paper_qdrant_map (arxiv_id, qdrant_point_id)
               VALUES (?, ?)""",
            (arxiv_id, qdrant_point_id),
        )
        await db.commit()


async def get_qdrant_ids_batch(arxiv_ids: list[str]) -> dict[str, int]:
    """Return {arxiv_id: qdrant_point_id} for all IDs found in cache."""
    if not arxiv_ids:
        return {}
    async with aiosqlite.connect(DB_PATH) as db:
        placeholders = ",".join("?" * len(arxiv_ids))
        cur = await db.execute(
            f"SELECT arxiv_id, qdrant_point_id FROM paper_qdrant_map WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        )
        rows = await cur.fetchall()
        return {r[0]: r[1] for r in rows}


# ── Metadata cache helpers ────────────────────────────────────────────────────

async def get_cached_metadata(arxiv_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM paper_metadata WHERE arxiv_id = ?", (arxiv_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def cache_metadata(paper: dict) -> None:
    """Upsert paper metadata dict into cache. Expects 'arxiv_id' key."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO paper_metadata
               (arxiv_id, title, abstract, authors, category, published)
               VALUES (:arxiv_id, :title, :abstract, :authors, :category, :published)""",
            paper,
        )
        await db.commit()


async def get_cached_metadata_batch(arxiv_ids: list[str]) -> dict[str, dict]:
    """Return {arxiv_id: metadata_dict} for all IDs found in cache."""
    if not arxiv_ids:
        return {}
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        placeholders = ",".join("?" * len(arxiv_ids))
        cur = await db.execute(
            f"SELECT * FROM paper_metadata WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        )
        rows = await cur.fetchall()
        return {r["arxiv_id"]: dict(r) for r in rows}


# ── User profile helpers (Phase 2a) ──────────────────────────────────────────

async def get_user_profile(user_id: str, profile_type: str) -> dict | None:
    """Return profile row as dict, or None if not found."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT vector, interaction_count FROM user_profiles "
            "WHERE user_id = ? AND profile_type = ?",
            (user_id, profile_type),
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def upsert_user_profile(
    user_id: str,
    profile_type: str,
    vector: bytes,
    interaction_count: int,
) -> None:
    """Insert or update a user profile embedding."""
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            """INSERT INTO user_profiles
               (user_id, profile_type, vector, interaction_count, updated_at)
               VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(user_id, profile_type) DO UPDATE SET
                 vector = excluded.vector,
                 interaction_count = excluded.interaction_count,
                 updated_at = excluded.updated_at""",
            (user_id, profile_type, vector, interaction_count),
        )
        await conn.commit()


# ── User cluster helpers (Phase 2b) ──────────────────────────────────────────

async def save_user_clusters(user_id: str, clusters: list[dict]) -> None:
    """Replace all clusters for a user with new ones."""
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "DELETE FROM user_clusters WHERE user_id = ?", (user_id,)
        )
        for c in clusters:
            await conn.execute(
                """INSERT INTO user_clusters
                   (user_id, cluster_idx, medoid_paper_id, importance, paper_ids,
                    medoid_embedding_blob)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, c["cluster_idx"], c["medoid_paper_id"],
                 c["importance"], c["paper_ids"],
                 c.get("medoid_embedding_blob")),
            )
        await conn.commit()


async def get_user_clusters(user_id: str) -> list[dict]:
    """Return clusters for a user, ordered by importance desc."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            """SELECT cluster_idx, medoid_paper_id, importance, paper_ids,
                      medoid_embedding_blob, computed_at
               FROM user_clusters
               WHERE user_id = ?
               ORDER BY importance DESC""",
            (user_id,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


# ── Phase 4.3: Category suppression helpers ───────────────────────────────────

async def cache_turso_metadata_batch(papers: list[dict]) -> None:
    """
    Write Turso paper dicts to the paper_metadata SQLite cache.

    Called after every Turso fetch so dismissal-category JOINs work.
    Silently skips rows missing required fields.
    """
    if not papers:
        return
    async with aiosqlite.connect(DB_PATH) as conn:
        for paper in papers:
            if not paper.get("arxiv_id"):
                continue
            try:
                await conn.execute(
                    """INSERT OR REPLACE INTO paper_metadata
                       (arxiv_id, title, abstract, authors, category, published)
                       VALUES (:arxiv_id, :title, :abstract, :authors, :category, :published)""",
                    {
                        "arxiv_id": paper.get("arxiv_id", ""),
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", ""),
                        "authors": paper.get("authors", "[]"),
                        "category": paper.get("category", ""),
                        "published": paper.get("published", ""),
                    },
                )
            except Exception:
                pass
        await conn.commit()


async def get_suppressed_categories(
    user_id: str,
    threshold: int = 3,
    window_days: int = 14,
) -> set[str]:
    """
    Return categories the user has strongly signalled disinterest in.

    A category is suppressed when the user has dismissed ≥ threshold papers
    in that category within the last window_days days.

    Requires paper_metadata to be populated (via cache_turso_metadata_batch).
    Returns an empty set if no suppressions are found.
    """
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(
            """SELECT pm.category, COUNT(*) AS cnt
               FROM interactions i
               JOIN paper_metadata pm ON i.paper_id = pm.arxiv_id
               WHERE i.user_id = ?
                 AND i.event_type = 'not_interested'
                 AND i.timestamp >= datetime('now', ? || ' days')
                 AND pm.category != ''
               GROUP BY pm.category
               HAVING COUNT(*) >= ?""",
            (user_id, f"-{window_days}", threshold),
        )
        rows = await cur.fetchall()
        return {row[0] for row in rows}


# ── Phase 5: Onboarding helpers ───────────────────────────────────────────────

async def save_onboarding_categories(
    user_id: str, categories: list[str]
) -> None:
    """Save or update user's selected category groups."""
    import json
    cats_json = json.dumps(categories)
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            """INSERT INTO user_onboarding (user_id, selected_categories, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(user_id) DO UPDATE SET
                   selected_categories = excluded.selected_categories,
                   updated_at = datetime('now')""",
            (user_id, cats_json),
        )
        await conn.commit()


async def get_onboarding_state(user_id: str) -> dict | None:
    """Fetch onboarding data for a user.  Returns None if no row exists."""
    import json
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT * FROM user_onboarding WHERE user_id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        # Parse categories JSON
        try:
            d["selected_categories"] = json.loads(d["selected_categories"] or "[]")
        except (json.JSONDecodeError, TypeError):
            d["selected_categories"] = []
        return d


async def complete_onboarding(user_id: str) -> None:
    """Mark user's onboarding as complete (upsert)."""
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            """INSERT INTO user_onboarding (user_id, onboarding_completed, updated_at)
               VALUES (?, 1, datetime('now'))
               ON CONFLICT(user_id) DO UPDATE SET
                   onboarding_completed = 1,
                   updated_at = datetime('now')""",
            (user_id,),
        )
        await conn.commit()


async def get_user_category_filter(user_id: str) -> set[str]:
    """Return the flat set of arXiv category codes for a user's selected groups."""
    state = await get_onboarding_state(user_id)
    if state is None:
        return set()
    from app.config import expand_category_groups
    return expand_category_groups(state["selected_categories"])


# ── Phase 6.5 B3: Cluster snapshot versioning ─────────────────────────────────

async def save_cluster_snapshot(user_id: str, clusters: list[dict]) -> str:
    """Append a new snapshot of the user's clusters. Returns snapshot_id.

    This is purely additive history — current-state queries still hit
    user_clusters. Retrospective queries hit cluster_snapshots.

    Each cluster dict must have: cluster_idx, medoid_paper_id, importance,
    paper_ids (list[str] or JSON string), optionally medoid_embedding_blob.
    """
    snapshot_id = str(_uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as conn:
        for c in clusters:
            paper_ids = c["paper_ids"]
            if isinstance(paper_ids, str):
                paper_ids = json.loads(paper_ids)
            paper_ids_hash = hashlib.sha256(
                json.dumps(sorted(paper_ids)).encode()
            ).hexdigest()[:16]
            await conn.execute(
                """INSERT INTO cluster_snapshots
                   (user_id, snapshot_id, cluster_idx, medoid_paper_id,
                    importance, paper_ids, medoid_embedding_blob, paper_ids_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, snapshot_id, c["cluster_idx"], c["medoid_paper_id"],
                 c["importance"], json.dumps(paper_ids),
                 c.get("medoid_embedding_blob"), paper_ids_hash),
            )
        await conn.commit()
    return snapshot_id


async def prune_old_snapshots(retention_days: int = 30) -> int:
    """Delete cluster snapshots older than retention_days. Returns rows deleted."""
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(
            "DELETE FROM cluster_snapshots WHERE snapshot_date < datetime('now', ?)",
            (f"-{retention_days} days",),
        )
        await conn.commit()
        return cur.rowcount


# ── Feed impressions ─────────────────────────────────────────────────────────
#
# See the feed_impressions DDL for why this exists and why it is not replicated.

async def record_impressions(user_id: str, paper_ids: list[str]) -> int:
    """Mark papers as having been shown to this user. Returns rows written.

    Upserts shown_at so a re-shown paper moves to the back of the eviction
    queue rather than accumulating rows.
    """
    if not user_id or not paper_ids:
        return 0
    rows = [(user_id, str(pid)) for pid in paper_ids if pid]
    if not rows:
        return 0
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.executemany(
            "INSERT INTO feed_impressions (user_id, paper_id, shown_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(user_id, paper_id) DO UPDATE SET shown_at = datetime('now')",
            rows,
        )
        await conn.commit()
    return len(rows)


async def count_feed_issues(user_id: str) -> int:
    """
    How many distinct days this user has been served a feed, counting today.

    Used for the masthead's issue number. Distinct DAYS rather than distinct
    requests on purpose: refreshing four times in an afternoon is one issue,
    not four, and a number that climbs every time you hit reload would be
    noise dressed up as a milestone.

    Best-effort — the masthead degrades to no number rather than failing the
    feed, which is why callers do not need to guard this.
    """
    if not user_id:
        return 1
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            cur = await conn.execute(
                "SELECT COUNT(DISTINCT date(shown_at)) FROM feed_impressions "
                "WHERE user_id = ?",
                (user_id,),
            )
            row = await cur.fetchone()
    except aiosqlite.Error:
        return 0
    prior = int(row[0]) if row and row[0] else 0
    # Today's impressions are written after the page renders, so on a user's
    # first-ever feed the table is still empty; count today explicitly.
    return prior if prior else 1


async def get_impressed_ids(user_id: str, within_days: int | None = None) -> set[str]:
    """Papers already shown to this user, optionally limited to a recent window.

    `within_days=None` means "everything on record", which is what the feed
    wants: a paper shown last week is still not news.
    """
    if not user_id:
        return set()
    sql = "SELECT paper_id FROM feed_impressions WHERE user_id = ?"
    args: list = [user_id]
    if within_days is not None:
        sql += " AND shown_at >= datetime('now', ?)"
        args.append(f"-{int(within_days)} days")
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(sql, args)
        return {r[0] for r in await cur.fetchall()}


async def forget_oldest_impressions(user_id: str, keep: int = 0) -> int:
    """Drop this user's oldest impressions, keeping the `keep` most recent.

    Called when the candidate pool is exhausted. Without it the feed would go
    empty for anyone who has worked through everything their categories offer,
    which is a far worse failure than showing an older paper again.
    """
    if not user_id:
        return 0
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(
            "DELETE FROM feed_impressions WHERE user_id = ? AND paper_id NOT IN ("
            "    SELECT paper_id FROM feed_impressions WHERE user_id = ?"
            "    ORDER BY shown_at DESC LIMIT ?)",
            (user_id, user_id, max(0, int(keep))),
        )
        await conn.commit()
        return cur.rowcount


async def prune_impressions(retention_days: int = 90) -> int:
    """Delete impressions older than retention_days. Returns rows deleted."""
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(
            "DELETE FROM feed_impressions WHERE shown_at < datetime('now', ?)",
            (f"-{retention_days} days",),
        )
        await conn.commit()
        return cur.rowcount


# ── Collection follows ───────────────────────────────────────────────────────
#
# Which curated collections a user follows. The collections themselves live in
# the repo (see app/collections_svc.py); this is the user-data half, so it is
# replicated by turso_sync.

async def follow_collection(user_id: str, slug: str) -> None:
    if not user_id or not slug:
        return
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO collection_follows (user_id, slug, followed_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(user_id, slug) DO UPDATE SET followed_at = datetime('now')",
            (user_id, slug),
        )
        await conn.commit()


async def unfollow_collection(user_id: str, slug: str) -> None:
    if not user_id or not slug:
        return
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "DELETE FROM collection_follows WHERE user_id = ? AND slug = ?",
            (user_id, slug),
        )
        await conn.commit()


async def get_followed_slugs(user_id: str) -> set[str]:
    if not user_id:
        return set()
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(
            "SELECT slug FROM collection_follows WHERE user_id = ?", (user_id,))
        return {r[0] for r in await cur.fetchall()}
