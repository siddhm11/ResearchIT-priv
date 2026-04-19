"""
SQLite database layer.

Tables
──────
interactions        – every user action (save, not_interested, click, view)
paper_qdrant_map    – arxiv_id → integer Qdrant point ID (cached lazily)
paper_metadata      – arXiv API response cache (title, abstract, …)
"""
import aiosqlite
from app.config import DB_PATH

# ── DDL ───────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS interactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    NOT NULL,
    paper_id      TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,   -- save | not_interested | click | view
    source        TEXT,               -- search | recommendation
    position      INTEGER,
    query_id      TEXT,
    timestamp     TEXT    NOT NULL DEFAULT (datetime('now'))
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
    computed_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, cluster_idx)
);
"""


async def init_db() -> None:
    """Create tables if they don't exist. Called once at startup."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


# ── Interaction helpers ───────────────────────────────────────────────────────

async def log_interaction(
    user_id: str,
    paper_id: str,
    event_type: str,
    source: str | None = None,
    position: int | None = None,
    query_id: str | None = None,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO interactions
               (user_id, paper_id, event_type, source, position, query_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, paper_id, event_type, source, position, query_id),
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
                   (user_id, cluster_idx, medoid_paper_id, importance, paper_ids)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, c["cluster_idx"], c["medoid_paper_id"],
                 c["importance"], c["paper_ids"]),
            )
        await conn.commit()


async def get_user_clusters(user_id: str) -> list[dict]:
    """Return clusters for a user, ordered by importance desc."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            """SELECT cluster_idx, medoid_paper_id, importance, paper_ids, computed_at
               FROM user_clusters
               WHERE user_id = ?
               ORDER BY importance DESC""",
            (user_id,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]
