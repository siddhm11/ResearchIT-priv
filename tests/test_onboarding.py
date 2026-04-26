"""
Tests for Phase 5: Onboarding (DB layer + config helpers).
"""
import os
import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    import app.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    return db_path


# ── user_onboarding table ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_onboarding_table_exists(tmp_db):
    """The user_onboarding table should be created by init_db."""
    import app.db as db
    import aiosqlite
    await db.init_db()
    async with aiosqlite.connect(tmp_db) as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {r[0] for r in await cur.fetchall()}
    assert "user_onboarding" in tables


@pytest.mark.asyncio
async def test_get_onboarding_state_returns_none_for_new_user(tmp_db):
    """A user with no onboarding row should return None."""
    import app.db as db
    await db.init_db()
    result = await db.get_onboarding_state("brand-new-user")
    assert result is None


@pytest.mark.asyncio
async def test_save_and_get_categories(tmp_db):
    """Saving categories should be retrievable via get_onboarding_state."""
    import app.db as db
    await db.init_db()
    await db.save_onboarding_categories("u1", ["nlp", "cv", "ml"])
    state = await db.get_onboarding_state("u1")
    assert state is not None
    assert state["selected_categories"] == ["nlp", "cv", "ml"]
    assert state["onboarding_completed"] == 0


@pytest.mark.asyncio
async def test_save_categories_upserts(tmp_db):
    """Second save should overwrite the first."""
    import app.db as db
    await db.init_db()
    await db.save_onboarding_categories("u1", ["nlp"])
    await db.save_onboarding_categories("u1", ["cv", "astro"])
    state = await db.get_onboarding_state("u1")
    assert state["selected_categories"] == ["cv", "astro"]


@pytest.mark.asyncio
async def test_complete_onboarding(tmp_db):
    """complete_onboarding should set the flag to 1."""
    import app.db as db
    await db.init_db()
    await db.save_onboarding_categories("u1", ["nlp"])
    await db.complete_onboarding("u1")
    state = await db.get_onboarding_state("u1")
    assert state["onboarding_completed"] == 1


@pytest.mark.asyncio
async def test_complete_onboarding_without_categories(tmp_db):
    """complete_onboarding should work even without prior categories (skip flow)."""
    import app.db as db
    await db.init_db()
    await db.complete_onboarding("u_skip")
    state = await db.get_onboarding_state("u_skip")
    assert state is not None
    assert state["onboarding_completed"] == 1
    assert state["selected_categories"] == []


@pytest.mark.asyncio
async def test_get_user_category_filter(tmp_db):
    """Category filter should expand group keys into arXiv codes."""
    import app.db as db
    await db.init_db()
    await db.save_onboarding_categories("u1", ["nlp", "cv"])
    cat_filter = await db.get_user_category_filter("u1")
    assert "cs.CL" in cat_filter
    assert "cs.IR" in cat_filter
    assert "cs.CV" in cat_filter
    # Not selected
    assert "cs.LG" not in cat_filter


@pytest.mark.asyncio
async def test_get_user_category_filter_empty_for_no_user(tmp_db):
    """Users without onboarding should return an empty set."""
    import app.db as db
    await db.init_db()
    result = await db.get_user_category_filter("nobody")
    assert result == set()


# ── config.expand_category_groups ─────────────────────────────────────────────

def test_expand_category_groups_basic():
    """expand_category_groups should flatten group keys into arXiv codes."""
    from app.config import expand_category_groups
    result = expand_category_groups(["nlp", "hep"])
    assert "cs.CL" in result
    assert "cs.IR" in result
    assert "hep-ph" in result
    assert "hep-th" in result


def test_expand_category_groups_empty():
    """Empty input → empty set."""
    from app.config import expand_category_groups
    assert expand_category_groups([]) == set()


def test_expand_category_groups_unknown_key():
    """Unknown keys should be silently skipped."""
    from app.config import expand_category_groups
    result = expand_category_groups(["nlp", "unknown_group_xyzzy"])
    assert "cs.CL" in result
    # unknown key produced nothing extra
    assert len(result) == 2  # cs.CL + cs.IR
