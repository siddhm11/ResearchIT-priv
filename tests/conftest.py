"""
Shared pytest configuration.

The critical thing here is that the test suite must never talk to the
production Turso database.

`app/config.py` calls `load_dotenv(".env.local")` at import time, so on a
developer machine the real `TURSO_URL` and `TURSO_DB_TOKEN` are live by the
time any test imports the app. Every test that builds a `TestClient(app)`
runs the FastAPI lifespan, and the lifespan starts `turso_sync`, which:

  - on startup calls `restore()`, pulling real users' interactions, profiles
    and clusters into the temporary test database, and
  - on shutdown calls `sync_once()`, pushing whatever the test just wrote
    back up to the production database.

That is how a `pytest` run silently became a read/write against live user
data. It went unnoticed because the tests that trigger it were erroring out
on an unrelated event-loop bug, so nobody read their output.

`turso_svc` (metadata reads) is deliberately left alone -- it is read-only
and several tests legitimately depend on it. Only the replication daemon is
disabled.
"""
import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_turso_replication():
    """Hard-disable the Turso sync daemon for the whole test session.

    Set RESEARCHIT_TEST_ALLOW_TURSO_SYNC=1 to opt back in, which should only
    ever be done against a scratch database.
    """
    if os.getenv("RESEARCHIT_TEST_ALLOW_TURSO_SYNC") == "1":
        yield
        return

    from app import turso_sync

    original = turso_sync.enabled
    # enabled() is consulted at call time by both start() and stop(), so
    # replacing it is sufficient to neutralise restore *and* push.
    turso_sync.enabled = lambda: False
    try:
        yield
    finally:
        turso_sync.enabled = original
