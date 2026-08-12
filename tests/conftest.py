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

    # Set the app's own kill switch rather than monkeypatching enabled(). Using
    # the real mechanism means the tests exercise the code path that protects a
    # dev server, instead of a stub that could drift away from it -- and it
    # leaves enabled() itself testable.
    previous = os.environ.get("TURSO_SYNC_DISABLED")
    os.environ["TURSO_SYNC_DISABLED"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TURSO_SYNC_DISABLED", None)
        else:
            os.environ["TURSO_SYNC_DISABLED"] = previous


@pytest.fixture(autouse=True, scope="session")
def _disable_rate_limiting():
    """Rate limiting off by default for the suite.

    app/rate_limit.py keeps its counters in module-level state, and the
    TestClient presents the same client identity for every request, so the whole
    session shares one bucket. Enough tests hitting /search would silently start
    getting 429s -- a failure that would look like a routing or template bug and
    would land on whichever test happened to run 31st.

    tests/test_rate_limit.py re-enables it per-test with monkeypatch, so the
    limiter itself is still covered.
    """
    from app import config

    original = config.RATE_LIMIT_ENABLED
    config.RATE_LIMIT_ENABLED = False
    try:
        yield
    finally:
        config.RATE_LIMIT_ENABLED = original
