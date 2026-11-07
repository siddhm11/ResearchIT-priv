"""One service failing to stop must not cost the final backup of user data.

The user database lives on the Space's temporary disk and reaches Turso every
60 s; the last flush runs at shutdown. Eight service stops ran one after
another with no guard of their own, so one raising skipped the flush and lost
the last minute of saves (audit F8).
"""
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import hn_svc, turso_sync
from app.main import app


def test_the_final_flush_runs_when_a_service_fails_to_stop():
    flush = AsyncMock()
    with patch.object(hn_svc, "stop", AsyncMock(side_effect=RuntimeError("boom"))), \
            patch.object(turso_sync, "stop", flush):
        with TestClient(app):
            pass
    flush.assert_awaited_once()


def test_every_other_service_still_stops():
    later = AsyncMock()
    from app import paper_source_svc
    with patch.object(hn_svc, "stop", AsyncMock(side_effect=RuntimeError("boom"))), \
            patch.object(paper_source_svc, "stop", later), \
            patch.object(turso_sync, "stop", AsyncMock()):
        with TestClient(app):
            pass
    later.assert_awaited_once()
