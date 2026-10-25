"""
The Turso databases this app writes to and reads from: a primary and,
optionally, a mirror on a second account that holds the same data.

Why two
-------
On 2026-10-24 the free plan's 500M monthly reads ran out and Turso refused
every read, even `SELECT 1`, until the next month. Writes still worked. The
running Space carried on from its sidecar and local SQLite, but a restart would
have booted with empty user tables, and paper details for new papers, trending
topics and the figure warm-up all stopped. A second database on another account
has its own quota, so one account's block or outage leaves the other answering.

Both are on Turso, in the same region, so this does not cover a Turso-wide
outage, and a bad write reaches both. It covers one account being blocked,
suspended or unreachable.

How each side uses it
---------------------
  * writes  turso_sync pushes user data to every remote, each with its own
            watermark, so a remote that was down catches up when it returns.
            The daily job copies its papers to the mirror the same way
            (scripts/daily_refresh.py, mirror stage).
  * reads   read_order() puts the primary first. A read that fails with an
            outage (BLOCKED, 5xx, 401/403/429, timeout, connection error)
            benches that remote for TURSO_FAILOVER_S, so the next reads go
            straight to the other instead of first waiting on the dead one:
            a Turso round trip from the Space is about a second.

With no mirror configured, remotes() is just the primary, bench() does nothing,
and every caller behaves exactly as it did before this module existed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from app import config


@dataclass(frozen=True)
class Remote:
    name: str        # "primary" or "mirror"
    url: str         # https:// base, no trailing slash
    token: str


def https_base(url: str) -> str:
    u = url.strip().rstrip("/")
    if u.startswith("libsql://"):
        return "https://" + u[len("libsql://"):]
    if u.startswith("http://"):
        return "https://" + u[len("http://"):]
    if not u.startswith("https://"):
        return "https://" + u
    return u


def remotes() -> list[Remote]:
    """Primary first. Empty when Turso is not configured at all."""
    out = []
    if config.TURSO_URL and config.TURSO_DB_TOKEN:
        out.append(Remote("primary", https_base(config.TURSO_URL), config.TURSO_DB_TOKEN))
    if out and config.TURSO_MIRROR_URL and config.TURSO_MIRROR_DB_TOKEN:
        out.append(Remote("mirror", https_base(config.TURSO_MIRROR_URL),
                          config.TURSO_MIRROR_DB_TOKEN))
    return out


def has_mirror() -> bool:
    return len(remotes()) > 1


# ── read failover ────────────────────────────────────────────────────────────

_benched_until: dict[str, float] = {}
_last_outage: dict[str, str] = {}
_failovers = {"n": 0}


def read_order() -> list[Remote]:
    """Remotes to try for a read: healthy ones first, primary before mirror.

    A benched remote is still tried last rather than dropped: when both are
    benched, trying one beats failing without asking.
    """
    now = time.monotonic()
    rs = remotes()
    return ([r for r in rs if _benched_until.get(r.name, 0) <= now]
            + [r for r in rs if _benched_until.get(r.name, 0) > now])


def bench(remote: Remote, reason: str) -> None:
    """Skip `remote` for reads for TURSO_FAILOVER_S. No-op without a mirror."""
    if not has_mirror():
        return
    now = time.monotonic()
    if _benched_until.get(remote.name, 0) <= now:
        _failovers["n"] += 1
        print(f"[turso] {remote.name} benched for {config.TURSO_FAILOVER_S}s: {reason[:160]}")
    _benched_until[remote.name] = now + config.TURSO_FAILOVER_S
    _last_outage[remote.name] = reason[:200]


def recovered(remote: Remote) -> None:
    _benched_until.pop(remote.name, None)


class TursoError(RuntimeError):
    """A Turso request failed. `outage` is True when the database refused
    service (blocked, down, rate-limited) rather than rejecting our SQL."""

    def __init__(self, message: str, outage: bool):
        super().__init__(message)
        self.outage = outage


def is_outage_status(status: int) -> bool:
    """HTTP statuses that mean "this database can't answer", not "bad query"."""
    return status >= 500 or status in (401, 403, 429)


def is_outage_error(error: dict | str | None) -> bool:
    """An in-band Turso error (HTTP 200, error inside the result) that means the
    database refuses service, as opposed to an error in our SQL.

    The 2026-10-24 block came back as
    {"code": "BLOCKED", "message": "Operation was blocked: SQL read operations are
    forbidden (reads are blocked, do you need to upgrade your plan?)"}.
    """
    if not error:
        return False
    if isinstance(error, dict):
        code = str(error.get("code") or "").upper()
        text = f"{code} {error.get('message') or ''}".lower()
    else:
        code, text = "", str(error).lower()
    return code == "BLOCKED" or "blocked" in text or "forbidden" in text or "quota" in text


def is_outage_exception(e: BaseException) -> bool:
    if isinstance(e, TursoError):
        return e.outage
    if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        return is_outage_status(e.response.status_code)
    return is_outage_error(str(e))


def status() -> dict:
    now = time.monotonic()
    return {
        "mirror_configured": has_mirror(),
        "read_order": [r.name for r in read_order()],
        "benched": {name: round(until - now) for name, until in _benched_until.items() if until > now},
        "last_outage": dict(_last_outage),
        "failovers": _failovers["n"],
    }
