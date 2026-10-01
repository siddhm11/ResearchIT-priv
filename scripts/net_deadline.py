"""A hard wall-clock limit for one blocking network call.

Socket timeouts bound each individual read, not the request: a server that
trickles bytes, or a library call made without a timeout, can block forever.
On 2026-10-01 a daily-refresh ingest sat 20 minutes at 0% CPU in an SSL read
(_buffered_readline -> SSLSocket.read -> poll) while arXiv answered other
requests in 0.2 s. SIGALRM interrupts a blocked syscall, and the handler's
exception aborts the call instead of letting it resume.

Main thread only (where the ingest makes all its requests); elsewhere it is a
no-op rather than an error.
"""
from __future__ import annotations

import contextlib
import signal
import threading


class DeadlineExceeded(TimeoutError):
    pass


@contextlib.contextmanager
def deadline(seconds: float, what: str = "network call"):
    if (seconds <= 0 or not hasattr(signal, "SIGALRM")
            or threading.current_thread() is not threading.main_thread()):
        yield
        return

    def _expire(_signum, _frame):
        raise DeadlineExceeded(f"{what} exceeded {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
