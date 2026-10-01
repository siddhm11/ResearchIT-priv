"""scripts/net_deadline.py interrupts calls that socket timeouts cannot."""
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from net_deadline import DeadlineExceeded, deadline  # noqa: E402


@pytest.fixture
def silent_server():
    """Accepts connections and never sends a byte: the hang seen on 2026-10-01."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    conns = []
    stop = threading.Event()

    def run():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conns.append(srv.accept()[0])
            except OSError:
                pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}/"
    stop.set()
    t.join(1)
    for c in conns:
        c.close()
    srv.close()


def test_deadline_interrupts_a_read_the_socket_timeout_would_allow(silent_server):
    t0 = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        with deadline(1.0, "test"), urllib.request.urlopen(silent_server, timeout=100) as r:
            r.read()
    assert time.monotonic() - t0 < 5


def test_no_stray_alarm_after_a_call_that_finishes():
    with deadline(0.3, "fast"):
        pass
    time.sleep(0.5)          # an alarm left armed would raise here


def test_off_the_main_thread_it_is_a_no_op():
    out = []

    def worker():
        with deadline(0.1, "thread"):
            time.sleep(0.3)
        out.append("ok")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert out == ["ok"]
