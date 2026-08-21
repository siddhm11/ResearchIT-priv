"""
A swallowed exception must still be diagnosable.

`grep -rn "traceback|exc_info" app/` returned ZERO hits across the whole
application, against 95 `except Exception` handlers. Every production failure
was one line — no traceback, no line number:

    [recommendations] multi-interest preprocessing failed: KeyError('2401.1')

And the handler printing that sits at the bottom of a 473-line try covering the
entire Tier-1 pipeline, so the message narrows the fault to "somewhere in the
feed" while the user silently drops to Tier 2.

Swallowing is the right CHOICE — a feed that degrades beats one that 500s.
Swallowing SILENTLY was not.
"""
import pathlib
import re

import pytest

from app import errors


def _raise_nested():
    def inner(d):
        return d["absent"]

    def middle(items):
        return [inner(i) for i in items]

    middle([{"present": 1}])


def test_report_names_the_failing_function_and_line(capsys):
    try:
        _raise_nested()
    except Exception as e:
        errors.report("recommendations", "Tier 1 failed", e)

    out = capsys.readouterr().out
    assert "KeyError" in out
    assert "inner()" in out, "does not say which function failed"
    assert re.search(r":\d+ in ", out), "no line number"


def test_report_keeps_the_module_tag_so_log_filters_still_work(capsys):
    try:
        _raise_nested()
    except Exception as e:
        errors.report("turso", "metadata request failed", e)
    for line in capsys.readouterr().out.strip().split("\n"):
        assert line.startswith("[turso]"), f"untagged line: {line!r}"


def test_describe_is_a_single_line():
    try:
        _raise_nested()
    except Exception as e:
        d = errors.describe(e)
    assert "\n" not in d
    assert "KeyError" in d


def test_traceback_is_trimmed_by_default(capsys):
    """An untrimmed trace through httpx and asyncio buries the one frame that
    matters."""
    try:
        _raise_nested()
    except Exception as e:
        errors.report("x", "failed", e)
    frames = [l for l in capsys.readouterr().out.split("\n") if "File \"" in l]
    assert 0 < len(frames) <= errors._FRAMES + 1


def test_a_chained_cause_is_surfaced(capsys):
    """"During handling of X, Y occurred" is where the answer usually is."""
    try:
        try:
            raise ValueError("the real cause")
        except ValueError as inner:
            raise RuntimeError("the visible symptom") from inner
    except Exception as e:
        errors.report("x", "failed", e)

    out = capsys.readouterr().out
    assert "the visible symptom" in out
    assert "caused by" in out and "the real cause" in out


def test_report_never_raises_on_an_exception_with_no_traceback(capsys):
    """Defensive: a handler that itself throws would turn a degraded feed into
    a 500, which is the outcome the swallowing exists to prevent."""
    errors.report("x", "failed", ValueError("never raised"))
    assert "ValueError" in capsys.readouterr().out


def test_the_feed_catch_all_reports_a_trace():
    """The single worst diagnosability point in the codebase."""
    src = pathlib.Path("app/routers/recommendations.py").read_text()
    assert 'errors.report("recommendations"' in src, (
        "the 473-line Tier-1 try still swallows without a trace")
