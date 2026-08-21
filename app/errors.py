"""
Report a swallowed exception with enough detail to actually diagnose it.

THE PROBLEM
-----------
`grep -rn "traceback|exc_info" app/` returned ZERO hits across the whole
application, against 95 `except Exception` handlers. Every failure in
production was a single line:

    [recommendations] multi-interest preprocessing failed: KeyError('2401.1')

No traceback, no line number, no module path. And the handler that prints that
particular line sits at the bottom of a **473-line** try block covering the
entire Tier-1 pipeline — clustering, quota, retrieval, metadata, reranking, MMR
and labelling — so the message narrows the fault to "somewhere in the feed".
The feed then silently degrades to Tier 2 and the user sees a worse feed with
no indication anything broke.

Swallowing is usually the right CHOICE here: a feed that degrades beats a feed
that 500s, and the graceful-degradation paths are deliberate. What was wrong is
swallowing SILENTLY. This keeps the behaviour and adds the diagnosis.

DESIGN
------
Still `print`, because that is the codebase convention (CLAUDE.md §5.3) and
because HF Spaces captures stdout. The change is what gets printed: the
exception type, the failing file and line, and a trimmed traceback pointing at
OUR frames rather than a hundred lines of library internals.

Trimmed, not full: an untrimmed traceback through httpx and asyncio buries the
one frame that matters. The last few application frames are what a person reads.
"""
from __future__ import annotations

import os
import traceback

# How many of the innermost frames to show. Enough to see the call into the
# failing helper, short enough to stay readable in a log tail.
_FRAMES = 4

# Set RESEARCHIT_FULL_TRACEBACKS=1 to keep everything, for the rare failure that
# is genuinely inside a library.
_FULL = os.getenv("RESEARCHIT_FULL_TRACEBACKS", "").strip().lower() in (
    "1", "true", "yes")


def _app_frames(exc: BaseException) -> list[traceback.FrameSummary]:
    """Our frames, preferentially — library internals are rarely the story."""
    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in frames if f"{os.sep}app{os.sep}" in f.filename
            or f.filename.endswith("run.py")]
    return ours or frames


def describe(exc: BaseException) -> str:
    """One line: type, message, and where it actually happened."""
    frames = _app_frames(exc)
    where = ""
    if frames:
        f = frames[-1]
        where = f" at {os.path.basename(f.filename)}:{f.lineno} in {f.name}()"
    return f"{type(exc).__name__}: {exc}{where}"


def report(module: str, message: str, exc: BaseException) -> None:
    """Print a swallowed exception with a usable trace.

    `module` is the bracket tag the codebase already uses ("recommendations",
    "turso", …) so existing log filters keep working.
    """
    print(f"[{module}] {message}: {describe(exc)}")

    frames = traceback.extract_tb(exc.__traceback__)
    if not _FULL:
        frames = frames[-_FRAMES:]
    for line in traceback.format_list(frames):
        for part in line.rstrip().split("\n"):
            print(f"[{module}]   {part}")

    # A chained cause is usually the real one — "during handling of X, Y
    # occurred" is where the answer lives more often than not.
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        print(f"[{module}]   caused by {describe(cause)}")
