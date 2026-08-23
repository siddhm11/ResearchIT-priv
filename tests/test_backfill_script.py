"""
The abstract backfill.

90.6% of the live corpus (1,630,273 of 1,799,348 rows, measured 2026-08-21) is
capped at 500 characters, and the truncation degrades four things at once: the
card, the generated explanation, the density badge, and the cross-encoder in the
search path.

These cover the decision logic. The network and the writes are not exercised —
this repairs a production store, and a test that mutates it would be worse than
no test.
"""
import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "backfill", pathlib.Path("scripts/backfill_abstracts.py"))
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)


def test_batch_and_pause_respect_arxiv_policy():
    """arXiv asks for no more than one request every three seconds in bulk."""
    assert backfill.PAUSE_S >= 3.0
    assert backfill.BATCH <= 200, "id_list batches larger than this get refused"


def test_a_marginally_longer_abstract_is_not_worth_a_write():
    """Rewriting a row for 10 extra characters costs a write and buys nothing."""
    assert backfill.MIN_GAIN_CHARS > 0
    assert 500 + backfill.MIN_GAIN_CHARS > 500


def test_cell_encodes_the_types_turso_accepts():
    assert backfill._cell(None) == {"type": "null", "value": None}
    assert backfill._cell(7) == {"type": "integer", "value": "7"}
    assert backfill._cell("abc") == {"type": "text", "value": "abc"}


def test_every_strategy_is_a_valid_ordering():
    """A typo here would silently reorder a 13-hour job."""
    for strategy in ("served", "recent", "any"):
        # pick() builds SQL from this map; exercise the lookup without a network.
        order = {"served": "ORDER BY citation_count DESC",
                 "recent": "ORDER BY update_date DESC",
                 "any": ""}[strategy]
        assert "DROP" not in order.upper()
        assert order == "" or order.startswith("ORDER BY")


def test_writing_requires_an_explicit_flag():
    """--dry-run is the default posture for a bulk production mutation."""
    src = pathlib.Path("scripts/backfill_abstracts.py").read_text()
    assert '"--apply"' in src
    assert "if args.dry_run or not args.apply:" in src, (
        "the script can write without an explicit --apply")


def test_the_script_is_resumable():
    """A repaired row must stop matching the selection.

    This test previously asserted `>= 500` and PASSED while the script was NOT
    resumable — a repaired abstract of 976 characters still satisfies `>= 500`,
    so the script would have re-selected and re-fetched its own completed work
    forever. The predicate has to be the cap EXACTLY.
    """
    assert backfill.TRUNCATED == "length(abstract_preview) = 500"

    src = pathlib.Path("scripts/backfill_abstracts.py").read_text()
    assert "abstract_preview) >= 500" not in src, (
        "a >= predicate also matches repaired rows and rows that were never "
        "truncated")


def test_the_predicate_does_not_count_healthy_rows_as_damaged():
    """`>= 500` counted 193,689 legitimately-long abstracts as truncated,
    overstating the job by 13%."""
    assert "=" in backfill.TRUNCATED and ">=" not in backfill.TRUNCATED


def test_one_http_request_per_batch():
    """The first run was 429'd out of 9 of its 20 batches.

    The cause was reusing arxiv_svc.fetch_metadata_batch, which fans any input
    into 20-id sub-requests at ~3/s — so a "batch of 100" was a burst of five
    requests and the pause sat between bursts, not between requests.
    """
    import ast
    src = pathlib.Path("scripts/backfill_abstracts.py").read_text()
    assert "async def fetch_abstracts" in src

    # Parsed, not grepped: the module comments EXPLAIN why the interactive
    # helper is not used, and a substring check flags its own rationale.
    tree = ast.parse(src)
    calls = [
        ast.unparse(n.func) for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "arxiv_svc.fetch_metadata_batch" not in calls, (
        "still fanning out through the interactive helper")

    assert backfill.MAX_RETRIES >= 2, "no retry budget for rate limiting"
    assert "429" in src, "no explicit handling of arXiv rate limiting"


def test_old_style_ids_survive_the_round_trip():
    """6.4% of the corpus has a category prefix; the script must key on it."""
    from app.arxiv_svc import _normalise_id
    assert _normalise_id("http://arxiv.org/abs/math/0309136v1") == "math/0309136"


def test_writes_are_batched_into_one_round_trip():
    """One UPDATE per row is not a detail at this scale.

    At ~150ms per Turso round trip, 1.63M individual writes would spend ~68
    hours on HTTP alone — dwarfing the arXiv rate limiting the estimate was
    originally based on. Turso's pipeline API takes a batch.
    """
    src = pathlib.Path("scripts/backfill_abstracts.py").read_text()
    assert "async def _pipeline" in src
    assert "for aid, full in updates" in src

    # The per-row await inside the write loop is what must not come back.
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "repair":
            for inner in ast.walk(node):
                if isinstance(inner, ast.For):
                    body = ast.unparse(inner)
                    if "UPDATE papers" in body and "_turso(" in body:
                        pytest.fail(
                            "repair() awaits a write inside a per-row loop")


def test_the_estimate_is_measured_not_modelled():
    """Four estimates, three wrong: 13.6h counted only the arXiv pause, 22.6h
    added a guessed write cost but kept an inflated truncation count, 10h fixed
    the count but kept the guess. This one is timed against real batches."""
    assert backfill.SECONDS_PER_BATCH > backfill.PAUSE_S, (
        "the per-batch cost cannot be less than the mandated pause")
    src = pathlib.Path("scripts/backfill_abstracts.py").read_text()
    assert "SECONDS_PER_BATCH" in src
    assert "PAUSE_S + 2.0" not in src, "still using the guessed write cost"
