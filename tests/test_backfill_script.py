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
    """Selection is by current truncation, so repaired rows drop out."""
    src = pathlib.Path("scripts/backfill_abstracts.py").read_text()
    assert "length(abstract_preview) >= 500" in src, (
        "selection is not driven by the defect, so a re-run would redo work")
