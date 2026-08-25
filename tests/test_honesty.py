"""The honesty invariant: a replayed run can never become a published number.

The cassette makes evals free, and free is exactly the condition under which a
number gets re-run casually and quoted without thinking. Everything here defends
the one boundary that matters — replay is for testing the pipeline, never for
measuring it.
"""

from __future__ import annotations

import json

import pytest

from evals import gate
from evals.schema import (
    MetricsRow,
    NotAMeasurement,
    append_row,
    latest_for_layer,
    load_history,
    provenance_of,
)
from graphs.llm import Usage


def usage_with(live: int = 0, replayed: int = 0) -> Usage:
    usage = Usage()
    usage.calls = live
    usage.replayed_calls = replayed
    return usage


def as_provenance(row: MetricsRow, provenance: str) -> MetricsRow:
    clone = row.model_copy(deep=True)
    clone.context.provenance = provenance  # type: ignore[assignment]
    return clone


# --- provenance is derived, never asserted ------------------------------------


@pytest.mark.parametrize(
    "live,replayed,expected",
    [(12, 0, "live"), (0, 12, "replayed"), (5, 7, "mixed"), (0, 0, "live")],
)
def test_provenance_is_derived_from_usage(live, replayed, expected):
    assert provenance_of(usage_with(live, replayed)) == expected


def test_a_live_run_is_the_default_for_a_fresh_usage():
    """Zero calls of either kind is a live run that happened to make none."""
    assert provenance_of(Usage()) == "live"


# --- append_row is the mechanism ----------------------------------------------


def test_a_live_row_records_normally(tmp_path, answer_row):
    path = tmp_path / "history.jsonl"
    append_row(answer_row, path)
    assert len(load_history(path)) == 1


@pytest.mark.parametrize("provenance", ["replayed", "mixed"])
def test_a_cassette_served_row_cannot_be_recorded(tmp_path, answer_row, provenance):
    """The single choke point. Every publishing path goes through append_row.

    Bypassing it means deleting a named exception in a reviewable diff, which is
    a much harder thing to do by accident than forgetting a flag.
    """
    path = tmp_path / "history.jsonl"
    with pytest.raises(NotAMeasurement, match="cassette"):
        append_row(as_provenance(answer_row, provenance), path)

    assert not path.exists(), "nothing may be written before the refusal"


def test_latest_for_layer_ignores_a_hand_edited_replayed_row(retrieval_row):
    """Defends the README generator against a history file edited by hand."""
    replayed = as_provenance(retrieval_row, "replayed")
    assert latest_for_layer([retrieval_row, replayed], "retrieval") is retrieval_row
    assert latest_for_layer([replayed], "retrieval") is None


# --- the gate -----------------------------------------------------------------


def test_the_gate_ignores_replayed_rows_and_says_so(retrieval_row):
    replayed = as_provenance(retrieval_row, "replayed")
    replayed.metrics.recall_at_k = {3: 0.01, 5: 0.01, 10: 0.01}  # type: ignore[union-attr]

    passed, lines = gate.check("retrieval", history=[retrieval_row, replayed])

    assert passed, "a replayed row must not be gated against"
    assert any("Ignoring 1 non-live" in line for line in lines)
    # It compared nothing, because only one live row remains.
    assert any("baseline" in line for line in lines)


def test_the_gate_reports_the_skip_even_when_nothing_is_left(retrieval_row):
    replayed = as_provenance(retrieval_row, "replayed")
    passed, lines = gate.check("retrieval", history=[replayed])
    assert passed
    assert any("Ignoring 1 non-live" in line for line in lines)
    assert any("nothing to gate" in line for line in lines)


# --- backwards compatibility --------------------------------------------------


def test_history_rows_written_before_provenance_existed_still_validate():
    """The three real rows in metrics/history.jsonl predate this field.

    Defaulting to "live" is not a convenience: they were live, and a default that
    invalidated them would mean rewriting recorded history to add a column.
    """
    payload = {
        "context": {
            "commit": "a" * 40,
            "branch": "main",
            "timestamp": "2026-08-21T17:07:07+00:00",
            "embedding_model": "intfloat/multilingual-e5-large",
            "embedding_revision": "deadbeef",
            "model_cheap": "claude-haiku-4-5-20251001",
            "model_strong": "claude-sonnet-5",
            "retrieval_config_hash": "5a945bc32acf",
            "corpus_manifest_sha": "c" * 64,
            "api_cost_usd": 0.0,
            "duration_s": 114.1,
        },
        "metrics": {
            "layer": "retrieval",
            "n_questions": 97,
            "recall_at_k": {"3": 0.77, "5": 0.85, "10": 0.92},
            "mrr": 0.73,
            "faithfulness": None,
            "misses_at_5": [],
        },
    }
    assert "provenance" not in payload["context"]

    row = MetricsRow.model_validate(payload)
    assert row.context.provenance == "live"


def test_the_committed_history_is_all_live():
    """The real file, not a fixture — it must keep loading after the schema change."""
    rows = load_history()
    assert rows, "metrics/history.jsonl should not be empty"
    assert all(row.context.provenance == "live" for row in rows)


def test_avoided_cost_is_not_a_field_a_row_can_carry():
    """It lives on Usage only.

    `append_row` refuses non-live rows, so a written row's avoided cost could
    only ever be zero — a field implying the cassette had been used to produce a
    metric, which is the one thing it must never do.
    """
    from evals.schema import RunContext

    assert "avoided_cost_usd" not in RunContext.model_fields
    assert "avoided_cost_usd" in Usage.__dataclass_fields__


def test_a_replayed_row_never_reaches_disk_through_the_public_path(tmp_path, answer_row):
    """Belt and braces: a written file must never gain a non-live line."""
    path = tmp_path / "history.jsonl"
    append_row(answer_row, path)
    with pytest.raises(NotAMeasurement):
        append_row(as_provenance(answer_row, "mixed"), path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["context"]["provenance"] == "live"
