from __future__ import annotations

import pytest

from evals.render_readme import EMPTY, render, render_answers, render_layer1, render_layer2

TEMPLATE = """# Title

<!-- METRICS:LAYER1:START -->
_No eval runs recorded yet._
<!-- METRICS:LAYER1:END -->

middle

<!-- METRICS:ANSWERS:START -->
_No eval runs recorded yet._
<!-- METRICS:ANSWERS:END -->

<!-- METRICS:LAYER2:START -->
_No eval runs recorded yet._
<!-- METRICS:LAYER2:END -->
"""


def test_empty_history_renders_placeholders():
    out = render(TEMPLATE, rows=[])
    # One placeholder per layer: retrieval, answers, audit.
    assert out.count("_No eval runs recorded yet._") == 3


def test_layer1_table_reports_the_documented_k_values(retrieval_row):
    table = render_layer1(retrieval_row)
    assert "k=3" in table and "k=5" in table and "k=10" in table
    assert "60.0%" in table and "75.0%" in table and "85.0%" in table
    assert "30" in table  # gold set size


def test_layer1_marks_unreported_k_rather_than_inventing_one(retrieval_row):
    retrieval_row.metrics.recall_at_k = {3: 0.6}
    table = render_layer1(retrieval_row)
    assert "—" in table


def test_layer2_table_includes_per_type_breakdown(audit_row):
    table = render_layer2(audit_row)
    assert "probation_too_long" in table
    assert "wage_below_minimum" in table
    assert "0.746" in table


def test_render_preserves_surrounding_content(retrieval_row, audit_row):
    out = render(TEMPLATE, rows=[retrieval_row, audit_row])
    assert out.startswith("# Title")
    assert "middle" in out
    assert out.count("<!-- METRICS:LAYER1:START -->") == 1


def test_render_is_idempotent(retrieval_row, audit_row):
    once = render(TEMPLATE, rows=[retrieval_row, audit_row])
    twice = render(once, rows=[retrieval_row, audit_row])
    assert once == twice


def test_missing_marker_is_a_loud_failure():
    with pytest.raises(ValueError, match="missing the METRICS:LAYER1 marker"):
        render("# no markers here", rows=[])


def test_provenance_is_always_attached(retrieval_row):
    """A number without its configuration is an anecdote."""
    table = render_layer1(retrieval_row)
    assert "abc123def456" in table  # retrieval config hash
    assert "multilingual-e5-large" in table


def test_answers_table_reports_defect_rates_with_their_direction(answer_row):
    """Precision-style and defect-style rows sit in one table; direction must be stated."""
    body = render_answers(answer_row)

    assert "| **Citation F1** | **0.755** |" in body
    assert "| False refusals | 4.0% |" in body
    assert "lower is better" in body
    # A refusal on an answerable question is a defect, and the table says so.
    assert "*false* refusal" in body


def test_answers_table_is_empty_before_any_run():
    assert render_answers(None) == EMPTY


def test_answers_table_ignores_a_row_from_another_layer(retrieval_row):
    """Layer sections must never render each other's numbers."""
    assert render_answers(retrieval_row) == EMPTY
