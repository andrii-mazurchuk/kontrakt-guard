from __future__ import annotations

import pytest

from evals.render_readme import render, render_layer1, render_layer2

TEMPLATE = """# Title

<!-- METRICS:LAYER1:START -->
_No eval runs recorded yet._
<!-- METRICS:LAYER1:END -->

middle

<!-- METRICS:LAYER2:START -->
_No eval runs recorded yet._
<!-- METRICS:LAYER2:END -->
"""


def test_empty_history_renders_placeholders():
    out = render(TEMPLATE, rows=[])
    assert out.count("_No eval runs recorded yet._") == 2


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
