from __future__ import annotations

import pytest
from pydantic import ValidationError

from evals.schema import (
    AuditMetrics,
    MetricsRow,
    RetrievalMetrics,
    append_row,
    latest_for_layer,
    load_history,
)


def test_roundtrip_through_jsonl(tmp_path, retrieval_row, audit_row):
    path = tmp_path / "history.jsonl"
    append_row(retrieval_row, path)
    append_row(audit_row, path)

    rows = load_history(path)
    assert [r.metrics.layer for r in rows] == ["retrieval", "audit"]
    assert isinstance(rows[0].metrics, RetrievalMetrics)
    assert isinstance(rows[1].metrics, AuditMetrics)
    # int keys survive the JSON round trip (JSON stringifies them)
    assert rows[0].metrics.recall_at_k[5] == pytest.approx(0.75)


def test_load_history_on_missing_file_is_empty(tmp_path):
    assert load_history(tmp_path / "nope.jsonl") == []


def test_latest_for_layer_picks_the_most_recent(tmp_path, retrieval_row, audit_row):
    newer = retrieval_row.model_copy(deep=True)
    newer.metrics.recall_at_k = {5: 0.99}

    rows = [retrieval_row, audit_row, newer]
    assert latest_for_layer(rows, "retrieval") is newer
    assert latest_for_layer(rows, "audit") is audit_row
    assert latest_for_layer([], "retrieval") is None


def test_rows_are_one_line_each(tmp_path, retrieval_row):
    """JSONL must stay line-oriented, or the committed history stops diffing cleanly."""
    path = tmp_path / "history.jsonl"
    append_row(retrieval_row, path)
    append_row(retrieval_row, path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_out_of_range_metrics_are_rejected():
    with pytest.raises(ValidationError):
        AuditMetrics(n_contracts=5, n_planted_violations=1, precision=1.4, recall=0.5, f1=0.5)


def test_layer_discriminator_rejects_unknown_layer(retrieval_row):
    payload = retrieval_row.model_dump()
    payload["metrics"]["layer"] = "vibes"
    with pytest.raises(ValidationError):
        MetricsRow.model_validate(payload)
