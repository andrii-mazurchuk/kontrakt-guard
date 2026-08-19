from __future__ import annotations

from evals import gate


def _with_recall(row, **recall):
    clone = row.model_copy(deep=True)
    clone.metrics.recall_at_k = {int(k[1:]): v for k, v in recall.items()}
    return clone


def test_no_history_passes(retrieval_row):
    passed, lines = gate.check("retrieval", history=[])
    assert passed
    assert "nothing to gate" in " ".join(lines)


def test_first_run_records_baseline_without_comparing(retrieval_row):
    passed, lines = gate.check("retrieval", history=[retrieval_row])
    assert passed
    assert "baseline" in " ".join(lines)


def test_improvement_passes(retrieval_row):
    better = _with_recall(retrieval_row, k3=0.70, k5=0.85, k10=0.95)
    passed, _ = gate.check("retrieval", history=[retrieval_row, better])
    assert passed


def test_regression_beyond_tolerance_fails(retrieval_row):
    worse = _with_recall(retrieval_row, k3=0.60, k5=0.55, k10=0.85)
    passed, lines = gate.check("retrieval", history=[retrieval_row, worse])
    assert not passed
    assert any("recall@5" in line and "FAILED" in line for line in lines)


def test_small_dip_within_tolerance_passes(retrieval_row):
    """LLM non-determinism jitters metrics; only real regressions should block."""
    jittered = _with_recall(retrieval_row, k3=0.60, k5=0.74, k10=0.85)
    passed, _ = gate.check("retrieval", history=[retrieval_row, jittered])
    assert passed


def test_gate_ignores_the_other_layer(retrieval_row, audit_row):
    worse = _with_recall(retrieval_row, k5=0.10)
    passed, _ = gate.check("audit", history=[retrieval_row, worse, audit_row])
    assert passed


def test_absolute_floor_blocks_even_without_regression(retrieval_row, monkeypatch):
    monkeypatch.setitem(gate.FLOORS, "recall@5", 0.80)
    passed, lines = gate.check("retrieval", history=[retrieval_row])
    assert not passed
    assert any("below the floor" in line for line in lines)


def test_floors_are_unset_by_default():
    """Floors invented before the first measurement would be made-up numbers."""
    assert all(v is None for v in gate.FLOORS.values())
