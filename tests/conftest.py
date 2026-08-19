from __future__ import annotations

import pytest

from evals.schema import AuditMetrics, MetricsRow, RetrievalMetrics, RunContext


def make_context(commit: str = "a" * 40, cost: float = 0.12) -> RunContext:
    return RunContext(
        commit=commit,
        branch="main",
        timestamp="2026-08-19T10:00:00Z",
        embedding_model="intfloat/multilingual-e5-large",
        embedding_revision="deadbeef",
        model_cheap="claude-haiku-4-5-20251001",
        model_strong="claude-sonnet-5",
        retrieval_config_hash="abc123def456",
        corpus_manifest_sha="c" * 64,
        api_cost_usd=cost,
        duration_s=42.0,
    )


@pytest.fixture
def retrieval_row() -> MetricsRow:
    return MetricsRow(
        context=make_context(),
        metrics=RetrievalMetrics(
            n_questions=30,
            recall_at_k={3: 0.60, 5: 0.75, 10: 0.85},
            mrr=0.52,
            faithfulness=0.9,
            misses_at_5=["q-07", "q-19"],
        ),
    )


@pytest.fixture
def audit_row() -> MetricsRow:
    return MetricsRow(
        context=make_context(commit="b" * 40),
        metrics=AuditMetrics(
            n_contracts=24,
            n_planted_violations=57,
            precision=0.80,
            recall=0.70,
            f1=0.746,
            per_violation_type={"probation_too_long": 0.9, "wage_below_minimum": 0.6},
        ),
    )
