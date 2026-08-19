"""Schemas for the append-only metrics log at ``metrics/history.jsonl``.

One JSON object per line, one line per eval run. The file is committed, so a
metric change shows up as a reviewable diff inside the pull request that caused
it. Every row carries full provenance: a number with no attached configuration
is an anecdote, not a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

HISTORY_PATH = Path("metrics/history.jsonl")


class RunContext(BaseModel):
    """Provenance shared by every eval run, regardless of layer."""

    commit: str = Field(description="Full git SHA the run was executed against.")
    branch: str
    timestamp: str = Field(description="UTC ISO-8601, recorded by the caller.")

    embedding_model: str
    embedding_revision: str
    model_cheap: str
    model_strong: str
    retrieval_config_hash: str = Field(
        description="Hash of retrieval-affecting settings; see Settings.retrieval_config_hash."
    )

    corpus_manifest_sha: str = Field(
        description="Checksum of the corpus manifest, so a corpus change is never invisible."
    )

    api_cost_usd: float = Field(ge=0.0)
    duration_s: float = Field(ge=0.0)


class RetrievalMetrics(BaseModel):
    """Layer 1 — does hybrid retrieval surface the ground-truth articles?"""

    layer: Literal["retrieval"] = "retrieval"

    n_questions: int = Field(gt=0)
    recall_at_k: dict[int, float] = Field(description="k -> recall. The README reports k=3, 5, 10.")
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    faithfulness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM-as-judge: is the answer supported by the retrieved chunks? "
        "Null when the run skipped the billable judging pass.",
    )

    # Which gold questions retrieval still misses at k=5. Kept in the row so the
    # dashboard can show *what* fails, not merely that the aggregate dipped.
    misses_at_5: list[str] = Field(default_factory=list)


class AuditMetrics(BaseModel):
    """Layer 2 — does the auditor find the violations that were planted?"""

    layer: Literal["audit"] = "audit"

    n_contracts: int = Field(gt=0)
    n_planted_violations: int = Field(ge=0)

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)

    # Aggregate F1 hides which violation classes are unsolved. A per-class
    # breakdown is what turns the table into something diagnostic.
    per_violation_type: dict[str, float] = Field(
        default_factory=dict, description="violation type -> F1"
    )


Metrics = Annotated[RetrievalMetrics | AuditMetrics, Field(discriminator="layer")]


class MetricsRow(BaseModel):
    context: RunContext
    metrics: Metrics


def append_row(row: MetricsRow, path: Path = HISTORY_PATH) -> None:
    """Append one run to the history log, creating it if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = row.model_dump_json()
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")


def load_history(path: Path = HISTORY_PATH) -> list[MetricsRow]:
    """Read the full history, oldest first. Missing file means no runs yet."""
    if not path.exists():
        return []
    rows: list[MetricsRow] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(MetricsRow.model_validate(json.loads(raw)))
    return rows


def latest_for_layer(
    rows: list[MetricsRow], layer: Literal["retrieval", "audit"]
) -> MetricsRow | None:
    """Most recent run for a layer, or None if that layer has never run."""
    for row in reversed(rows):
        if row.metrics.layer == layer:
            return row
    return None
