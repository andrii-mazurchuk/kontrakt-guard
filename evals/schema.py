"""Schemas for the append-only metrics log at ``metrics/history.jsonl``.

One JSON object per line, one line per eval run. The file is committed, so a
metric change shows up as a reviewable diff inside the pull request that caused
it. Every row carries full provenance: a number with no attached configuration
is an anecdote, not a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from graphs.llm import Usage

HISTORY_PATH = Path("metrics/history.jsonl")

Provenance = Literal["live", "replayed", "mixed"]


class NotAMeasurement(RuntimeError):  # noqa: N818
    """A run served wholly or partly from a cassette tried to record a metric."""


def provenance_of(usage: Usage) -> Provenance:
    """Where a run's model responses came from, derived — never asserted.

    Deliberately computed from `Usage` rather than passed by the caller. A flag a
    caller sets is a flag a caller can forget, and the failure mode is a replayed
    number published as a measurement.
    """
    if usage.replayed_calls == 0:
        return "live"
    if usage.calls == 0:
        return "replayed"
    return "mixed"


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

    # Defaults to "live" because it must: the rows already in the history file
    # predate the cassette and were every one of them live, and a default that
    # invalidated them would rewrite history to make room for a new field.
    #
    # There is deliberately no `avoided_cost_usd` here. `append_row` refuses any
    # row that is not live, so the field could never be non-zero in a written
    # row — it would be decoration implying the cassette had been used to
    # produce a metric, which is the one thing it must never do.
    provenance: Provenance = "live"


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


class AnswerMetrics(BaseModel):
    """Layer 1b — does the Q&A graph answer from the articles it was given?

    Scored against the same gold set as retrieval, but end to end through the
    graph, so it measures what retrieval, grading and generation do *together*.
    Every figure here is computed by comparing article ids, not by asking a model
    to judge prose: an LLM judge is a second opinion worth having later, and a
    poor substitute for a check that can simply be computed.
    """

    layer: Literal["answers"] = "answers"

    n_questions: int = Field(gt=0)

    # Every gold question is answerable from the corpus by construction, so a
    # refusal here is a *false* refusal. The guardrail is only valuable while
    # this stays low; a system that refuses everything is trivially faithful.
    refusal_rate: float = Field(ge=0.0, le=1.0)

    # Cited articles against ground-truth articles.
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)
    citation_f1: float = Field(ge=0.0, le=1.0)

    # How often generation reached for an article that was never retrieved. The
    # verifier strips these before anyone sees them, so this is the rate at which
    # the guardrail actually fires rather than the rate at which it fails.
    hallucinated_citation_rate: float = Field(
        ge=0.0, le=1.0, description="Share of answers citing at least one unretrieved article."
    )

    # An answer with no citation is ungrounded even when it is correct, and the
    # brief makes citations mandatory rather than encouraged.
    uncited_answer_rate: float = Field(ge=0.0, le=1.0)

    false_refusals: list[str] = Field(
        default_factory=list, description="Gold question ids the graph refused to answer."
    )


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


Metrics = Annotated[RetrievalMetrics | AnswerMetrics | AuditMetrics, Field(discriminator="layer")]

Layer = Literal["retrieval", "answers", "audit"]


class MetricsRow(BaseModel):
    context: RunContext
    metrics: Metrics


def append_row(row: MetricsRow, path: Path = HISTORY_PATH) -> None:
    """Append one run to the history log, creating it if absent.

    **A replayed run is not a measurement and cannot be recorded.** This is the
    mechanism, not a convention: every path that publishes a number goes through
    this one function, so making a cassette-served run publishable would mean
    deleting a named exception in a diff someone has to approve.

    Replay reproduces what the model said last time. That is exactly what makes
    it useful for testing the pipeline and exactly what makes it worthless as
    evidence about the pipeline's current behaviour.
    """
    if row.context.provenance != "live":
        raise NotAMeasurement(
            f"refusing to record a '{row.context.provenance}' run: responses came from a "
            "cassette, so this reproduces an old measurement rather than making a new one. "
            "Re-run with --cassette off (or --cassette record) to record a metric."
        )
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


def latest_for_layer(rows: list[MetricsRow], layer: Layer) -> MetricsRow | None:
    """Most recent *live* run for a layer, or None if that layer has never run.

    Non-live rows cannot reach the history file through `append_row`, so this
    filter defends against a hand-edited file rather than against the harness —
    which is precisely where a fabricated README number would come from.
    """
    for row in reversed(rows):
        if row.metrics.layer == layer and row.context.provenance == "live":
            return row
    return None
