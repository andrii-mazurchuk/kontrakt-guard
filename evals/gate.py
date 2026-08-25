"""Regression gate over ``metrics/history.jsonl``.

This is the point of the whole metrics setup: a number that nothing defends is
a footnote, and footnotes rot. CI runs this on every pull request so a change
that quietly degrades retrieval or judgement fails the build instead of being
discovered three weeks later.

    uv run python -m evals.gate --layer retrieval

Comparison is against the previous run of the same layer already recorded in
the history file. Because the history is committed, a pull-request branch
carries main's rows plus its own, so "previous run" is main's baseline without
any extra bookkeeping.
"""

from __future__ import annotations

import argparse
import sys

from evals.schema import (
    AnswerMetrics,
    AuditMetrics,
    Layer,
    MetricsRow,
    RetrievalMetrics,
    load_history,
)

# How far a metric may fall before the build fails. Small drops are noise from
# LLM non-determinism; a 3-point drop is a real regression worth blocking on.
TOLERANCE = 0.03

# Binary floating point cannot represent these decimals exactly, so a drop of
# precisely TOLERANCE lands a hair beyond it: 0.85 - 0.88 == -0.030000000000000027.
# Without this slack a run sitting exactly at the limit fails, which makes the
# gate's verdict depend on representation error rather than on the measurement.
EPSILON = 1e-9


def _is_regression(delta: float) -> bool:
    """A fall of exactly TOLERANCE is allowed; anything beyond it is not."""
    return delta < -TOLERANCE - EPSILON


# Absolute floors. Left at None until the first honest numbers exist — a floor
# invented before any measurement is a number pulled out of the air, and the
# brief is explicit that the metrics must be honest.
FLOORS: dict[str, float | None] = {
    "recall@5": None,
    "f1": None,
}


def _tracked(row: MetricsRow) -> dict[str, float]:
    """Flatten a row into the metric names the gate defends."""
    m = row.metrics
    if isinstance(m, RetrievalMetrics):
        tracked = {f"recall@{k}": v for k, v in m.recall_at_k.items()}
        if m.faithfulness is not None:
            tracked["faithfulness"] = m.faithfulness
        return tracked
    if isinstance(m, AnswerMetrics):
        # Refusal and hallucination are defects, so they are negated: the gate
        # only understands "higher is better", and a rising false-refusal rate
        # must read as a regression rather than as an improvement.
        return {
            "citation_f1": m.citation_f1,
            "citation_precision": m.citation_precision,
            "citation_recall": m.citation_recall,
            "not_refused": 1.0 - m.refusal_rate,
            "not_hallucinated": 1.0 - m.hallucinated_citation_rate,
            "cited": 1.0 - m.uncited_answer_rate,
        }
    if isinstance(m, AuditMetrics):
        return {"precision": m.precision, "recall": m.recall, "f1": m.f1}
    raise TypeError(f"unhandled metrics type: {type(m)!r}")


def check(layer: Layer, history: list[MetricsRow] | None = None) -> tuple[bool, list[str]]:
    """Return (passed, human-readable report lines)."""
    history = load_history() if history is None else history
    for_layer = [r for r in history if r.metrics.layer == layer]
    # A cassette-served row reproduces an old measurement, so gating against it
    # would compare the current build to a recording of itself. `append_row`
    # already refuses to write one; this covers a hand-edited history file.
    rows = [r for r in for_layer if r.context.provenance == "live"]
    lines: list[str] = []

    skipped = len(for_layer) - len(rows)
    if skipped:
        lines.append(f"Ignoring {skipped} non-live (cassette-replayed) `{layer}` row(s).")

    if not rows:
        return True, [*lines, f"No `{layer}` runs recorded yet — nothing to gate."]

    current = _tracked(rows[-1])
    baseline = _tracked(rows[-2]) if len(rows) >= 2 else None

    failures: list[str] = []

    if baseline is None:
        lines.append(f"First `{layer}` run — recording baseline, no comparison possible.")
        for name, value in sorted(current.items()):
            lines.append(f"  {name}: {value:.4f}")
    else:
        lines.append(f"`{layer}` vs previous run (tolerance {TOLERANCE:+.0%}):")
        for name, value in sorted(current.items()):
            prev = baseline.get(name)
            if prev is None:
                lines.append(f"  {name}: {value:.4f} (new)")
                continue
            delta = value - prev
            regressed = _is_regression(delta)
            mark = "REGRESSION" if regressed else "OK"
            lines.append(f"  {name}: {prev:.4f} -> {value:.4f} ({delta:+.4f}) {mark}")
            if regressed:
                failures.append(f"{name} fell {abs(delta):.4f} (tolerance {TOLERANCE})")

    for name, floor in FLOORS.items():
        if floor is not None and name in current and current[name] < floor:
            failures.append(f"{name} = {current[name]:.4f} is below the floor of {floor}")

    if failures:
        lines.append("")
        lines.extend(f"FAILED: {f}" for f in failures)

    return not failures, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=["retrieval", "answers", "audit"], required=True)
    args = parser.parse_args()

    passed, lines = check(args.layer)
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
