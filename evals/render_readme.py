"""Regenerate the README metric tables from ``metrics/history.jsonl``.

Run by CI after every eval run. Hand-typed metrics go stale and quietly become
lies; generating them means the README cannot drift from the recorded evidence.

    uv run python -m evals.render_readme [--check]

``--check`` renders without writing and exits non-zero if the README is out of
date, which is how CI catches a stale table.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from evals.schema import (
    AuditMetrics,
    MetricsRow,
    RetrievalMetrics,
    latest_for_layer,
    load_history,
)

README_PATH = Path("README.md")
REPORTED_K = (3, 5, 10)

EMPTY = "_No eval runs recorded yet._"


def _provenance(row: MetricsRow) -> str:
    ctx = row.context
    return (
        f"\n<sub>Run `{ctx.commit[:8]}` · {ctx.timestamp} · "
        f"embeddings `{ctx.embedding_model}@{ctx.embedding_revision or 'unpinned'}` · "
        f"config `{ctx.retrieval_config_hash}` · ${ctx.api_cost_usd:.2f} · "
        f"{ctx.duration_s:.0f}s</sub>\n"
    )


def render_layer1(row: MetricsRow | None) -> str:
    if row is None or not isinstance(row.metrics, RetrievalMetrics):
        return EMPTY
    m = row.metrics

    header = "| Metric | " + " | ".join(f"k={k}" for k in REPORTED_K) + " |"
    sep = "|---|" + "---|" * len(REPORTED_K)
    cells = []
    for k in REPORTED_K:
        value = m.recall_at_k.get(k)
        cells.append("—" if value is None else f"{value:.1%}")
    body = f"| Recall@k on article IDs | {' | '.join(cells)} |"

    extra = [f"\nGold set: **{m.n_questions}** PIP-derived questions."]
    if m.mrr is not None:
        extra.append(f" MRR **{m.mrr:.3f}**.")
    if m.faithfulness is not None:
        extra.append(f" Answer faithfulness (LLM-as-judge) **{m.faithfulness:.1%}**.")

    return "\n".join([header, sep, body]) + "".join(extra) + "\n" + _provenance(row)


def render_layer2(row: MetricsRow | None) -> str:
    if row is None or not isinstance(row.metrics, AuditMetrics):
        return EMPTY
    m = row.metrics

    lines = [
        "| Violation type | F1 |",
        "|---|---|",
        f"| **Overall** | **{m.f1:.3f}** |",
    ]
    for name, f1 in sorted(m.per_violation_type.items()):
        lines.append(f"| {name} | {f1:.3f} |")

    summary = (
        f"\nPrecision **{m.precision:.1%}** · Recall **{m.recall:.1%}** · F1 **{m.f1:.3f}** "
        f"over **{m.n_contracts}** synthetic contracts carrying "
        f"**{m.n_planted_violations}** planted violations.\n"
    )
    return "\n".join(lines) + "\n" + summary + _provenance(row)


def _splice(text: str, marker: str, body: str) -> str:
    """Replace the content between a START/END marker pair."""
    pattern = re.compile(
        rf"(<!-- METRICS:{marker}:START -->\n).*?(\n<!-- METRICS:{marker}:END -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"README is missing the METRICS:{marker} marker pair")
    return pattern.sub(lambda m: m.group(1) + body.strip("\n") + m.group(2), text)


def render(readme: str, rows: list[MetricsRow] | None = None) -> str:
    rows = load_history() if rows is None else rows
    readme = _splice(readme, "LAYER1", render_layer1(latest_for_layer(rows, "retrieval")))
    return _splice(readme, "LAYER2", render_layer2(latest_for_layer(rows, "audit")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the README tables are stale, without writing.",
    )
    args = parser.parse_args()

    current = README_PATH.read_text(encoding="utf-8")
    updated = render(current)

    if args.check:
        if current != updated:
            print("README metric tables are stale. Run: uv run python -m evals.render_readme")
            return 1
        print("README metric tables are up to date.")
        return 0

    if current != updated:
        README_PATH.write_text(updated, encoding="utf-8", newline="\n")
        print("README metric tables updated.")
    else:
        print("README metric tables already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
