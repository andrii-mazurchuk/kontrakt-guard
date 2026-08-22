"""Layer 1b: does the Q&A graph answer from the articles it was given?

    uv run python -m evals.qa_eval --sample 10     # cheap first
    uv run python -m evals.qa_eval --record        # full set, appends a row

BILLABLE. Runs the whole graph — rewrite, retrieve, grade, answer — over the
gold set, which is roughly twelve API calls per question. Check the cost first:

    uv run python -m evals.cost_estimate

Every figure is computed by comparing **article ids**, never by asking a model
whether an answer looks good. That keeps the layer deterministic apart from the
generation itself, and it means the numbers are attributable: Layer 1 says
whether the right article was retrieved, this says whether the answer was built
out of it.

The four things measured, and why each is a defect rather than a statistic:

- **false refusals** — every gold question is answerable from the corpus by
  construction, so any refusal here is wrong. Without this the refusal guardrail
  could be made perfect by refusing everything.
- **citation precision / recall** — against the same ground-truth articles
  Layer 1 scores. Recall bounded above by retrieval; precision is generation's
  own.
- **hallucinated citations** — the rate at which generation reached for an
  article that was never retrieved. The verifier strips these before anyone sees
  them, so this is how often the guardrail *fires*, not how often it fails.
- **uncited answers** — an answer with no citation is ungrounded even when it is
  correct, and the brief makes citations mandatory.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime

from evals.gold import GoldQuestion, load_gold
from evals.retrieval_eval import _git
from evals.schema import HISTORY_PATH, AnswerMetrics, MetricsRow, RunContext, append_row
from graphs.llm import Usage
from graphs.qa_flow import QAState, ask
from ingestion.db import connect
from ingestion.embed import Embedder
from ingestion.manifest import load_manifest
from kontrakt_guard.config import Settings, get_settings

# Questions in flight at once. Each opens its own connection, so this is also the
# ceiling on concurrent database sessions.
CONCURRENCY = 4


@dataclass
class Outcome:
    """One question's result, reduced to what the metrics need."""

    question_id: str
    refused: bool
    cited: set[str]
    truth: set[str]
    hallucinated: bool

    @property
    def uncited(self) -> bool:
        return not self.refused and not self.cited


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)

    def metrics(self) -> AnswerMetrics:
        n = len(self.outcomes)
        answered = [o for o in self.outcomes if not o.refused]

        # Micro-averaged: pooled over all citations rather than averaged per
        # question, so a question requiring three articles counts for three times
        # as much as one requiring one. Macro-averaging would let a long tail of
        # single-article questions hide failures on the multi-article ones.
        true_positive = sum(len(o.cited & o.truth) for o in answered)
        cited_total = sum(len(o.cited) for o in answered)
        truth_total = sum(len(o.truth) for o in self.outcomes)

        precision = true_positive / cited_total if cited_total else 0.0
        recall = true_positive / truth_total if truth_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return AnswerMetrics(
            n_questions=n,
            refusal_rate=sum(o.refused for o in self.outcomes) / n,
            citation_precision=precision,
            citation_recall=recall,
            citation_f1=f1,
            hallucinated_citation_rate=sum(o.hallucinated for o in self.outcomes) / n,
            uncited_answer_rate=sum(o.uncited for o in self.outcomes) / n,
            false_refusals=[o.question_id for o in self.outcomes if o.refused],
        )


def run_one(
    question: GoldQuestion, embedder: Embedder, settings: Settings, usage: Usage
) -> Outcome:
    """One question through the full graph, on its own connection.

    A connection per question rather than one shared: psycopg connections are not
    safe to use concurrently, and the questions are run in parallel because a
    serial run over the gold set is twelve sequential API calls deep, ninety-seven
    times over.
    """
    with connect(settings) as conn:
        state: QAState = ask(question.question, conn, embedder, settings, usage)

    return Outcome(
        question_id=question.id,
        refused=bool(state.get("refused", False)),
        cited={c.article for c in state.get("citations") or []},
        truth=set(question.ground_truth_articles),
        hallucinated=bool(state.get("unsupported_citations")),
    )


def score(
    questions: Sequence[GoldQuestion], embedder: Embedder, settings: Settings, usage: Usage
) -> Report:
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        outcomes = list(pool.map(lambda q: run_one(q, embedder, settings, usage), questions))
    return Report(outcomes=outcomes)


def build_row(
    metrics: AnswerMetrics, duration: float, settings: Settings, embedder: Embedder, cost: float
) -> MetricsRow:
    revision = embedder.resolved_revision()
    pinned = settings.model_copy(update={"embedding_revision": revision})
    return MetricsRow(
        context=RunContext(
            commit=_git("rev-parse", "HEAD"),
            branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            embedding_model=settings.embedding_model,
            embedding_revision=revision,
            model_cheap=settings.model_cheap,
            model_strong=settings.model_strong,
            retrieval_config_hash=pinned.retrieval_config_hash(),
            corpus_manifest_sha=load_manifest().digest(),
            api_cost_usd=cost,
            duration_s=duration,
        ),
        metrics=metrics,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0, help="Questions to run (0 = all).")
    parser.add_argument("--record", action="store_true", help="Append to metrics/history.jsonl.")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.anthropic_api_key.get_secret_value():
        print("this eval is billable and ANTHROPIC_API_KEY is unset", file=sys.stderr)
        return 1

    questions = load_gold()
    selected = questions[: args.sample] if args.sample else questions
    if not selected:
        print("gold set is empty", file=sys.stderr)
        return 1

    usage = Usage()
    embedder = Embedder(settings)

    started = time.monotonic()
    report = score(selected, embedder, settings, usage)
    duration = time.monotonic() - started
    metrics = report.metrics()

    print(f"Layer 1b — answers, {metrics.n_questions} gold questions\n")
    print(f"  citation precision   {metrics.citation_precision:>7.1%}")
    print(f"  citation recall      {metrics.citation_recall:>7.1%}")
    print(f"  citation F1          {metrics.citation_f1:>7.3f}")
    print(f"\n  false refusals       {metrics.refusal_rate:>7.1%}")
    print(f"  hallucinated cites   {metrics.hallucinated_citation_rate:>7.1%}")
    print(f"  uncited answers      {metrics.uncited_answer_rate:>7.1%}")
    print(f"\n{usage.summary()}   duration: {duration:.1f}s")

    if metrics.false_refusals:
        print(f"\nrefused ({len(metrics.false_refusals)}): {', '.join(metrics.false_refusals)}")

    if args.record:
        if args.sample:
            # A number from a subset is a different number. Recording it beside
            # full-set rows would let the gate compare them as though they were
            # the same measurement.
            print("\nrefusing to record a sampled run; drop --sample", file=sys.stderr)
            return 1
        append_row(build_row(metrics, duration, settings, embedder, usage.cost_usd))
        print(f"\nrecorded to {HISTORY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
