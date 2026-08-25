"""Layer 1b: does the Q&A graph answer from the articles it was given?

    uv run python -m evals.qa_eval --sample 10     # cheap first
    uv run python -m evals.qa_eval --record        # full set, appends a row
    uv run python -m evals.qa_eval --cassette replay --cassette-name qa-eval   # free

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
from evals.schema import (
    HISTORY_PATH,
    AnswerMetrics,
    MetricsRow,
    RunContext,
    append_row,
    provenance_of,
)
from graphs.cassette import CassetteMiss, active_cassette
from graphs.llm import Usage
from graphs.qa_flow import QAState, ask
from ingestion.db import connect
from ingestion.embed import Embedder
from ingestion.manifest import load_manifest
from kontrakt_guard.config import Settings, get_settings

# Questions in flight at once. Each opens its own connection, so this is also the
# ceiling on concurrent database sessions.
CONCURRENCY = 4


# If more than this share of questions fail outright, the run is treated as
# broken rather than scored. A metric computed only from the questions that
# happened to succeed carries a survivorship bias invisible in the number.
MAX_FAILURE_RATE = 0.05


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
) -> tuple[Report, list[tuple[str, str]]]:
    """Run every question, surviving individual failures.

    An exception used to abort the whole run and discard every completed
    question — which on a billable eval means throwing away work already paid
    for. The first full-set attempt died on question ~90 when the API credit ran
    out, taking the other 89 results with it and leaving not even a usage total
    to say what had been spent.

    Failures are collected instead, and `main` refuses to score a run where too
    many of them occurred.
    """
    failures: list[tuple[str, str]] = []

    def attempt(question: GoldQuestion) -> Outcome | None:
        try:
            return run_one(question, embedder, settings, usage)
        # Re-raised BEFORE the broad handler, and the order is load-bearing.
        # Swallowed into `failures`, a cassette miss would look like an ordinary
        # per-question error: under the 5% threshold the run would go on to
        # score and publish metrics computed from a half-recorded cassette —
        # numbers partly fabricated by omission. A miss means the tape is
        # incomplete, which is a fact about the whole run, not about one question.
        except CassetteMiss:
            raise
        # Broad on purpose: any failure costs one question, never the run.
        except Exception as exc:
            failures.append((question.id, f"{type(exc).__name__}: {exc}"))
            return None

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(attempt, questions))

    return Report(outcomes=[o for o in results if o is not None]), failures


def build_row(
    metrics: AnswerMetrics, duration: float, settings: Settings, embedder: Embedder, usage: Usage
) -> MetricsRow:
    """One history row. Takes the `Usage` object, not a bare cost.

    Provenance is derived from the same object the cost comes from, so the two
    can never disagree about whether the run actually spent anything.
    """
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
            api_cost_usd=usage.cost_usd,
            duration_s=duration,
            provenance=provenance_of(usage),
        ),
        metrics=metrics,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0, help="Questions to run (0 = all).")
    parser.add_argument("--record", action="store_true", help="Append to metrics/history.jsonl.")
    parser.add_argument(
        "--cassette",
        choices=["off", "record", "replay", "auto"],
        default=None,
        help="Record LLM calls to disk, or replay them for $0.00. See cassettes/README.md.",
    )
    parser.add_argument("--cassette-name", default=None, help="Which cassette directory to use.")
    args = parser.parse_args()

    # Early, before the corpus is touched or a single call is made. Finding out
    # at the end that a two-minute run cannot be recorded is finding out too
    # late — and the cost of the mistake is the operator re-running it live.
    if args.record and args.cassette in ("replay", "auto"):
        print(
            f"--record and --cassette {args.cassette} are contradictory: a run served from a "
            "cassette is not a measurement and cannot be published.",
            file=sys.stderr,
        )
        return 1

    settings = get_settings()
    update: dict[str, str] = {}
    if args.cassette is not None:
        update["cassette_mode"] = args.cassette
    if args.cassette_name is not None:
        update["cassette_name"] = args.cassette_name
    if update:
        settings = settings.model_copy(update=update)

    # Replay never opens a socket, so it needs no key. Every other mode does.
    if settings.cassette_mode != "replay" and not settings.anthropic_api_key.get_secret_value():
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
    report, failures = score(selected, embedder, settings, usage)
    duration = time.monotonic() - started

    if failures:
        print(f"{len(failures)} of {len(selected)} questions failed:", file=sys.stderr)
        for question_id, error in failures[:5]:
            print(f"  {question_id}: {error}", file=sys.stderr)
        # Always report the spend, even on a broken run. The run that provoked
        # this had already spent real money and reported nothing at all.
        print(f"\nspent before failing: {usage.summary()}", file=sys.stderr)

        if len(failures) / len(selected) > MAX_FAILURE_RATE:
            print(
                f"\nrefusing to score: more than {MAX_FAILURE_RATE:.0%} of questions failed, "
                "so any metric would be computed from the survivors only.",
                file=sys.stderr,
            )
            return 1

    if not report.outcomes:
        print("no question completed", file=sys.stderr)
        return 1

    metrics = report.metrics()

    print(f"Layer 1b — answers, {metrics.n_questions} gold questions\n")
    print(f"  citation precision   {metrics.citation_precision:>7.1%}")
    print(f"  citation recall      {metrics.citation_recall:>7.1%}")
    print(f"  citation F1          {metrics.citation_f1:>7.3f}")
    print(f"\n  false refusals       {metrics.refusal_rate:>7.1%}")
    print(f"  hallucinated cites   {metrics.hallucinated_citation_rate:>7.1%}")
    print(f"  uncited answers      {metrics.uncited_answer_rate:>7.1%}")
    print(f"\n{usage.summary()}   duration: {duration:.1f}s")

    cassette = active_cassette()
    if cassette is not None:
        print(cassette.summary())

    if metrics.false_refusals:
        print(f"\nrefused ({len(metrics.false_refusals)}): {', '.join(metrics.false_refusals)}")

    if args.record:
        if args.sample:
            # A number from a subset is a different number. Recording it beside
            # full-set rows would let the gate compare them as though they were
            # the same measurement.
            print("\nrefusing to record a sampled run; drop --sample", file=sys.stderr)
            return 1
        append_row(build_row(metrics, duration, settings, embedder, usage))
        print(f"\nrecorded to {HISTORY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
