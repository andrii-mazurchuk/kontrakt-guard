"""Layer 1: does retrieval surface the articles that answer the question?

    uv run python -m evals.retrieval_eval
    uv run python -m evals.retrieval_eval --leg lexical   # what one leg contributes
    uv run python -m evals.retrieval_eval --record        # append to metrics/history.jsonl

Deterministic and free: embeddings and SQL, no LLM call. That is what lets this
run as a per-pull-request gate rather than a nightly job.

Two definitions, reported separately because they answer different questions and
are routinely conflated:

- **recall@k** — of the articles a question requires, what fraction appeared in
  the top k? Macro-averaged over questions. This is the headline; it is the
  stricter figure and the one that degrades when a multi-article answer is only
  half found.
- **hit_rate@k** — in what fraction of questions did *at least one* required
  article appear? Always the higher number, and the one usually quoted without
  saying so.

A note on k: it counts **articles, not chunks**. An article split into three
chunks would otherwise consume three of the five slots at k=5 and flatter the
score.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import psycopg
from psycopg.rows import DictRow

from evals.gold import GoldQuestion, load_gold
from evals.schema import HISTORY_PATH, MetricsRow, RetrievalMetrics, RunContext, append_row
from ingestion.chunk import query_input
from ingestion.db import connect, lexical_index_is_stale
from ingestion.embed import Embedder
from ingestion.manifest import load_manifest
from kontrakt_guard.config import Settings, get_settings
from retrieval.search import Ranking, dense_search, lexical_search, merge

REPORTED_K = (3, 5, 10)
Leg = Literal["hybrid", "lexical", "dense"]

# Effectively unbounded: the merged pool is at most bm25_candidates +
# vector_candidates chunks, which collapse to fewer articles still.
POOL_LIMIT = 10_000


@dataclass(frozen=True)
class Scores:
    """One evaluation of the gold set.

    `pool_recall` is the fraction of required articles that reached the candidate
    pool at any rank. Nothing downstream — fusion weight, k, a reranker, the LLM
    grading node — can recover an article that neither leg proposed, so this is
    the ceiling on the whole pipeline, and the gap between it and recall@5 is
    what better ranking is worth.
    """

    recall: dict[int, float]
    hit_rate: dict[int, float]
    mrr: float
    pool_recall: float
    missed: list[str]


def retrieved_articles(
    conn: psycopg.Connection[DictRow],
    question: str,
    embedder: Embedder,
    settings: Settings,
    leg: Leg,
    limit: int,
    fusion: Literal["rrf", "weighted"] = "rrf",
    alpha: float | None = None,
    ranking: Ranking | None = None,
) -> list[str]:
    """Ranked, de-duplicated article ids for one question.

    Chunks collapse to articles keeping the best rank of each, so `limit` means
    distinct articles.
    """
    lexical = (
        lexical_search(
            conn,
            question,
            settings.bm25_candidates,
            ranking=ranking or settings.lexical_ranking,
            k1=settings.bm25_k1,
            b=settings.bm25_b,
        )
        if leg in ("hybrid", "lexical")
        else []
    )
    dense = (
        dense_search(conn, embedder.encode_query(query_input(question)), settings.vector_candidates)
        if leg in ("hybrid", "dense")
        else []
    )

    if leg == "lexical":
        hits = lexical
    elif leg == "dense":
        hits = dense
    else:
        # Merge over the full candidate pool, then collapse — collapsing first
        # would discard the second leg's evidence for an article.
        hits = merge(
            lexical,
            dense,
            k=len(lexical) + len(dense),
            fusion=fusion,
            alpha=settings.hybrid_alpha if alpha is None else alpha,
        )

    ordered: list[str] = []
    for hit in hits:
        if hit.article not in ordered:
            ordered.append(hit.article)
        if len(ordered) >= limit:
            break
    return ordered


def score(
    questions: Sequence[GoldQuestion],
    conn: psycopg.Connection[DictRow],
    embedder: Embedder,
    settings: Settings,
    leg: Leg = "hybrid",
    ks: Sequence[int] = REPORTED_K,
    fusion: Literal["rrf", "weighted"] = "rrf",
    alpha: float | None = None,
    ranking: Ranking | None = None,
) -> Scores:
    recall_totals = dict.fromkeys(ks, 0.0)
    hit_totals = dict.fromkeys(ks, 0.0)
    reciprocal = 0.0
    pool_total = 0.0
    missed: list[str] = []

    for question in questions:
        # The whole candidate pool, unbounded, alongside the top-k ordering. No
        # fusion or reranker can retrieve an article that neither leg proposed,
        # so pool recall is the ceiling on everything downstream — and the number
        # that says whether to invest in the legs or in the ranking between them.
        ranked = retrieved_articles(
            conn, question.question, embedder, settings, leg, POOL_LIMIT, fusion, alpha, ranking
        )
        truth = set(question.ground_truth_articles)
        pool_total += len(truth & set(ranked)) / len(truth)

        for k in ks:
            found = truth & set(ranked[:k])
            recall_totals[k] += len(found) / len(truth)
            hit_totals[k] += 1.0 if found else 0.0

        rank = next((i for i, a in enumerate(ranked, start=1) if a in truth), None)
        reciprocal += 1.0 / rank if rank else 0.0

        if not truth & set(ranked[:5]):
            missed.append(question.id)

    n = len(questions)
    return Scores(
        recall={k: recall_totals[k] / n for k in ks},
        hit_rate={k: hit_totals[k] / n for k in ks},
        mrr=reciprocal / n,
        pool_recall=pool_total / n,
        missed=missed,
    )


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def build_row(
    questions: Sequence[GoldQuestion],
    recall: dict[int, float],
    mrr: float,
    missed: list[str],
    duration: float,
    settings: Settings,
    embedder: Embedder,
) -> MetricsRow:
    """Assemble one history row, with enough provenance to reproduce the number."""
    # Resolve the embedding revision rather than recording the empty default.
    # retrieval_config_hash includes it, so an unresolved revision produces a hash
    # that silently fails to distinguish two different versions of the same model.
    revision = embedder.resolved_revision()
    pinned = settings.model_copy(update={"embedding_revision": revision})

    context = RunContext(
        commit=_git("rev-parse", "HEAD"),
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        embedding_model=settings.embedding_model,
        embedding_revision=revision,
        model_cheap=settings.model_cheap,
        model_strong=settings.model_strong,
        retrieval_config_hash=pinned.retrieval_config_hash(),
        corpus_manifest_sha=load_manifest().digest(),
        # Layer 1 makes no LLM calls; that is what lets it gate every pull request.
        api_cost_usd=0.0,
        duration_s=duration,
    )
    return MetricsRow(
        context=context,
        metrics=RetrievalMetrics(
            n_questions=len(questions),
            recall_at_k=dict(recall),
            mrr=mrr,
            faithfulness=None,
            misses_at_5=missed,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", choices=["hybrid", "lexical", "dense"], default="hybrid")
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default=None)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dense weight for weighted fusion. Defaults to Settings.hybrid_alpha.",
    )
    parser.add_argument(
        "--ranking",
        choices=["bm25", "ts_rank_cd"],
        default=None,
        help="Lexical scoring function. Defaults to Settings.lexical_ranking.",
    )
    parser.add_argument("--record", action="store_true", help="Append to metrics/history.jsonl.")
    args = parser.parse_args()

    questions = load_gold()
    if not questions:
        print("gold set is empty; nothing to evaluate", file=sys.stderr)
        return 1

    settings = get_settings()
    embedder = Embedder(settings)
    fusion = args.fusion or settings.fusion

    ranking = args.ranking or settings.lexical_ranking
    uses_lexical = args.leg in ("hybrid", "lexical")

    started = time.monotonic()
    with connect(settings) as conn:
        # A materialised view is a snapshot. Scoring against statistics left over
        # from a previous corpus would move the number with nothing to indicate
        # why, so refuse rather than report.
        if uses_lexical and ranking == "bm25" and lexical_index_is_stale(conn):
            print(
                "BM25 statistics are stale — chunk_terms/corpus_stats do not match "
                "the corpus. Run: uv run python -m ingestion.build",
                file=sys.stderr,
            )
            return 1

        scores = score(
            questions,
            conn,
            embedder,
            settings,
            args.leg,
            fusion=fusion,
            alpha=args.alpha,
            ranking=ranking,
        )
    duration = time.monotonic() - started

    label = args.leg if args.leg != "hybrid" else f"{args.leg}/{fusion}"
    if uses_lexical:
        label += f", {ranking}"
    print(f"Layer 1 — retrieval [{label}], {len(questions)} gold questions\n")
    print(f"{'k':>4}  {'recall@k':>10}  {'hit_rate@k':>11}")
    for k in REPORTED_K:
        print(f"{k:>4}  {scores.recall[k]:>9.1%}  {scores.hit_rate[k]:>10.1%}")
    print(f"\nMRR: {scores.mrr:.3f}   duration: {duration:.1f}s")

    # The ceiling and the distance to it. A large gap says the ranking is losing
    # articles the legs already found, which is a reranker's job; a small one says
    # the legs themselves are the limit and no amount of reordering will help.
    headroom = scores.pool_recall - scores.recall[5]
    print(
        f"candidate-pool recall: {scores.pool_recall:.1%} ({headroom:+.1%} headroom above recall@5)"
    )
    if scores.missed:
        print(f"\nmissed at k=5 ({len(scores.missed)}): {', '.join(scores.missed)}")

    if args.record:
        row = build_row(
            questions=questions,
            recall=scores.recall,
            mrr=scores.mrr,
            missed=scores.missed,
            duration=duration,
            settings=settings,
            embedder=embedder,
        )
        append_row(row)
        print(f"\nrecorded to {HISTORY_PATH} (corpus {row.context.corpus_manifest_sha[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
