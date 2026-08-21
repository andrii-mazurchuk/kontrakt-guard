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
import sys
import time
from collections.abc import Sequence
from typing import Literal

import psycopg
from psycopg.rows import DictRow

from evals.gold import GoldQuestion, load_gold
from ingestion.chunk import query_input
from ingestion.db import connect
from ingestion.embed import Embedder
from kontrakt_guard.config import Settings, get_settings
from retrieval.search import dense_search, lexical_search, merge

REPORTED_K = (3, 5, 10)
Leg = Literal["hybrid", "lexical", "dense"]


def retrieved_articles(
    conn: psycopg.Connection[DictRow],
    question: str,
    embedder: Embedder,
    settings: Settings,
    leg: Leg,
    limit: int,
) -> list[str]:
    """Ranked, de-duplicated article ids for one question.

    Chunks collapse to articles keeping the best rank of each, so `limit` means
    distinct articles.
    """
    lexical = (
        lexical_search(conn, question, settings.bm25_candidates)
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
        hits = merge(lexical, dense, k=len(lexical) + len(dense), fusion="rrf")

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
) -> tuple[dict[int, float], dict[int, float], dict[int, float], list[str]]:
    """Return (recall@k, hit_rate@k, mrr, ids missed at the largest reported k)."""
    top = max(ks)
    recall_totals = dict.fromkeys(ks, 0.0)
    hit_totals = dict.fromkeys(ks, 0.0)
    reciprocal = 0.0
    missed: list[str] = []

    for question in questions:
        ranked = retrieved_articles(conn, question.question, embedder, settings, leg, top)
        truth = set(question.ground_truth_articles)

        for k in ks:
            found = truth & set(ranked[:k])
            recall_totals[k] += len(found) / len(truth)
            hit_totals[k] += 1.0 if found else 0.0

        rank = next((i for i, a in enumerate(ranked, start=1) if a in truth), None)
        reciprocal += 1.0 / rank if rank else 0.0

        if not truth & set(ranked[:5]):
            missed.append(question.id)

    n = len(questions)
    return (
        {k: recall_totals[k] / n for k in ks},
        {k: hit_totals[k] / n for k in ks},
        {0: reciprocal / n},
        missed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", choices=["hybrid", "lexical", "dense"], default="hybrid")
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--record", action="store_true", help="Append to metrics/history.jsonl.")
    args = parser.parse_args()

    questions = load_gold()
    if not questions:
        print("gold set is empty; nothing to evaluate", file=sys.stderr)
        return 1

    settings = get_settings()
    embedder = Embedder(settings)

    started = time.monotonic()
    with connect(settings) as conn:
        recall, hit_rate, mrr, missed = score(questions, conn, embedder, settings, args.leg)
    duration = time.monotonic() - started

    print(f"Layer 1 — retrieval [{args.leg}], {len(questions)} gold questions\n")
    print(f"{'k':>4}  {'recall@k':>10}  {'hit_rate@k':>11}")
    for k in REPORTED_K:
        print(f"{k:>4}  {recall[k]:>9.1%}  {hit_rate[k]:>10.1%}")
    print(f"\nMRR: {mrr[0]:.3f}   duration: {duration:.1f}s")
    if missed:
        print(f"\nmissed at k=5 ({len(missed)}): {', '.join(missed)}")

    if args.record:
        print("\n--record is not wired up until the gold set is reviewed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
