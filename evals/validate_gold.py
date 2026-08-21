"""Check the gold set against the corpus before anything is measured against it.

    uv run python -m evals.validate_gold

This is the only defence against a silently broken metric. A gold question whose
ground-truth article does not exist in the corpus is unachievable by
construction: retrieval can never surface it, so recall@k is permanently capped
and the shortfall looks like a retrieval weakness rather than a data error.

Deliberately *not* checked here: whether the article is the legally correct
answer. No script can establish that. It is what human review of the gold set is
for, and why the set is reviewed before it is trusted.
"""

from __future__ import annotations

import sys
from collections import Counter

import psycopg
from psycopg.rows import DictRow

from evals.gold import GoldQuestion, load_gold
from ingestion.db import connect


class GoldValidationError(RuntimeError):
    """The gold set is not usable as ground truth."""


def _corpus_articles(conn: psycopg.Connection[DictRow], act: str) -> dict[str, bool]:
    """Map article id -> repealed, for one act."""
    rows = conn.execute(
        "SELECT DISTINCT article, repealed FROM chunks WHERE act = %s", (act,)
    ).fetchall()
    return {row["article"]: row["repealed"] for row in rows}


def validate(
    questions: list[GoldQuestion], conn: psycopg.Connection[DictRow]
) -> tuple[list[str], list[str]]:
    """Return (report lines, problems)."""
    report: list[str] = []
    problems: list[str] = []

    if not questions:
        raise GoldValidationError("gold set is empty")

    duplicates = sorted(k for k, n in Counter(q.id for q in questions).items() if n > 1)
    if duplicates:
        problems.append(f"duplicate question ids: {duplicates}")

    acts = {q.act for q in questions}
    corpus = {act: _corpus_articles(conn, act) for act in acts}

    for act, articles in corpus.items():
        if not articles:
            problems.append(f"act {act!r} has no chunks in the corpus; has it been loaded?")

    missing: list[str] = []
    repealed: list[str] = []
    for question in questions:
        known = corpus.get(question.act, {})
        for article in question.ground_truth_articles:
            if article not in known:
                missing.append(f"{question.id}: Art. {article} ({question.act}) not in corpus")
            # A repealed article is not automatically wrong — a question may
            # genuinely be about one — but it is nearly always an error, so it has
            # to be acknowledged in `notes` rather than passed over in silence.
            elif known[article] and "repealed" not in question.notes.lower():
                repealed.append(f"{question.id}: Art. {article} is repealed")

    problems.extend(missing)
    problems.extend(repealed)

    topics = Counter(q.topic for q in questions)
    single = sum(1 for q in questions if len(q.ground_truth_articles) == 1)
    unsourced = [q.id for q in questions if not q.source_url]

    report.append(f"gold set: {len(questions)} questions across {len(topics)} topics")
    report.append(f"  single-article answers: {single}")
    report.append(f"  multi-article answers:  {len(questions) - single}")
    report.append(f"  without a source url:   {len(unsourced)}")
    report.append("  topics: " + ", ".join(f"{t}={n}" for t, n in topics.most_common()))

    # A gold set clustered on two topics measures two topics, however large it is.
    thin = [t for t, n in topics.items() if n < 2]
    if thin:
        report.append(f"  thin topics (n=1): {', '.join(thin)}")

    return report, problems


def main() -> int:
    questions = load_gold()
    with connect() as conn:
        try:
            report, problems = validate(questions, conn)
        except GoldValidationError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    print("\n".join(report))
    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("\nevery ground-truth article exists in the corpus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
