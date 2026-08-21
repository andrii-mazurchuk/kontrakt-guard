"""Fetch an article by its citation.

    uv run python -m retrieval.lookup 25
    uv run python -m retrieval.lookup 151^1 --act kp

"What does Art. 25 say" is a lookup, not a similarity search. Answering it with
the vector index would be both slower and capable of returning the wrong article,
which for a citation is not a ranking error but a factual one.

This is also the tool for verifying gold-set ground truth: it shows the statute
text for a cited article without going anywhere near retrieval, so the check
stays independent of the system being measured.
"""

from __future__ import annotations

import argparse
import sys

import psycopg
from psycopg.rows import DictRow

from evals.gold import normalise_article
from ingestion.db import connect


def by_citation(conn: psycopg.Connection[DictRow], article: str, act: str = "kp") -> list[DictRow]:
    """Every chunk of one article, in reading order."""
    return conn.execute(
        """
        SELECT article, article_display, paragraph, part_index,
               title_path, content, repealed, repeal_kind, page_start
        FROM chunks
        WHERE act = %s AND article = %s
        ORDER BY part_index
        """,
        (act, normalise_article(article)),
    ).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("article", help="Article id: 25, 151^1, 151-1 or Art. 151(1).")
    parser.add_argument("--act", default="kp")
    args = parser.parse_args()

    with connect() as conn:
        rows = by_citation(conn, args.article, args.act)

    if not rows:
        canonical = normalise_article(args.article)
        print(
            f"no article {canonical!r} in act {args.act!r}. "
            "Note that Art. 151-1 and Art. 1511 are different articles.",
            file=sys.stderr,
        )
        return 1

    head = rows[0]
    print(f"{head['article_display']}  (act={args.act}, page {head['page_start']})")
    if head["title_path"]:
        print(" > ".join(head["title_path"]))
    if head["repealed"]:
        print(f"[NOT IN FORCE: {head['repeal_kind']}]")
    print()
    for row in rows:
        print(row["content"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
