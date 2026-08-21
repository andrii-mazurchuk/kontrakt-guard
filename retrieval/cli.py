"""Ask the corpus a question from the command line.

    uv run python -m retrieval.cli "czy 6-miesięczny okres próbny jest legalny?"
    uv run python -m retrieval.cli --leg lexical "wynagrodzenie za nadgodziny"

Exists to make retrieval quality inspectable by eye before any metric is
computed. A recall@k figure tells you how often the right article appears; this
shows *which* articles come back and *which leg* found them — the difference
between a number and an explanation.
"""

from __future__ import annotations

import argparse
import sys

from ingestion.chunk import query_input
from ingestion.db import connect
from ingestion.embed import Embedder
from kontrakt_guard.config import get_settings
from retrieval.search import Hit, dense_search, lexical_search, merge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question in Polish.")
    parser.add_argument("--k", type=int, default=5, help="Results to show.")
    parser.add_argument(
        "--leg",
        choices=["hybrid", "lexical", "dense"],
        default="hybrid",
        help="Run one leg alone to see what it contributes.",
    )
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--include-repealed", action="store_true")
    args = parser.parse_args()

    settings = get_settings()

    with connect(settings) as conn:
        lexical: list[Hit] = []
        dense: list[Hit] = []

        if args.leg in ("hybrid", "lexical"):
            lexical = lexical_search(
                conn, args.question, settings.bm25_candidates, args.include_repealed
            )
        if args.leg in ("hybrid", "dense"):
            vector = Embedder(settings).encode_query(query_input(args.question))
            dense = dense_search(conn, vector, settings.vector_candidates, args.include_repealed)

        if args.leg == "lexical":
            hits = lexical[: args.k]
        elif args.leg == "dense":
            hits = dense[: args.k]
        else:
            hits = merge(lexical, dense, args.k, args.fusion, settings.hybrid_alpha)

    if not hits:
        print("no results — the corpus could not ground this question", file=sys.stderr)
        return 1

    print(f'"{args.question}"  [{args.leg}, k={args.k}]\n')
    for position, hit in enumerate(hits, start=1):
        legs = []
        if hit.lexical_rank:
            legs.append(f"lex#{hit.lexical_rank}")
        if hit.dense_rank:
            legs.append(f"dense#{hit.dense_rank}")
        found_by = f"  ({', '.join(legs)})" if legs else ""

        print(f"{position}. {hit.citation}   score={hit.score:.4f}{found_by}")
        if hit.title_path:
            print(f"   {' > '.join(hit.title_path)}")
        body = hit.content.split(". ", 1)[-1]
        print(f"   {body[:200]}{'…' if len(body) > 200 else ''}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
