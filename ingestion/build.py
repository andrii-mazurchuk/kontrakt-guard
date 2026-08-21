"""Run the ingestion pipeline.

    uv run python -m ingestion.build            # parse, validate, chunk — no DB, no model
    uv run python -m ingestion.build --load     # also embed and write to Postgres

The dry run is deliberately free and offline: it exercises every parsing decision
and reports the chunk shape without downloading a 2 GB model or requiring a
database, which makes it usable as a fast check after any parser change.

The corpus must already be pinned; run `python -m ingestion.fetch` first if not.
"""

from __future__ import annotations

import argparse
import sys

from ingestion.chunk import TOKEN_BUDGET, chunk_articles, whitespace_counter
from ingestion.db import apply_schema, connect, polish_config_available
from ingestion.embed import Embedder
from ingestion.load import corpus_stats, load_chunks
from ingestion.manifest import load_manifest
from ingestion.parse import parse_articles
from ingestion.pdf_text import extract_pages
from ingestion.validate import ParseValidationError, validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load",
        action="store_true",
        help="Embed the chunks and upsert them into Postgres.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    report: list[str] = [f"corpus manifest digest: {manifest.digest()}", ""]

    # The real tokenizer only when it will be used; otherwise a whitespace count,
    # which is close enough to report chunk shape and costs nothing.
    embedder = Embedder() if args.load else None
    count = embedder.count_tokens if embedder else whitespace_counter

    all_chunks = []
    for act in manifest.acts:
        if not act.path.exists():
            print(
                f"{act.slug}: {act.path} is missing. Run: uv run python -m ingestion.fetch",
                file=sys.stderr,
            )
            return 1

        pages = extract_pages(act.path)
        articles = parse_articles(pages, act=act.slug)

        try:
            report.extend(validate(articles, act=act.slug))
        except ParseValidationError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        chunks = chunk_articles(articles, count=count)
        all_chunks.extend(chunks)

        split = sum(1 for c in chunks if c.part_index > 0)
        report.append(f"  pages:         {len(pages)}")
        report.append(f"  chunks:        {len(chunks)} ({split} from split articles)")
        report.append(f"  token budget:  {TOKEN_BUDGET}")
        report.append(f"  source:        {act.eli} variant {act.variant}")
        report.append("")

    if not args.load:
        report.append("dry run — pass --load to embed and write to Postgres")
        print("\n".join(report))
        return 0

    assert embedder is not None
    with connect() as conn:
        if not polish_config_available(conn):
            print(
                "The 'polish' text-search configuration is missing. This database was not "
                "built from docker/db/Dockerfile. Run: docker compose up -d --build --wait db",
                file=sys.stderr,
            )
            return 1

        apply_schema(conn)
        report.extend(load_chunks(conn, all_chunks, embedder))
        report.append(f"  in database: {corpus_stats(conn)}")

    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
