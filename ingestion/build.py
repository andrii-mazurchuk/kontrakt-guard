"""Run the ingestion pipeline as far as it currently goes.

    uv run python -m ingestion.build

Today: fetch (verifying checksums) -> extract -> parse -> validate -> report.
The embedding and pgvector load steps attach here as they land.

The corpus must already be pinned; run `python -m ingestion.fetch` first if not.
"""

from __future__ import annotations

import sys

from ingestion.manifest import load_manifest
from ingestion.parse import parse_articles
from ingestion.pdf_text import extract_pages
from ingestion.validate import ParseValidationError, validate


def main() -> int:
    manifest = load_manifest()
    report: list[str] = [f"corpus manifest digest: {manifest.digest()}", ""]

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

        report.append(f"  pages:         {len(pages)}")
        report.append(f"  source:        {act.eli} variant {act.variant}")
        report.append(f"  pdf created:   {act.pdf_creation_date}")
        report.append("")

    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
