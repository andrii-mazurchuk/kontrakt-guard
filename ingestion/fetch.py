"""Download the corpus and pin it by checksum.

    uv run python -m ingestion.fetch            # fetch and verify against the manifest
    uv run python -m ingestion.fetch --repin    # accept new bytes and rewrite the pins

A checksum mismatch here means **the law changed**, not that a file is corrupt:
ISAP regenerates the consolidated PDF whenever an amendment lands. That is why
re-pinning is an explicit flag rather than automatic — the new corpus is not
comparable to metrics recorded under the old one.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ingestion.manifest import (
    CORPUS_DIR,
    ActEntry,
    Manifest,
    load_manifest,
    save_manifest,
    sha256_file,
)

TIMEOUT = httpx.Timeout(60.0, connect=15.0)


class ChecksumMismatchError(RuntimeError):
    """Fetched bytes do not match the pinned checksum."""


def _pdf_creation_date(path: Path) -> str | None:
    """Read the PDF's own CreationDate, which dates the consolidation itself.

    The checksum says the bytes changed; this says how current the law in them is.
    """
    try:
        from pypdf import PdfReader

        meta = PdfReader(path).metadata
        if meta is None:
            return None
        raw = meta.get("/CreationDate")
        return str(raw) if raw else None
    except Exception:
        return None


def _page_count(path: Path) -> int | None:
    try:
        from pypdf import PdfReader

        return len(PdfReader(path).pages)
    except Exception:
        return None


def download(act: ActEntry, dest_dir: Path = CORPUS_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / act.file_name

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        response = client.get(act.url)
        response.raise_for_status()
        target.write_bytes(response.content)

    return target


def fetch_act(act: ActEntry, *, repin: bool, dest_dir: Path = CORPUS_DIR) -> list[str]:
    """Fetch one act, verify or pin it, and return report lines."""
    lines = [f"{act.slug}: {act.eli} variant {act.variant}"]
    path = download(act, dest_dir)
    digest = sha256_file(path)
    size_kb = path.stat().st_size / 1024

    lines.append(f"  downloaded {size_kb:,.0f} KiB -> {path}")

    if not act.is_pinned:
        lines.append(f"  first fetch, pinning sha256 {digest[:16]}...")
    elif digest == act.sha256:
        lines.append(f"  checksum matches pin {digest[:16]}...")
        return lines
    elif not repin:
        raise ChecksumMismatchError(
            f"{act.slug}: pinned {act.sha256[:16] if act.sha256 else None}... "
            f"but fetched {digest[:16]}...\n"
            "ISAP regenerates the consolidated text when an amendment lands, so this most "
            "likely means the law changed rather than the file being corrupt.\n"
            "Re-pin deliberately with --repin, and treat metrics recorded under the previous "
            "checksum as no longer comparable."
        )
    else:
        lines.append(
            f"  RE-PINNED {act.sha256[:16] if act.sha256 else None}... -> {digest[:16]}..."
        )

    act.sha256 = digest
    act.pdf_creation_date = _pdf_creation_date(path)
    act.pages = _page_count(path)
    act.fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    lines.append(f"  pages={act.pages} pdf_created={act.pdf_creation_date}")
    return lines


def fetch_all(manifest: Manifest, *, repin: bool, dest_dir: Path = CORPUS_DIR) -> list[str]:
    lines: list[str] = []
    for act in manifest.acts:
        lines.extend(fetch_act(act, repin=repin, dest_dir=dest_dir))
    lines.append(f"corpus manifest digest: {manifest.digest()}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repin",
        action="store_true",
        help="Accept changed bytes and rewrite the checksums. Invalidates prior metrics.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    try:
        lines = fetch_all(manifest, repin=args.repin)
    except ChecksumMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    save_manifest(manifest)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
