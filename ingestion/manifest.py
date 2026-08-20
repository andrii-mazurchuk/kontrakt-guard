"""The corpus manifest: what the corpus is, and how to know it has changed.

A metric is only meaningful against a known corpus. These models are what let a
recorded eval run name the exact bytes it was computed over.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_PATH = Path("ingestion/corpus_manifest.json")
CORPUS_DIR = Path("data/corpus")


class ActEntry(BaseModel):
    """One act in the corpus."""

    slug: str = Field(description="Short stable key used as the `act` column on every chunk.")
    title_pl: str
    title_en: str

    eli: str = Field(description="European Legislation Identifier, e.g. DU/2025/277.")
    publisher: str
    year: int
    pos: int

    variant: str = Field(description="ISAP text variant: U = tekst ujednolicony.")
    variant_meaning: str
    url: str
    file_name: str

    # Null until the first fetch pins them.
    sha256: str | None = None
    pdf_creation_date: str | None = None
    pages: int | None = None
    fetched_at: str | None = None

    notes: list[str] = Field(default_factory=list)

    @property
    def path(self) -> Path:
        return CORPUS_DIR / self.file_name

    @property
    def is_pinned(self) -> bool:
        return self.sha256 is not None


class Manifest(BaseModel):
    acts: list[ActEntry]

    def by_slug(self, slug: str) -> ActEntry:
        for act in self.acts:
            if act.slug == slug:
                return act
        known = ", ".join(a.slug for a in self.acts) or "none"
        raise KeyError(f"no act with slug {slug!r} in the manifest (known: {known})")

    def digest(self) -> str:
        """Checksum of the manifest's pinned state.

        Recorded on every metrics row as `corpus_manifest_sha`, so a corpus change
        can never be invisible when comparing two runs.
        """
        payload = [
            {"slug": a.slug, "eli": a.eli, "variant": a.variant, "sha256": a.sha256}
            for a in sorted(self.acts, key=lambda a: a.slug)
        ]
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # `$comment` keys document the file for a human reader; drop them before parsing.
    raw.pop("$comment", None)
    return Manifest.model_validate(raw)


def save_manifest(manifest: Manifest, path: Path = MANIFEST_PATH) -> None:
    """Write the manifest back, preserving the `$comment` block if present."""
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out: dict[str, object] = {}
    if "$comment" in existing:
        out["$comment"] = existing["$comment"]
    out["acts"] = [a.model_dump() for a in manifest.acts]
    # newline="\n" explicitly: the default rewrites to CRLF on Windows, and the
    # manifest is committed, so every fetch would otherwise show as a whole-file diff.
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
