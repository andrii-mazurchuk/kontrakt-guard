"""Extract Polish statute text from ISAP PDFs with superscripts preserved.

Why this is not `pypdf.extract_text()`
--------------------------------------
Polish legislation inserts articles by superscripting: ``Art. 23²`` sits between
``Art. 23`` and ``Art. 24``, while a wholly unrelated ``Art. 232`` exists later in
the same act. A plain text extractor flattens both to ``"Art. 232"``. Since the
article identifier is the ground-truth key for retrieval evaluation (ADR 0002),
that collision would silently corrupt recall@k — the metric would be measuring
something other than what it reports.

`pdfplumber` exposes per-glyph font size and baseline offset. In these documents
superscripts are consistently ~8pt glyphs raised ~5.4pt against 12pt body text,
which is a deterministic geometric test rather than a heuristic. They are emitted
here as ``^``-prefixed runs (``Art. 23^2``), so everything downstream is plain
string work.

Word spacing is rebuilt from glyph positions rather than from the PDF's own space
characters. The source is justified text produced by Word, which yields both
doubled spaces between words and spurious spaces inside them (``pracown ika``).
Geometry is the more reliable signal.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber

# A glyph must be this much smaller than body text, and raised at least this far
# above the line's baseline, to count as a superscript. The gap between body
# (12pt, offset 0) and superscript (8.04pt, offset +5.42) is wide, so these
# thresholds sit comfortably between the two populations rather than on an edge.
SIZE_MARGIN = 0.6
RAISE_MIN = 2.0

# Fraction of body font size above which a horizontal gap becomes a word break.
# Intra-word glyph gaps are near zero; inter-word gaps in this justified text run
# from roughly 0.25 to 0.5 em.
SPACE_RATIO = 0.18

# Page furniture repeated on every page of an ISAP consolidated text.
FURNITURE = (
    re.compile(r"^©\s*Kancelaria\s+Sejmu\s+s\.\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
)


@dataclass(frozen=True)
class PageText:
    number: int  # 1-indexed, as printed
    text: str


def _dominant_size(chars: list[dict[str, Any]]) -> float:
    """Modal glyph size — the body text size for this document."""
    sizes = Counter(round(float(c["size"]), 2) for c in chars)
    if not sizes:
        raise ValueError("page contains no glyphs; is this a scanned PDF?")
    return float(sizes.most_common(1)[0][0])


def _is_furniture(line: str) -> bool:
    stripped = line.strip()
    return any(pattern.match(stripped) for pattern in FURNITURE)


def _render_line(chars: list[dict[str, Any]], body_size: float) -> str:
    """Assemble one visual line, marking superscript runs with a leading ``^``.

    The PDF's own space glyphs are kept — they are mostly correct, and discarding
    them in favour of pure geometry loses narrow but genuine spaces (``w miejscu``
    becomes ``wmiejscu``). Geometry is used only to *add* a space where a wide gap
    carries no space glyph. Runs of whitespace collapse afterwards, which disposes
    of the doubled spaces that justified Word output produces.
    """
    if not chars:
        return ""

    # The baseline is where most glyphs sit; superscripts are the exceptions.
    inked = [c for c in chars if c["text"].strip()]
    if not inked:
        return ""
    baseline = statistics.median(c["bottom"] for c in inked)

    out: list[str] = []
    in_superscript = False
    prev: dict[str, Any] | None = None

    for c in chars:
        text = c["text"]
        is_space = not text.strip()

        if prev is not None and not is_space and prev["text"].strip():
            gap = c["x0"] - prev["x1"]
            # Scale the threshold to the local glyph size, not the document body
            # size: footnotes and marginal notes are set several points smaller,
            # and a body-sized threshold swallows their genuine spaces
            # ("w miejscu" -> "wmiejscu").
            local = max(c["size"], prev["size"])
            if gap > SPACE_RATIO * local:
                out.append(" ")
                in_superscript = False

        if is_space:
            out.append(" ")
            in_superscript = False
            prev = c
            continue

        raised = baseline - c["bottom"]
        superscript = c["size"] < body_size - SIZE_MARGIN and raised > RAISE_MIN

        if superscript and not in_superscript:
            out.append("^")
        in_superscript = superscript

        out.append(text)
        prev = c

    return re.sub(r"\s+", " ", "".join(out)).strip()


def _body_right_edge(pdf: Any, body_size: float) -> float:
    """Rightmost extent of the main text column.

    ISAP consolidated texts carry amendment notes ("Nowe brzmienie", the date a
    change takes effect) in a narrow right-hand margin. Those notes sit at the
    same vertical positions as the body, so line grouping interleaves them into
    the sentences they annotate — a paragraph acquires fragments like
    "^art. ^18[3d] ^we" and "5.11.2026 r. (Dz." spliced mid-clause, and the
    article heading they precede stops being recognisable.

    The two columns separate cleanly by geometry: body text ends around x=471 and
    the margin begins near x=480. Measuring the boundary rather than assuming it
    keeps this working if the layout changes.
    """
    edges = [
        float(char["x1"])
        for page in pdf.pages
        for char in page.chars
        if abs(char["size"] - body_size) < 0.5
    ]
    if not edges:
        raise ValueError("no body-size glyphs found; is this a scanned PDF?")
    return max(edges)


def extract_pages(path: Path) -> list[PageText]:
    """Extract every page, superscripts marked, furniture and margin notes removed."""
    pages: list[PageText] = []

    with pdfplumber.open(path) as pdf:
        # Calibrate against the whole document rather than per page: a page that
        # happens to be mostly headings would otherwise mis-detect its body size.
        sample = [c for page in pdf.pages[:20] for c in page.chars]
        body_size = _dominant_size(sample)
        right_edge = _body_right_edge(pdf, body_size)

        for index, page in enumerate(pdf.pages, start=1):
            lines: list[str] = []
            for line in page.extract_text_lines(return_chars=True, strip=True):
                # A glyph starting beyond the body column belongs to the margin.
                # Superscripts inside the text flow always start within it.
                chars = [c for c in line["chars"] if c["x0"] <= right_edge]
                rendered = _render_line(chars, body_size)
                if rendered and not _is_furniture(rendered):
                    lines.append(rendered)
            pages.append(PageText(number=index, text="\n".join(lines)))

    return pages


def extract_text(path: Path) -> str:
    """Whole document as one string, pages joined by a blank line."""
    return "\n".join(page.text for page in extract_pages(path))
