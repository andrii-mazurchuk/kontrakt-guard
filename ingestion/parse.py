"""Turn extracted statute text into articles — the unit of citation and of evaluation.

The article is the natural chunk for legal text (ADR 0002): it is what a citation
names, and its identifier is the ground-truth key that recall@k is computed
against. Everything here exists to produce that identifier reliably.

Canonical vs. display form
--------------------------
Internally an article is ``25``, ``9^1``, ``18^3a`` — ASCII, sortable, safe in a
database column and in a JSON gold set. For humans it renders as ``Art. 9¹``.
The canonical form is the contract; changing it later would invalidate every
recorded metric, since the gold set is keyed on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ingestion.pdf_text import PageText

# An article heading opens a line: "Art. 25.", "Art. 9^1.", "Art. 18^3a.".
# Anchored at line start and requiring the capital A, so that in-text references
# ("w rozumieniu art. 22 § 1") and footnote markers ("Kodeks pracy^1)") do not match.
ARTICLE_RE = re.compile(r"^Art\.\s*(\d+(?:\^[0-9a-ząćęłńóśźż]+)*)\.\s*")

# Paragraph markers within an article: "§ 1.", "§ 2^1.".
PARAGRAPH_RE = re.compile(r"§\s*(\d+(?:\^[0-9a-z]+)*)\.\s*")

# Structural headings. Polish statutes nest DZIAŁ > Rozdział > Oddział.
HEADING_RE = re.compile(r"^(DZIAŁ|Rozdział|Oddział)\s+(\S+)\s*$", re.IGNORECASE)
HEADING_LEVEL = {"dział": 0, "rozdział": 1, "oddział": 2}

# An article can stop carrying operative law in more than one way, and the ways
# are legally distinct: "uchylony" is repealed by the legislature, "utracił moc"
# has typically been struck down by the Constitutional Tribunal, "pominięty" was
# omitted when the consolidated text was compiled. All three must be kept out of
# retrieval as governing law, but a question naming one deserves the specific
# reason rather than silence — so the kind is preserved, not just the fact.
_NOT_IN_FORCE = r"uchylony|uchylona|uchylone|utracił[ay]?\s+moc|utraciło\s+moc|pominięty|pominięte"

# Anchored to the WHOLE body, not merely present in it.
#
# Statutes repeal individual paragraphs constantly: Art. 171 § 2 is repealed while
# paragraphs 1 and 3 to 5 remain the operative law on holiday pay. A substring search
# marks the entire article dead, and since retrieval excludes repealed articles by
# default that article becomes permanently unreachable — a silent, uncorrectable
# recall miss that looks like a retrieval weakness. It wrongly condemned 34 of 477
# articles, among them Art. 94 (employer obligations) and Art. 87 (wage deductions).
#
# The trailing group absorbs footnote references, as in "(utracił moc)^5)".
NOT_IN_FORCE_BODY_RE = re.compile(
    rf"^\(({_NOT_IN_FORCE})\)(\s*\^?\d*\)?)*$",
    re.IGNORECASE,
)

SUPERSCRIPT = str.maketrans(
    "0123456789abcdefghijklmnoprstuwxyz", "⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘʷˣʸᶻ"
)


def to_display(canonical: str) -> str:
    """``9^1`` -> ``Art. 9¹``. Presentation only; never stored as the key."""
    parts = canonical.split("^")
    rendered = parts[0] + "".join(p.translate(SUPERSCRIPT) for p in parts[1:])
    return f"Art. {rendered}"


COMPONENT_RE = re.compile(r"(\d*)([a-ząćęłńóśźż]*)")


def _component(part: str) -> tuple[int, str]:
    """Split '3ca' into (3, 'ca') so digits sort numerically and letters break ties."""
    match = COMPONENT_RE.match(part)
    if match is None:  # pragma: no cover - the pattern matches the empty string
        return (0, part)
    digits, letters = match.groups()
    return (int(digits) if digits else 0, letters)


class Paragraph(BaseModel):
    marker: str = Field(
        description="Canonical paragraph id, e.g. '1' or '2^1'. Empty if undivided."
    )
    text: str
    repealed: bool = Field(
        default=False,
        description="This paragraph alone is not in force; the article may still be.",
    )


class Article(BaseModel):
    act: str
    article: str = Field(description="Canonical id: '25', '9^1', '18^3a'.")
    display: str
    title_path: list[str] = Field(
        default_factory=list,
        description="Structural ancestry, outermost first: DZIAŁ / Rozdział / Oddział.",
    )
    paragraphs: list[Paragraph] = Field(default_factory=list)
    repealed: bool = Field(
        default=False, description="True when the article carries no operative law."
    )
    repeal_kind: str = Field(
        default="",
        description="Why it is not in force: 'uchylony', 'utracił moc', 'pominięty'.",
    )
    page_start: int
    text: str

    @property
    def sort_key(self) -> tuple[tuple[int, str], ...]:
        """Order articles the way the statute does.

        Every component is compared numerically before alphabetically, at every
        level of superscripting. Comparing the superscript as a string instead
        would place Art. 94¹⁰ before Art. 94⁹, because "10" < "9" lexicographically
        — which reads as a parser fault when it is only a sorting one.
        """
        return tuple(_component(part) for part in self.article.split("^"))


@dataclass
class _Pending:
    """The article currently being accumulated, before its body is complete."""

    article: str
    title_path: list[str]
    page: int


@dataclass
class _Heading:
    """Mutable structural cursor while walking the document."""

    path: list[str] = field(default_factory=lambda: ["", "", ""])

    def set(self, level: int, text: str) -> None:
        self.path[level] = text
        # Entering a new section resets everything nested beneath it.
        for deeper in range(level + 1, len(self.path)):
            self.path[deeper] = ""

    def current(self) -> list[str]:
        return [p for p in self.path if p]


def _split_paragraphs(body: str) -> list[Paragraph]:
    """Split an article body at § markers, preserving the marker as the id."""
    matches = list(PARAGRAPH_RE.finditer(body))
    if not matches:
        text = body.strip()
        if not text:
            return []
        return [Paragraph(marker="", text=text, repealed=_is_not_in_force(text))]

    paragraphs: list[Paragraph] = []
    # Text before the first § belongs to the article, not to any paragraph; in
    # practice it is empty, because a divided article opens with "§ 1.".
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[match.end() : end].strip()
        if text:
            paragraphs.append(
                Paragraph(marker=match.group(1), text=text, repealed=_is_not_in_force(text))
            )
    return paragraphs


def _is_not_in_force(text: str) -> bool:
    """Whether this text is *nothing but* a repeal marker."""
    return NOT_IN_FORCE_BODY_RE.match(text.strip()) is not None


def parse_articles(pages: list[PageText], act: str) -> list[Article]:
    """Walk the document, emitting one Article per article heading.

    Articles routinely span a page break, so the walk is over a flat line stream
    with the page number carried alongside rather than page by page.
    """
    lines: list[tuple[int, str]] = [
        (page.number, line) for page in pages for line in page.text.split("\n")
    ]

    heading = _Heading()
    articles: list[Article] = []

    current: _Pending | None = None
    buffer: list[str] = []
    pending_heading_level: int | None = None

    def flush() -> None:
        if current is None:
            return
        body = " ".join(buffer).strip()
        # Only when the entire body is the marker — see NOT_IN_FORCE_BODY_RE.
        whole_body = NOT_IN_FORCE_BODY_RE.match(body)
        articles.append(
            Article(
                act=act,
                article=current.article,
                display=to_display(current.article),
                title_path=list(current.title_path),
                paragraphs=_split_paragraphs(body),
                repealed=whole_body is not None,
                repeal_kind=whole_body.group(1).lower() if whole_body else "",
                page_start=current.page,
                text=body,
            )
        )

    for page_no, line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            level = HEADING_LEVEL[heading_match.group(1).lower()]
            heading.set(level, stripped)
            pending_heading_level = level
            continue

        # A heading's title sits on the following line ("Rozdział II" / "Umowa o pracę").
        if pending_heading_level is not None:
            if not ARTICLE_RE.match(stripped):
                level = pending_heading_level
                heading.set(level, f"{heading.path[level]} — {stripped}")
                pending_heading_level = None
                continue
            pending_heading_level = None

        article_match = ARTICLE_RE.match(stripped)
        if article_match:
            flush()
            current = _Pending(
                article=article_match.group(1),
                title_path=heading.current(),
                page=page_no,
            )
            buffer = [stripped[article_match.end() :]]
            continue

        if current is not None:
            buffer.append(stripped)

    flush()
    return articles
