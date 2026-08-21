"""Turn articles into embeddable chunks.

One chunk per article (ADR 0002), split at ``§`` boundaries only when the article
exceeds the embedding model's context window. Splitting mid-paragraph is avoided
because a fragment that begins in the middle of a legal sentence is not citable,
and citation is the product.

The chunk carries its own citation in the text: an article opens with
``Art. 25 § 1.`` exactly as the statute prints it. That serves both retrieval legs
— the lexical index sees the article number, and the dense encoder sees the
context rather than an anonymous paragraph.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from pydantic import BaseModel, Field

from ingestion.parse import Article

# multilingual-e5-large truncates at 512 tokens. Exceeding it does not raise: the
# encoder silently drops the tail, which would remove the end of an article from
# the index while the load still reports success.
MODEL_MAX_TOKENS = 512

# Safety margin below the hard limit, absorbing tokenizer differences between the
# fast and slow implementations.
TOKEN_BUDGET = 480

TokenCounter = Callable[[str], int]

_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


class Chunk(BaseModel):
    act: str
    article: str
    article_display: str
    paragraph: str = ""
    part_index: int = 0

    title_path: list[str] = Field(default_factory=list)
    page_start: int
    repealed: bool = False
    repeal_kind: str = ""

    content: str
    n_tokens: int = 0

    @property
    def citation(self) -> str:
        """How this chunk is cited in an answer."""
        if self.paragraph:
            return f"{self.article_display} § {self.paragraph.replace('^', '')}"
        return self.article_display


def whitespace_counter(text: str) -> int:
    """Cheap stand-in for the real tokenizer, for tests and dry runs."""
    return len(text.split())


def _pack(
    pieces: list[tuple[str, str]], count: TokenCounter, budget: int
) -> list[list[tuple[str, str]]]:
    """Greedily group (marker, text) pieces into parts that fit the budget."""
    parts: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_tokens = 0

    for marker, text in pieces:
        size = count(text)
        if current and current_tokens + size > budget:
            parts.append(current)
            current, current_tokens = [], 0
        current.append((marker, text))
        current_tokens += size

    if current:
        parts.append(current)
    return parts


def _split_oversized(text: str, count: TokenCounter, budget: int) -> list[str]:
    """Last resort for a single paragraph larger than the whole budget.

    Splits at sentence boundaries. Polish statutes enumerate at length — one
    paragraph can run to a dozen numbered points — so this does happen.
    """
    if count(text) <= budget:
        return [text]

    out: list[str] = []
    buffer: list[str] = []
    size = 0
    for sentence in _SENTENCE_END.split(text):
        s = count(sentence)
        if buffer and size + s > budget:
            out.append(" ".join(buffer))
            buffer, size = [], 0
        buffer.append(sentence)
        size += s
    if buffer:
        out.append(" ".join(buffer))
    return out


def chunk_article(
    article: Article,
    count: TokenCounter = whitespace_counter,
    budget: int = TOKEN_BUDGET,
) -> list[Chunk]:
    """One chunk if the article fits; otherwise several, split at § boundaries."""

    def build(paragraph: str, part: int, body: str) -> Chunk:
        header = (
            article.display
            if not paragraph
            else f"{article.display} § {paragraph.replace('^', '')}"
        )
        content = f"{header}. {body}".strip()
        return Chunk(
            act=article.act,
            article=article.article,
            article_display=article.display,
            paragraph=paragraph,
            part_index=part,
            title_path=list(article.title_path),
            page_start=article.page_start,
            repealed=article.repealed,
            repeal_kind=article.repeal_kind,
            content=content,
            n_tokens=count(content),
        )

    pieces = [(p.marker, p.text) for p in article.paragraphs if p.text]

    # Budget against what the encoder actually receives, not against the raw body.
    # embedding_input prepends "passage: " and the structural path, and build()
    # prepends the citation header — together often 40+ tokens. Measuring the body
    # alone lets a chunk sail past the model limit and be silently truncated.
    widest_marker = max((m for m, _ in pieces), key=len, default="")
    overhead = count(embedding_input(build(widest_marker, 0, "")))
    budget = max(budget - overhead, 64)

    if not pieces:
        return [build("", 0, article.text)]

    whole = " ".join(f"§ {m}. {t}" if m else t for m, t in pieces)
    if count(whole) <= budget:
        # The common case: the entire article is one chunk, which is what makes
        # a retrieved result directly citable.
        marker = pieces[0][0] if len(pieces) == 1 else ""
        body = pieces[0][1] if len(pieces) == 1 else whole
        return [build(marker, 0, body)]

    chunks: list[Chunk] = []
    for part_index, group in enumerate(_pack(pieces, count, budget)):
        if len(group) == 1:
            marker, text = group[0]
            fragments = _split_oversized(text, count, budget)
            if len(fragments) == 1:
                chunks.append(build(marker, part_index, fragments[0]))
            else:
                # Renumber so part_index stays unique within the article.
                for fragment in fragments:
                    chunks.append(build(marker, len(chunks), fragment))
        else:
            body = " ".join(f"§ {m}. {t}" if m else t for m, t in group)
            chunks.append(build("", part_index, body))

    # part_index is part of the row identity, so it must be dense and unique.
    for index, chunk in enumerate(chunks):
        chunk.part_index = index
    return chunks


def chunk_articles(
    articles: Iterable[Article],
    count: TokenCounter = whitespace_counter,
    budget: int = TOKEN_BUDGET,
) -> list[Chunk]:
    return [c for article in articles for c in chunk_article(article, count, budget)]


def search_text(chunk: Chunk) -> str:
    """What the lexical index is built over.

    The structural path is included so a question about remote work reaches
    articles filed under "Rozdział IIc — Praca zdalna" even when the article body
    never repeats the chapter's wording.
    """
    path = " ".join(chunk.title_path)
    return f"{path} {chunk.content}".strip()


def embedding_input(chunk: Chunk) -> str:
    """What actually goes to the encoder.

    e5 requires the ``passage: `` prefix at index time and ``query: `` at search
    time; omitting them degrades retrieval quietly rather than failing, which is
    exactly the class of bug recall@k exists to catch.

    The structural path is prepended so an article inherits its chapter's subject
    — "Praca zdalna" — even when its own text never states it.
    """
    path = " > ".join(chunk.title_path)
    body = f"{path}\n{chunk.content}" if path else chunk.content
    return f"passage: {body}"


def query_input(question: str) -> str:
    """The matching ``query: `` prefix. Must stay paired with embedding_input."""
    return f"query: {question}"


def assert_within_model_limit(
    chunks: Iterable[Chunk],
    count: TokenCounter,
    limit: int = MODEL_MAX_TOKENS,
) -> None:
    """Refuse to index a chunk the encoder would silently truncate.

    Truncation loses the tail of an article while the load still reports success,
    so the corpus ends up with holes that retrieval evaluation cannot distinguish
    from ordinary misses. Better to fail here and name the article.
    """
    oversized = [
        (chunk.citation, size)
        for chunk in chunks
        if (size := count(embedding_input(chunk))) > limit
    ]
    if oversized:
        detail = ", ".join(f"{citation} ({size} tokens)" for citation, size in oversized[:5])
        raise ValueError(
            f"{len(oversized)} chunk(s) exceed the model limit of {limit} tokens "
            f"and would be truncated: {detail}"
        )
