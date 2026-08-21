from __future__ import annotations

import pytest

from ingestion.chunk import (
    Chunk,
    chunk_article,
    chunk_articles,
    embedding_input,
    query_input,
    whitespace_counter,
)
from ingestion.parse import Article, Paragraph


def make_article(
    article: str = "25",
    paragraphs: list[tuple[str, str]] | None = None,
    text: str = "",
    title_path: list[str] | None = None,
    repealed: bool = False,
) -> Article:
    return Article(
        act="kp",
        article=article,
        display=f"Art. {article}",
        title_path=title_path or ["DZIAŁ DRUGI", "Rozdział II — Umowa o pracę"],
        paragraphs=[Paragraph(marker=m, text=t) for m, t in (paragraphs or [])],
        repealed=repealed,
        page_start=3,
        text=text,
    )


# --- the common case: one article, one chunk ----------------------------------


def test_short_article_becomes_exactly_one_chunk():
    """A retrieved result must be directly citable, so articles stay whole."""
    article = make_article(paragraphs=[("1", "Umowę o pracę zawiera się na okres próbny.")])
    chunks = chunk_article(article)
    assert len(chunks) == 1
    assert chunks[0].part_index == 0


def test_chunk_content_opens_with_its_own_citation():
    article = make_article(paragraphs=[("1", "Umowę o pracę zawiera się.")])
    assert chunk_article(article)[0].content.startswith("Art. 25 § 1.")


def test_undivided_article_carries_only_the_article_header():
    article = make_article(article="8", text="Nie można czynić ze swego prawa użytku.")
    content = chunk_article(article)[0].content
    assert content.startswith("Art. 8.")
    assert "§" not in content


def test_multi_paragraph_article_that_fits_stays_one_chunk():
    article = make_article(paragraphs=[("1", "Pierwszy."), ("2", "Drugi."), ("3", "Trzeci.")])
    chunks = chunk_article(article)
    assert len(chunks) == 1
    assert "§ 1." in chunks[0].content and "§ 3." in chunks[0].content


def test_superscript_paragraph_renders_without_the_caret():
    """'2^1' is the storage form; a citation reads 'Art. 25 § 21'."""
    article = make_article(paragraphs=[("2^1", "Strony mogą uzgodnić.")])
    assert "^" not in chunk_article(article)[0].content


# --- splitting ------------------------------------------------------------------


def test_oversized_article_splits_at_paragraph_boundaries():
    paragraphs = [(str(i), "słowo " * 100) for i in range(1, 6)]
    chunks = chunk_article(make_article(paragraphs=paragraphs), budget=150)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 400 for c in chunks)


def test_split_chunks_get_dense_unique_part_indexes():
    """part_index is part of the row identity, so gaps or repeats break the upsert."""
    paragraphs = [(str(i), "słowo " * 100) for i in range(1, 8)]
    chunks = chunk_article(make_article(paragraphs=paragraphs), budget=150)
    assert [c.part_index for c in chunks] == list(range(len(chunks)))


def test_a_single_oversized_paragraph_splits_at_sentence_boundaries():
    """Polish statutes enumerate at length; one paragraph can exceed the budget alone."""
    long_text = " ".join(f"Zdanie numer {i} z pewną treścią." for i in range(60))
    chunks = chunk_article(make_article(paragraphs=[("1", long_text)]), budget=60)
    assert len(chunks) > 1
    # No fragment should begin mid-sentence.
    for chunk in chunks:
        body = chunk.content.split(". ", 1)[-1]
        assert body[0].isupper() or body[0].isdigit()


def test_every_chunk_keeps_full_provenance():
    paragraphs = [(str(i), "słowo " * 100) for i in range(1, 6)]
    chunks = chunk_article(make_article(paragraphs=paragraphs), budget=150)
    for chunk in chunks:
        assert chunk.act == "kp"
        assert chunk.article == "25"
        assert chunk.title_path
        assert chunk.page_start == 3


# --- repealed -------------------------------------------------------------------


def test_repealed_articles_are_chunked_not_discarded():
    """Kept so a question naming one gets the reason instead of silence."""
    article = make_article(article="24", text="(uchylony)", repealed=True)
    chunks = chunk_article(article)
    assert len(chunks) == 1
    assert chunks[0].repealed


# --- e5 prefixes ----------------------------------------------------------------


def test_passage_and_query_prefixes_are_asymmetric():
    """e5 is trained asymmetrically. Using one prefix for both degrades recall silently."""
    chunk = chunk_article(make_article(paragraphs=[("1", "Treść.")]))[0]
    assert embedding_input(chunk).startswith("passage: ")
    assert query_input("czy to legalne?").startswith("query: ")


def test_embedding_input_carries_the_structural_path():
    """An article inherits its chapter's subject even when its text never states it."""
    chunk = chunk_article(
        make_article(paragraphs=[("1", "Treść.")], title_path=["DZIAŁ DRUGI", "Praca zdalna"])
    )[0]
    assert "Praca zdalna" in embedding_input(chunk)


def test_stored_content_is_free_of_the_prefix():
    """The prefix is an encoder artefact and must never reach a citation."""
    chunk = chunk_article(make_article(paragraphs=[("1", "Treść.")]))[0]
    assert "passage:" not in chunk.content


# --- citations ------------------------------------------------------------------


def test_citation_property():
    article = make_article(paragraphs=[("2^1", "Treść.")])
    assert chunk_article(article)[0].citation == "Art. 25 § 21"
    assert chunk_article(make_article(article="8", text="Treść."))[0].citation == "Art. 8"


def test_chunk_articles_flattens():
    articles = [make_article(article=str(i), text="Treść artykułu.") for i in range(1, 4)]
    assert len(chunk_articles(articles)) == 3


def test_token_counts_are_recorded():
    chunk = chunk_article(make_article(paragraphs=[("1", "jedno dwa trzy")]))[0]
    assert chunk.n_tokens == whitespace_counter(chunk.content)


def test_chunk_model_defaults():
    chunk = Chunk(act="kp", article="1", article_display="Art. 1", page_start=1, content="x")
    assert (chunk.paragraph, chunk.part_index, chunk.repealed) == ("", 0, False)


@pytest.mark.parametrize("budget", [50, 100, 200])
def test_no_chunk_is_empty_at_any_budget(budget):
    paragraphs = [(str(i), "słowo " * 80) for i in range(1, 5)]
    for chunk in chunk_article(make_article(paragraphs=paragraphs), budget=budget):
        assert chunk.content.strip()
