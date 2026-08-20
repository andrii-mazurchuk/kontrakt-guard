"""Extraction tests against a committed 3-page slice of the real act.

The fixture is pages 1, 4 and 26 of the consolidated Kodeks pracy — chosen because
they carry the four cases that break naive extraction: a superscript article
(Art. 9¹), a superscript that collides with a real article number (Art. 23²),
superscript paragraph markers (§ 2¹), and a repealed article.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.pdf_text import extract_pages, extract_text

FIXTURE = Path("tests/fixtures/kp_slice.pdf")


@pytest.fixture(scope="module")
def text() -> str:
    return extract_text(FIXTURE)


def test_pages_are_returned_in_order():
    pages = extract_pages(FIXTURE)
    assert [p.number for p in pages] == [1, 2, 3]
    assert all(p.text for p in pages)


def test_superscript_article_is_recovered(text):
    """Art. 9¹ must not flatten to 'Art. 91'.

    Article ids are the ground-truth key for recall@k, so a collision here would
    corrupt the metric rather than merely the display.
    """
    assert "Art. 9^1" in text
    assert "Art. 91" not in text


def test_superscript_that_collides_with_a_real_article_number(text):
    """Art. 23² and Art. 232 are different articles; extraction must distinguish them."""
    assert "Art. 23^2" in text
    assert "Art. 232" not in text


def test_superscript_paragraph_markers_are_recovered(text):
    """The range 'para 2-1 to 2-3' flattens to the nonsensical '21-23' otherwise."""
    assert "2^1" in text
    assert "§ 21" not in text


def test_repealed_articles_are_retained(text):
    """Kept, not dropped: a question naming one deserves an answer, not silence."""
    assert "Art. 24. (uchylony)" in text


def test_page_furniture_is_stripped(text):
    assert "Kancelaria Sejmu" not in text
    assert "s. 26/190" not in text


def test_genuine_narrow_spaces_survive(text):
    """Rebuilding spacing purely from geometry loses these, yielding 'wmiejscu'."""
    assert "w miejscu pracy" in text
    assert "wmiejscu" not in text


def test_spurious_intra_word_spaces_are_repaired(text):
    """The PDF text layer splits this token; a naive extractor emits 'pracown ika'."""
    assert "pracownika" in text
    assert "pracown ika" not in text


def test_doubled_spaces_from_justification_are_collapsed(text):
    assert "  " not in text


def test_structural_headings_survive(text):
    assert "Rozdział II" in text
    assert "Oddział 1" in text
