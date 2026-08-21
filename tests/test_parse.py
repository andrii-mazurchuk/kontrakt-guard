from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.parse import Article, Paragraph, parse_articles, to_display
from ingestion.pdf_text import PageText, extract_pages
from ingestion.validate import ParseValidationError, validate

FIXTURE = Path("tests/fixtures/kp_slice.pdf")


@pytest.fixture(scope="module")
def articles() -> list[Article]:
    return parse_articles(extract_pages(FIXTURE), act="kp")


def _by_id(articles: list[Article], article_id: str) -> Article:
    return next(a for a in articles if a.article == article_id)


# --- canonical vs display form -------------------------------------------------


def test_display_rendering():
    assert to_display("25") == "Art. 25"
    assert to_display("9^1") == "Art. 9¹"
    assert to_display("18^3a") == "Art. 18³ᵃ"


def test_canonical_form_is_ascii_and_stable():
    """The gold set is keyed on this, so it must survive a JSON round trip intact."""
    for canonical in ("25", "9^1", "18^3ca"):
        assert canonical.isascii()


# --- ordering ------------------------------------------------------------------


def _article(article_id: str) -> Article:
    return Article(act="kp", article=article_id, display="", page_start=1, text="x" * 50)


def test_superscripts_sort_numerically_not_lexicographically():
    """Art. 94⁹ precedes Art. 94¹⁰. Comparing '9' > '10' as strings inverts them."""
    assert _article("94^9").sort_key < _article("94^10").sort_key


def test_bare_article_precedes_its_superscripted_insertions():
    assert _article("9").sort_key < _article("9^1").sort_key
    assert _article("9^1").sort_key < _article("10").sort_key


def test_letter_suffixes_break_ties_after_digits():
    assert _article("18^3").sort_key < _article("18^3a").sort_key
    assert _article("18^3a").sort_key < _article("18^3b").sort_key
    assert _article("151^9b").sort_key < _article("151^10").sort_key


# --- parsing the fixture -------------------------------------------------------


def test_superscript_article_is_parsed_as_its_own_article(articles):
    art = _by_id(articles, "9^1")
    assert art.display == "Art. 9¹"
    assert len(art.paragraphs) == 3


def test_collision_article_keeps_its_superscript(articles):
    art = _by_id(articles, "23^2")
    assert art.display == "Art. 23²"
    assert "232" not in art.article


def test_repealed_articles_are_flagged_not_dropped(articles):
    art = _by_id(articles, "24")
    assert art.repealed
    assert "uchylony" in art.text


def test_substantive_articles_are_not_flagged_repealed(articles):
    assert not _by_id(articles, "25").repealed


def test_lost_force_is_recognised_and_distinguished_from_repealed():
    """Art. 103 of the real act reads '(utracił moc)', not '(uchylony)'.

    Legally distinct: repealed by the legislature versus struck down, typically by
    the Constitutional Tribunal. Both carry no operative law and must stay out of
    retrieval as governing law, but a question naming one deserves the specific
    reason. Matching only 'uchylony' left Art. 103 looking like a parse failure.
    """
    pages = [
        PageText(
            number=1,
            text=(
                "DZIAŁ CZWARTY\n"
                "Obowiązki pracodawcy\n"
                "Art. 103. (utracił moc)^5)\n"
                "Art. 104. (uchylony)\n"
                "Art. 105. (pominięty)"
            ),
        )
    ]
    lost, repealed, omitted = parse_articles(pages, act="kp")

    assert (lost.repealed, lost.repeal_kind) == (True, "utracił moc")
    assert (repealed.repealed, repealed.repeal_kind) == (True, "uchylony")
    assert (omitted.repealed, omitted.repeal_kind) == (True, "pominięty")


def test_in_force_articles_carry_no_repeal_kind(articles):
    assert _by_id(articles, "25").repeal_kind == ""


def test_a_repealed_paragraph_does_not_condemn_its_article():
    """Art. 171 § 2 is repealed; §§ 1 and 3 are the operative law on holiday pay.

    Matching the repeal marker anywhere in the body marked the whole article dead.
    Because retrieval excludes repealed articles by default, that made 34 of 477
    live articles permanently unreachable — including Art. 94 and Art. 87 — a
    silent recall cap indistinguishable from a retrieval weakness.
    """
    pages = [
        PageText(
            number=1,
            text=(
                "DZIAŁ SIÓDMY\n"
                "Urlopy pracownicze\n"
                "Art. 171. § 1. W przypadku niewykorzystania urlopu przysługuje ekwiwalent "
                "pieniężny. § 2. (uchylony) § 3. Pracodawca nie ma obowiązku wypłacenia "
                "ekwiwalentu, gdy strony postanowią o wykorzystaniu urlopu."
            ),
        )
    ]
    article = parse_articles(pages, act="kp")[0]

    assert not article.repealed
    assert article.repeal_kind == ""
    assert [p.repealed for p in article.paragraphs] == [False, True, False]


def test_bracketed_heading_is_recognised_as_an_article():
    """ISAP wraps text with a pending amendment in [square brackets].

    Without matching the bracket the heading is invisible and the article's whole
    text is absorbed into its predecessor. That is how Art. 94³ (mobbing)
    disappeared while its text was served under Art. 94² — a wrong citation, which
    for a legal answer is a factual error rather than a ranking one.
    """
    pages = [
        PageText(
            number=1,
            text=(
                "DZIAŁ CZWARTY\n"
                "Obowiązki\n"
                "Art. 94^2. Pracodawca jest obowiązany informować pracowników.\n"
                "[Art. 94^3. § 1. Pracodawca jest obowiązany przeciwdziałać mobbingowi.]"
            ),
        )
    ]
    articles = parse_articles(pages, act="kp")

    assert [a.article for a in articles] == ["94^2", "94^3"]
    assert "mobbing" not in _by_id(articles, "94^2").text
    assert "mobbingowi" in _by_id(articles, "94^3").text


def test_future_law_in_angle_brackets_is_not_ingested():
    """A <...> block takes effect on a future date.

    Answering today's question from it is wrong in the same way as answering from
    repealed law, and it would collide with the in-force version's article id.
    """
    pages = [
        PageText(
            number=1,
            text=(
                "DZIAŁ CZWARTY\n"
                "Obowiązki\n"
                "[Art. 94^3. § 1. Obecne brzmienie przepisu o mobbingu.]\n"
                "<Art. 94^[3]. § 1. Przyszłe brzmienie przepisu o mobbingu.>\n"
                "<Art. 94^[3a]. § 1. Zupełnie nowy przepis.>"
            ),
        )
    ]
    articles = parse_articles(pages, act="kp")

    assert [a.article for a in articles] == ["94^3"]
    assert "Obecne brzmienie" in articles[0].text
    assert "Przyszłe" not in articles[0].text


def test_amendment_brackets_are_stripped_from_article_text():
    """The brackets delimit the change, not the statute's wording."""
    pages = [PageText(number=1, text="DZIAŁ\nX\n[Art. 55. § 1. Pracownik może rozwiązać umowę.]")]
    article = parse_articles(pages, act="kp")[0]
    assert "[" not in article.text and "]" not in article.text


def test_bracketed_superscript_id_is_normalised():
    """Inside an amendment block ISAP writes the superscript as 18^[3d]."""
    pages = [PageText(number=1, text="DZIAŁ\nX\n<Art. 18^[3f]. § 1. Nowy przepis.>")]
    assert parse_articles(pages, act="kp") == []


def test_an_article_that_is_only_a_marker_is_still_repealed():
    pages = [
        PageText(number=1, text="DZIAŁ PIERWSZY\nPrzepisy\nArt. 24. (uchylony)"),
    ]
    article = parse_articles(pages, act="kp")[0]
    assert article.repealed and article.repeal_kind == "uchylony"


def test_footnote_marker_after_a_repeal_marker_is_tolerated():
    """The real text reads 'Art. 103. (utracił moc)^5)'."""
    pages = [PageText(number=1, text="DZIAŁ CZWARTY\nObowiązki\nArt. 103. (utracił moc)^5)")]
    article = parse_articles(pages, act="kp")[0]
    assert article.repealed and article.repeal_kind == "utracił moc"


def test_paragraphs_are_split_on_the_section_marker(articles):
    art = _by_id(articles, "9")
    assert [p.marker for p in art.paragraphs] == ["1", "2", "3", "4"]
    assert art.paragraphs[0].text.startswith("Ilekroć w Kodeksie pracy")


def test_undivided_article_yields_one_unmarked_paragraph(articles):
    art = _by_id(articles, "8")
    assert [p.marker for p in art.paragraphs] == [""]


def test_structural_path_is_captured(articles):
    art = _by_id(articles, "25")
    assert art.title_path == [
        "Rozdział II — Umowa o pracę",
        "Oddział 1 — Zawarcie umowy o pracę",
    ]


def test_in_text_references_do_not_start_a_new_article(articles):
    """'w rozumieniu art. 22 § 1' is a citation, not a heading."""
    assert "22" not in {a.article for a in articles}


def test_page_provenance_is_recorded(articles):
    assert _by_id(articles, "25").page_start == 3


# --- validation ----------------------------------------------------------------


def test_validate_rejects_an_empty_parse():
    with pytest.raises(ParseValidationError, match="zero articles"):
        validate([], act="kp")


def test_validate_rejects_duplicate_ids():
    dupes = [_article("5"), _article("5")]
    for a in dupes:
        a.title_path = ["DZIAŁ PIERWSZY"]
    with pytest.raises(ParseValidationError, match="duplicate article ids"):
        validate(dupes, act="kp")


def test_validate_rejects_an_implausible_article_count():
    one = _article("1")
    one.title_path = ["DZIAŁ PIERWSZY"]
    with pytest.raises(ParseValidationError, match="outside expected band"):
        validate([one], act="kp")


def test_validate_rejects_out_of_order_articles():
    a, b = _article("10"), _article("9")
    for art in (a, b):
        art.title_path = ["DZIAŁ PIERWSZY"]
    with pytest.raises(ParseValidationError, match="out of statutory order"):
        validate([a, b], act="unknown-act")


def test_validate_rejects_an_article_that_swallowed_the_next_one():
    """The check that would have caught the bracketed-heading bug immediately.

    Article numbering has legal gaps, so a missing id proves nothing. A capitalised
    heading sitting inside another article's body does.
    """
    swallowed = Article(
        act="kp",
        article="94^2",
        display="Art. 94²",
        page_start=1,
        text="Pracodawca informuje pracowników. Art. 94^3. Pracodawca przeciwdziała mobbingowi.",
    )
    swallowed.title_path = ["DZIAŁ CZWARTY"]
    with pytest.raises(ParseValidationError, match="swallowed a following article"):
        validate([swallowed], act="unknown-act")


def test_validate_tolerates_lowercase_in_text_references():
    """'w rozumieniu art. 22 § 1' is a citation, not a missed heading."""
    citing = Article(
        act="kp",
        article="25",
        display="Art. 25",
        page_start=1,
        text="Umowę zawiera się zgodnie z art. 22 § 1 oraz art. 29 niniejszego kodeksu.",
    )
    citing.title_path = ["DZIAŁ DRUGI"]
    report = validate([citing], act="unknown-act")
    assert report


def test_validate_rejects_hollow_articles():
    thin = Article(act="kp", article="1", display="", page_start=1, text="")
    thin.title_path = ["DZIAŁ PIERWSZY"]
    with pytest.raises(ParseValidationError, match="suspiciously little text"):
        validate([thin], act="unknown-act")


def test_paragraph_model_defaults():
    assert Paragraph(marker="", text="t").marker == ""
