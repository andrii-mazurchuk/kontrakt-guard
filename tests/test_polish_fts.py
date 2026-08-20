"""Integration tests for the Polish full-text configuration.

Marked `integration`: needs a running database from `docker compose up -d db`.

These exist because the lexical half of hybrid retrieval is worthless without
lemmatisation. Polish inflects heavily, so an unlemmatised index means a question
about "wynagrodzenia" never matches a statute saying "wynagrodzenie", and hybrid
retrieval quietly degrades into dense retrieval alone — with no error to show for
it, only a lower recall@k that looks like an embedding problem.
"""

from __future__ import annotations

import pytest

from ingestion.db import TS_CONFIG, connect, polish_config_available

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    with connect() as connection:
        yield connection


def _lexize(conn, word: str) -> list[str]:
    row = conn.execute("SELECT ts_lexize('polish_hunspell', %s) AS lex", (word,)).fetchone()
    return list(row["lex"] or [])


def _tsvector(conn, text: str) -> str:
    row = conn.execute("SELECT to_tsvector(%s, %s)::text AS v", (TS_CONFIG, text)).fetchone()
    return str(row["v"])


def test_polish_configuration_exists(conn):
    """Stock Postgres ships 29 configurations and Polish is not among them."""
    assert polish_config_available(conn)


def test_pgvector_is_installed(conn):
    row = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None


def test_inflected_nouns_reduce_to_one_lemma(conn):
    for form in ("wynagrodzenia", "wynagrodzeniu", "wynagrodzenie"):
        assert "wynagrodzenie" in _lexize(conn, form)

    for form in ("godzina", "godzinach", "godzinami"):
        assert "godzina" in _lexize(conn, form)

    for form in ("umowa", "umowie", "umowami"):
        assert "umowa" in _lexize(conn, form)


def test_adjectives_reduce_to_their_base_form(conn):
    assert "nadliczbowy" in _lexize(conn, "nadliczbowych")


def test_stopwords_are_removed(conn):
    for word in ("za", "w", "oraz", "który"):
        assert _lexize(conn, word) == []


def test_ascii_only_words_are_lemmatised(conn):
    """The mapping must cover `asciiword`, not only `word`.

    Postgres emits `asciiword` for tokens of pure ASCII letters and `word` only
    for tokens carrying non-ASCII characters. Most Polish vocabulary has no
    diacritics, so a mapping listing only `word` lemmatises "pracę" while leaving
    "wynagrodzenia" untouched — which reads as a broken dictionary rather than a
    broken mapping.
    """
    ascii_only = _tsvector(conn, "wynagrodzenia godzinach umowy")
    assert "wynagrodzenie" in ascii_only
    assert "godzina" in ascii_only
    assert "wynagrodzenia" not in ascii_only


def test_words_with_diacritics_are_lemmatised(conn):
    assert "praca" in _tsvector(conn, "pracę")


def test_colloquial_question_matches_statutory_wording(conn):
    """The whole point of the lexical leg, in one assertion."""
    row = conn.execute(
        """
        SELECT to_tsvector(%s, 'wynagrodzenie za pracę w godzinach nadliczbowych')
               @@ plainto_tsquery(%s, 'wynagrodzenia za godziny nadliczbowe') AS matched
        """,
        (TS_CONFIG, TS_CONFIG),
    ).fetchone()
    assert row["matched"] is True


def test_unknown_tokens_are_still_indexed(conn):
    """Hunspell falls through to `simple`, so citations and proper nouns survive."""
    assert "pjatk" in _tsvector(conn, "PJATK")


def test_numbers_survive_for_citations(conn):
    """Article identifiers carry the citation and must not be dropped."""
    vector = _tsvector(conn, "art 25 par 2")
    assert "25" in vector
