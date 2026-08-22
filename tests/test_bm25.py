"""Integration tests for the BM25 lexical leg.

Marked `integration`: needs `docker compose up -d --build --wait db`.

These exist because Postgres has no BM25 and its own ranking functions have no
inverse document frequency term at all. `ts_rank_cd` rewards a chunk for saying a
word often and for saying it in a tight cluster, but never asks whether the word
distinguishes anything. On a corpus of employment law that is close to fatal:
'pracownik' occurs in three quarters of the chunks and 'art' in every single one,
and those are precisely the words a question about employment law contains.

The fixtures use invented vocabulary ('zorblak', 'kwintrol') rather than real
Polish. The production corpus shares the database, so a test querying real
statutory wording would be scored against 543 competing chunks and would assert
on the corpus rather than on the ranking function.
"""

from __future__ import annotations

import pytest

from ingestion.chunk import Chunk
from ingestion.db import apply_schema, connect, lexical_index_is_stale, refresh_lexical_stats
from ingestion.load import load_chunks
from retrieval.search import lexical_search
from tests.test_load import StubEmbedder

pytestmark = pytest.mark.integration

ACT = "bm25-test"

# Frequent in the fixture act, so its inverse document frequency is low.
COMMON = "zorblak"
# Present in exactly one chunk, so its inverse document frequency is high.
RARE = "kwintrol"


def make_chunk(article: str, content: str) -> Chunk:
    return Chunk(
        act=ACT,
        article=article,
        article_display=f"Art. {article}",
        paragraph="",
        part_index=0,
        # Deliberately empty: the structural path is indexed too, and a shared
        # path would put the same lexemes in every fixture chunk.
        title_path=[],
        page_start=1,
        content=content,
        n_tokens=len(content.split()),
    )


@pytest.fixture
def conn():
    with connect() as connection:
        apply_schema(connection)
        connection.execute("DELETE FROM chunks WHERE act = %s", (ACT,))
        refresh_lexical_stats(connection)
        connection.commit()
        yield connection
        connection.rollback()
        connection.execute("DELETE FROM chunks WHERE act = %s", (ACT,))
        # Refreshed on the way out as well as the way in. Deleting the fixture
        # rows without rebuilding the statistics would leave the developer's
        # database permanently stale, and the next eval run would refuse to start.
        refresh_lexical_stats(connection)
        connection.commit()


@pytest.fixture
def corpus(conn):
    """Ten chunks carrying the common term; one also carrying the rare one.

    Article 1 repeats the common term thirty times — it is the chunk a
    frequency-only ranking would put first.
    """
    chunks = [make_chunk("1", " ".join([COMMON] * 30))]
    chunks += [make_chunk(str(i), f"{COMMON} tekst numer {i}") for i in range(2, 11)]
    chunks.append(make_chunk("11", f"{COMMON} {RARE}"))
    load_chunks(conn, chunks, StubEmbedder(), show_progress=False)
    return conn


def articles(hits) -> list[str]:
    return [h.article for h in hits if h.act == ACT]


def test_a_rare_term_outweighs_a_repeated_common_one(corpus):
    """The whole point of the IDF term, in one assertion.

    Article 1 says the common word thirty times; article 11 says it once and adds
    a word that appears nowhere else. Only article 11 narrows anything down.
    """
    hits = lexical_search(corpus, f"{COMMON} {RARE}", limit=25)
    assert articles(hits)[0] == "11"


def test_the_repeated_common_term_still_wins_under_ts_rank_cd(corpus):
    """The behaviour being replaced, pinned so the comparison stays honest.

    If this ever starts agreeing with BM25, the fixture has stopped exercising
    the difference and the test above proves nothing.
    """
    hits = lexical_search(corpus, f"{COMMON} {RARE}", limit=25, ranking="ts_rank_cd")
    assert articles(hits)[0] == "1"


def test_a_shorter_chunk_wins_at_equal_term_frequency(conn):
    """BM25's `b` parameter: length normalisation.

    Two chunks mention the term exactly once. The shorter one is more *about* it,
    and a ranking that ignores length would call them equal.
    """
    load_chunks(
        conn,
        [
            make_chunk("1", f"{RARE} krótko"),
            make_chunk("2", f"{RARE} " + " ".join(f"wypełniacz{i}" for i in range(200))),
        ],
        StubEmbedder(),
        show_progress=False,
    )
    assert articles(lexical_search(conn, RARE, limit=25))[0] == "1"


def test_length_normalisation_can_be_switched_off(conn):
    """With b=0 the two chunks tie, which is what makes the previous test a test."""
    load_chunks(
        conn,
        [
            make_chunk("1", f"{RARE} krótko"),
            make_chunk("2", f"{RARE} " + " ".join(f"wypełniacz{i}" for i in range(200))),
        ],
        StubEmbedder(),
        show_progress=False,
    )
    hits = [h for h in lexical_search(conn, RARE, limit=25, b=0.0) if h.act == ACT]
    assert hits[0].score == pytest.approx(hits[1].score)


def test_scores_are_never_negative(corpus):
    """The `1 +` inside the logarithm is load-bearing.

    The raw Robertson/Sparck-Jones IDF goes negative for a term held by more than
    half the corpus. Without the shift, matching a common word would push a chunk
    *down* the ranking — worse than ignoring it.
    """
    hits = lexical_search(corpus, f"{COMMON} {RARE}", limit=25)
    assert hits
    assert all(h.score >= 0 for h in hits)


def test_a_question_of_only_stopwords_returns_nothing(corpus):
    """No lexemes means no query; it must return empty rather than everything."""
    assert lexical_search(corpus, "za w oraz który", limit=25) == []


def test_repealed_chunks_are_excluded_by_default(conn):
    chunk = make_chunk("1", f"{RARE} przepis")
    chunk.repealed = True
    chunk.repeal_kind = "uchylony"
    load_chunks(conn, [chunk], StubEmbedder(), show_progress=False)

    assert articles(lexical_search(conn, RARE, limit=25)) == []
    assert articles(lexical_search(conn, RARE, limit=25, include_repealed=True)) == ["1"]


def test_loading_refreshes_the_statistics(conn):
    """The loader must leave the index describing the corpus it just wrote."""
    load_chunks(conn, [make_chunk("1", f"{RARE} przepis")], StubEmbedder(), show_progress=False)
    assert not lexical_index_is_stale(conn)

    row = conn.execute(
        "SELECT count(*) AS n FROM chunk_terms WHERE lexeme = %s", (RARE,)
    ).fetchone()
    assert row["n"] == 1


def test_deleting_rows_behind_the_index_is_detected(conn):
    """A materialised view is a snapshot, and a stale one still returns results.

    Nothing errors: the search scores rows that no longer exist against a corpus
    size that no longer holds. The eval refuses to run rather than record it.
    """
    load_chunks(conn, [make_chunk("1", f"{RARE} przepis")], StubEmbedder(), show_progress=False)
    assert not lexical_index_is_stale(conn)

    conn.execute("DELETE FROM chunks WHERE act = %s", (ACT,))
    assert lexical_index_is_stale(conn)

    refresh_lexical_stats(conn)
    assert not lexical_index_is_stale(conn)


def test_statistics_describe_the_whole_corpus_not_one_act(corpus):
    """IDF is only meaningful against the full document count."""
    row = corpus.execute(
        "SELECT (SELECT n_docs FROM corpus_stats) AS n, "
        "(SELECT count(*)::float8 FROM chunks) AS actual"
    ).fetchone()
    assert row["n"] == row["actual"]
