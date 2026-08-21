"""Integration tests for the schema and loader.

Marked `integration`: needs `docker compose up -d --build --wait db`.

A stub embedder stands in for the real model — these tests are about SQL and
idempotency, and loading 2 GB of weights to assert an upsert would make them
useless as a fast check.
"""

from __future__ import annotations

import pytest

from ingestion.chunk import Chunk, assert_within_model_limit, embedding_input, search_text
from ingestion.db import apply_schema, connect
from ingestion.load import column_dimension, corpus_stats, load_chunks

pytestmark = pytest.mark.integration

DIM = 1024


class StubEmbedder:
    """Deterministic vectors, so a reload can be compared byte for byte."""

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        return [[float((len(t) + i) % 7) / 7.0 for i in range(self._dim)] for t in texts]


def make_chunk(article: str = "25", paragraph: str = "1", part: int = 0) -> Chunk:
    return Chunk(
        act="kp",
        article=article,
        article_display=f"Art. {article}",
        paragraph=paragraph,
        part_index=part,
        title_path=["DZIAŁ DRUGI — Stosunek pracy", "Rozdział IIc — Praca zdalna"],
        page_start=7,
        content=f"Art. {article} § {paragraph}. Umowę o pracę zawiera się na okres próbny.",
        n_tokens=12,
    )


@pytest.fixture
def conn():
    with connect() as connection:
        apply_schema(connection)
        connection.execute("DELETE FROM chunks WHERE act = 'kp-test'")
        connection.commit()
        yield connection
        # Roll back first: a test that provoked a database error leaves the
        # transaction aborted, and the cleanup would then fail too, reporting a
        # teardown error that masks the real failure.
        connection.rollback()
        connection.execute("DELETE FROM chunks WHERE act = 'kp-test'")
        connection.commit()


def _as_test_act(chunks: list[Chunk]) -> list[Chunk]:
    for chunk in chunks:
        chunk.act = "kp-test"
    return chunks


def test_schema_is_idempotent(conn):
    apply_schema(conn)
    apply_schema(conn)


def test_load_inserts_rows(conn):
    chunks = _as_test_act([make_chunk("25", "1"), make_chunk("26", "1")])
    load_chunks(conn, chunks, StubEmbedder(), show_progress=False)

    row = conn.execute("SELECT count(*) AS n FROM chunks WHERE act='kp-test'").fetchone()
    assert row["n"] == 2


def test_load_is_idempotent(conn):
    """Re-running after a parser change must update in place, not duplicate.

    A corpus that can only be built once is a corpus nobody re-derives.
    """
    chunks = _as_test_act([make_chunk("25", "1")])
    load_chunks(conn, chunks, StubEmbedder(), show_progress=False)
    load_chunks(conn, chunks, StubEmbedder(), show_progress=False)

    row = conn.execute("SELECT count(*) AS n FROM chunks WHERE act='kp-test'").fetchone()
    assert row["n"] == 1


def test_reload_updates_changed_content(conn):
    chunk = make_chunk("25", "1")
    _as_test_act([chunk])
    load_chunks(conn, [chunk], StubEmbedder(), show_progress=False)

    chunk.content = "Art. 25 § 1. Zupełnie nowa treść przepisu."
    load_chunks(conn, [chunk], StubEmbedder(), show_progress=False)

    row = conn.execute("SELECT content FROM chunks WHERE act='kp-test' AND article='25'").fetchone()
    assert "nowa treść" in row["content"]


def test_parts_of_one_article_coexist(conn):
    """part_index is part of the identity, so split articles must not collide."""
    chunks = _as_test_act([make_chunk("25", "1", 0), make_chunk("25", "1", 1)])
    load_chunks(conn, chunks, StubEmbedder(), show_progress=False)

    row = conn.execute("SELECT count(*) AS n FROM chunks WHERE act='kp-test'").fetchone()
    assert row["n"] == 2


def test_embeddings_are_stored_at_the_expected_width(conn):
    load_chunks(conn, _as_test_act([make_chunk()]), StubEmbedder(), show_progress=False)

    row = conn.execute(
        "SELECT vector_dims(embedding) AS dims FROM chunks WHERE act='kp-test'"
    ).fetchone()
    assert row["dims"] == DIM


def test_dimension_mismatch_is_refused_before_insert(conn):
    """Changing the embedding model is a migration, not a config tweak.

    The check compares the model against the column width read from the database.
    Comparing it against the model's own reported dimension would be tautological
    and could never fail — which is what it did before this test existed.
    """
    with pytest.raises(RuntimeError, match="migration"):
        load_chunks(conn, _as_test_act([make_chunk()]), StubEmbedder(dim=384), show_progress=False)


def test_stale_chunks_from_a_previous_parse_are_pruned(conn):
    """A chunking change that produces fewer pieces must not leave orphans.

    The upsert cannot remove them: with nothing to overwrite them they survive,
    still indexed and still retrievable, carrying text from a corpus version that
    no longer exists. Nothing errors — the stale chunk just competes for a slot.
    """
    split = _as_test_act([make_chunk("25", "1", 0), make_chunk("25", "1", 1)])
    load_chunks(conn, split, StubEmbedder(), show_progress=False)
    assert conn.execute("SELECT count(*) AS n FROM chunks WHERE act='kp-test'").fetchone()["n"] == 2

    # Re-parsed: the article now fits in one chunk.
    whole = _as_test_act([make_chunk("25", "1", 0)])
    load_chunks(conn, whole, StubEmbedder(), show_progress=False)

    rows = conn.execute(
        "SELECT part_index FROM chunks WHERE act='kp-test' ORDER BY part_index"
    ).fetchall()
    assert [r["part_index"] for r in rows] == [0]


def test_prune_can_be_disabled_for_a_partial_load(conn):
    load_chunks(conn, _as_test_act([make_chunk("25")]), StubEmbedder(), show_progress=False)
    load_chunks(
        conn, _as_test_act([make_chunk("26")]), StubEmbedder(), show_progress=False, prune=False
    )
    row = conn.execute("SELECT count(*) AS n FROM chunks WHERE act='kp-test'").fetchone()
    assert row["n"] == 2


def test_prune_leaves_other_acts_untouched(conn):
    """Loading one act must never delete another."""
    load_chunks(conn, _as_test_act([make_chunk("25")]), StubEmbedder(), show_progress=False)
    before = conn.execute("SELECT count(*) AS n FROM chunks WHERE act='kp'").fetchone()["n"]

    load_chunks(conn, _as_test_act([make_chunk("26")]), StubEmbedder(), show_progress=False)

    after = conn.execute("SELECT count(*) AS n FROM chunks WHERE act='kp'").fetchone()["n"]
    assert after == before


def test_column_dimension_matches_the_production_model(conn):
    """vector(1024) is multilingual-e5-large. The two must not drift apart."""
    assert column_dimension(conn) == DIM


def test_tsvector_is_generated_and_lemmatised(conn):
    load_chunks(conn, _as_test_act([make_chunk()]), StubEmbedder(), show_progress=False)

    row = conn.execute("SELECT tsv::text AS tsv FROM chunks WHERE act='kp-test'").fetchone()
    # "pracę" and "umowę" must appear as lemmas, proving the Polish config is applied.
    assert "praca" in row["tsv"]
    assert "umowa" in row["tsv"]


def test_structural_path_is_searchable(conn):
    """An article inherits its chapter's subject even if its text never says it."""
    load_chunks(conn, _as_test_act([make_chunk()]), StubEmbedder(), show_progress=False)

    row = conn.execute(
        """
        SELECT count(*) AS n FROM chunks
        WHERE act='kp-test' AND tsv @@ plainto_tsquery('polish', 'praca zdalna')
        """
    ).fetchone()
    assert row["n"] == 1


def test_corpus_stats_reports_shape(conn):
    load_chunks(conn, _as_test_act([make_chunk("25"), make_chunk("26")]), StubEmbedder(), False)
    stats = corpus_stats(conn)
    assert stats["chunks"] >= 2
    assert stats["missing_embeddings"] == 0


# --- guards that run without a database ---------------------------------------


def test_oversized_chunk_is_refused_rather_than_truncated():
    """Silent truncation would drop the tail of an article from the index."""
    chunk = make_chunk()
    chunk.content = "słowo " * 900
    with pytest.raises(ValueError, match="exceed the model limit"):
        assert_within_model_limit([chunk], lambda t: len(t.split()), limit=512)


def test_search_text_includes_path_and_content():
    chunk = make_chunk()
    text = search_text(chunk)
    assert "Praca zdalna" in text
    assert "okres próbny" in text


def test_search_text_has_no_encoder_prefix():
    """`passage:` is an encoder artefact and must not pollute the lexical index."""
    assert "passage:" not in search_text(make_chunk())
    assert embedding_input(make_chunk()).startswith("passage: ")
