"""Load chunks and their embeddings into Postgres.

Idempotent by design: the loader upserts on the natural key
``(act, article, paragraph, part_index)``, so re-running after a parser change
updates rows in place rather than accumulating duplicates or requiring a manual
truncate. A corpus that can only be built once is a corpus nobody re-derives.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import psycopg
from psycopg.rows import DictRow

from ingestion.chunk import Chunk, assert_within_model_limit, embedding_input, search_text


class SupportsEmbedding(Protocol):
    """What the loader needs from an encoder.

    A protocol rather than the concrete Embedder so tests can substitute a stub:
    loading 2 GB of weights to assert that an upsert deduplicates would make the
    integration suite useless as a fast check.
    """

    @property
    def dimension(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def encode_passages(
        self, texts: list[str], show_progress: bool = False
    ) -> list[list[float]]: ...


def column_dimension(conn: psycopg.Connection[DictRow]) -> int:
    """Width the `embedding` column was declared with.

    Read from the database rather than assumed, because this is the number the
    model must agree with. Comparing the model against its own reported dimension
    is tautological and can never fail.
    """
    row = conn.execute(
        """
        SELECT atttypmod AS dim
        FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("chunks.embedding column not found; has the schema been applied?")
    return int(row["dim"])


UPSERT = """
INSERT INTO chunks (
    act, article, article_display, paragraph, part_index,
    title_path, page_start, repealed, repeal_kind,
    content, search_text, n_tokens, embedding
)
VALUES (
    %(act)s, %(article)s, %(article_display)s, %(paragraph)s, %(part_index)s,
    %(title_path)s, %(page_start)s, %(repealed)s, %(repeal_kind)s,
    %(content)s, %(search_text)s, %(n_tokens)s, %(embedding)s
)
ON CONFLICT (act, article, paragraph, part_index) DO UPDATE SET
    article_display = EXCLUDED.article_display,
    title_path      = EXCLUDED.title_path,
    page_start      = EXCLUDED.page_start,
    repealed        = EXCLUDED.repealed,
    repeal_kind     = EXCLUDED.repeal_kind,
    content         = EXCLUDED.content,
    search_text     = EXCLUDED.search_text,
    n_tokens        = EXCLUDED.n_tokens,
    embedding       = EXCLUDED.embedding
"""


def load_chunks(
    conn: psycopg.Connection[DictRow],
    chunks: Sequence[Chunk],
    embedder: SupportsEmbedding,
    show_progress: bool = True,
    prune: bool = True,
) -> list[str]:
    """Embed and upsert every chunk. Returns report lines.

    `prune` removes rows for these acts that this load did not produce, which is
    correct when loading a whole act and wrong when loading a subset. Pass False
    for a partial load.
    """
    if not chunks:
        return ["no chunks to load"]

    # Before spending minutes encoding, confirm nothing will be truncated.
    assert_within_model_limit(chunks, embedder.count_tokens)

    # Checked against the schema before encoding, so a model swap fails in seconds
    # with a message naming the cause, rather than after a long encode with an
    # opaque type error from Postgres.
    expected = column_dimension(conn)
    if embedder.dimension != expected:
        raise RuntimeError(
            f"the embedding model produces {embedder.dimension}-dimensional vectors "
            f"but chunks.embedding is vector({expected}). Changing the embedding "
            f"model is a migration, not a configuration change — and it invalidates "
            f"every metric recorded under the previous model."
        )

    vectors = embedder.encode_passages(
        [embedding_input(c) for c in chunks], show_progress=show_progress
    )

    if len(vectors) != len(chunks):
        raise RuntimeError(f"embedded {len(vectors)} vectors for {len(chunks)} chunks")

    rows = [
        {
            "act": c.act,
            "article": c.article,
            "article_display": c.article_display,
            "paragraph": c.paragraph,
            "part_index": c.part_index,
            "title_path": c.title_path,
            "page_start": c.page_start,
            "repealed": c.repealed,
            "repeal_kind": c.repeal_kind,
            "content": c.content,
            "search_text": search_text(c),
            "n_tokens": c.n_tokens,
            "embedding": str(vector),
        }
        for c, vector in zip(chunks, vectors, strict=True)
    ]

    with conn.cursor() as cur:
        cur.executemany(UPSERT, rows)

    report = [
        f"loaded {len(rows)} chunks ({expected}-dim embeddings)",
        f"  repealed:  {sum(c.repealed for c in chunks)}",
        f"  max tokens: {max(c.n_tokens for c in chunks)}",
    ]

    if prune:
        removed = prune_stale(conn, chunks)
        if removed:
            report.append(f"  pruned {removed} stale chunk(s) from a previous parse")

    conn.commit()
    return report


def prune_stale(conn: psycopg.Connection[DictRow], chunks: Sequence[Chunk]) -> int:
    """Delete rows for these acts that this load did not produce.

    The upsert alone cannot do this. When a chunking change makes an article
    split into fewer pieces, the surplus rows from the previous parse survive
    with nothing to overwrite them — still indexed, still retrievable, and
    carrying text from a corpus version that no longer exists. Nothing errors;
    the stale chunk simply competes for a slot at k=5.
    """
    acts = sorted({c.act for c in chunks})
    removed = 0
    for act in acts:
        keys = [(c.article, c.paragraph, c.part_index) for c in chunks if c.act == act]
        result = conn.execute(
            """
            DELETE FROM chunks c
            WHERE c.act = %(act)s
              AND NOT EXISTS (
                  SELECT 1
                  FROM unnest(
                      %(articles)s::text[], %(paragraphs)s::text[], %(parts)s::int[]
                  ) AS k(article, paragraph, part_index)
                  WHERE k.article = c.article
                    AND k.paragraph = c.paragraph
                    AND k.part_index = c.part_index
              )
            """,
            {
                "act": act,
                "articles": [k[0] for k in keys],
                "paragraphs": [k[1] for k in keys],
                "parts": [k[2] for k in keys],
            },
        )
        removed += result.rowcount
    return removed


def corpus_stats(conn: psycopg.Connection[DictRow]) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT count(*)                                AS chunks,
               count(DISTINCT article)                 AS articles,
               count(*) FILTER (WHERE embedding IS NULL) AS missing_embeddings,
               count(*) FILTER (WHERE repealed)          AS repealed
        FROM chunks
        """
    ).fetchone()
    return {k: int(v) for k, v in dict(row or {}).items()}
