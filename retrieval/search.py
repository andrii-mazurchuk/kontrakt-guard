"""Lexical, dense and hybrid search over the corpus.

Fusion
------
The two legs produce scores on incomparable scales: `ts_rank_cd` is an unbounded
relevance figure, cosine similarity lives in [-1, 1]. Combining them by weighted
sum therefore requires normalising first, and normalisation over a short candidate
list is unstable — one outlier rescales everything.

Reciprocal Rank Fusion sidesteps this by combining *ranks* rather than scores, and
is the default here. The weighted variant is kept because which one wins on Polish
legal questions is an empirical question, and this repository answers those with
recall@k rather than with argument.
"""

from __future__ import annotations

from typing import Literal

import psycopg
from psycopg.rows import DictRow
from pydantic import BaseModel, Field

from ingestion.db import TS_CONFIG

# Standard RRF constant. Damps the influence of the top ranks just enough that a
# single leg cannot dominate the merged ordering.
RRF_K = 60

Leg = Literal["lexical", "dense", "hybrid"]
Ranking = Literal["bm25", "ts_rank_cd"]


class Hit(BaseModel):
    """One retrieved chunk, with enough provenance to cite it."""

    chunk_id: int
    act: str
    article: str
    article_display: str
    paragraph: str = ""
    title_path: list[str] = Field(default_factory=list)
    content: str
    repealed: bool = False
    repeal_kind: str = ""

    score: float = 0.0
    lexical_rank: int | None = None
    dense_rank: int | None = None

    @property
    def citation(self) -> str:
        if self.paragraph:
            return f"{self.article_display} § {self.paragraph.replace('^', '')}"
        return self.article_display


_SELECT = """
    id, act, article, article_display, paragraph, title_path,
    content, repealed, repeal_kind
"""


def _to_hit(row: DictRow, score: float) -> Hit:
    return Hit(
        chunk_id=row["id"],
        act=row["act"],
        article=row["article"],
        article_display=row["article_display"],
        paragraph=row["paragraph"],
        title_path=list(row["title_path"] or []),
        content=row["content"],
        repealed=row["repealed"],
        repeal_kind=row["repeal_kind"],
        score=score,
    )


# Lemmas are OR-ed, not AND-ed, and the ranking decides.
#
# `plainto_tsquery` and `websearch_to_tsquery` both combine terms with AND, which
# is right for a search box and wrong for retrieval. A natural-language question —
# "czy 6-miesięczny okres próbny jest zgodny z prawem?" — demands that one chunk
# contain every lemma at once, and no statute article does. The lexical leg then
# returns nothing at all and hybrid retrieval silently degrades to dense-only,
# with no error to show for it.
#
# Lemmas are taken from to_tsvector rather than split from the raw string, so
# punctuation and stopwords are already gone and nothing can break to_tsquery.
_LEXICAL_SQL = f"""
WITH lexemes AS (
    SELECT string_agg(quote_literal(lexeme), ' | ') AS expr
    FROM unnest(to_tsvector(%(cfg)s, %(q)s))
),
q AS (
    SELECT CASE
             WHEN expr IS NULL OR expr = '' THEN NULL
             ELSE to_tsquery(%(cfg)s, expr)
           END AS query
    FROM lexemes
)
SELECT {_SELECT}, ts_rank_cd(tsv, q.query) AS rank
FROM chunks, q
WHERE q.query IS NOT NULL
  AND tsv @@ q.query
  AND (%(include_repealed)s OR NOT repealed)
ORDER BY rank DESC
LIMIT %(limit)s
"""


# BM25, computed against the materialised inverted index rather than by
# `ts_rank_cd`.
#
# The problem being fixed: Postgres's ranking functions have no inverse document
# frequency term. `ts_rank_cd` measures how densely the matched lexemes cluster
# within a document and how often they occur, but it never asks how *rare* they
# are. Every question in the gold set contains 'pracownik' or 'praca'; those
# lexemes sit in three quarters of the corpus and in 'art'/'dział' the figure is
# 543 chunks out of 543. Ranking treated them as evidence, so the lexical leg was
# largely sorting by how often an article says "employee".
#
#     score(d, q) = Σ  IDF(t) · ( tf · (k1 + 1) )
#                  t∈q          ───────────────────────────────────
#                               tf + k1 · (1 - b + b · |d| / avgdl)
#
# with the Robertson/Sparck-Jones IDF, `ln(1 + (N - df + 0.5)/(df + 0.5))`. The
# +1 inside the logarithm is what keeps it non-negative: the raw form goes
# negative for a term in more than half the corpus, which would let a common word
# actively push a matching chunk *down* the ranking.
#
# Lexemes come from `to_tsvector`, so the query is analysed by the same Polish
# Hunspell configuration that built the index — a lemma matched against a raw
# token would silently match nothing.
_BM25_SQL = f"""
WITH q AS (
    SELECT DISTINCT lexeme FROM unnest(to_tsvector(%(cfg)s, %(q)s))
),
stats AS (
    SELECT n_docs, avgdl FROM corpus_stats
),
df AS (
    SELECT ct.lexeme, count(*)::float8 AS df
    FROM chunk_terms ct
    JOIN q USING (lexeme)
    GROUP BY ct.lexeme
),
scored AS (
    SELECT
        ct.chunk_id,
        sum(
            ln(1 + (s.n_docs - df.df + 0.5) / (df.df + 0.5))
            * (ct.tf * (%(k1)s + 1))
            / (ct.tf + %(k1)s * (1 - %(b)s + %(b)s * ct.doc_len / s.avgdl))
        ) AS score
    FROM chunk_terms ct
    JOIN df USING (lexeme)
    CROSS JOIN stats s
    GROUP BY ct.chunk_id
)
SELECT {_SELECT}, scored.score AS rank
FROM chunks
JOIN scored ON scored.chunk_id = chunks.id
WHERE (%(include_repealed)s OR NOT repealed)
ORDER BY rank DESC
LIMIT %(limit)s
"""


def lexical_search(
    conn: psycopg.Connection[DictRow],
    question: str,
    limit: int = 25,
    include_repealed: bool = False,
    ranking: Ranking = "bm25",
    k1: float = 1.2,
    b: float = 0.75,
) -> list[Hit]:
    """Rank by Polish full-text relevance, OR-ing the question's lemmas.

    `ranking` selects between BM25 and Postgres's built-in `ts_rank_cd`. The
    latter is kept so the earlier measurement stays reproducible, not because it
    is a supported alternative.
    """
    params = {
        "cfg": TS_CONFIG,
        "q": question,
        "limit": limit,
        "include_repealed": include_repealed,
    }
    if ranking == "bm25":
        params |= {"k1": k1, "b": b}
    rows = conn.execute(_BM25_SQL if ranking == "bm25" else _LEXICAL_SQL, params).fetchall()
    return [_to_hit(row, float(row["rank"])) for row in rows]


def dense_search(
    conn: psycopg.Connection[DictRow],
    query_vector: list[float],
    limit: int = 25,
    include_repealed: bool = False,
) -> list[Hit]:
    """Rank by cosine similarity in pgvector.

    `<=>` is cosine distance, so smaller is closer; the score is inverted to keep
    "higher is better" consistent across both legs.
    """
    rows = conn.execute(
        f"""
        SELECT {_SELECT}, 1 - (embedding <=> %(v)s::vector) AS similarity
        FROM chunks
        WHERE embedding IS NOT NULL
          AND (%(include_repealed)s OR NOT repealed)
        ORDER BY embedding <=> %(v)s::vector
        LIMIT %(limit)s
        """,
        {"v": str(query_vector), "limit": limit, "include_repealed": include_repealed},
    ).fetchall()
    return [_to_hit(row, float(row["similarity"])) for row in rows]


def _reciprocal_rank_fusion(
    lexical: list[Hit], dense: list[Hit], alpha: float = 0.5, rrf_k: int = RRF_K
) -> dict[int, float]:
    """Combine ranks rather than scores. `alpha` is the dense weight.

    Classic RRF weights both legs equally, which is `alpha = 0.5` — the weights
    scale every score by a constant there, so the ordering is identical to the
    unweighted formula and the number recorded under it stays comparable.

    Weighting matters because the two legs are not equally trustworthy, and how
    unequal they are is a property of the corpus rather than of the method.
    """
    scores: dict[int, float] = {}
    for weight, leg in ((1.0 - alpha, lexical), (alpha, dense)):
        for rank, hit in enumerate(leg, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (rrf_k + rank)
    return scores


def _weighted_fusion(lexical: list[Hit], dense: list[Hit], alpha: float) -> dict[int, float]:
    """Min-max normalise each leg, then blend. `alpha` is the dense weight."""

    def normalised(hits: list[Hit]) -> dict[int, float]:
        if not hits:
            return {}
        values = [h.score for h in hits]
        low, high = min(values), max(values)
        span = high - low
        if span == 0:
            return {h.chunk_id: 1.0 for h in hits}
        return {h.chunk_id: (h.score - low) / span for h in hits}

    lex, den = normalised(lexical), normalised(dense)
    scores: dict[int, float] = {}
    for chunk_id in set(lex) | set(den):
        scores[chunk_id] = alpha * den.get(chunk_id, 0.0) + (1 - alpha) * lex.get(chunk_id, 0.0)
    return scores


def merge(
    lexical: list[Hit],
    dense: list[Hit],
    k: int,
    fusion: Literal["rrf", "weighted"] = "rrf",
    alpha: float = 0.5,
) -> list[Hit]:
    """Combine the two candidate lists into a single ranking of length k."""
    scores = (
        _reciprocal_rank_fusion(lexical, dense, alpha)
        if fusion == "rrf"
        else _weighted_fusion(lexical, dense, alpha)
    )

    lexical_ranks = {h.chunk_id: i for i, h in enumerate(lexical, start=1)}
    dense_ranks = {h.chunk_id: i for i, h in enumerate(dense, start=1)}

    by_id: dict[int, Hit] = {h.chunk_id: h for h in dense}
    by_id.update({h.chunk_id: h for h in lexical})

    merged: list[Hit] = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        hit = by_id[chunk_id].model_copy()
        hit.score = score
        # Which leg found it is diagnostic: a corpus where every hit comes from
        # one leg means hybrid retrieval is not actually hybrid.
        hit.lexical_rank = lexical_ranks.get(chunk_id)
        hit.dense_rank = dense_ranks.get(chunk_id)
        merged.append(hit)

    return merged[:k]
