# 0008 — BM25 over `ts_rank_cd` for the lexical leg

**Status:** Accepted

## Context

The lexical leg was scored by Postgres's `ts_rank_cd`. Neither `ts_rank` nor `ts_rank_cd` contains
an **inverse document frequency term**. They measure how often the query's lexemes occur in a
document and, for `_cd`, how tightly they cluster — but never how rare those lexemes are in the
corpus. Postgres does not know: a `tsvector` carries no corpus-wide statistics, and the ranking
functions see one document at a time.

On a corpus of employment law this is close to fatal, because the words a question about employment
law contains are exactly the words every article contains:

| lexeme | documents containing it | share of the 543 chunks |
|---|---|---|
| `art`, `dział`, `dziać` | 543 | **100%** |
| `rozdział` | 444 | 82% |
| `praca` | 425 | 78% |
| `pracownik` | 403 | 74% |
| `pracodawca` | 299 | 55% |

A term present in every single chunk distinguishes nothing, and `ts_rank_cd` was weighting it as
strongly as the one term the question actually turned on. The leg was substantially sorting by how
often an article says "employee".

## Decision

Compute **BM25** against a materialised inverted index instead:

```
score(d, q) = Σ  IDF(t) · ( tf · (k1 + 1) ) / ( tf + k1 · (1 - b + b · |d| / avgdl) )
              t∈q
```

with `IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`, `k1 = 1.2`, `b = 0.75`.

BM25 needs per-`(chunk, lexeme)` term frequency, per-chunk length, and corpus-wide document
frequency. None is reachable from a `tsvector` without unnesting it, and unnesting at query time is
O(corpus) per search. Two materialised views hold them: `chunk_terms` (the inverted index) and
`corpus_stats` (`n_docs`, `avgdl`). The loader refreshes both **inside the same transaction** as the
upsert and the prune, and `lexical_index_is_stale` re-checks before any eval run.

`ts_rank_cd` is kept behind `lexical_ranking` so the earlier measurement stays reproducible. It is
not a supported alternative.

## Consequences

- No extension. `pg_search`/ParadeDB provides BM25 natively but is not available in the `pgvector`
  image, and adding a second extension to get one ranking function is a larger commitment than
  forty lines of SQL.
- **A materialised view is a snapshot, and a stale one still returns plausible results** — the
  failure mode this project keeps meeting, where nothing errors and only the metric moves. Hence
  the same-transaction refresh, the staleness check before every eval, and the refresh in the
  integration-test teardown.
- `lexical_ranking`, `bm25_k1` and `bm25_b` join `retrieval_config_hash`, so a metric produced under
  one ranking can never be confused with a metric produced under the other.

## Measured, 2026-08-22

Same corpus, same 97-question gold set, same embeddings. The control reproduced the previously
recorded `ts_rank_cd` figures exactly, so the comparison is like-for-like.

### The leg in isolation

| lexical ranking | recall@3 | recall@5 | recall@10 | MRR | duration |
|---|---|---|---|---|---|
| `ts_rank_cd` | 42.8% | 46.9% | 58.2% | 0.410 | 14.0 s |
| **BM25** | **60.8%** | **69.6%** | **82.5%** | **0.610** | **4.8 s** |

**+22.7 points of recall@5, and three times faster** — the index is scanned for the handful of
lexemes in the question rather than every row that matches being ranked.

<!-- ADR-0008-FUSION -->

## What this did not fix

Pool recall — the fraction of required articles reaching the candidate pool at any rank — is
**96.9% under both rankings, identical**. BM25 reorders the pool; it does not enlarge it.

That number is the ceiling on everything downstream. No fusion weight, no `k`, no reranker and no
LLM grading node can surface an article that neither leg proposed. With recall@5 at ~86% against a
96.9% ceiling, roughly **eleven points are being lost to ranking rather than to retrieval** — which
says the next investment belongs in ordering the pool, not in widening it. That is the reranker
deferred in ADR 0003, and it now has a measured budget rather than an assumption behind it.
