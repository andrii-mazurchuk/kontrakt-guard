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

### The merged system

Barely at all — and this is the result worth keeping.

| configuration | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| `ts_rank_cd`, weighted α=0.7 *(previous default)* | **77.1%** | 84.8% | 92.0% | **0.736** |
| **BM25, weighted α=0.8** *(new default)* | 75.5% | **85.3%** | **93.0%** | 0.712 |
| BM25, RRF α=0.75 | 78.6% | 85.8% | 89.9% | 0.697 |
| BM25, RRF α=0.5 *(classic, unweighted)* | 74.7% | 84.3% | — | 0.698 |
| dense only | 72.9% | 80.2% | 93.0% | 0.700 |

A 22.7-point improvement to the leg bought **half a point of recall@5**, and MRR and recall@3 both
went *down*. Adopted regardless, for reasons that are not the headline number: the component is
correct rather than defective, it is three times faster, and the lexical leg is what the citation
lookup and the coming LLM grading node lean on directly rather than through the merge.

The MRR regression is real and unexplained. The plausible reading is that `ts_rank_cd`'s cover
density is a **proximity** signal — it rewards a chunk for mentioning the query's terms close
together — and BM25 discards proximity entirely in exchange for rarity. Proximity appears to be a
decent guide to which chunk belongs at rank 1, even while being a poor guide to which chunks belong
in the top ten at all. Recorded as an open question, not as a settled explanation.

α was re-swept because the previous optimum had been tuned against a far weaker leg. Recall@5
across 0.6/0.7/0.8/0.85 is 84.8/84.3/85.3/83.8 — a plateau about one question wide on a 97-question
set, so no value inside it is meaningfully better than its neighbours. 0.8 is taken because it is
the joint best across recall@5, recall@10 and MRR rather than the winner on any single one.

RRF gained a weight in the same change (α=0.5 reproduces the classic unweighted ordering exactly,
so the figure already recorded under it stays comparable). It wins at k=3 and k=5 and loses at k=10.
`retrieval_top_k` is 10 and that pool is what the grading node will receive, so weighted fusion
keeps the default.



## What this did not fix

Pool recall — the fraction of required articles reaching the candidate pool at any rank — is
**96.9% under both rankings, identical**. BM25 reorders the pool; it does not enlarge it.

That number is the ceiling on everything downstream. No fusion weight, no `k`, no reranker and no
LLM grading node can surface an article that neither leg proposed. With recall@5 at ~86% against a
96.9% ceiling, roughly **eleven points are being lost to ranking rather than to retrieval** — which
says the next investment belongs in ordering the pool, not in widening it. That is the reranker
deferred in ADR 0003, and it now has a measured budget rather than an assumption behind it.
