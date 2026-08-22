# 0003 — Hybrid BM25 + dense retrieval, no reranker in v1

**Status:** Accepted

## Context

Polish legal questions arrive in two registers. Users write colloquially ("szef każe mi zostawać po
godzinach"); the statute uses formal terminology ("praca w godzinach nadliczbowych"). Dense
retrieval bridges the vocabulary gap. Lexical retrieval is what reliably finds an exact article
reference or a statutory term of art. Neither alone is sufficient.

A cross-encoder reranker (for example BGE-reranker) would plausibly improve precision further.

## Decision

Run both legs and merge the candidate lists, then grade the survivors with a cheap LLM pass. No
reranker in v1.

## Consequences

- The LLM relevance-grading node already performs the reranker's job — discarding plausible but
  irrelevant candidates — at the cost of latency rather than a second model in the stack.
- Keeping the reranker out means the v2 line item can be presented with **before/after numbers**
  from the same eval harness. A reranker added at the start is an unmeasured assumption; added
  later it is a measured improvement. That is worth more than the precision points.
- Cost: `hybrid_alpha`, the dense/sparse merge weight, becomes a tuned hyperparameter. It is part
  of `retrieval_config_hash`, so a metric change caused by tuning it is never mistaken for a
  change caused by something else.

## Measured, 2026-08-21 — the fusion method mattered more than the legs

The first Layer-1 run over 97 gold questions nearly refuted this decision:

| Configuration | recall@5 | MRR |
|---|---|---|
| Lexical only | 46.9% | 0.410 |
| Hybrid, RRF | 75.5% | 0.627 |
| Dense only | 80.2% | 0.699 |
| Hybrid, weighted α=0.7 | **84.8%** | **0.735** |

**Hybrid retrieval with RRF was worse than dense retrieval alone.** RRF combines ranks with equal
weight, and on this corpus the lexical leg is far weaker than the dense one, so its noise was being
promoted into the top ranks. The claim "neither is sufficient alone" survived; the assumption that
any reasonable fusion realises the benefit did not.

Weighting the dense leg at 0.7 recovers the gain and beats dense alone by 4.6 points, so the default
is now weighted fusion rather than RRF. `fusion` and `hybrid_alpha` are both part of
`retrieval_config_hash`, so a metric recorded under one cannot be confused with the other.

The lexical leg's weakness is itself a finding worth chasing: Postgres `ts_rank_cd` has no IDF term,
so common legal vocabulary — *pracownik*, *pracodawca*, *umowa* — contributes as much to the score as
a distinguishing term does. That, rather than a reranker, now looks like the highest-value next change.

**Followed up 2026-08-22 in [ADR 0008](0008-bm25-over-ts-rank-cd.md), and it was not.** BM25 took the
lexical leg from 46.9% to 69.6% recall@5 and moved the merged system by half a point. The
prediction — that fixing the weaker leg was worth more than a reranker — was wrong, and the
measurement that refuted it also explains why: candidate-pool recall is 96.9% under both rankings, so
both legs were already proposing the right articles and the loss is in the ordering. The default
merge weight moved from α=0.7 to α=0.8 in the same change, re-swept because the old optimum had been
fitted against a much weaker leg.

**Caveat:** α was tuned on the same 97 questions the headline number is reported over. With one
parameter and no held-out split, some of the 4.6-point gain is fitted to this set. Recorded in
`LIMITATIONS.md` rather than left for a reader to infer.

## What would change our mind

Layer-1 recall@10 being materially higher than recall@3 — that gap is exactly the space a reranker
recovers. It is (93.0% against 75.5%), and candidate-pool recall of 96.9% puts a firm number on the
headroom: about eleven points sit between what the legs already propose and what survives to k=5.
The lexical leg has since been given an IDF-aware ranking, so that caveat is discharged and the case
for a reranker is now the strongest remaining one.
