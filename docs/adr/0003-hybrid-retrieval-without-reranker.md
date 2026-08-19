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

## What would change our mind

Layer-1 recall@10 being materially higher than recall@3 — that gap is exactly the space a reranker
recovers, and would make it the highest-value next change.
