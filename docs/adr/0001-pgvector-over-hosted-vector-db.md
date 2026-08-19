# 0001 — Postgres + pgvector as the only vector store

**Status:** Accepted

## Context

Retrieval needs both a dense (embedding similarity) and a sparse (keyword) leg. The obvious
alternative was a dedicated vector database — Qdrant, Weaviate, Pinecone — paired with a separate
keyword index.

## Decision

Use Postgres with the `pgvector` extension as the single store, and Postgres full-text search with
the Polish configuration for the sparse leg.

## Consequences

- **One system holds both legs of hybrid retrieval.** The merge happens over two indexes in the
  same database, against the same rows, in one query. A separate vector DB would mean keeping two
  stores consistent and reconciling two sets of IDs at merge time.
- Chunk metadata (`act`, `article`, `paragraph`) lives in ordinary relational columns, so
  filtering retrieval by act is a `WHERE` clause rather than a metadata-filter dialect.
- The eval harness can compute recall@k with a SQL join against ground-truth article IDs.
- Cost: Postgres full-text search is not literally BM25. It is a comparable lexical ranking
  (`ts_rank_cd`) and is treated as the sparse leg throughout; the README says so rather than
  claiming BM25 it does not implement.
- At this corpus size (a few thousand articles) a dedicated vector DB would buy nothing. `pgvector`
  with an HNSW index is far past sufficient.

## What would change our mind

Corpus growth past roughly a million chunks, or a latency requirement that `pgvector` HNSW cannot
meet under concurrency.
