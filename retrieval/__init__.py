"""Hybrid retrieval over the statute corpus.

Two legs, both served by Postgres:

- **Lexical** — `tsvector` ranking under the Polish Hunspell configuration. Finds
  statutory terms of art and exact citations.
- **Dense** — `pgvector` cosine similarity over multilingual-e5-large. Bridges the
  gap between how people ask ("szef każe mi zostawać po godzinach") and how the
  statute writes ("praca w godzinach nadliczbowych").

Neither is sufficient alone, which is the entire argument for hybrid (ADR 0003).
"""
