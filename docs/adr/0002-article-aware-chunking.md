# 0002 — Chunk by legal article, not by token window

**Status:** Accepted

## Context

The default RAG recipe splits documents into fixed token windows with an overlap. Legal text has a
structure that predates the technique by about two thousand years: the article, subdivided into
paragraphs (`§`), points and letters.

## Decision

Chunk on the article boundary. Split an over-long article at its paragraph boundaries rather than
mid-sentence. Store `{act, article, paragraph, title_path}` as columns on every chunk.

## Consequences

- **A retrieved chunk is a citable unit.** "Art. 25¹ § 1 Kodeksu pracy" is what a citation must
  look like; a token window spanning the tail of one article and the head of the next cannot be
  cited without lying about where it starts.
- **The metadata is the eval ground truth.** Recall@k is computed on article IDs, which requires
  that a chunk map to exactly one article. Token windows would make the ground-truth key ambiguous
  and the metric unmeasurable — this is the decisive reason.
- Legal articles are semantically self-contained by drafting convention, so the usual argument for
  overlap (a concept split across a boundary) mostly does not apply.
- Cost: the parser must understand ISAP's document structure, including the amendment numbering
  (`25¹`, `25²`) that Polish legislation uses when inserting articles. This is real work and the
  main risk in the ingestion layer.

## What would change our mind

Evidence from the Layer-1 evals that questions needing cross-article context systematically fail —
which would argue for a parent-document retrieval strategy layered on top, not for abandoning
article boundaries.
