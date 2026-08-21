# 0007 — Ingest the law in force, exclude future-dated amendments

**Status:** Accepted

## Context

ISAP consolidated texts do not contain one version of the law. Where an amendment
has been enacted but has not yet taken effect, the document carries both: the text
currently in force wrapped in `[square brackets]`, and the replacement wrapped in
`<angle brackets>`, with a marginal note giving the date. In the Kodeks pracy
snapshot used here, five such blocks take effect on 5 November 2026.

Both versions share an article number. `Art. 94³` appears twice — once as the
mobbing provision in force today, once as its replacement.

## Decision

Ingest the bracketed (in-force) version. Do not ingest the angle-bracketed
(future) version. Report how many were skipped rather than dropping them silently.

## Consequences

- **The corpus answers "what is the law today".** Grounding an answer in a
  provision that takes effect in three months is wrong in the same way as
  grounding it in a repealed one, and more dangerous, because the text reads as
  current and carries no repeal marker to warn a reader.
- The alternative — ingesting both — is not merely redundant. The two versions
  share an article id, so they would collide on the `(act, article, paragraph,
  part_index)` key. Resolving that needs a version discriminator in the primary
  key, in the citation format, and in the gold set. That is a real feature, and
  it is not the one this project is timeboxed to build.
- Cost: a question about an imminent change gets the current answer with no
  indication that the law is about to move. For an auditor of contracts signed
  today that is the correct behaviour; for advice about a contract starting next
  year it is not. Recorded in `LIMITATIONS.md` rather than hidden.
- The brackets are stripped from the stored text. They delimit the scope of a
  change, not the statute's wording, and would otherwise be quoted back as if
  they were part of the law.

## What would change our mind

A question in the gold set that can only be answered correctly by knowing about a
pending change — which would mean the product genuinely needs temporal
versioning, and the schema should carry an `in_force_from` column rather than a
filter.
