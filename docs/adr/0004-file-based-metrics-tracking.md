# 0004 — Metrics in a committed JSONL, not Weights & Biases or MLflow

**Status:** Accepted

## Context

The project's central claim is quantitative: recall@k on a gold set, F1 on violation detection. The
numbers need to be recorded, compared across runs, and shown to a reader. The conventional answer
is an experiment tracker — W&B or MLflow.

## Decision

Append one JSON object per eval run to `metrics/history.jsonl`, commit it, and generate both the
README tables and the dashboard from it.

## Consequences

- **The numbers are public without a login.** A reader following a link to a W&B project they
  cannot open learns nothing. For a repository whose purpose is to be read, that is disqualifying.
- **A metric change appears as a diff inside the pull request that caused it.** Reviewing a change
  and its effect on retrieval quality in one view is the property that makes the regression gate
  (`evals/gate.py`) possible at all.
- Zero infrastructure: no server, no account, no API key, no second system to keep alive.
- Every row carries `retrieval_config_hash`, the embedding model *and its revision*, and the pinned
  model IDs. A number without its configuration is an anecdote; the hash is what makes it a
  measurement.
- Cost: no run-comparison UI out of the box — hence the generated dashboard. And the file grows
  monotonically, though at a few hundred bytes per run that is irrelevant for years.
- W&B remains a plausible later addition for the CV keyword alone; it would sit alongside this file
  rather than replace it, because the public-readability property is the whole point.

## What would change our mind

Multiple people running evals concurrently, where append-to-a-committed-file becomes a merge
conflict generator rather than a review aid.
