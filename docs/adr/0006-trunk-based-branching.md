# 0006 — Trunk-based branching with mandatory pull requests

**Status:** Accepted

## Context

The repository has exactly one contributor. The conventional choices are GitFlow (`develop`,
`release/*`, `hotfix/*`), GitHub Flow (branch → PR → main), or committing straight to `main`.

## Decision

Trunk-based development. `main` is protected and always green. Work happens on short-lived
`feat/ fix/ eval/ docs/ chore/` branches merged by squash. Commit subjects follow Conventional
Commits.

## Consequences

- GitFlow is rejected outright: `develop` and `release/*` exist to coordinate parallel teams and
  batch releases. Neither pressure exists here, and the ceremony would read as cargo-culting to
  precisely the reader this repository is written for.
- Direct commits to `main` are rejected for a less obvious reason: **the pull request is where CI
  results and the eval metric delta render.** Someone reading the PR history sees retrieval
  quality move, change by change. That history is a deliverable, not overhead.
- Squash merge keeps `main` at one readable commit per unit of work, so `git log` stays a narrative
  rather than a transcript of intermediate saves.
- Conventional Commits are what `release.yml --generate-notes` turns into a changelog, so the
  convention pays for itself rather than being discipline for its own sake.
- Cost: a solo developer waiting on their own CI. Mitigated by keeping `ci.yml` free and fast, and
  by keeping billable eval runs in a separate workflow.

## What would change our mind

Nothing at this scale. A second contributor would reinforce the decision rather than weaken it.
