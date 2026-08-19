# 0005 — Publish container images now, defer the cloud target

**Status:** Accepted

## Context

The project brief lists cloud deployment (AWS Bedrock + a deployed container) as the first v2 item,
explicitly outside the build timebox. It is also one of the three market gaps the project exists to
address, so it will happen — just not now.

## Decision

Build the artifact half of the pipeline immediately: a multi-stage `Dockerfile`, a
`docker-compose.yml` that brings up the whole stack in one command, an image built on every pull
request, and a versioned image published to GHCR on every tag. Choose no cloud target.

## Consequences

- Building the image on every pull request stops the `Dockerfile` from rotting silently between
  releases — the standard failure where the container is broken for two months and nobody notices
  because it is only built at release time.
- When v2 arrives there is already a versioned, public, pullable image, so deployment is a matter
  of pointing a runtime at it. That is what makes the brief's one-day estimate real rather than
  optimistic.
- The Anthropic client is kept behind a thin internal seam so that swapping it for Bedrock is a
  localised change and not a search-and-replace across the graphs.
- Cost: `release.yml` cannot be fully exercised until the first tag, so it carries more unverified
  surface than the rest of CI.

## What would change our mind

Nothing within the timebox. This is a sequencing decision, not a technical one.
