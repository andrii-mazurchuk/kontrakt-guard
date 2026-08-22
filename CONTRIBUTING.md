# Working in this repository

## Setup

```bash
uv sync                  # creates .venv on Python 3.13, installs from uv.lock
uv run pre-commit install
cp .env.example .env     # then fill in ANTHROPIC_API_KEY and POSTGRES_PASSWORD
```

## The database

```bash
docker compose up -d --build --wait db
```

Built rather than pulled. Stock PostgreSQL ships 29 text-search configurations
and **Polish is not among them** — Snowball has no Polish stemmer. `docker/db/Dockerfile`
adds a Hunspell Polish dictionary and `docker/initdb/02-polish-fts.sql` assembles a
`polish` configuration from it. Without that, the lexical half of hybrid retrieval
does no lemmatisation, and in a language this heavily inflected a question about
`wynagrodzenia` never matches a statute saying `wynagrodzenie`.

**The database is published on host port `55432`, not `5432`.** A locally installed
PostgreSQL is common and binds 5432 first, and when it wins the race every
connection lands on the wrong server — reported as `password authentication failed`
rather than as a port conflict. Set `POSTGRES_PORT=55432` in `.env`; the container
port is unchanged, so the `api` service still talks to 5432 inside the network.

## The commands CI runs

These four are the whole gate. `.pre-commit-config.yaml` and `.github/workflows/ci.yml` invoke
them identically on purpose — a hook that disagrees with CI is a hook that lies.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src evals tests
uv run pytest
```

Two more guard the metrics:

```bash
uv run python -m evals.render_readme --check   # README tables match the recorded evidence
uv run python -m evals.gate --layer retrieval  # no regression past tolerance
```

### What does not run on a pull request

The retrieval eval itself. It is free of API cost — embeddings and SQL, no LLM call — but it fetches
the corpus, embeds 543 chunks with a 2.2 GB model and scores 97 questions, which is 8–12 minutes to
defend a number that only moves when retrieval changes. `evals.yml` runs it nightly and on demand
(`gh workflow run evals.yml`); run it locally after any retrieval change:

```bash
docker compose up -d --build --wait db
uv run python -m ingestion.build --load        # ~10 min, embeds the corpus
uv run python -m evals.retrieval_eval          # add --record to append a row
```

The nightly job appends its row on the runner and does not commit it: `main` is protected, and a
record of results that CI can write to itself is a record nobody has read. The row is uploaded as an
artifact so a genuine improvement is promoted through a pull request like anything else.

A **checksum mismatch in the fetch step is not a broken build — it means the law changed.** ISAP
regenerates the consolidated PDF whenever an amendment lands, so re-pinning is deliberate, and the
new corpus is not comparable to metrics recorded under the old one.

### Test selection

The default `pytest` run is free and fast: markers `integration` (needs Postgres) and `llm` (makes
billable Claude calls) are excluded. Opt in explicitly.

```bash
uv run pytest -m integration    # requires docker compose up db
uv run pytest -m llm            # costs money
uv run pytest -m ""             # everything
uv run pytest tests/test_gate.py::test_regression_beyond_tolerance_fails
```

## Branches

`main` is protected and always green. Everything else is short-lived — hours, not days.

| Prefix | For |
|---|---|
| `feat/` | new capability |
| `fix/` | a defect |
| `eval/` | gold set, generators, metric changes |
| `docs/` | README, ADRs, limitations |
| `chore/` | tooling, CI, dependencies |

```bash
git switch -c feat/hybrid-retrieval
# ... work, commit ...
git push -u origin feat/hybrid-retrieval
gh pr create --fill
```

Merge by **squash** only — the platform enforces it; merge and rebase commits are disabled, and
the branch is deleted automatically on merge.

### What protection actually enforces

`main` requires a pull request, both CI checks passing, the branch up to date with `main`, and
linear history. Force-pushes and deletions are refused.

**Administrator enforcement is on.** This is deliberate: with it off, GitHub lets a repository
admin push straight to `main` and merely notes "bypassed rule violations" afterwards, which makes
the whole arrangement advisory. Discipline you bypass by habit is not discipline.

The consequence is that a broken CI configuration can lock the repository — you cannot push the
fix directly. The escape hatch is deliberate rather than habitual:

```bash
gh api -X DELETE repos/andrii-mazurchuk/kontrakt-guard/branches/main/protection/enforce_admins
# push the fix
gh api -X POST   repos/andrii-mazurchuk/kontrakt-guard/branches/main/protection/enforce_admins
```

Re-enable it in the same sitting. An escape hatch left open is just an open door.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): subject`. The subject
is imperative and lowercase. Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `eval`.

This is not decoration — `release.yml` turns these subjects into release notes.

The body explains **why**. The diff already shows what changed.

## Evals

Layer 1 (retrieval) is deterministic and near-free, so it gates every pull request. Layer 2 (audit)
is LLM-heavy and runs nightly or on request.

Rules that are not negotiable:

- **Never weaken an eval to make a number look better.** Fix the system or report the honest
  number. A gold question that turns out to be wrong gets corrected in its own commit, with the
  reason in the message, never bundled into a change that also moves the metric.
- **Never hand-edit the README metric tables.** They are generated from `metrics/history.jsonl`.
- **Never float a model ID or an embedding revision.** A metric is only comparable across runs if
  the configuration that produced it is pinned. See `Settings.retrieval_config_hash`.

## Decisions

Anything that would be re-litigated later, or that a reader would reasonably question, gets an ADR
in [`docs/adr/`](docs/adr/). Short: context, decision, consequences, what would change our mind.
