# Working in this repository

## Setup

```bash
uv sync                  # creates .venv on Python 3.13, installs from uv.lock
uv run pre-commit install
cp .env.example .env     # then fill in ANTHROPIC_API_KEY
```

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

Merge by **squash** only, so `main` reads as one commit per unit of work.

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
