# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kontrakt-Guard — Polish employment contract auditor. Upload a contract (umowa o pracę / zlecenie / B2B) → clause segmentation → per-clause retrieval of relevant law → structured verdict (`compliant / risky / illegal`) with article citations. Byproduct: a grounded legal Q&A endpoint. Full specification lives in `kontrakt_guard_project_brief.md` — **read it before any non-trivial work; it is the source of truth for scope decisions.**

This is a portfolio project with a hard 5–6 day timebox. Resist all scope growth. The v2 list in the brief (§10) is explicitly NOT to be built now (no cloud deployment, no reranker, no UI, no case-law corpus).

## Non-negotiable technical decisions (from the brief)

- **Python** — approved exception to the global TypeScript preference.
- **LangGraph** for orchestration — hard requirement, checkbox #1 the project exists to earn.
- **Postgres + pgvector** as the only vector store (also provides BM25 via Polish full-text config). No hosted vector DBs.
- **Embeddings:** free local multilingual model (`multilingual-e5-large` or `BGE-M3` via sentence-transformers). No Anthropic embeddings API exists.
- **Generation:** Claude API — Haiku 4.5 (`claude-haiku-4-5-20251001`) for high-volume cheap steps (query rewrite, chunk grading, segmentation); Sonnet for verdicts, final answers, faithfulness judging.
- **Chunking is article-aware**, never naive token splitting. Chunk = article (split long ones by paragraph), with metadata `{act, article, paragraph, title_path}` — this metadata is the ground-truth key for evals.
- **Corpus:** consolidated acts from ISAP (isap.sejm.gov.pl); Kodeks pracy is the core. Statutory numbers (2026 minimum wage, probation limits) come from the corpus, never from LLM memory.

## Architecture (big picture)

One shared engine underlies everything: *given a legal question, return the exact articles that answer it and an answer grounded in them.* The auditor is that engine called in a loop over contract clauses.

- `ingestion/` — ISAP fetch, article-aware parser, embedder, pgvector loader (offline, run once)
- `retrieval/` — hybrid search (BM25 + vector), merge, LLM relevance grading
- `graphs/` — LangGraph flows: `qa_flow.py` (understand → retrieve → grade → answer/refuse) and `audit_flow.py` (ingest → segment → per-clause QA loop → verdict → aggregate). Refusal-when-ungrounded is a required, demonstrable conditional edge, not an error case.
- `evals/` — `gold_qa.jsonl` (PIP-derived questions with ground-truth article IDs), `contract_generator.py` (plants known violations into legit templates), `labeled_contracts/`, `metrics.py`
- `api/` — FastAPI: `POST /audit` (headline), `POST /ask` (byproduct)

All LLM steps that feed downstream logic (segmentation, verdicts) use schema-enforced structured outputs. Every user-visible answer/verdict carries `{act, article}` citations and a not-legal-advice disclaimer.

## Evaluation is a first-class deliverable

Two independent layers so failures are attributable to a layer:
1. **Retrieval:** recall@k (k=3,5,10) on article IDs against the PIP gold set; secondary LLM-as-judge faithfulness.
2. **Audit:** precision/recall/F1 on violation detection over the synthetic labeled contract set.

Both metrics tables go in the README — the single highest-value artifact of the project. Never weaken or narrow an eval to make numbers look better. If a CV-style claim can't end with a real number, the eval isn't done.

## Commands

```bash
uv sync                                        # venv on Python 3.13 from uv.lock
uv run pre-commit install
uv run pytest                                  # default run excludes integration + llm markers
uv run pytest -m integration                   # needs: docker compose up db
uv run pytest -m llm                           # billable
uv run pytest tests/test_gate.py::test_name    # single test
uv run ruff format --check . && uv run ruff check . && uv run mypy src evals tests
uv run python -m evals.render_readme --check   # README tables match recorded evidence
uv run python -m evals.gate --layer retrieval  # regression gate
docker compose up                              # postgres+pgvector + API
```

`.pre-commit-config.yaml` and `.github/workflows/ci.yml` run these identically on purpose. **If you change one, change both** — a hook that disagrees with CI is a hook that lies.

Use `git commit -F <file>` for multi-line messages on this machine; PowerShell mangles here-strings containing double quotes when passing them to native commands.

## Working agreement (overrides the global planning rules for this repo)

Agreed 2026-08-20. The global `CLAUDE.md` requires a written plan and explicit sign-off before any multi-file task and between phases. **That is suspended here**, deliberately, because approving a plan one does not yet understand teaches nothing and merely slows the build.

Instead: **ship in milestones, teach in the pull request.** One PR per working capability, each ending in something runnable and, wherever possible, a number. Merge when CI is green without waiting. The PR body carries the reasoning — what was built, what was rejected, what the metric says — because that is the record Andrey reads to learn the system.

The line for when to stop is **empirical vs. not**. This repo has an eval harness, so most architecture questions are settled by recall@k rather than by taste:

- **A number can decide it** (chunk size, `§` splitting, hybrid alpha, `k`, reranking) → measure, report, proceed. Asking for approval here would be asking someone to guess ahead of the measurement.
- **No number can decide it** → stop and ask. Specifically: **the gold set content** and **the violation catalogue** (these define what "correct" means, so an error there makes every downstream number confidently wrong), **cost** before any large billable eval run, **scope changes**, and anything **irreversible**.

A study guide with interview-shaped questions is owed at the end of the build, mapping RAG/LangGraph concepts to where they live in this code. The metrics are destined for a CV, so being unable to defend them in an interview would make the project a liability rather than an asset — that artifact is the mitigation, not an optional extra.

## Repository state

Workflow foundation is complete and verified: CI green, `main` protected with admin enforcement on, squash-merge only. `CONTRIBUTING.md` documents the branch strategy and the protection escape hatch; `docs/adr/` records the load-bearing decisions; `LIMITATIONS.md` was written before any numbers exist so the caveats stay honest.

Not yet built — this is the actual project, and its design is Andrey's call, not something to assume: ingestion, retrieval, the LangGraph flows, the gold set, and the contract generator. The API carries only `/health` and `/`.

**Installed LangGraph is 1.2.11 and langchain-core 1.5.6 — the v1 APIs.** Nearly every tutorial and blog post online targets 0.x and its idioms differ. Check the installed version's actual API rather than pattern-matching from memory or from search results.

## Conventions

- Code and README in English; corpus, prompts touching legal text, and test data in Polish.
- Repo will be public under github.com/andrii-mazurchuk — keep it clean: no TODO markers, no dead code, no "coming soon" except the explicit v2 list.
