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

Target state (per brief §9): `docker compose up` brings up postgres+pgvector and the API. As build/test/eval commands are established during the build, record them here.

## Conventions

- Code and README in English; corpus, prompts touching legal text, and test data in Polish.
- Repo will be public under github.com/andrii-mazurchuk — keep it clean: no TODO markers, no dead code, no "coming soon" except the explicit v2 list.
