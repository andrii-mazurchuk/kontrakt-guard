# Kontrakt-Guard

Polish employment-contract auditor. Upload an `umowa o pracę`, `umowa zlecenie` or B2B contract;
the system segments it into clauses, retrieves the governing law per clause from a corpus of
consolidated Polish acts, and returns a structured verdict — `compliant` / `risky` / `illegal` —
with an explanation and exact article citations.

A grounded legal Q&A endpoint falls out of the same engine: ask an employment-law question, get a
citation-backed answer or an explicit refusal when the corpus cannot ground it.

> **This is not legal advice.** Verdicts are produced by a language model over a static snapshot of
> Polish legislation and may be wrong, incomplete, or out of date. Consult a qualified lawyer or the
> Państwowa Inspekcja Pracy before acting on anything here.

---

## Status

🚧 In development. This section is replaced by a quickstart once `docker compose up` serves the API.

---

## Metrics

Both tables are **generated from `metrics/history.jsonl` by CI** — no number here is typed by hand.
Every row is attributable to an exact configuration (embedding model + revision, model IDs,
retrieval config hash) recorded alongside it.

### Layer 1 — retrieval quality

Recall@k on ground-truth article IDs, over a gold set of employment-law questions derived from
Państwowa Inspekcja Pracy guidance.

<!-- METRICS:LAYER1:START -->
_No eval runs recorded yet._
<!-- METRICS:LAYER1:END -->

### Layer 2 — audit quality

Precision / recall / F1 on violation detection, over a synthetic labeled contract set built by
planting known violations into legitimate contract templates.

<!-- METRICS:LAYER2:START -->
_No eval runs recorded yet._
<!-- METRICS:LAYER2:END -->

The two layers are evaluated independently on purpose: when a number moves, it is attributable to
retrieval or to judgement, not to an undifferentiated blob.

---

## Architecture

_Diagram lands here once the graphs are built._

One engine underlies both products: *given a legal question, return the exact articles that answer
it and an answer grounded in them.* The auditor is that engine called in a loop over contract
clauses.

- `src/kontrakt_guard/ingestion` — ISAP acquisition, article-aware parsing, embedding, pgvector load
- `src/kontrakt_guard/retrieval` — hybrid search (Postgres full-text BM25 + pgvector dense), merge, LLM relevance grading
- `src/kontrakt_guard/graphs` — LangGraph flows: Q&A and contract audit, both with an explicit refusal path
- `src/kontrakt_guard/api` — FastAPI: `POST /audit`, `POST /ask`
- `evals/` — gold set, synthetic contract generator, metric computation

---

## Development

See [`CLAUDE.md`](CLAUDE.md) for the binding technical decisions and
[`docs/adr/`](docs/adr/) for why they were made.

---

## License

MIT — see [`LICENSE`](LICENSE).
