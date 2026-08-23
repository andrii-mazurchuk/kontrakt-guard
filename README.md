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

Recall@k on ground-truth article IDs. **k counts articles, not chunks** — an article split across
several chunks would otherwise consume several of the five slots at k=5 and flatter the score.

The gold set is 97 questions: 62 taken from Państwowa Inspekcja Pracy guidance with the article
citations PIP itself gives, and 35 written against the statute text. Ground truth never comes from
this system's own retrieval, which would make the metric measure the system against itself.

<!-- METRICS:LAYER1:START -->
| Metric | k=3 | k=5 | k=10 |
|---|---|---|---|
| Recall@k on article IDs | 75.5% | 85.3% | 93.0% |
Gold set: **97** questions. MRR **0.712**.

<sub>Run `865e7df7` · 2026-08-22T10:05:50+00:00 · embeddings `intfloat/multilingual-e5-large@3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` · config `7976e955a84d` · $0.00 · 84s</sub>
<!-- METRICS:LAYER1:END -->

**What each retrieval leg contributes**, measured over the same gold set. This is the argument for
hybrid retrieval stated as evidence rather than as assertion — and it very nearly went the other way:

| Configuration | recall@5 | MRR |
|---|---|---|
| Lexical only, `ts_rank_cd` | 46.9% | 0.410 |
| Lexical only, BM25 | 69.6% | 0.610 |
| Hybrid, Reciprocal Rank Fusion, equal weights | 84.3% | 0.698 |
| Dense only (multilingual-e5-large) | 80.2% | 0.699 |
| **Hybrid, weighted α=0.8** | **85.3%** | **0.712** |

Two findings, both of which contradicted an assumption the design started with.

**RRF made hybrid retrieval worse than dense alone.** It weights both legs equally, and the lexical
leg was far weaker, so its noise was promoted into the top ranks. Weighting the dense leg recovers
the gain. ([ADR 0003](docs/adr/0003-hybrid-retrieval-without-reranker.md))

**The lexical leg was weak for a specific, fixable reason: Postgres ranking has no IDF term.**
Neither `ts_rank` nor `ts_rank_cd` knows how rare a lexeme is — and on this corpus `art` and `dział`
occur in *all* 543 chunks while `pracownik` occurs in 403, which are precisely the words an
employment-law question contains. Replacing it with BM25 over a materialised inverted index took the
leg from 46.9% to 69.6%. ([ADR 0008](docs/adr/0008-bm25-over-ts-rank-cd.md))

That second fix improved the *merged* system by half a point, and MRR fell from 0.736 to 0.712. It
was adopted anyway, and both figures are reported here rather than only the favourable one.

**Where the remaining error actually is.** Candidate-pool recall — the share of required articles
reaching the pool at any rank — is **96.9%**. Nothing downstream can retrieve an article neither leg
proposed, so that is the ceiling on the whole pipeline. With recall@5 at 85.3%, about **eleven
points are being lost to ranking rather than to retrieval**, which is what a reranker recovers and
what makes it the next thing worth building.

### Layer 1b — answer quality

The same gold set, but end to end through the LangGraph flow, so this measures what retrieval,
relevance grading and generation do *together*. Every figure compares **article IDs** — none of it
asks a model whether an answer reads well, which keeps failures attributable: Layer 1 says whether
the right article was retrieved, this says whether the answer was built out of it.

<!-- METRICS:ANSWERS:START -->
| Measure | Value |
|---|---|
| Citation precision | 54.2% |
| Citation recall | 78.3% |
| **Citation F1** | **0.641** |
| False refusals | 2.1% |
| Hallucinated citations | 0.0% |
| Uncited answers | 5.2% |
End to end through the graph over **97** gold questions. The last three rows are failure rates — lower is better. A refusal here is a *false* refusal, since every gold question is answerable from the corpus by construction; without that row the refusal guardrail could be made perfect by refusing everything.

<sub>Run `ade2ab74` · 2026-08-23T06:10:34+00:00 · embeddings `intfloat/multilingual-e5-large@3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` · config `aeb463a95ec1` · $3.23 · 289s</sub>
<!-- METRICS:ANSWERS:END -->

Two of these deserve reading carefully rather than at face value.

**Hallucinated citations are 0.0% across 97 questions** — no answer cited an article that was not
retrieved. That is enforced in code, not requested in a prompt: the answer node names its citations
in a structured field, and they are checked against the retrieved set before the answer is returned.
A guardrail that lives only inside a prompt is one the model can decline to apply.

**Citation precision of 54.2% is not straightforwardly an error rate.** The system cites about 1.4
articles for every article the gold set names, and the gold set names only what is needed to *answer*
the question — not every provision a careful answer might reasonably reference. Some of that gap is
over-citation and some is the metric being stricter than a lawyer would be. It is reported as
measured, and how much of it is real is an open question rather than a settled one.

The actionable number is the third: **citation recall is 78.3% against retrieval's 93.0% recall@10**.
About fifteen points go missing *after* retrieval has already found the right article — discarded by
relevance grading or ignored by generation. That, not retrieval, is where this pipeline currently
loses most of its ground truth.

### Layer 2 — audit quality

Precision / recall / F1 on violation detection, over a synthetic labeled contract set built by
planting known violations into legitimate contract templates.

<!-- METRICS:LAYER2:START -->
_No eval runs recorded yet._
<!-- METRICS:LAYER2:END -->

The two layers are evaluated independently on purpose: when a number moves, it is attributable to
retrieval or to judgement, not to an undifferentiated blob. CI fails a pull request that regresses
either metric past tolerance, so these numbers are defended rather than merely reported.

**Read [`LIMITATIONS.md`](LIMITATIONS.md) alongside them.** The audit dataset is synthetic, the gold
set is small and single-annotator, and the corpus excludes case law — all of which bounds what these
scores can be taken to mean.

---

## Architecture

_Diagram lands here once the graphs are built._

One engine underlies both products: *given a legal question, return the exact articles that answer
it and an answer grounded in them.* The auditor is that engine called in a loop over contract
clauses.

- `src/kontrakt_guard/ingestion` — ISAP acquisition, article-aware parsing, embedding, pgvector load
- `src/kontrakt_guard/retrieval` — hybrid search (BM25 over Postgres full-text + pgvector dense), merge, LLM relevance grading
- `src/kontrakt_guard/graphs` — LangGraph flows: Q&A and contract audit, both with an explicit refusal path
- `src/kontrakt_guard/api` — FastAPI: `POST /audit`, `POST /ask`
- `evals/` — gold set, synthetic contract generator, metric computation

---

## Development

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, the exact commands CI runs, branch and commit
conventions, and the rules around evals. [`docs/adr/`](docs/adr/) records why the load-bearing
decisions were made — pgvector over a hosted vector database, article-aware chunking, hybrid
retrieval without a reranker, and metrics in a committed file rather than an experiment tracker.

```bash
uv sync && uv run pre-commit install
uv run pytest
```

---

## License

MIT — see [`LICENSE`](LICENSE).
