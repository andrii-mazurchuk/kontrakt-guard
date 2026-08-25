# Cassettes — recorded Claude calls

A cassette is every request/response pair from one eval run, written to disk once so the run can be
repeated for **$0.00**. See [ADR 0010](../docs/adr/0010-cassette-replay-for-llm-evals.md) for why
this exists and what it deliberately does not do.

```
cassettes/<name>/calls.jsonl   the recorded calls — gitignored, one JSON object per line
cassettes/<name>/index.json    committed metadata: what was recorded, when, and what it cost
```

`calls.jsonl` is not committed: it is large, and it is reproducible for money. `index.json` is,
because *what a cassette contains and what it cost* should be reviewable without carrying megabytes
of transcript in the repository history.

**A replayed run is not a measurement.** `evals.schema.append_row` raises `NotAMeasurement` for any
row whose provenance is not `live`, and `--record` with `--cassette replay` fails at argument
parsing. Replay reproduces what the model said last time; that is what makes it useful for testing
the pipeline and worthless as evidence about the pipeline's current behaviour.

## Using one

```bash
# free — every call served from disk, no API key needed
uv run python -m evals.qa_eval --cassette replay --cassette-name qa-eval

# top up a cassette after adding gold questions: hits are free, misses are bought
uv run python -m evals.qa_eval --cassette auto --cassette-name qa-eval
```

## Re-recording

Re-recording is the only thing here that costs money. `record` mode never serves from the tape — it
calls through every time — so what you get back is a genuinely live run.

```bash
uv run python -m evals.qa_eval --cassette record --cassette-name qa-eval
```

**Cost of a full Layer 1b re-record: ~$3.23** (97 gold questions, ~1100 calls). A Layer 1 rewrite
cassette is ~$0.12. Check `index.json` for what the existing recording actually cost.

Re-record when — and only when — the answer would genuinely differ:

- a prompt in `graphs/prompts.py` changed
- a structured-output schema changed (`Grade`, `Answer`, `SearchQuery`)
- `model_cheap` or `model_strong` changed
- the corpus changed, so retrieval feeds different passages to the model

A refactor that does not touch any of those does not need a re-record; if it did, the cassette would
miss and tell you so, loudly.

## When a replay misses

`CassetteMiss` names the model, the messages, their content hashes, and how many recorded requests
share the same system-message hash. That last number is the diagnosis: **many shared** means the
system prompt is unchanged and this particular question was never recorded; **none shared** means the
system prompt itself changed and the whole cassette is stale.

There is no fuzzy matching, on purpose. A cassette that guessed which recording you meant would be a
cassette that could serve one question's answer for another.
