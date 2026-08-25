# 0010 — Record/replay cassettes for LLM calls, and why a replayed run is unpublishable

**Status:** Accepted

## Context

A full Layer 1b run is 97 gold questions, roughly 1100 Claude calls, and **$3.23**. Layer 1 with
`--rewrite` is another $0.12. Those are the real recorded numbers, not estimates — they are in
`metrics/history.jsonl`.

Almost every reason to run the harness again has nothing to do with what the model would say:
renaming a field, refactoring the graph, checking that a CLI flag still parses, demonstrating the
system to someone, or debugging the eval harness itself. Each of those cost $3.23 to try, which is a
tax on touching the code — and a tax that is paid most heavily by exactly the careful, iterative
work the repository is supposed to encourage.

There is also a second-order problem. Because a run is expensive, it is run rarely; because it is run
rarely, the harness is under-tested; and an under-tested harness produces numbers nobody can fully
defend. The brief is explicit that the metrics are the highest-value artifact here, so that is not a
small thing.

## Decision

Record every LLM request/response pair to an append-only JSONL file once, and serve it from disk
thereafter. Four modes — `off` (default), `record`, `replay`, `auto` — selected by
`Settings.cassette_mode` or `--cassette` on either eval.

Four sub-decisions carry most of the weight:

**The seam is a `ChatAnthropic` subclass overriding `_generate`, not a wrapper around
`structured_call`.** Wrapping `structured_call` would leak — `rewrite_question` does not go through
it and neither will the audit flow — and it would mean recording *parsed* Pydantic objects. Replay
would then skip langchain's real parser, so a replayed run would stop testing the code and start
testing the cassette. Recording the raw `AIMessage` means a replayed call still parses, still
validates, and still fails the way a live one would. `graphs/qa_flow.py` is unchanged.

**The cassette key excludes `message.id`.** LangChain and LangGraph stamp fresh random UUIDs on
messages. A key computed over them differs on every run, so a fully recorded cassette replays at
exactly 0% — and in `auto` mode that failure is silent and expensive: it pays full price while
appearing to be cached. This has a dedicated regression test that names the trap.

**Serialisation is field by field, not `langchain_core.load.dumpd`.** The dumpd envelope embeds a
class path and a serialisation version, so a langchain upgrade can invalidate an otherwise perfectly
good recording, and the result is not readable in a diff.

**Every append is flushed under the lock, immediately.** `evals/qa_eval.py` already carries a scar
from a run that died at question ~90 and took the 89 already paid for with it (see the docstring on
`score`). A cassette that serialised at exit would reproduce that failure exactly, and the second
time it would be self-inflicted.

## The load-bearing decision: replayed runs are structurally unpublishable

Making evals free makes them casual, and casual is precisely the condition under which a number gets
re-run without thinking and quoted as though it were fresh. A replayed run reproduces what the model
said on some earlier day. That is what makes it useful for testing the pipeline and worthless as
evidence about the pipeline's *current* behaviour — and the difference is invisible in the output,
which is a table of percentages either way.

So this is enforced, not documented:

- `RunContext.provenance` is `live` / `replayed` / `mixed`, **derived** by `provenance_of(usage)` from
  the `Usage` object. A caller cannot pass it, so a caller cannot forget it.
- `append_row()` raises `NotAMeasurement` for anything that is not `live`. Every publishing path in
  the repository goes through that one function, so making a replayed run publishable requires
  deleting a named exception in a diff someone has to approve.
- `latest_for_layer` and `evals.gate.check` filter non-live rows, which covers a hand-edited history
  file rather than the harness.
- `--record` together with `--cassette replay` or `--cassette auto` fails at argument parsing, before
  any work happens.
- A banner goes to stderr on every non-`off` process start, saying in words that the numbers are not
  a measurement.

`Usage.cost_usd` keeps its existing meaning exactly: money actually spent. Replayed calls accumulate
into `replayed_calls` / `avoided_cost_usd` instead, and `avoided_cost_usd` deliberately does **not**
exist on `RunContext` — since `append_row` refuses non-live rows, the field could only ever be zero in
a written row, where it would imply the cassette had been used to produce a metric.

The default is `off`, and a test asserts `type(cheap_model(Settings())) is ChatAnthropic` — so the
replay seam cannot reach production by way of a changed default.

## What was rejected

- **A `prompts_sha` in the index.** Heuristic, and redundant: the key hashes message content, so a
  changed prompt already misses. A second, weaker signal would only invite trusting it.
- **A fuzzy nearest-match on a miss.** A cassette that guessed which recording you meant is a
  cassette that can serve one question's answer for another. The miss report gives the model, the
  message hashes, and how many recorded requests share the system-message hash — enough to tell "the
  prompt changed" from "this question was never recorded", and no guessing.
- **Erroring on an `index.json` that disagrees with the body.** A `kill -9` skips the atexit index
  rewrite while the body is already flushed, so the mismatch describes a perfectly good cassette. The
  index is advisory; counts and token totals are recomputed from the body, and a mismatch warns.
- **A single response per key.** Generation is not deterministic even at temperature 0, so responses
  are pooled per key and replay always serves the first — two replays of one cassette are therefore
  bit-identical, and a second distinct response is recorded rather than silently discarded.
- **Cassette settings inside `retrieval_config_hash`.** How a response was obtained is provenance,
  not configuration; including it would make a replayed run look like a differently-*configured*
  measurement instead of the same one replayed.

## Consequences

- A full Layer 1b re-run costs $0.00 and needs no API key. Only a deliberate `record` costs money.
- `calls.jsonl` is gitignored; `index.json` is committed, so what a cassette holds and what it cost
  stays reviewable without megabytes of transcript in the repository history.
- `key_version` / `cassette_version` mismatches are hard errors that say "re-record"; model, corpus
  and config-hash mismatches are warnings naming both values.
- A cassette miss inside `qa_eval.score.attempt` is re-raised *before* the broad handler. Swallowed
  into `failures`, a miss would sit under the 5% threshold and let the run score and publish numbers
  computed from a half-recorded cassette.
- New surface to maintain: `graphs/cassette.py`, two test modules, and the discipline of re-recording
  when a prompt, schema, model or corpus changes.

## What would change our mind

If replay ever produced a *different* result from the live run it recorded, the seam is in the wrong
place and the cassette is lying — that would be grounds to move it down to the HTTP transport
(`anthropic`'s client) and record wire bytes instead, which is stricter but far more opaque.

If the cassette starts being re-recorded on nearly every change — because prompts and schemas turn
out to churn faster than the code around them — the saving evaporates and the honest conclusion would
be that it costs more to maintain than the $3.23 it saves.
