# metrics/

`history.jsonl` is the append-only record of every evaluation run — one JSON object per line,
created by the first eval and never rewritten. Schema: [`evals/schema.py`](../evals/schema.py).

It is committed deliberately. Three things follow from that, and they are the entire reason the
file exists rather than a tracker service:

1. A metric change shows up as a **diff inside the pull request that caused it**.
2. [`evals/gate.py`](../evals/gate.py) compares a run against the previous one already in the file,
   so a pull-request branch carries main's baseline with no extra bookkeeping.
3. The README tables and the dashboard are **generated** from it, so neither can drift from the
   evidence.

Every row carries full provenance — commit, embedding model *and revision*, pinned model IDs,
retrieval config hash, corpus manifest checksum, cost, duration. A number without its configuration
is an anecdote.

**Never edit this file by hand.** Rewriting history here is falsifying results, not tidying up.
