# 0009 — Query rewriting is off by default

**Status:** Accepted

## Context

The brief specifies the Q&A flow's first node as:

> **Understand** — LLM rewrites a messy user question into search-friendly legal Polish
> (e.g. "szef każe mi zostawać po godzinach bez zapłaty" → "wynagrodzenie za pracę w godzinach
> nadliczbowych")

The reasoning is sound on its face. Users write colloquially; the statute uses terms of art; the
lexical leg matches lemmas, so a question that never says *nadliczbowy* cannot match an article that
does. Reading the rewrites the node produces, they look exactly right.

## Decision

Ship the node, keep it in the graph, and **default it off**.

## Measured, 2026-08-22

Same corpus, same 97-question gold set, same embeddings, same fusion. The only change is whether the
question reaching retrieval is the user's or the model's.

| | recall@3 | recall@5 | recall@10 | MRR | pool recall |
|---|---|---|---|---|---|
| **question as asked** | **75.5%** | **85.3%** | **93.0%** | **0.712** | **96.9%** |
| rewritten first | 63.7% | 77.3% | 89.7% | 0.635 | 95.4% |

**Rewriting cost 8 points of recall@5 and 11.8 at k=3.** Cost of finding out: $0.12.

The most diagnostic figure is the last column. Candidate-pool recall fell too — from 96.9% to 95.4%
— which means rewriting did not merely reorder the results. It removed correct articles from
consideration entirely, before any ranking decision was taken. Nothing downstream could have
recovered them.

The mechanism is visible in the rewrites themselves. They are *shorter* than the questions. Asked
"jaka jest stolica Francji i jak ugotować rosół?", the node returned "stolica Francji" — half the
question discarded. A rewrite that compresses a sentence into a keyword phrase throws away terms
BM25 would have scored and context the dense encoder would have used. Both legs are worse off, which
is why the loss shows up in the pool rather than only in the ranking.

## The caveat, which bounds the claim

**The gold set cannot settle the question the node was built to answer.** Its 97 questions come from
PIP guidance and from statute text, so they are already phrased formally. There is no register gap
for a rewrite to close, and nothing to gain against a real cost.

So this measures rewriting on *formal* questions and finds it harmful. It does not measure rewriting
on the colloquial input the brief describes, because the gold set contains none. Claiming otherwise
would be overreading a real number — and the honest position is that the node's original motivation
remains untested rather than refuted.

Testing it properly needs colloquial paraphrases of the gold questions with the same ground-truth
article ids. That is gold-set content, which the working agreement says to stop and ask about rather
than generate unilaterally.

## Consequences

- `query_rewrite` defaults to `False` and is part of `retrieval_config_hash`, so a metric recorded
  with rewriting on can never be confused with one recorded without it.
- The node stays in the graph. It costs nothing when off, the finding is worth keeping visible, and
  removing it would make this ADR unreproducible.
- Roughly $0.12 and one API call per question saved on every run.

## What would change our mind

A colloquial gold set. If rewriting recovers recall on questions phrased the way users actually
write, the node earns its default back — and the honest result would then be that it should be
enabled for user-facing traffic and disabled for the formal questions this eval scores.

A cheaper variant is also worth measuring first: retrieve with **both** the original and the
rewritten query and merge the candidate pools. The failure here is subtraction — terms being
discarded — and a union cannot subtract. That costs one extra retrieval per question and no extra
LLM call beyond the rewrite itself.
