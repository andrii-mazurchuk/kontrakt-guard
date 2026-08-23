# Limitations

What the numbers in the README do **not** prove. Written before the numbers exist, so that the
limitations are honest rather than retrofitted to whatever came out.

## This is not legal advice

Verdicts come from a language model reasoning over retrieved statute text. They are not a legal
opinion, carry no professional liability, and have not been reviewed by a lawyer. Polish employment
law turns on facts, case law and context that a static corpus does not contain.

## The audit dataset is synthetic

There is no public labeled corpus of Polish employment contracts, so Layer 2 is measured against
contracts built by planting known violations into legitimate templates. This has consequences that
no F1 score can paper over:

- **Planted violations are cleaner than real ones.** A generator writes a probation period of six
  months plainly. A real contract buries the problem in a cross-reference to an annex.
- **The violation catalogue is closed.** Precision and recall are computed over the classes we
  chose to plant. A violation type absent from the catalogue is invisible to the metric — it cannot
  be counted as a miss, so recall is measured against our imagination, not against Polish law.
- **Generator and grader share assumptions.** The same understanding shaped both what gets planted
  and what counts as a correct detection. Genuinely independent labelling would be stronger.

## The fusion weight was tuned on the set it is reported over

`hybrid_alpha` was swept across 0.3–0.9 against the same 97 gold questions the headline recall@k is
computed on, and the fusion method and lexical ranking were chosen the same way. There is no
held-out split. Some part of the gain over dense-only retrieval is therefore fitted to this
particular set rather than to Polish legal retrieval in general.

The α sweep makes the size of that effect visible rather than hiding it. Recall@5 across
0.6/0.7/0.8/0.85 is 84.8/84.3/85.3/83.8 — **non-monotonic, and spanning about one question on a
97-question set**. Differences of that size inside the plateau are not real; 0.8 was taken because
it is the joint best across recall@5, recall@10 and MRR, not because 85.3% is meaningfully above
84.8%.

Treat the leg comparison in the README as sound — the gap between 46.9% and 85.3% is far too wide to
be an artefact — and the precise value of α as the softest number in the table. By the same measure,
the BM25-versus-`ts_rank_cd` comparison **on the lexical leg alone** (46.9% → 69.6%) is solid, while
its half-point effect on the merged system is not distinguishable from noise, and is not claimed as
an improvement.

## Citation precision measures agreement with the gold set, not correctness

Layer 1b scores cited articles against `ground_truth_articles`, so a citation counts as wrong
whenever the gold set does not list it. The gold set lists what is needed to **answer** the question,
which is not the same as every provision a careful answer might reference.

At 54.2% precision and 78.3% recall the system cites roughly **1.4 articles for every one the gold
set names**. Some of that surplus is genuine over-citation; some is a legitimately broader answer
being marked down. Nothing here separates the two, and no human has read the answers to find out.

So treat **hallucinated citations (0.0%)** as the solid number — it is a mechanical check against the
retrieved set, with no judgement in it — and **citation precision as the softest figure in the
table**. Deciding the rest needs reference answers written by someone who knows the law, which the
gold set does not currently carry.

## What in the gold set was human-checked, and what was not

The 97 questions come from two sources and carry different evidentiary weight.
Reporting one number over both without saying so would overstate the weaker half.

- **62 PIP-derived questions** use the article citations that Państwowa Inspekcja
  Pracy gives in its own answers. They were accepted **on the source's authority**
  and machine-checked for existence in the corpus, not independently re-derived
  from the statute by a human. Where PIP states a rule without naming an article,
  the question was dropped rather than have an article guessed for it.
- **35 statute-derived questions** were written against article text read directly
  through `retrieval/lookup.py`, and reviewed.

Neither half was checked by a lawyer. Ground truth in both was taken from a source
or from the statute — never from this system's own retrieval, which would make
recall@k measure the system against itself.

## The gold set is small and single-annotator

Layer 1 uses roughly 30–50 questions derived from Państwowa Inspekcja Pracy material, with
ground-truth articles assigned by one person who is not a lawyer. At that size, confidence
intervals on recall@k are wide: a handful of questions moving flips a percentage point that looks
meaningful. Treat differences of a few points between configurations as noise unless the eval says
otherwise.

## The corpus is the law as at one moment, and only one

The snapshot holds the text **in force on the day it was fetched**. Amendments
already enacted but not yet effective are present in the source and deliberately
excluded (see [ADR 0007](docs/adr/0007-pending-amendments.md)) — five such blocks
in this snapshot take effect on 5 November 2026.

The consequence is asymmetric and worth stating plainly: a question about a
contract signed today gets the right answer, while a question about one starting
after a pending change gets an answer that is confidently out of date, with
nothing in the text to signal it. The system has no notion of time.

## Corpus scope and staleness

The corpus is a snapshot of consolidated acts, pinned by checksum. It therefore excludes:

- **Case law.** How courts have actually ruled is often what decides whether a clause survives.
  Statute alone overstates how determinate the answer is.
- **Amendments after the snapshot date.** The manifest records when each act was fetched; anything
  later is simply absent, and the system will answer confidently from superseded text.
- **Collective agreements and workplace regulations**, which can lawfully alter many defaults.

## Retrieval, not reasoning

The system finds articles and grounds an answer in them. It does not reason about interactions
between clauses that are individually lawful and jointly abusive — a known gap, listed as a v2 item
rather than quietly omitted.

## Polish-language NLP

The embedding model is multilingual, not Polish-specific, and the sparse leg uses PostgreSQL's
Polish text-search configuration rather than a true BM25 implementation. Polish is heavily
inflected; stemming failures on legal terminology are a plausible source of retrieval misses, and
`misses_at_5` is recorded per run precisely so that this is inspectable rather than assumed.
