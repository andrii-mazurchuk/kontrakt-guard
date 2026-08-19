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

## The gold set is small and single-annotator

Layer 1 uses roughly 30–50 questions derived from Państwowa Inspekcja Pracy material, with
ground-truth articles assigned by one person who is not a lawyer. At that size, confidence
intervals on recall@k are wide: a handful of questions moving flips a percentage point that looks
meaningful. Treat differences of a few points between configurations as noise unless the eval says
otherwise.

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
