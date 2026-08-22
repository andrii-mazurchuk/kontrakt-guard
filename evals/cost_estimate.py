"""Estimate what a billable run will cost, before running it.

    uv run python -m evals.cost_estimate                 # whole gold set
    uv run python -m evals.cost_estimate --sample 10     # faster, extrapolated

Makes no API call. It runs the free half of the flow — retrieval — for real,
builds the exact prompt strings the nodes would send, and prices them. The
guesswork is therefore confined to one number: characters per token.

Why this exists as a tool rather than as arithmetic in a message: "single-digit
dollars" is the brief's budget, and a budget nobody re-checks after the prompts
grow is a budget that quietly stops holding. Run it again whenever a prompt,
`retrieval_top_k`, or a model changes.
"""

from __future__ import annotations

import argparse
import sys

from evals.gold import load_gold
from graphs.llm import PRICES
from graphs.prompts import (
    ANSWER_SYSTEM,
    GRADE_SYSTEM,
    REWRITE_SYSTEM,
    answer_user_prompt,
    grade_user_prompt,
)
from graphs.qa_flow import MAX_PASSAGES
from ingestion.chunk import query_input
from ingestion.db import connect
from ingestion.embed import Embedder
from kontrakt_guard.config import get_settings
from retrieval.search import dense_search, lexical_search, merge

# Characters per token, Polish, Claude tokenizer. Both constants below were
# calibrated against a measured 10-question run of `evals.qa_eval`, because the
# first version of this estimator — 3.0 chars per token and no per-call overhead
# — came in **77% under** the real bill. An estimator that under-predicts is
# worse than no estimator, since its only job is to be the number someone is
# asked to approve.
#
# 2.2 rather than 3.0: Polish tokenises worse than English. Diacritics and long
# inflected forms split more often, and the corpus is full of both.
CHARS_PER_TOKEN = 2.2

# What `with_structured_output` costs beyond the prompt itself. Every call ships
# the schema as a tool definition, plus the message envelope — invisible in the
# prompt strings, and charged on every one of the ~12 calls a question makes.
STRUCTURED_OUTPUT_OVERHEAD = 450

# Output tokens per call, also calibrated against the measured run rather than
# assumed. Grading was the surprise: the schema asks for a "krótkie uzasadnienie"
# and the model writes a considered paragraph, so grading output ran nearly three
# times the first guess of 60. With ten grades per question that is the single
# largest line in the bill — larger than the Sonnet answer it protects.
OUTPUT_TOKENS = {"rewrite": 40, "grade": 170, "answer": 500}


def tokens(text: str) -> int:
    """Prompt tokens for one call, including what the API adds to it."""
    return int(len(text) / CHARS_PER_TOKEN) + STRUCTURED_OUTPUT_OVERHEAD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0, help="Questions to measure (0 = all).")
    args = parser.parse_args()

    settings = get_settings()
    questions = load_gold()
    measured = questions[: args.sample] if args.sample else questions
    if not measured:
        print("gold set is empty", file=sys.stderr)
        return 1

    embedder = Embedder(settings)
    cheap_in = cheap_out = strong_in = strong_out = 0
    grade_calls = 0

    with connect(settings) as conn:
        for question in measured:
            text = question.question

            # understand — only billable when the rewrite is switched on, which
            # it is not by default: it cost 8 points of recall@5. See ADR 0009.
            if settings.query_rewrite:
                cheap_in += tokens(REWRITE_SYSTEM + text)
                cheap_out += OUTPUT_TOKENS["rewrite"]

            # retrieve — free, and run for real so the chunk sizes below are the
            # sizes this corpus actually produces rather than an assumed average.
            lexical = lexical_search(
                conn,
                text,
                settings.bm25_candidates,
                ranking=settings.lexical_ranking,
                k1=settings.bm25_k1,
                b=settings.bm25_b,
            )
            dense = dense_search(
                conn, embedder.encode_query(query_input(text)), settings.vector_candidates
            )
            hits = merge(
                lexical,
                dense,
                k=settings.retrieval_top_k,
                fusion=settings.fusion,
                alpha=settings.hybrid_alpha,
            )

            # grade — one call per retrieved chunk.
            for hit in hits:
                prompt = grade_user_prompt(
                    text, hit.citation, " > ".join(hit.title_path), hit.content
                )
                cheap_in += tokens(GRADE_SYSTEM + prompt)
                cheap_out += OUTPUT_TOKENS["grade"]
                grade_calls += 1

            # answer — worst case, every graded chunk survives up to the cap.
            passages = [f"{h.citation}\n{h.content}" for h in hits[:MAX_PASSAGES]]
            strong_in += tokens(ANSWER_SYSTEM + answer_user_prompt(text, passages))
            strong_out += OUTPUT_TOKENS["answer"]

    scale = len(questions) / len(measured)
    cheap_rate_in, cheap_rate_out = PRICES[settings.model_cheap]
    strong_rate_in, strong_rate_out = PRICES[settings.model_strong]

    cheap_cost = (cheap_in * cheap_rate_in + cheap_out * cheap_rate_out) / 1_000_000
    strong_cost = (strong_in * strong_rate_in + strong_out * strong_rate_out) / 1_000_000
    total = (cheap_cost + strong_cost) * scale

    per_question_fixed = 2 if settings.query_rewrite else 1
    calls = (len(measured) * per_question_fixed + grade_calls) * scale
    print(f"Measured {len(measured)} of {len(questions)} gold questions.\n")
    print(f"{'step':<14}{'model':<28}{'in':>12}{'out':>10}{'USD':>10}")
    print(
        f"{'rewrite+grade':<14}{settings.model_cheap:<28}"
        f"{int(cheap_in * scale):>12,}{int(cheap_out * scale):>10,}{cheap_cost * scale:>10.3f}"
    )
    print(
        f"{'answer':<14}{settings.model_strong:<28}"
        f"{int(strong_in * scale):>12,}{int(strong_out * scale):>10,}{strong_cost * scale:>10.3f}"
    )
    print(f"\n{int(calls):,} API calls, estimated **${total:.2f}** for the full gold set.")
    print(f"Budget ceiling (eval_max_cost_usd): ${settings.eval_max_cost_usd:.2f}")

    if total > settings.eval_max_cost_usd:
        print("\nOVER BUDGET — raise eval_max_cost_usd deliberately or reduce the run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
