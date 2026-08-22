"""Q&A flow tests. No database, no model, no billable call.

Every LLM response is scripted, which is what makes these tests about the
*graph* — routing, grounding checks, fallbacks — rather than about whether Claude
happened to answer well on the day. Answer quality is Layer 1's and the
faithfulness judge's job, and neither belongs in a unit test.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from graphs.llm import PRICES, Usage
from graphs.qa_flow import (
    Answer,
    Citation,
    Grade,
    QAFlow,
    QAState,
    SearchQuery,
    _verify_citations,
    build_qa_graph,
    rewrite_question,
    route_after_grading,
)
from kontrakt_guard.config import Settings
from retrieval.search import Hit

CHEAP = "claude-haiku-4-5-20251001"


def hit(article: str = "25", content: str = "Umowę o pracę zawiera się na okres próbny.") -> Hit:
    return Hit(
        chunk_id=int(article) if article.isdigit() else 1,
        act="kp",
        article=article,
        article_display=f"Art. {article}",
        content=content,
        title_path=["DZIAŁ DRUGI", "Rozdział II"],
    )


def ai(model: str = CHEAP, input_tokens: int = 100, output_tokens: int = 20) -> AIMessage:
    return AIMessage(
        content="",
        response_metadata={"model_name": model},
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class ScriptedModel:
    """Returns queued `{parsed, raw}` payloads, one per invoke.

    Mirrors the shape `with_structured_output(..., include_raw=True)` produces,
    because that shape is exactly what the flow's cost accounting depends on.
    """

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[Any] = []

    def with_structured_output(self, schema: type, include_raw: bool = False) -> ScriptedModel:
        return self

    def invoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        # Repeat the last response rather than running dry: the grading node
        # makes one call per chunk and the count is not the thing under test.
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return response


def parsed(obj: Any, model: str = CHEAP) -> dict[str, Any]:
    return {"parsed": obj, "raw": ai(model), "parsing_error": None}


def make_flow(cheap: Any = None, strong: Any = None, hits: list[Hit] | None = None) -> QAFlow:
    """A flow with its two models scripted and retrieval stubbed out.

    `object()` stands in for the connection and embedder: with `retrieve`
    replaced they are never touched, and a real one would drag a database and a
    2 GB model into a unit test.
    """
    flow = QAFlow.__new__(QAFlow)
    flow.conn = object()  # type: ignore[assignment]
    flow.embedder = object()  # type: ignore[assignment]
    flow.settings = Settings()
    flow.usage = Usage()
    flow._cheap = cheap
    flow._strong = strong
    if hits is not None:
        flow.retrieve = lambda state: {"hits": hits}  # type: ignore[method-assign]
    return flow


# --- the conditional edge -----------------------------------------------------


def test_routes_to_refusal_when_nothing_survives_grading():
    assert route_after_grading(QAState(graded=[])) == "refuse"


def test_routes_to_answer_when_something_survives():
    assert route_after_grading(QAState(graded=[hit()])) == "answer"


def test_ungrounded_question_refuses_instead_of_answering():
    """The whole guardrail, end to end through the compiled graph.

    An empty context handed to a generative model is how a RAG system starts
    inventing law. The refusal is a designed route, so it is asserted on the
    graph rather than on the node.
    """
    strong = ScriptedModel(parsed(Answer(answer="wymyślona odpowiedź", citations=[])))
    flow = make_flow(
        cheap=ScriptedModel(parsed(Grade(relevant=False, reason="inna instytucja"))),
        strong=strong,
        hits=[hit()],
    )

    state = build_qa_graph(flow).invoke({"question": "jaka jest stolica Francji?"})

    assert state["refused"] is True
    assert state["citations"] == []
    assert "Nie znalazłem" in state["answer"]
    # The expensive model must never have been reached.
    assert strong.calls == []


def test_grounded_question_reaches_the_answer_node():
    flow = make_flow(
        # No rewrite response queued: with query_rewrite off the understand node
        # passes the question straight through and never calls the model.
        cheap=ScriptedModel(parsed(Grade(relevant=True))),
        strong=ScriptedModel(
            parsed(
                Answer(
                    answer="Okres próbny nie może przekraczać 3 miesięcy.",
                    citations=[Citation(act="kp", article="25")],
                ),
                model="claude-sonnet-5",
            )
        ),
        hits=[hit("25")],
    )

    state = build_qa_graph(flow).invoke({"question": "czy 6-miesięczny okres próbny jest legalny?"})

    assert state["refused"] is False
    assert [c.article for c in state["citations"]] == ["25"]
    assert state["search_query"] == "czy 6-miesięczny okres próbny jest legalny?"


# --- grounding is checked, not requested --------------------------------------


def test_a_citation_to_an_unretrieved_article_is_stripped_and_flagged():
    """The most damaging failure this system can produce, and the least visible.

    A fluent sentence carrying a real-looking article number the model was never
    shown is worse than a refusal and worse than an obvious error, because it is
    the one a reader has no way to catch.
    """
    supported, unsupported = _verify_citations(
        [Citation(act="kp", article="25"), Citation(act="kp", article="999")],
        [hit("25")],
    )
    assert [c.article for c in supported] == ["25"]
    assert unsupported == ["kp 999"]


def test_the_answer_node_reports_unsupported_citations():
    flow = make_flow(
        strong=ScriptedModel(
            parsed(
                Answer(
                    answer="Zgodnie z art. 300 …",
                    citations=[Citation(act="kp", article="300")],
                ),
                model="claude-sonnet-5",
            )
        )
    )
    result = flow.answer(QAState(question="pytanie", graded=[hit("25")]))

    assert result["citations"] == []
    assert result["unsupported_citations"] == ["kp 300"]


def test_a_repealed_passage_is_labelled_for_the_model():
    """The model cannot know a chunk is no longer law unless it is told."""
    repealed = hit("94")
    repealed.repealed = True
    strong = ScriptedModel(parsed(Answer(answer="…", citations=[]), model="claude-sonnet-5"))

    make_flow(strong=strong).answer(QAState(question="pytanie", graded=[repealed]))

    assert "PRZEPIS UCHYLONY" in strong.calls[0][1].content


# --- degradation rather than failure ------------------------------------------


def test_a_failed_rewrite_falls_back_to_the_original_question():
    """A rewrite is an optimisation; losing it should cost recall, not the run."""
    flow = make_flow(cheap=ScriptedModel({"parsed": None, "raw": ai(), "parsing_error": "boom"}))
    result = flow.understand(QAState(question="szef nie płaci za nadgodziny"))
    assert result["search_query"] == "szef nie płaci za nadgodziny"


def test_an_empty_rewrite_falls_back_too():
    flow = make_flow(cheap=ScriptedModel(parsed(SearchQuery(query="   "))))
    result = flow.understand(QAState(question="pytanie"))
    assert result["search_query"] == "pytanie"


def test_a_failed_answer_refuses_rather_than_returning_unvalidated_prose():
    flow = make_flow(
        strong=ScriptedModel({"parsed": None, "raw": ai("claude-sonnet-5"), "parsing_error": "x"})
    )
    result = flow.answer(QAState(question="pytanie", graded=[hit()]))

    assert result["refused"] is True
    assert result["citations"] == []


def test_grading_with_no_hits_does_not_call_the_model():
    cheap = ScriptedModel(parsed(Grade(relevant=True)))
    result = make_flow(cheap=cheap).grade(QAState(question="pytanie", hits=[]))

    assert result["graded"] == []
    assert cheap.calls == []


def test_grading_preserves_the_merge_ranking():
    """Grading filters the ranking; it does not reorder it.

    The chunks are judged concurrently, so completion order is arbitrary and
    relying on it would silently discard the fusion weight that was measured.
    """
    ordered = [hit("1"), hit("2"), hit("3")]
    flow = make_flow(cheap=ScriptedModel(parsed(Grade(relevant=True))), hits=ordered)
    result = flow.grade(QAState(question="pytanie", hits=ordered))

    assert [h.article for h in result["graded"]] == ["1", "2", "3"]


def test_rejected_chunks_are_kept_with_their_reasons():
    """The cheapest way to discover that grading is discarding the right article."""
    flow = make_flow(cheap=ScriptedModel(parsed(Grade(relevant=False, reason="inny przepis"))))
    result = flow.grade(QAState(question="pytanie", hits=[hit("25")]))

    assert result["graded"] == []
    assert result["rejected"] == [("Art. 25", "inny przepis")]


# --- cost accounting ----------------------------------------------------------


def test_usage_costs_a_call_at_the_published_rate():
    usage = Usage()
    usage.record(ai(CHEAP, input_tokens=1_000_000, output_tokens=1_000_000), CHEAP)

    rate_in, rate_out = PRICES[CHEAP]
    assert usage.cost_usd == pytest.approx(rate_in + rate_out)
    assert usage.calls == 1


def test_an_unpriced_model_is_charged_at_the_most_expensive_rate():
    """A run that looks cheaper than it was is the failure that matters here."""
    usage = Usage()
    usage.record(ai("claude-something-new", input_tokens=1_000_000, output_tokens=0), "x")

    assert usage.cost_usd == pytest.approx(max(rate for rate, _ in PRICES.values()))


def test_usage_accumulates_across_the_whole_run():
    flow = make_flow(
        cheap=ScriptedModel(parsed(Grade(relevant=True))),
        strong=ScriptedModel(parsed(Answer(answer="a", citations=[]), model="claude-sonnet-5")),
        hits=[hit("25"), hit("26")],
    )
    build_qa_graph(flow).invoke({"question": "pytanie"})

    # One grade per chunk, plus the answer. No rewrite call: it is off by default.
    assert flow.usage.calls == 3
    assert flow.usage.cost_usd > 0
    assert flow.usage.per_model["claude-sonnet-5"] == 1


# --- the rewrite, as the eval harness uses it ---------------------------------


def test_rewrite_question_is_usable_without_the_whole_flow():
    """`evals.retrieval_eval --rewrite` scores this node with the Layer 1 harness.

    A query rewrite is a retrieval change, so it is settled by recall@k like
    every other retrieval decision here — not by reading the rewrites and
    finding them plausible.
    """
    usage = Usage()
    model = ScriptedModel(parsed(SearchQuery(query="praca w godzinach nadliczbowych")))

    result = rewrite_question(
        cast(BaseChatModel, model), "szef każe mi zostawać po godzinach", usage
    )

    assert result == "praca w godzinach nadliczbowych"
    assert usage.calls == 1


def test_rewrite_question_falls_back_when_the_call_fails():
    usage = Usage()
    model = ScriptedModel({"parsed": None, "raw": ai(), "parsing_error": "boom"})

    assert rewrite_question(cast(BaseChatModel, model), "pytanie", usage) == "pytanie"


# --- citation canonicalisation ------------------------------------------------


@pytest.mark.parametrize(
    "act,article",
    [
        ("kp", "25"),
        ("Kodeks pracy", "25 § 2"),
        ("KP", "art. 25"),
        ("k.p.", "25"),
    ],
)
def test_a_grounded_citation_is_recognised_however_it_is_spelled(act, article):
    """Found by the first live call, which cost the run its entire citation list.

    The model answered correctly, cited `Kodeks pracy art. 25 § 2`, and the
    verifier — comparing raw strings against a corpus keyed ("kp", "25") —
    reported every citation as unsupported. Strictness was right; the comparison
    was not.
    """
    supported, unsupported = _verify_citations([Citation(act=act, article=article)], [hit("25")])

    assert unsupported == []
    # Returned canonically, so a caller never sees the model's spelling.
    assert (supported[0].act, supported[0].article) == ("kp", "25")


def test_normalising_does_not_soften_the_guardrail():
    """The point of the check survives: an unretrieved article is still caught."""
    _, unsupported = _verify_citations(
        [Citation(act="Kodeks pracy", article="999 § 1")], [hit("25")]
    )
    assert unsupported == ["Kodeks pracy 999 § 1"]


def test_passages_carry_the_corpus_key_the_model_must_copy():
    strong = ScriptedModel(parsed(Answer(answer="…", citations=[]), model="claude-sonnet-5"))
    make_flow(strong=strong).answer(QAState(question="pytanie", graded=[hit("29^3")]))

    assert "[kp:29^3]" in strong.calls[0][1].content


def test_the_rewrite_is_off_by_default_and_the_node_passes_through():
    """Measured: rewriting cost 8 points of recall@5 on the gold set (ADR 0009).

    The node stays in the graph — the finding is worth keeping visible, and the
    setting keeps it reproducible — but it must not call the model when off.
    """
    cheap = ScriptedModel(parsed(SearchQuery(query="cokolwiek")))
    flow = make_flow(cheap=cheap)
    assert flow.settings.query_rewrite is False

    result = flow.understand(QAState(question="szef nie płaci za nadgodziny"))

    assert result["search_query"] == "szef nie płaci za nadgodziny"
    assert cheap.calls == []


def test_the_rewrite_still_runs_when_switched_on():
    cheap = ScriptedModel(parsed(SearchQuery(query="praca w godzinach nadliczbowych")))
    flow = make_flow(cheap=cheap)
    flow.settings = Settings(query_rewrite=True)

    result = flow.understand(QAState(question="szef nie płaci za nadgodziny"))

    assert result["search_query"] == "praca w godzinach nadliczbowych"
    assert len(cheap.calls) == 1
