"""Grounded legal Q&A as a LangGraph flow.

    understand → retrieve → grade → ┬→ answer → END
                                    └→ refuse → END

    uv run python -m graphs.qa_flow "czy 6-miesięczny okres próbny jest legalny?"

Four nodes and one conditional edge. The conditional edge is the point: when no
retrieved chunk survives relevance grading, the flow takes a **refusal** path
rather than asking the model to answer from an empty context. That is a designed
route through the graph, not an error handler — an ungrounded question has a
correct answer, and it is "I cannot ground this".

Why a graph rather than a function calling four functions in sequence: the state
is explicit and inspectable at every step, so a bad answer is attributable to a
node (the rewrite went wrong / retrieval missed it / grading discarded it /
generation ignored it) instead of to "the pipeline". The audit flow reuses these
same nodes per contract clause.

Grounding is enforced in code, not requested in the prompt. The answer node is
told to cite only the supplied articles; `_verify_citations` then checks that it
did, and downgrades the result if it did not. A guardrail that only exists inside
a prompt is a guardrail the model can decline to apply.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal, TypedDict, cast

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from graphs.llm import Usage, cheap_model, strong_model
from graphs.prompts import (
    ANSWER_SYSTEM,
    DISCLAIMER,
    GRADE_SYSTEM,
    REFUSAL,
    REWRITE_SYSTEM,
    answer_user_prompt,
    grade_user_prompt,
)
from ingestion.chunk import query_input
from ingestion.db import connect
from ingestion.embed import Embedder
from kontrakt_guard.config import Settings, get_settings
from retrieval.search import Hit, dense_search, lexical_search, merge

# How many graded chunks reach the answer node. Grading already removed the
# irrelevant ones, so this is a context-size bound rather than a quality filter.
MAX_PASSAGES = 6

# Grading is one independent call per chunk, so the wall-clock cost is one call
# deep rather than `retrieval_top_k` calls deep.
GRADE_CONCURRENCY = 8

# --- structured outputs -------------------------------------------------------
# Schema-enforced rather than parsed out of prose: these feed downstream control
# flow, and a parser that mostly works is a control flow that mostly works.


class SearchQuery(BaseModel):
    """The user's question, restated in the register the statute uses."""

    query: str = Field(description="Fraza wyszukiwania w języku ustawy.")


class Grade(BaseModel):
    """One chunk's relevance verdict."""

    relevant: bool = Field(description="Czy fragment jest przydatny do odpowiedzi.")
    reason: str = Field(default="", description="Krótkie uzasadnienie.")


class Citation(BaseModel):
    act: str
    article: str


class Answer(BaseModel):
    """The answer, with the articles it rests on named separately.

    Citations are a structured field rather than something to extract from the
    prose, because they are checked against the retrieved set before the answer
    is returned.
    """

    answer: str = Field(description="Odpowiedź po polsku, zwięzła.")
    citations: list[Citation] = Field(
        default_factory=list, description="Artykuły, na których opiera się odpowiedź."
    )


# --- state --------------------------------------------------------------------


def _keep_last(_current: object, incoming: object) -> object:
    """Last write wins. The nodes here are sequential; none merge concurrently."""
    return incoming


class QAState(TypedDict, total=False):
    """What flows between nodes.

    Deliberately fat: `search_query`, `hits` and `graded` are all kept even
    though only the last is needed downstream, because they are what makes a
    wrong answer diagnosable. `rejected` carries the grader's reasons, which is
    the cheapest way to find out that grading is the thing discarding the right
    article.
    """

    question: str
    search_query: str
    hits: list[Hit]
    graded: list[Hit]
    rejected: list[tuple[str, str]]
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str
    unsupported_citations: Annotated[list[str], _keep_last]


# --- nodes --------------------------------------------------------------------


class QAFlow:
    """The nodes, bound to their dependencies.

    A class rather than closures over module state so a test can substitute a
    fake model and a fake connection without patching imports, and so `usage`
    accumulates against one owner per run.
    """

    def __init__(
        self,
        conn: psycopg.Connection[psycopg.rows.DictRow],
        embedder: Embedder,
        settings: Settings,
        usage: Usage | None = None,
    ) -> None:
        self.conn = conn
        self.embedder = embedder
        self.settings = settings
        self.usage = usage if usage is not None else Usage()
        self._cheap = cheap_model(settings)
        self._strong = strong_model(settings)

    # -- understand ------------------------------------------------------------

    def understand(self, state: QAState) -> QAState:
        """Rewrite a colloquial question into statutory Polish.

        A retrieval change, so its value is settled by recall@k rather than by
        reading the rewrites and finding them plausible.
        """
        return {"search_query": rewrite_question(self._cheap, state["question"], self.usage)}

    # -- retrieve --------------------------------------------------------------

    def retrieve(self, state: QAState) -> QAState:
        """Hybrid search. No LLM call, so this node is free and deterministic."""
        query = state.get("search_query") or state["question"]
        settings = self.settings

        lexical = lexical_search(
            self.conn,
            query,
            settings.bm25_candidates,
            ranking=settings.lexical_ranking,
            k1=settings.bm25_k1,
            b=settings.bm25_b,
        )
        dense = dense_search(
            self.conn,
            self.embedder.encode_query(query_input(query)),
            settings.vector_candidates,
        )
        hits = merge(
            lexical,
            dense,
            k=settings.retrieval_top_k,
            fusion=settings.fusion,
            alpha=settings.hybrid_alpha,
        )
        return {"hits": hits}

    # -- grade -----------------------------------------------------------------

    def grade(self, state: QAState) -> QAState:
        """Judge each retrieved chunk independently.

        Independently, and in parallel, for the same reason: a single call shown
        all candidates at once would let their order colour the verdicts, and
        this node exists precisely to be a second opinion on that order.
        """
        hits = state.get("hits") or []
        if not hits:
            return {"graded": [], "rejected": []}

        question = state["question"]

        def judge(hit: Hit) -> tuple[Hit, Grade | None]:
            return hit, self._structured(
                self._cheap,
                Grade,
                [
                    SystemMessage(content=GRADE_SYSTEM),
                    HumanMessage(
                        content=grade_user_prompt(
                            question,
                            hit.citation,
                            " > ".join(hit.title_path),
                            hit.content,
                        )
                    ),
                ],
            )

        with ThreadPoolExecutor(max_workers=GRADE_CONCURRENCY) as pool:
            verdicts = list(pool.map(judge, hits))

        # Order is restored from `hits` rather than taken from completion order:
        # the merge already ranked these, and grading is a filter over that
        # ranking, not a re-ranking of it.
        graded = [hit for hit, grade in verdicts if grade is not None and grade.relevant]
        rejected = [
            (hit.citation, grade.reason)
            for hit, grade in verdicts
            if grade is not None and not grade.relevant
        ]
        return {"graded": graded[:MAX_PASSAGES], "rejected": rejected}

    # -- answer / refuse -------------------------------------------------------

    def answer(self, state: QAState) -> QAState:
        graded = state["graded"]
        passages = [
            f"{hit.citation}"
            + (f" [{' > '.join(hit.title_path)}]" if hit.title_path else "")
            + (" [PRZEPIS UCHYLONY]" if hit.repealed else "")
            + f"\n{hit.content}"
            for hit in graded
        ]

        parsed = self._structured(
            self._strong,
            Answer,
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=answer_user_prompt(state["question"], passages)),
            ],
        )
        if parsed is None:
            # Structured output failed. Refusing beats returning unvalidated
            # prose as though it were a grounded answer.
            return {
                "refused": True,
                "refusal_reason": "structured output failed",
                "answer": REFUSAL,
                "citations": [],
            }

        supported, unsupported = _verify_citations(parsed.citations, graded)
        return {
            "answer": parsed.answer,
            "citations": supported,
            "unsupported_citations": unsupported,
            "refused": False,
        }

    def refuse(self, state: QAState) -> QAState:
        """The designed route for an ungrounded question, not an error path."""
        return {
            "refused": True,
            "refusal_reason": state.get("refusal_reason") or "no chunk survived grading",
            "answer": REFUSAL,
            "citations": [],
        }

    # -- internals -------------------------------------------------------------

    def _structured[SchemaT: BaseModel](
        self, model: BaseChatModel, schema: type[SchemaT], messages: list[BaseMessage]
    ) -> SchemaT | None:
        return structured_call(model, schema, messages, self.usage)


def structured_call[SchemaT: BaseModel](
    model: BaseChatModel,
    schema: type[SchemaT],
    messages: list[BaseMessage],
    usage: Usage,
) -> SchemaT | None:
    """One schema-enforced call, with its token usage banked.

    `include_raw=True` is what makes cost accounting possible at all: the parsed
    object carries no usage metadata, so without the raw message alongside it
    there is nothing to bill against. It also converts a validation failure from
    an exception into a None, which each caller decides how to handle — the
    answer node refuses, the rewrite node falls back to the original question.
    """
    result = model.with_structured_output(schema, include_raw=True).invoke(messages)
    if not isinstance(result, dict):
        return None

    raw = result.get("raw")
    if isinstance(raw, AIMessage):
        usage.record(raw, str(raw.response_metadata.get("model_name") or ""))

    parsed = result.get("parsed")
    return parsed if isinstance(parsed, schema) else None


def rewrite_question(model: BaseChatModel, question: str, usage: Usage) -> str:
    """Restate a question in the register the statute uses.

    Standalone rather than only a method, because a query rewrite is a
    *retrieval* change and `evals.retrieval_eval --rewrite` scores it with the
    same harness that scored fusion and BM25. Measuring it any other way would
    make it the one retrieval decision in this repository settled by reading the
    output and finding it plausible.
    """
    parsed = structured_call(
        model,
        SearchQuery,
        [SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=question)],
        usage,
    )
    # Falling back to the original question rather than failing: a rewrite is an
    # optimisation, and losing it should degrade retrieval, not the run.
    return (parsed.query.strip() if parsed else "") or question


def _verify_citations(
    citations: list[Citation], graded: list[Hit]
) -> tuple[list[Citation], list[str]]:
    """Split cited articles into those that were actually retrieved and those not.

    The prompt instructs the model to cite only the supplied articles. This
    checks. A citation to an article the model never saw is the most damaging
    failure this system can produce, because it is the most convincing: a
    plausible sentence carrying a real-looking article number is worse than a
    refusal and worse than an obvious error.
    """
    available = {(hit.act, hit.article) for hit in graded}
    supported = [c for c in citations if (c.act, c.article) in available]
    unsupported = [f"{c.act} {c.article}" for c in citations if (c.act, c.article) not in available]
    return supported, unsupported


# --- graph --------------------------------------------------------------------


def route_after_grading(state: QAState) -> Literal["answer", "refuse"]:
    """The conditional edge.

    Nothing survived grading means the corpus does not answer this question. The
    alternative — handing an empty context to the model and asking anyway — is
    how a RAG system starts inventing law.
    """
    return "answer" if state.get("graded") else "refuse"


def build_qa_graph(flow: QAFlow) -> CompiledStateGraph[QAState]:
    """Wire the nodes. Returns a compiled graph ready to `invoke`."""
    graph = StateGraph(QAState)

    graph.add_node("understand", flow.understand)
    graph.add_node("retrieve", flow.retrieve)
    graph.add_node("grade", flow.grade)
    graph.add_node("answer", flow.answer)
    graph.add_node("refuse", flow.refuse)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route_after_grading, {"answer": "answer", "refuse": "refuse"}
    )
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)

    return graph.compile()


def ask(
    question: str,
    conn: psycopg.Connection[psycopg.rows.DictRow],
    embedder: Embedder,
    settings: Settings,
    usage: Usage | None = None,
) -> QAState:
    """Run one question through the flow."""
    flow = QAFlow(conn, embedder, settings, usage)
    result = build_qa_graph(flow).invoke({"question": question})
    return cast(QAState, result)


# --- cli ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Pytanie po polsku.")
    parser.add_argument("--show-rejected", action="store_true", help="Print graded-out chunks.")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.anthropic_api_key.get_secret_value():
        print("ANTHROPIC_API_KEY is not set; this flow makes billable calls.", file=sys.stderr)
        return 1

    usage = Usage()
    with connect(settings) as conn:
        state = ask(args.question, conn, Embedder(settings), settings, usage)

    print(f'"{args.question}"')
    print(f'rewritten → "{state.get("search_query", "")}"\n')

    if state.get("refused"):
        print(f"REFUSED ({state.get('refusal_reason')})\n")
    print(state.get("answer", ""))

    citations = state.get("citations") or []
    if citations:
        print("\nPodstawa prawna: " + ", ".join(f"{c.act} art. {c.article}" for c in citations))

    # Surfaced rather than swallowed: a model citing an article it was not shown
    # is the failure worth knowing about immediately.
    unsupported = state.get("unsupported_citations") or []
    if unsupported:
        print(f"\n⚠ cited without support in the retrieved set: {', '.join(unsupported)}")

    if args.show_rejected:
        for citation, reason in state.get("rejected") or []:
            print(f"  graded out: {citation} — {reason}")

    print(f"\n{DISCLAIMER}")
    print(f"\n[{usage.summary()}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
