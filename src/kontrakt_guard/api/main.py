"""API surface.

``POST /ask`` serves the Q&A graph. ``POST /audit`` arrives with the audit flow;
adding its route before the engine exists would only produce a passing health
check over a hollow API.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graphs.llm import Usage
from graphs.prompts import DISCLAIMER as DISCLAIMER_ANSWER
from graphs.qa_flow import QAState, ask
from ingestion.db import connect
from ingestion.embed import Embedder
from kontrakt_guard import DISCLAIMER_EN, DISCLAIMER_PL, __version__
from kontrakt_guard.config import get_settings

app = FastAPI(
    title="Kontrakt-Guard",
    version=__version__,
    description=(
        "Polish employment-contract auditor. Clause-level verdicts grounded in "
        "consolidated Polish labour law, with article-level citations. " + DISCLAIMER_EN
    ),
)


class Health(BaseModel):
    status: str
    version: str


class Meta(BaseModel):
    name: str
    version: str
    disclaimer_en: str
    disclaimer_pl: str


@app.get("/health", response_model=Health, tags=["ops"])
async def health() -> Health:
    """Liveness probe. Deliberately does not touch the database.

    A readiness check that also verifies pgvector belongs next to the retrieval
    layer, so that "ready" means "can actually answer", not merely "process up".
    """
    return Health(status="ok", version=__version__)


@app.get("/", response_model=Meta, tags=["ops"])
async def root() -> Meta:
    return Meta(
        name="kontrakt-guard",
        version=__version__,
        disclaimer_en=DISCLAIMER_EN,
        disclaimer_pl=DISCLAIMER_PL,
    )


# --- Q&A ----------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, description="Pytanie po polsku.")


class AskCitation(BaseModel):
    act: str
    article: str


class AskResponse(BaseModel):
    """The answer, plus enough of the graph's state to explain how it got there.

    `search_query`, `refused` and `considered` are returned rather than hidden
    because an answer nobody can attribute to a step is an answer nobody can
    debug. `cost_usd` is included for the same reason the metrics rows carry it.
    """

    answer: str
    citations: list[AskCitation]
    refused: bool
    refusal_reason: str | None = None
    search_query: str
    considered: int = Field(description="Chunks retrieved before relevance grading.")
    grounded_in: int = Field(description="Chunks that survived grading.")
    disclaimer: str
    cost_usd: float


@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    """Loaded once per process. The model is ~2 GB; per-request is not an option."""
    return Embedder(get_settings())


@app.post("/ask", response_model=AskResponse, tags=["qa"])
async def ask_endpoint(request: AskRequest) -> AskResponse:
    """Answer an employment-law question, grounded in the corpus, or refuse.

    A refusal is a 200 with `refused: true`, not an error status. "I cannot
    ground this in the corpus" is a successful outcome of the graph — treating it
    as a failure would push callers toward retrying rather than reading it.
    """
    settings = get_settings()
    if not settings.anthropic_api_key.get_secret_value():
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")

    usage = Usage()

    def run() -> QAState:
        with connect(settings) as conn:
            return ask(request.question, conn, _embedder(), settings, usage)

    # The graph is synchronous and both its legs block — psycopg on the socket,
    # sentence-transformers on the CPU. Running it inline would stall the event
    # loop for every other request for the duration.
    state = await asyncio.to_thread(run)

    return AskResponse(
        answer=state.get("answer", ""),
        citations=[AskCitation(act=c.act, article=c.article) for c in state.get("citations") or []],
        refused=state.get("refused", False),
        refusal_reason=state.get("refusal_reason") or None,
        search_query=state.get("search_query", ""),
        considered=len(state.get("hits") or []),
        grounded_in=len(state.get("graded") or []),
        disclaimer=DISCLAIMER_ANSWER,
        cost_usd=round(usage.cost_usd, 6),
    )
