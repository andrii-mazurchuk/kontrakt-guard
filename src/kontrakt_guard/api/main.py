"""API surface.

Only liveness and metadata exist so far. The two real endpoints — ``POST /audit``
and ``POST /ask`` — arrive with the graphs they serve; adding their routes before
the engine exists would only produce a passing health check over a hollow API.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from kontrakt_guard import DISCLAIMER_EN, DISCLAIMER_PL, __version__

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
