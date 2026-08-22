"""Runtime configuration, loaded from environment or `.env`.

Every field that can influence an eval number is carried here and hashed into
`retrieval_config_hash` on each metrics row. That is what makes a recorded
metric attributable to an exact configuration rather than to a vague "some run".
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Claude API ---------------------------------------------------------
    anthropic_api_key: SecretStr = SecretStr("")

    # Pinned, never floated to an alias. An eval number is only comparable
    # across runs if the model that produced it is fixed.
    model_cheap: str = "claude-haiku-4-5-20251001"
    model_strong: str = "claude-sonnet-5"

    # --- Postgres + pgvector ------------------------------------------------
    postgres_host: str = "localhost"
    # 55432, not 5432: docker-compose publishes the database on a non-standard
    # host port so it cannot collide with a locally installed PostgreSQL. Inside
    # the compose network the api service overrides this to the container's 5432.
    postgres_port: int = 55432
    postgres_db: str = "kontrakt_guard"
    postgres_user: str = "kontrakt"
    postgres_password: SecretStr = SecretStr("")

    # --- Embeddings ---------------------------------------------------------
    embedding_model: str = "intfloat/multilingual-e5-large"
    # Empty means "whatever HEAD is today", which silently invalidates
    # historical numbers. Pin it once the corpus is first embedded.
    embedding_revision: str = ""
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    # --- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = 10
    bm25_candidates: int = 25
    vector_candidates: int = 25

    # How the two candidate lists are combined. Chosen by measurement, not taste:
    # Reciprocal Rank Fusion weights both legs equally, and on this corpus the
    # lexical leg is far weaker than the dense one (recall@5 of 46.9% against
    # 80.2%), so equal weighting injected noise into the top ranks and made hybrid
    # retrieval *worse* than dense alone. See ADR 0003.
    # Whether the Q&A graph rewrites a question into statutory Polish before
    # searching. Off, by measurement: on the gold set it cost 8 points of
    # recall@5 (85.3% -> 77.3%) and 11.8 at k=3, and even lowered candidate-pool
    # recall, meaning it removed correct articles from consideration entirely.
    #
    # The caveat matters as much as the number. The gold set is PIP guidance and
    # statute text, so its questions are *already* formal — the rewrite has no
    # register gap to close there and can only discard terms. It may still earn
    # its place on genuinely colloquial input, which this gold set does not
    # contain and therefore cannot settle. See ADR 0009.
    query_rewrite: bool = False

    # How the lexical leg scores a match. Postgres's ts_rank_cd has no inverse
    # document frequency term, so lexemes present in most of the corpus ranked as
    # strongly as distinguishing ones — on a corpus of employment law, that means
    # "pracownik" carried the same weight as the term the question turns on.
    # BM25 is computed over a materialised inverted index instead. See ADR 0008.
    lexical_ranking: Literal["bm25", "ts_rank_cd"] = "bm25"

    # BM25 free parameters, at their standard values. k1 governs how quickly term
    # frequency saturates, b how strongly length normalisation applies. Statute
    # articles vary in length by an order of magnitude, so b is load-bearing here.
    bm25_k1: float = Field(default=1.2, ge=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    fusion: Literal["weighted", "rrf"] = "weighted"

    # Weight of the dense leg when merging. Re-swept after the lexical leg moved
    # to BM25, because the old optimum was tuned against a much weaker leg.
    # Recall@5 across 0.6/0.7/0.8/0.85 is 84.8/84.3/85.3/83.8 — a plateau roughly
    # one question wide on 97 questions, so the differences inside it are not
    # meaningful on their own. 0.8 is taken because it is the joint best on
    # recall@5, recall@10 and MRR rather than on any one of them. See ADR 0008.
    hybrid_alpha: float = Field(default=0.8, ge=0.0, le=1.0)

    # --- Eval cost control --------------------------------------------------
    eval_max_cost_usd: float = 5.0

    @property
    def postgres_dsn(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def retrieval_config_hash(self) -> str:
        """Stable hash of everything that can move a retrieval metric.

        Deliberately excludes credentials and host/port: moving the same corpus
        to a different machine must not read as a configuration change.
        """
        payload = {
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "retrieval_top_k": self.retrieval_top_k,
            "bm25_candidates": self.bm25_candidates,
            "vector_candidates": self.vector_candidates,
            "query_rewrite": self.query_rewrite,
            "lexical_ranking": self.lexical_ranking,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "fusion": self.fusion,
            "hybrid_alpha": self.hybrid_alpha,
            "model_cheap": self.model_cheap,
            "model_strong": self.model_strong,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
