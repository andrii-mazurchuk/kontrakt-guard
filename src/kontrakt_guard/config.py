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
    # Weight of the dense leg when merging the two candidate lists.
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

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
            "hybrid_alpha": self.hybrid_alpha,
            "model_cheap": self.model_cheap,
            "model_strong": self.model_strong,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
