"""Database access for ingestion and retrieval.

Connection settings come from `Settings`, so the DSN is assembled in exactly one
place and never drifts between the loader, the retriever and the tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import DictRow, dict_row

from kontrakt_guard.config import Settings, get_settings

SCHEMA_PATH = Path("ingestion/schema.sql")

# The text-search configuration created by docker/initdb/02-polish-fts.sql.
# Named once here so a rename cannot silently desynchronise the indexed tsvector
# from the query-time one — which would degrade recall without any error.
TS_CONFIG = "polish"


@contextmanager
def connect(settings: Settings | None = None) -> Iterator[psycopg.Connection[DictRow]]:
    settings = settings or get_settings()
    with psycopg.connect(settings.postgres_dsn, row_factory=dict_row) as conn:
        yield conn


def apply_schema(conn: psycopg.Connection[DictRow], path: Path = SCHEMA_PATH) -> None:
    """Create the schema. Idempotent — every statement is IF NOT EXISTS."""
    conn.execute(path.read_text(encoding="utf-8"))
    conn.commit()


def polish_config_available(conn: psycopg.Connection[DictRow]) -> bool:
    """Whether the Polish text-search configuration exists.

    Stock Postgres has none, so a database started from the wrong image would
    otherwise fail deep inside a query with a confusing message.
    """
    row = conn.execute("SELECT 1 FROM pg_ts_config WHERE cfgname = %s", (TS_CONFIG,)).fetchone()
    return row is not None
