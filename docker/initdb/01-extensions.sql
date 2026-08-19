-- Runs once, on first initialisation of an empty data volume.
-- Schema (tables, indexes) is owned by the ingestion layer, not by this file:
-- a schema defined in two places drifts.

CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram matching backs fuzzy lookups on article identifiers, where Polish
-- legal citations vary in punctuation ("art. 29 § 1" vs "Art. 29 par. 1").
CREATE EXTENSION IF NOT EXISTS pg_trgm;
