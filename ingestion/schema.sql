-- Corpus schema. Owned solely by the ingestion layer; the extension and
-- text-search setup live in docker/initdb/ because they must exist before any
-- table that depends on them.
--
-- Idempotent: safe to run against a populated database.

CREATE TABLE IF NOT EXISTS chunks (
    id               bigserial PRIMARY KEY,

    -- Provenance. `article` is the canonical ASCII id ('25', '9^1', '18^3ca')
    -- and is the ground-truth key that recall@k is computed against, so it must
    -- never be reformatted for display purposes.
    act              text        NOT NULL,
    article          text        NOT NULL,
    article_display  text        NOT NULL,
    paragraph        text        NOT NULL DEFAULT '',
    part_index       int         NOT NULL DEFAULT 0,

    title_path       text[]      NOT NULL DEFAULT '{}',
    page_start       int,

    -- An article that carries no operative law is kept, not dropped: a question
    -- naming one deserves the specific reason rather than silence. Retrieval
    -- filters on this rather than the row being absent.
    repealed         boolean     NOT NULL DEFAULT false,
    repeal_kind      text        NOT NULL DEFAULT '',

    content          text        NOT NULL,
    n_tokens         int         NOT NULL DEFAULT 0,

    -- Lexical leg of hybrid retrieval. Generated rather than maintained by the
    -- application, so it can never drift from `content`.
    --
    -- The structural path is indexed alongside the text: a question about remote
    -- work should reach articles sitting under "Rozdział IIc — Praca zdalna"
    -- even when the article body never repeats the chapter's wording.
    -- What the lexical index is built over: the structural path followed by the
    -- article text. Written by the loader rather than derived in SQL, because
    -- `array_to_string` is only STABLE (its element output functions may be), and
    -- a generated column requires every input to be IMMUTABLE. Postgres reports
    -- this as "generation expression is not immutable", which names the symptom
    -- and not the offending function.
    search_text      text        NOT NULL,

    -- Lexical leg of hybrid retrieval. Generated from search_text so the two can
    -- never drift apart.
    --
    -- 'polish'::regconfig, not 'polish': without the cast Postgres resolves the
    -- single-argument to_tsvector, which reads the configuration from a session
    -- setting and is therefore also not immutable.
    tsv              tsvector GENERATED ALWAYS AS (
        to_tsvector('polish'::regconfig, search_text)
    ) STORED,

    -- Dense leg. 1024 dimensions = multilingual-e5-large. Changing the model
    -- changes this width, which is a migration rather than a config tweak — and
    -- invalidates every recorded metric, hence the pin in retrieval_config_hash.
    embedding        vector(1024),

    -- One row per (article, paragraph, part). `part_index` distinguishes the
    -- pieces of an article too long to embed whole.
    CONSTRAINT chunks_identity UNIQUE (act, article, paragraph, part_index)
);

-- Lexical search.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

-- Dense search. HNSW over cosine distance: e5 embeddings are normalised, so
-- cosine and inner product rank identically, and cosine is the documented
-- pairing for this model.
--
-- Built here for convenience. At corpus scale (thousands of rows) index build
-- time is negligible; for a far larger corpus this would move to after the load.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Citation lookup: "what does Art. 25 say" must not become a similarity search.
CREATE INDEX IF NOT EXISTS chunks_citation_idx ON chunks (act, article);

-- Fuzzy matching on article ids, for the many ways a Polish citation can be
-- written ("art. 29 § 1", "Art. 29 par. 1", "art 29 ust 1").
CREATE INDEX IF NOT EXISTS chunks_article_trgm_idx
    ON chunks USING gin (article gin_trgm_ops);


-- ---------------------------------------------------------------------------
-- BM25 statistics
-- ---------------------------------------------------------------------------
-- Postgres has no BM25. `ts_rank` and `ts_rank_cd` score term frequency and
-- cover density but carry **no inverse document frequency term**, so a lexeme
-- occurring in every chunk counts for as much as one occurring in three. On this
-- corpus that is not a subtle effect: 'art', 'dział' and 'dziać' each appear in
-- 543 of 543 chunks, and 'pracownik' in 403 — the exact words a Polish
-- employment-law question is most likely to contain.
--
-- The inverted index is materialised because BM25 needs per-(chunk, lexeme) term
-- frequency and per-chunk length, neither of which is reachable from a tsvector
-- without unnesting it. Unnesting at query time is O(corpus) per search.
--
-- Refreshed by the loader in the same transaction as the upsert, so the two can
-- never be committed out of step; `lexical_index_is_stale` re-checks before any
-- eval, because a stale index would move a metric with nothing to show for it.
CREATE MATERIALIZED VIEW IF NOT EXISTS chunk_terms AS
SELECT
    c.id AS chunk_id,
    t.lexeme,
    -- Term frequency. A tsvector position list is capped at 256 entries per
    -- lexeme; nothing here approaches that (the longest chunk is 279 tokens
    -- in total). `positions` is NULL only for a stripped tsvector, which this
    -- schema never produces — the coalesce is a floor, not an expected path.
    coalesce(array_length(t.positions, 1), 1)::int AS tf,
    -- Document length, denormalised onto every term row so scoring needs no
    -- second join. BM25's length normalisation compares this against avgdl.
    sum(coalesce(array_length(t.positions, 1), 1))
        OVER (PARTITION BY c.id)::int AS doc_len
FROM chunks c, unnest(c.tsv) t;

-- Lexeme first: a search looks up the handful of lexemes in the question, so the
-- leading column must be the one being filtered. Unique overall, which is also
-- what REFRESH ... CONCURRENTLY would require were the corpus ever large enough
-- to need it.
CREATE UNIQUE INDEX IF NOT EXISTS chunk_terms_lexeme_idx
    ON chunk_terms (lexeme, chunk_id);

-- Corpus-level constants. One row, so scoring reads them without scanning.
--
-- n_docs counts chunks rather than distinct chunk_ids in chunk_terms: a chunk
-- whose tsvector came out empty still exists and still belongs in the
-- denominator of the IDF term.
CREATE MATERIALIZED VIEW IF NOT EXISTS corpus_stats AS
SELECT
    (SELECT count(*) FROM chunks)::float8 AS n_docs,
    (SELECT coalesce(avg(doc_len), 1)
       FROM (SELECT DISTINCT chunk_id, doc_len FROM chunk_terms) d)::float8 AS avgdl;
