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
