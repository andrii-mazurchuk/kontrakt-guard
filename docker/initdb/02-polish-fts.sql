-- Polish full-text search configuration, for the lexical half of hybrid retrieval.
--
-- Postgres has no built-in Polish configuration (Snowball provides no Polish
-- stemmer), so one is assembled here from the Hunspell dictionary installed by
-- docker/db/Dockerfile.
--
-- Why this matters for retrieval: Polish inflects heavily. A user asks about
-- "wynagrodzenia za nadgodziny" while the statute says "wynagrodzenie za pracę
-- w godzinach nadliczbowych". Without lemmatisation those are different tokens
-- and the lexical leg contributes nothing — leaving hybrid retrieval as dense
-- retrieval wearing a costume.

CREATE TEXT SEARCH DICTIONARY polish_hunspell (
    TEMPLATE  = ispell,
    DictFile  = polish,
    AffFile   = polish,
    StopWords = polish
);

CREATE TEXT SEARCH DICTIONARY polish_simple (
    TEMPLATE  = pg_catalog.simple,
    StopWords = polish
);

CREATE TEXT SEARCH CONFIGURATION polish (COPY = pg_catalog.simple);

-- Hunspell first; fall through to `simple` so that a token the dictionary does
-- not recognise is still indexed verbatim rather than dropped. Legal texts are
-- full of such tokens — proper nouns, Latin tags, and the abbreviations that
-- make up citations.
--
-- `asciiword` must be listed explicitly and is easy to miss. Postgres's parser
-- emits `asciiword` for tokens made purely of ASCII letters and `word` only for
-- tokens carrying non-ASCII characters. Most Polish vocabulary has no diacritics
-- at all — "wynagrodzenia", "umowy", "godzinach" — so a mapping that covers only
-- `word` lemmatises "pracę" while leaving "wynagrodzenia" untouched, which looks
-- like a broken dictionary when it is a broken mapping.
ALTER TEXT SEARCH CONFIGURATION polish
    ALTER MAPPING FOR
        asciiword, asciihword, hword_asciipart,
        word, hword, hword_part
    WITH polish_hunspell, polish_simple;

-- Numbers, and the article identifiers built from them, must survive intact:
-- they carry the citation.
ALTER TEXT SEARCH CONFIGURATION polish
    ALTER MAPPING FOR int, uint, numword, numhword, hword_numpart
    WITH simple;
