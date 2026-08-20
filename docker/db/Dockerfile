# syntax=docker/dockerfile:1

# Postgres with pgvector AND a Polish full-text search configuration.
#
# Stock Postgres ships 29 text-search configurations and Polish is not among
# them: Snowball has no Polish stemmer, because Polish morphology is too
# irregular for the suffix-stripping approach Snowball uses. Without a
# configuration, the lexical half of hybrid retrieval would have to run on the
# `simple` dictionary, which does no lemmatisation at all — and in a heavily
# inflected language that means "wynagrodzenia" in a question fails to match
# "wynagrodzenie" in the statute. That is precisely the kind of match the
# lexical leg exists to catch.
#
# Hunspell handles it. The Debian `hunspell-pl` package carries a full Polish
# morphological dictionary, which Postgres can load through its `ispell`
# template.

FROM pgvector/pgvector:pg17

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends hunspell-pl; \
    SHAREDIR="$(pg_config --sharedir)/tsearch_data"; \
    # Postgres requires the dictionary in the database encoding (UTF-8) and
    # named .dict/.affix. Debian ships it as ISO8859-2, so convert both the
    # bytes and the encoding declaration on the affix file's first line.
    iconv -f ISO8859-2 -t UTF-8 /usr/share/hunspell/pl_PL.dic > "$SHAREDIR/polish.dict"; \
    iconv -f ISO8859-2 -t UTF-8 /usr/share/hunspell/pl_PL.aff \
      | sed '1s/^SET ISO8859-2$/SET UTF-8/' > "$SHAREDIR/polish.affix"; \
    head -n 1 "$SHAREDIR/polish.affix" | grep -qx 'SET UTF-8'; \
    apt-get purge -y --auto-remove hunspell-pl; \
    rm -rf /var/lib/apt/lists/*

COPY docker/db/polish.stop /tmp/polish.stop
RUN cp /tmp/polish.stop "$(pg_config --sharedir)/tsearch_data/polish.stop" && rm /tmp/polish.stop
