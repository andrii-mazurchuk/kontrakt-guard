from __future__ import annotations

from kontrakt_guard.config import Settings


def test_dsn_is_assembled_from_parts():
    s = Settings(postgres_host="db", postgres_port=5433, postgres_db="kg", postgres_user="u")
    assert s.postgres_dsn.startswith("postgresql://u:")
    assert s.postgres_dsn.endswith("@db:5433/kg")


def test_config_hash_is_stable_across_instances():
    assert Settings().retrieval_config_hash() == Settings().retrieval_config_hash()


def test_config_hash_tracks_retrieval_affecting_settings():
    base = Settings()
    # Must differ from the default, or the assertion passes for the wrong reason.
    assert base.retrieval_config_hash() != Settings(hybrid_alpha=0.4).retrieval_config_hash()
    assert base.retrieval_config_hash() != Settings(retrieval_top_k=20).retrieval_config_hash()
    assert (
        base.retrieval_config_hash() != Settings(embedding_revision="xyz").retrieval_config_hash()
    )


def test_config_hash_tracks_the_fusion_method():
    """Weighted and RRF fusion produced 84.8% and 75.5% recall@5 on the same corpus.

    A metric recorded under one must never be mistaken for the other.
    """
    assert Settings().retrieval_config_hash() != Settings(fusion="rrf").retrieval_config_hash()


def test_config_hash_tracks_the_lexical_ranking_function():
    """BM25 and ts_rank_cd scored 69.6% and 46.9% recall@5 on the same corpus."""
    base = Settings()
    assert (
        base.retrieval_config_hash()
        != Settings(lexical_ranking="ts_rank_cd").retrieval_config_hash()
    )
    assert base.retrieval_config_hash() != Settings(bm25_k1=2.0).retrieval_config_hash()
    assert base.retrieval_config_hash() != Settings(bm25_b=0.4).retrieval_config_hash()
    assert base.retrieval_config_hash() != Settings(query_rewrite=True).retrieval_config_hash()


def test_defaults_match_the_measured_best_configuration():
    """Guards against a default silently reverting to the worse-performing setup."""
    settings = Settings()
    assert settings.fusion == "weighted"
    assert settings.lexical_ranking == "bm25"
    assert settings.hybrid_alpha == 0.8
    # Rewriting cost 8 points of recall@5 on the gold set. See ADR 0009.
    assert settings.query_rewrite is False


def test_config_hash_ignores_where_the_corpus_happens_to_live():
    """Moving the same corpus to another host must not read as a config change."""
    base = Settings()
    moved = Settings(postgres_host="prod.example.com", postgres_port=6543)
    assert base.retrieval_config_hash() == moved.retrieval_config_hash()


def test_secrets_do_not_leak_into_repr():
    s = Settings(anthropic_api_key="sk-ant-supersecret", postgres_password="hunter2")
    assert "supersecret" not in repr(s)
    assert "hunter2" not in repr(s)
