"""Fusion tests. Pure functions — no database, no model."""

from __future__ import annotations

from retrieval.search import Hit, merge


def hit(chunk_id: int, score: float, article: str = "25") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        act="kp",
        article=article,
        article_display=f"Art. {article}",
        content="treść",
        score=score,
    )


def test_a_chunk_found_by_both_legs_outranks_one_found_by_either():
    """The core claim of hybrid retrieval, stated as a test.

    Agreement between an independent lexical and dense signal is evidence; RRF
    exists to reward it.
    """
    lexical = [hit(1, 9.0), hit(2, 8.0), hit(3, 7.0)]
    dense = [hit(4, 0.9), hit(5, 0.8), hit(2, 0.7)]

    merged = merge(lexical, dense, k=5)
    assert merged[0].chunk_id == 2


def test_merge_deduplicates():
    lexical = [hit(1, 9.0), hit(2, 8.0)]
    dense = [hit(1, 0.9), hit(2, 0.8)]
    merged = merge(lexical, dense, k=10)
    assert [h.chunk_id for h in merged] == [1, 2]


def test_merge_respects_k():
    lexical = [hit(i, 10.0 - i) for i in range(1, 8)]
    assert len(merge(lexical, [], k=3)) == 3


def test_provenance_records_which_leg_found_each_hit():
    """A run where every hit comes from one leg means hybrid is not actually hybrid."""
    merged = merge([hit(1, 9.0)], [hit(2, 0.9)], k=5)
    by_id = {h.chunk_id: h for h in merged}

    assert by_id[1].lexical_rank == 1 and by_id[1].dense_rank is None
    assert by_id[2].dense_rank == 1 and by_id[2].lexical_rank is None


def test_either_leg_alone_still_returns_results():
    assert len(merge([hit(1, 9.0)], [], k=5)) == 1
    assert len(merge([], [hit(2, 0.9)], k=5)) == 1
    assert merge([], [], k=5) == []


def test_rrf_ignores_the_incomparable_raw_scales():
    """ts_rank_cd is unbounded, cosine lives in [-1, 1]; only ranks are comparable."""
    lexical = [hit(1, 900.0), hit(2, 0.001)]
    dense = [hit(2, 0.99), hit(1, 0.98)]

    modest = merge(lexical, dense, k=2)
    lexical_inflated = [hit(1, 9_000_000.0), hit(2, 0.001)]
    inflated = merge(lexical_inflated, dense, k=2)

    assert [h.chunk_id for h in modest] == [h.chunk_id for h in inflated]


def test_weighted_fusion_is_available_for_comparison():
    """Which fusion wins on Polish legal questions is settled by recall@k."""
    lexical = [hit(1, 9.0), hit(2, 1.0)]
    dense = [hit(2, 0.9), hit(1, 0.1)]

    dense_heavy = merge(lexical, dense, k=2, fusion="weighted", alpha=1.0)
    lexical_heavy = merge(lexical, dense, k=2, fusion="weighted", alpha=0.0)

    assert dense_heavy[0].chunk_id == 2
    assert lexical_heavy[0].chunk_id == 1


def test_weighted_fusion_survives_a_flat_leg():
    """Min-max normalisation divides by the range, which is zero when all scores tie."""
    lexical = [hit(1, 5.0), hit(2, 5.0)]
    merged = merge(lexical, [], k=2, fusion="weighted", alpha=0.5)
    assert len(merged) == 2


def test_citation_rendering():
    h = hit(1, 1.0)
    h.paragraph = "2^1"
    assert h.citation == "Art. 25 § 21"
    assert hit(2, 1.0).citation == "Art. 25"
