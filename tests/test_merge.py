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


def test_rrf_at_alpha_half_is_the_classic_unweighted_formula():
    """Equal weights scale every score by a constant, so the ordering is unchanged.

    Pinned because the RRF figure already recorded in `metrics/history.jsonl` was
    produced by the unweighted formula, and it stays comparable only while this
    holds.
    """
    lexical = [hit(1, 9.0), hit(2, 8.0), hit(3, 7.0)]
    dense = [hit(3, 0.9), hit(4, 0.8), hit(1, 0.7)]

    ranked = [h.chunk_id for h in merge(lexical, dense, k=4, fusion="rrf", alpha=0.5)]
    unweighted = sorted(
        {1: 1 / 61 + 1 / 63, 2: 1 / 62, 3: 1 / 63 + 1 / 61, 4: 1 / 62}.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    assert ranked == [chunk_id for chunk_id, _ in unweighted]


def test_weighting_rrf_shifts_the_balance_between_the_legs():
    lexical = [hit(1, 9.0)]
    dense = [hit(2, 0.9)]

    assert merge(lexical, dense, k=2, fusion="rrf", alpha=0.9)[0].chunk_id == 2
    assert merge(lexical, dense, k=2, fusion="rrf", alpha=0.1)[0].chunk_id == 1


def test_both_fusions_demote_a_hit_only_one_leg_found():
    """Neither method rescues an exclusive find, and RRF is the harsher of the two.

    Worth pinning because the intuition runs the other way. Under weighted
    fusion the lexical leg's top hit normalises to 1.0 and keeps `1 - alpha` of
    it, so it still outranks the dense tail. Under RRF every dense rank down to
    the candidate cutoff earns `alpha / (60 + rank)`, and with alpha at 0.8 that
    beats `0.2 / 61` all the way down. A leg cannot promote what the other leg
    never saw — which is why the candidate pool bounds the merged result.
    """
    lexical = [hit(1, 9.0)]
    dense = [hit(2, 0.9), hit(3, 0.85), hit(4, 0.8)]

    weighted = [h.chunk_id for h in merge(lexical, dense, k=4, fusion="weighted", alpha=0.8)]
    reciprocal = [h.chunk_id for h in merge(lexical, dense, k=4, fusion="rrf", alpha=0.8)]

    assert weighted.index(1) == 2
    assert reciprocal.index(1) == 3


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
