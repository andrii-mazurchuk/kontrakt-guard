from __future__ import annotations

import pytest
from pydantic import ValidationError

from evals.gold import GoldQuestion, load_gold, normalise_article, save_gold


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("25", "25"),
        ("Art. 25", "25"),
        ("art 25", "25"),
        ("151-1", "151^1"),
        ("151(1)", "151^1"),
        ("Art. 18-3a", "18^3a"),
        # Unicode superscripts need a separator inserted, not a character swap:
        # translating glyph-for-glyph turns "151¹" into "1511", which is a real
        # and entirely different article.
        ("151¹", "151^1"),
        ("18³ᵃ", "18^3a"),
        ("Art. 94¹⁰", "94^10"),
    ],
)
def test_article_references_normalise_to_the_corpus_form(raw, expected):
    assert normalise_article(raw) == expected


def test_superscript_expansion_does_not_collide_with_a_real_article():
    """Art. 151¹ and Art. 1511 must not normalise to the same thing."""
    assert normalise_article("151¹") != normalise_article("1511")


def test_normalisation_is_idempotent():
    once = normalise_article("Art. 151-1")
    assert normalise_article(once) == once


def test_question_requires_at_least_one_ground_truth_article():
    """A question with no ground truth cannot be scored, so it must not be storable."""
    with pytest.raises(ValidationError):
        GoldQuestion(id="q1", question="Czy to jest legalne?", ground_truth_articles=[])


def test_trivially_short_questions_are_rejected():
    with pytest.raises(ValidationError):
        GoldQuestion(id="q1", question="Czemu?", ground_truth_articles=["25"])


def test_articles_are_canonicalised_on_construction():
    q = GoldQuestion(
        id="q1",
        question="Ile wynosi okres próbny?",
        ground_truth_articles=["Art. 25", "151-1"],
    )
    assert q.ground_truth_articles == ["25", "151^1"]


def test_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "gold.jsonl"
    questions = [
        GoldQuestion(
            id="pip-001",
            question="Jaki jest maksymalny okres próbny?",
            ground_truth_articles=["25"],
            topic="okres_probny",
            source_url="https://example.gov.pl",
        )
    ]
    save_gold(questions, path)
    loaded = load_gold(path)
    assert loaded == questions


def test_missing_file_is_an_empty_set(tmp_path):
    assert load_gold(tmp_path / "absent.jsonl") == []


def test_one_question_per_line(tmp_path):
    """The gold set is committed and must diff cleanly, one question at a time."""
    path = tmp_path / "gold.jsonl"
    questions = [
        GoldQuestion(
            id=f"q{i}", question=f"Pytanie numer {i} o pracę?", ground_truth_articles=["1"]
        )
        for i in range(3)
    ]
    save_gold(questions, path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_unknown_topic_is_rejected():
    """Topics are a closed set so the coverage report cannot be quietly diluted."""
    with pytest.raises(ValidationError):
        GoldQuestion(
            id="q1",
            question="Czy to jest zgodne z prawem?",
            ground_truth_articles=["25"],
            topic="wymyślony_temat",
        )
