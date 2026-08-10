import pytest

from src.syntax import (
    analyze_syntax,
    average_sentence_length,
    complex_sentence_percentage,
    count_pos,
    count_sentences,
    count_subordinate_clauses,
    longest_sentence_length,
)


def test_empty_text_returns_zero_values():
    result = analyze_syntax("")

    assert result == {
        "sentences": 0,
        "average_sentence_length": 0.0,
        "sentence_length_std": 0.0,
        "total_clauses": 0,
        "clause_density": 0.0,
        "subordination_ratio": 0.0,
    }


def test_sentence_lengths_and_variation_are_calculated():
    result = analyze_syntax("Birds fly. Cats chase mice.")

    assert result["sentences"] == 2
    assert result["average_sentence_length"] == 2.5
    assert result["sentence_length_std"] == 0.5
    assert result["total_clauses"] == 2
    assert result["clause_density"] == 1.0
    assert result["subordination_ratio"] == 0.0


def test_clause_density_uses_verb_tokens():
    result = analyze_syntax("People think animals need protection.")

    assert result["sentences"] == 1
    assert result["total_clauses"] == 2
    assert result["clause_density"] == 2.0


def test_subordination_ratio_counts_affected_sentences():
    text = "People think animals need protection. Birds fly."
    result = analyze_syntax(text)

    assert result["sentences"] == 2
    assert result["subordination_ratio"] == pytest.approx(0.5)


def test_migrated_metrics_handle_empty_text():
    assert count_sentences("") == 0
    assert average_sentence_length("") == 0.0
    assert count_pos("") == {}
    assert count_subordinate_clauses("") == 0
    assert longest_sentence_length("") == 0
    assert complex_sentence_percentage("") == 0.0


def test_migrated_metrics_ignore_punctuation_and_whitespace():
    text = "Hello!!! Cats, dogs... Why?"

    assert count_sentences(text) == 2
    assert average_sentence_length(text) == 2.0
    assert count_pos(text) == {"INTJ": 1, "NOUN": 2, "SCONJ": 1}
    assert count_subordinate_clauses(text) == 0
    assert longest_sentence_length(text) == 3
    assert complex_sentence_percentage(text) == 0.0


def test_migrated_metrics_preserve_subordination_behavior():
    text = (
        "People think that animals need protection because habitats disappear."
    )

    assert count_sentences(text) == 1
    assert average_sentence_length(text) == 9.0
    assert count_pos(text) == {"NOUN": 4, "SCONJ": 2, "VERB": 3}
    assert count_subordinate_clauses(text) == 2
    assert longest_sentence_length(text) == 9
    assert complex_sentence_percentage(text) == 100.0
