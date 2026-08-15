import pytest

from src.syntax import (
    analyze_syntax,
    average_sentence_length,
    complex_sentence_percentage,
    count_pos,
    count_sentences,
    count_subordinate_clauses,
    longest_sentence_length,
    max_dependency_tree_depth,
    mean_dependency_distance,
    mean_dependency_tree_depth,
)


def test_empty_text_returns_zero_values():
    result = analyze_syntax("")

    assert result == {
        "sentences": 0,
        "average_sentence_length": 0.0,
        "sentence_length_std": 0.0,
        "total_clauses": 0,
        "finite_clauses": 0,
        "dependent_clauses": 0,
        "non_finite_predicative_units": 0,
        "clause_density": 0.0,
        "dependent_clause_ratio": 0.0,
        "subordination_ratio": 0.0,
        "mean_dependency_distance": 0.0,
        "mean_dependency_tree_depth": 0.0,
        "max_dependency_tree_depth": 0,
    }

    assert mean_dependency_distance("") == 0.0
    assert mean_dependency_tree_depth("") == 0.0
    assert max_dependency_tree_depth("") == 0


def test_sentence_lengths_and_variation_are_calculated():
    result = analyze_syntax("Birds fly. Cats chase mice.")

    assert result["sentences"] == 2
    assert result["average_sentence_length"] == 2.5
    assert result["sentence_length_std"] == 0.5
    assert result["total_clauses"] == 2
    assert result["finite_clauses"] == 2
    assert result["dependent_clauses"] == 0
    assert result["non_finite_predicative_units"] == 0
    assert result["clause_density"] == 1.0
    assert result["dependent_clause_ratio"] == 0.0
    assert result["subordination_ratio"] == 0.0


def test_clause_density_uses_finite_clauses():
    result = analyze_syntax("People think animals need protection.")

    assert result["sentences"] == 1
    assert result["total_clauses"] == 2
    assert result["finite_clauses"] == 2
    assert result["dependent_clauses"] == 1
    assert result["clause_density"] == 2.0
    assert result["dependent_clause_ratio"] == 0.5


def test_subordination_ratio_counts_affected_sentences():
    text = "People think animals need protection. Birds fly."
    result = analyze_syntax(text)

    assert result["sentences"] == 2
    assert result["subordination_ratio"] == pytest.approx(0.5)


def test_non_finite_unit_is_separate_from_finite_clause_counts():
    text = "I want to study abroad."
    result = analyze_syntax(text)

    assert result["total_clauses"] == 1
    assert result["finite_clauses"] == 1
    assert result["dependent_clauses"] == 0
    assert result["non_finite_predicative_units"] == 1
    assert result["dependent_clause_ratio"] == 0.0
    assert result["subordination_ratio"] == 0.0
    assert count_subordinate_clauses(text) == 0
    assert complex_sentence_percentage(text) == 0.0


@pytest.mark.parametrize(
    (
        "text",
        "expected_dependent_clauses",
        "expected_non_finite_units",
        "expected_subordination_ratio",
    ),
    (
        (
            "I think that education is important.",
            1,
            0,
            1.0,
        ),
        (
            "I want to study abroad.",
            0,
            1,
            0.0,
        ),
        (
            "Having finished the task, I went home.",
            0,
            1,
            0.0,
        ),
        (
            "Although I was tired, I continued working.",
            1,
            1,
            1.0,
        ),
        (
            "Anything or anyone has the right to decide.",
            0,
            1,
            0.0,
        ),
        (
            "I think that students want to study abroad.",
            1,
            1,
            1.0,
        ),
    ),
)
def test_sentence_subordination_requires_a_finite_dependent_clause(
    text,
    expected_dependent_clauses,
    expected_non_finite_units,
    expected_subordination_ratio,
):
    result = analyze_syntax(text)

    assert result["dependent_clauses"] == expected_dependent_clauses
    assert (
        result["non_finite_predicative_units"]
        == expected_non_finite_units
    )
    assert result["subordination_ratio"] == expected_subordination_ratio
    assert complex_sentence_percentage(text) == (
        expected_subordination_ratio * 100
    )


def test_non_finite_fragment_handles_zero_finite_clause_denominator():
    result = analyze_syntax("To study.")

    assert result["sentences"] == 1
    assert result["total_clauses"] == 0
    assert result["finite_clauses"] == 0
    assert result["non_finite_predicative_units"] == 1
    assert result["clause_density"] == 0.0
    assert result["dependent_clause_ratio"] == 0.0


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


def test_simple_sentence_dependency_metrics():
    result = analyze_syntax("Birds fly.")

    assert result["mean_dependency_distance"] == 1.0
    assert result["mean_dependency_tree_depth"] == 1.0
    assert result["max_dependency_tree_depth"] == 1


def test_modifiers_increase_dependency_tree_depth():
    text = "The bird with bright blue feathers sings."
    result = analyze_syntax(text)

    assert result["mean_dependency_distance"] == pytest.approx(13 / 6)
    assert result["mean_dependency_tree_depth"] == 4.0
    assert result["max_dependency_tree_depth"] == 4


def test_subordinate_clause_dependency_metrics():
    text = "People think that animals need protection."
    result = analyze_syntax(text)

    assert result["mean_dependency_distance"] == 1.6
    assert result["mean_dependency_tree_depth"] == 2.0
    assert result["max_dependency_tree_depth"] == 2
    assert result["subordination_ratio"] == 1.0


def test_multiple_sentence_depths_are_averaged():
    text = "Birds fly. Cats chase mice."
    result = analyze_syntax(text)

    assert result["mean_dependency_distance"] == 1.0
    assert result["mean_dependency_tree_depth"] == 1.0
    assert result["max_dependency_tree_depth"] == 1
    assert result["sentences"] == 2


def test_punctuation_dependencies_are_excluded():
    result = analyze_syntax("Birds fly!!!")

    assert result["mean_dependency_distance"] == 1.0
    assert result["mean_dependency_tree_depth"] == 1.0
    assert result["max_dependency_tree_depth"] == 1
