from pathlib import Path
import subprocess
import sys

import pytest

from src.efllex import load_efllex_vocabulary
from src.lexical.efllex import efllex_analysis


MOCK_VOCABULARY = {
    "important": {
        "A1": 0.0,
        "A2": 0.5,
        "B1": 0.5,
        "B2": 0.0,
        "C1": 0.0,
    },
    "significant": {
        "A1": 0.0,
        "A2": 0.0,
        "B1": 0.0,
        "B2": 0.7,
        "C1": 0.3,
    },
    "sustainable": {
        "A1": 0.0,
        "A2": 0.0,
        "B1": 0.0,
        "B2": 0.2,
        "C1": 0.8,
    },
}


def test_known_word_is_found():
    result = efllex_analysis("Important.", MOCK_VOCABULARY)

    assert result["total_words"] == 1
    assert result["known_words"] == 1
    assert result["unknown_words"] == []


def test_unknown_word_is_separate_from_learner_calculations():
    result = efllex_analysis("The technlogy.", MOCK_VOCABULARY)

    assert result["total_words"] == 1
    assert result["known_words"] == 0
    assert result["unknown_words"] == ["technlogy"]
    assert result["average_learner_index"] == 0.0
    assert result["high_c1_associated_words"] == []


def test_repeated_word_counts_each_occurrence():
    text = "Important ideas remain important."
    result = efllex_analysis(text, MOCK_VOCABULARY)

    assert result["total_words"] == 4
    assert result["known_words"] == 2
    assert result["learner_association_profile"]["A2"] == 50.0
    assert result["learner_association_profile"]["B1"] == 50.0


def test_association_profile_and_average_index_are_weighted_correctly():
    text = "Important changes are significant."
    result = efllex_analysis(text, MOCK_VOCABULARY)

    assert result["known_words"] == 2
    assert result["learner_association_profile"] == {
        "A1": 0.0,
        "A2": 25.0,
        "B1": 25.0,
        "B2": 35.0,
        "C1": 15.0,
    }
    assert result["average_learner_index"] == pytest.approx(3.4)
    assert result["high_c1_associated_words"] == []


def test_common_word_is_not_high_c1_associated():
    result = efllex_analysis("Important.", MOCK_VOCABULARY)

    assert result["high_c1_associated_words"] == []


def test_b2_association_is_preserved_without_creating_a_word_level():
    result = efllex_analysis("Significant.", MOCK_VOCABULARY)

    assert result["learner_association_profile"]["B2"] == 70.0
    assert result["learner_association_profile"]["C1"] == 30.0
    assert result["high_c1_associated_words"] == []


def test_high_c1_word_is_described_as_an_association_not_a_word_level():
    result = efllex_analysis("Sustainable.", MOCK_VOCABULARY)

    assert result["learner_association_profile"]["B2"] == 20.0
    assert result["learner_association_profile"]["C1"] == 80.0
    assert result["high_c1_associated_words"] == [
        {
            "word": "sustainable",
            "c1_association": 0.8,
            "distribution": {
                "A1": 0.0,
                "A2": 0.0,
                "B1": 0.0,
                "B2": 0.2,
                "C1": 0.8,
            },
        }
    ]


def test_empty_text_returns_zero_values():
    result = efllex_analysis("", MOCK_VOCABULARY)

    assert result == {
        "total_words": 0,
        "known_words": 0,
        "unknown_words": [],
        "learner_association_profile": {
            "A1": 0.0,
            "A2": 0.0,
            "B1": 0.0,
            "B2": 0.0,
            "C1": 0.0,
        },
        "average_learner_index": 0.0,
        "high_c1_associated_words": [],
    }


def test_loader_shaped_vocabulary_is_supported():
    loader_shaped_vocabulary = {
        "important": {
            "JJ": [
                {
                    "level_frequencies": {
                        "A1": 0.0,
                        "A2": 5.0,
                        "B1": 5.0,
                        "B2": 0.0,
                        "C1": 0.0,
                    }
                }
            ]
        }
    }

    result = efllex_analysis("Important.", loader_shaped_vocabulary)

    assert result["known_words"] == 1
    assert result["learner_association_profile"]["A2"] == 50.0
    assert result["learner_association_profile"]["B1"] == 50.0


def test_main_efllex_output_uses_association_language():
    project_directory = Path(__file__).resolve().parents[1]
    completed_process = subprocess.run(
        [sys.executable, "main.py"],
        cwd=project_directory,
        capture_output=True,
        text=True,
        check=True,
    )
    efllex_output = completed_process.stdout.split(
        "EFLLex Learner Association Profile:",
        maxsplit=1,
    )[1]

    assert "Advanced learner" not in efllex_output
    assert "learner tendency" not in efllex_output
    assert " - C1" not in efllex_output
    assert " - B2" not in efllex_output
    assert "C1 association:" in efllex_output
    assert "Average EFLLex learner index:" in efllex_output
    assert "Index scale: 1 = A1, 2 = A2, 3 = B1, 4 = B2, 5 = C1" in (
        efllex_output
    )


def test_requested_words_are_associations_not_cefr_word_levels():
    vocabulary = load_efllex_vocabulary()
    text = "Housing industry can develop a specific method."
    result = efllex_analysis(text, vocabulary)
    expected_words = ("housing", "industry", "develop", "specific", "method")
    found_words = []

    for word_association in result["high_c1_associated_words"]:
        found_words.append(word_association["word"])
        assert "level" not in word_association
        assert word_association["c1_association"] >= 0.5
        assert sum(word_association["distribution"].values()) == pytest.approx(1.0)

    for word in expected_words:
        assert word in found_words
