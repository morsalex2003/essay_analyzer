import json

import src.analysis
from src.analysis import ANALYSIS_SCHEMA_VERSION, analyze_writing


def test_analysis_returns_a_versioned_application_result():
    result = analyze_writing("Students learn because teachers explain ideas.")

    assert result["analysis_schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert result["input_summary"] == {
        "text_sha256": (
            "0900326df857649a74a5ac85c8472ad1796136d479f4b3ea379fcf23109f556e"
        ),
        "character_count": 46,
        "word_count": 6,
        "sentence_count": 1,
    }
    assert result["syntax"]["finite_clauses"] == 2
    assert result["syntax"]["dependent_clauses"] == 1
    assert result["descriptive_statistics"] == {
        "pos_counts": {
            "NOUN": 3,
            "VERB": 2,
            "SCONJ": 1,
        },
        "longest_sentence_length": 6,
        "complex_sentence_percentage": 100.0,
    }
    assert result["cefr"] is None
    assert result["research_metrics"] == {}


def test_analysis_runs_the_shared_spacy_pipeline_once(monkeypatch):
    original_nlp = src.analysis.nlp
    call_count = 0

    def counting_nlp(text):
        nonlocal call_count
        call_count += 1
        return original_nlp(text)

    monkeypatch.setattr(src.analysis, "nlp", counting_nlp)

    analyze_writing("Students learn because teachers explain ideas.")

    assert call_count == 1


def test_diversity_token_policies_are_explicit():
    result = analyze_writing("The student studies in London.")
    diversity = result["lexical"]["diversity"]

    assert diversity["ttr_token_policy"] == "all_non_punctuation_lemmas"
    assert diversity["mattr_token_policy"] == "content_words"
    assert diversity["ttr_token_count"] == 5
    assert diversity["mattr_token_count"] == 2


def test_surface_form_fallback_prevents_bad_species_lemma_in_word_lists():
    result = analyze_writing("Species and zoos protect species.")

    assert result["lexical"]["most_common_words"][0] == ("species", 2)
    assert result["lexical"]["repeated_words"] == {"species": 2}


def test_user_written_specie_is_not_silently_corrected():
    result = analyze_writing("Specie and specie.")

    assert result["lexical"]["repeated_words"] == {"specie": 2}


def test_repetition_uses_observed_surface_forms():
    result = analyze_writing("Studies and study require different methods.")

    assert result["lexical"]["repeated_words"] == {}
    assert result["lexical"]["repetition_token_policy"] == (
        "non_stop_surface_forms"
    )


def test_lexical_evidence_points_to_exact_text_occurrences():
    text = "Technlogy helps. Technlogy changes."
    result = analyze_writing(text)
    evidence = result["lexical"]["evidence"]

    repeated_entry = evidence["repeated_surface_forms"][0]
    unknown_entry = evidence["unknown_frequency_words"][0]

    assert repeated_entry["word"] == "technlogy"
    assert repeated_entry["count"] == 2
    assert repeated_entry["occurrences"] == [
        {
            "text": "Technlogy",
            "start_char": 0,
            "end_char": 9,
            "sentence_index": 1,
        },
        {
            "text": "Technlogy",
            "start_char": 17,
            "end_char": 26,
            "sentence_index": 2,
        },
    ]
    assert unknown_entry["word"] == "technlogy"
    assert unknown_entry["occurrences"] == repeated_entry["occurrences"]


def test_application_result_is_json_serializable():
    result = analyze_writing("Students learn clearly.")

    json.dumps(result)


def test_sentence_reliability_contains_user_facing_text_offsets():
    text = "I want to study abroad. Students learn quickly."
    result = analyze_writing(text)
    sentences = result["syntax_reliability"]["sentences"]

    assert len(sentences) == 2
    assert sentences[0]["text"] == "I want to study abroad."
    assert sentences[0]["start_char"] == 0
    assert sentences[0]["end_char"] == 23
    assert sentences[0]["reliability"] == "high"
    assert sentences[1]["text"] == "Students learn quickly."


def test_efllex_is_excluded_without_explicit_research_option():
    vocabulary = {
        "student": {
            "NN": [
                {
                    "level_frequencies": {
                        "A1": 1.0,
                        "A2": 0.0,
                        "B1": 0.0,
                        "B2": 0.0,
                        "C1": 0.0,
                    }
                }
            ]
        }
    }

    default_result = analyze_writing(
        "Students learn.",
        efllex_vocabulary=vocabulary,
    )
    research_result = analyze_writing(
        "Students learn.",
        efllex_vocabulary=vocabulary,
        include_research_metrics=True,
    )

    assert default_result["research_metrics"] == {}
    assert "efllex" in research_result["research_metrics"]


def test_empty_text_is_safe():
    result = analyze_writing("")

    assert result["input_summary"] == {
        "text_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "character_count": 0,
        "word_count": 0,
        "sentence_count": 0,
    }
    assert result["lexical"]["diversity"]["ttr"] == 0.0
    assert result["lexical"]["diversity"]["mattr"] == 0.0
    assert result["syntax_reliability"]["syntax_reliability"] == (
        "not_analyzable"
    )
    assert result["syntax_reliability"]["sentences"] == []
