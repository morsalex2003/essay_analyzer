import copy
import json

import pytest

from src.analysis import analyze_writing
from src.revisions import compare_revisions


def test_revision_comparison_tracks_unrecognized_words_neutrally():
    original = analyze_writing("Technlogy helps students.")
    revised = analyze_writing("Technology helps students.")
    comparison = compare_revisions(original, revised)
    unknown_changes = comparison["review_changes"][
        "unknown_frequency_words"
    ]

    assert unknown_changes["no_longer_present"][0]["word"] == "technlogy"
    assert unknown_changes["still_present"] == []
    assert unknown_changes["newly_present"] == []
    assert "not automatically an improvement" in comparison["scope_note"]


def test_revision_comparison_tracks_repetition_counts_and_occurrences():
    original = analyze_writing("People learn. People change. People improve.")
    revised = analyze_writing("People learn. Students change. Writers improve.")
    comparison = compare_revisions(original, revised)
    repetition_changes = comparison["review_changes"][
        "repeated_surface_forms"
    ]

    assert repetition_changes["no_longer_present"][0]["word"] == "people"
    assert repetition_changes["no_longer_present"][0]["count"] == 3
    assert repetition_changes["still_present"] == []
    assert repetition_changes["newly_present"] == []


def test_revision_comparison_reports_metric_deltas_without_direction_labels():
    original = analyze_writing("Cats sleep.")
    revised = analyze_writing("Cats sleep. Dogs learn.")
    comparison = compare_revisions(original, revised)

    assert comparison["metric_changes"]["word_count"] == {
        "before": 2,
        "after": 4,
        "delta": 2,
    }
    assert comparison["metric_changes"]["sentence_count"] == {
        "before": 1,
        "after": 2,
        "delta": 1,
    }
    assert "direction" not in comparison["metric_changes"]["word_count"]


def test_retained_evidence_keeps_before_and_after_locations():
    original = analyze_writing("Technlogy helps. Technlogy changes.")
    revised = analyze_writing("Today technlogy helps. Technlogy changes.")
    comparison = compare_revisions(original, revised)
    retained = comparison["review_changes"]["unknown_frequency_words"][
        "still_present"
    ][0]

    assert retained["word"] == "technlogy"
    assert retained["before_count"] == 2
    assert retained["after_count"] == 2
    assert retained["before_occurrences"][0]["start_char"] == 0
    assert retained["after_occurrences"][0]["start_char"] == 6


def test_cefr_comparison_is_optional():
    original = analyze_writing("Students learn.")
    revised = analyze_writing("Students learn clearly.")

    comparison = compare_revisions(original, revised)

    assert comparison["cefr_changes"] is None


def test_cefr_comparison_reports_descriptive_percentage_changes():
    vocabulary = {
        "people": {"NOUN": ["A1"]},
        "significant": {"ADJ": ["B2"]},
    }
    original = analyze_writing("People.", cefr_vocabulary=vocabulary)
    revised = analyze_writing(
        "People are significant.",
        cefr_vocabulary=vocabulary,
    )

    comparison = compare_revisions(original, revised)

    assert comparison["cefr_changes"]["advanced_percentage"] == {
        "before": 0.0,
        "after": 50.0,
        "delta": 50.0,
    }
    assert "descriptive" in comparison["cefr_changes"][
        "interpretation_note"
    ]


def test_revision_comparison_rejects_different_analysis_schemas():
    original = analyze_writing("Students learn.")
    revised = analyze_writing("Students learn clearly.")
    revised["analysis_schema_version"] = "2.0"

    with pytest.raises(ValueError, match="matching analysis schema versions"):
        compare_revisions(original, revised)


def test_revision_report_is_json_serializable_and_does_not_mutate_inputs():
    original = analyze_writing("Students learn.")
    revised = analyze_writing("Students learn clearly.")
    original_copy = copy.deepcopy(original)
    revised_copy = copy.deepcopy(revised)

    comparison = compare_revisions(original, revised)

    json.dumps(comparison)
    assert original == original_copy
    assert revised == revised_copy


def test_revision_comparison_reports_mattr_method_for_each_revision():
    original = analyze_writing("Students learn.")
    revised = analyze_writing("Students learn because teachers explain.")
    comparison = compare_revisions(original, revised)

    assert comparison["mattr_method"]["before"] == (
        "ttr_fallback_below_window"
    )
    assert comparison["mattr_method"]["after"] == (
        "ttr_fallback_below_window"
    )
    assert "only comparable" in comparison["mattr_method"][
        "interpretation_note"
    ]
