import copy
import json

from src.analysis import analyze_writing
from src.reporting import (
    MIN_STABLE_METRIC_WORD_COUNT,
    build_student_report,
    build_teacher_report,
)


def test_student_report_uses_evidence_without_claiming_spelling_error():
    analysis = analyze_writing("Technlogy helps. Technlogy changes.")
    report = build_student_report(analysis)
    priority = report["priorities"][0]

    assert report["report_type"] == "student"
    assert priority["priority_id"] == "review_unrecognized_words"
    assert "does not by itself mean" in priority["observation"]
    assert priority["evidence"][0]["word"] == "technlogy"
    assert priority["evidence"][0]["occurrences"][1][
        "sentence_index"
    ] == 2


def test_student_report_only_prioritizes_repetition_at_conservative_count():
    two_occurrences = build_student_report(
        analyze_writing("People learn and people change.")
    )
    three_occurrences = build_student_report(
        analyze_writing("People learn. People change. People improve.")
    )

    two_ids = []
    for priority in two_occurrences["priorities"]:
        two_ids.append(priority["priority_id"])

    three_ids = []
    for priority in three_occurrences["priorities"]:
        three_ids.append(priority["priority_id"])

    assert "review_repeated_wording" not in two_ids
    assert "review_repeated_wording" in three_ids


def test_student_report_does_not_prioritize_low_share_topic_repetition():
    analysis = analyze_writing("Planet planet planet.")
    analysis["input_summary"]["word_count"] = 100
    report = build_student_report(analysis)

    priority_ids = []
    for priority in report["priorities"]:
        priority_ids.append(priority["priority_id"])

    assert "review_repeated_wording" not in priority_ids


def test_syntax_warning_is_a_measurement_note_not_writing_feedback():
    analysis = analyze_writing(
        "For example, to help with the difficult question on the class "
        "or do the science article."
    )
    report = build_student_report(analysis)
    syntax_notes = []
    for note in report["measurement_notes"]:
        if note["note_id"] == "syntax_measurement_reliability":
            syntax_notes.append(note)

    assert len(syntax_notes) == 1
    assert "not a grammar judgement" in syntax_notes[0]["message"]


def test_teacher_report_preserves_diagnostic_flags_and_full_metrics():
    analysis = analyze_writing(
        "For example, to help with the difficult question on the class "
        "or do the science article."
    )
    report = build_teacher_report(analysis)

    assert report["report_type"] == "teacher"
    assert report["metrics"]["syntax"] == analysis["syntax"]
    assert report["syntax_reliability"] == analysis["syntax_reliability"]
    assert len(report["review_queue"]["syntax_sentences"]) == 1
    assert "flags" in report["review_queue"]["syntax_sentences"][0]


def test_reports_are_json_serializable_and_do_not_mutate_analysis():
    analysis = analyze_writing("Students learn clearly.")
    original_analysis = copy.deepcopy(analysis)

    student_report = build_student_report(analysis)
    teacher_report = build_teacher_report(analysis)

    json.dumps(student_report)
    json.dumps(teacher_report)
    assert analysis == original_analysis


def _note_ids(report):
    note_ids = []
    for note in report["measurement_notes"]:
        note_ids.append(note["note_id"])

    return note_ids


def _sentences_with_distinct_words(sentence_count):
    # Each sentence adds new content words, so long inputs reach the MATTR
    # window without repeating a single filler word.
    sentences = []
    for sentence_index in range(sentence_count):
        sentences.append(
            "Student number "
            + str(sentence_index)
            + " carefully explains another interesting idea today."
        )

    return " ".join(sentences)


def test_short_text_gets_length_note_with_mattr_fallback():
    analysis = analyze_writing("Students learn quickly when teachers explain.")
    report = build_student_report(analysis)
    length_note = report["measurement_notes"][0]

    assert length_note["note_id"] == "text_length_measurement"
    assert length_note["word_count"] < MIN_STABLE_METRIC_WORD_COUNT
    assert length_note["mattr_method"] == "ttr_fallback_below_window"
    assert "plain type-token ratio" in length_note["message"]
    assert "not a judgement" in length_note["message"]
    assert report["overview"]["mattr_method"] == "ttr_fallback_below_window"


def test_short_text_with_full_mattr_window_still_gets_length_note():
    analysis = analyze_writing(_sentences_with_distinct_words(10))
    report = build_student_report(analysis)
    length_note = report["measurement_notes"][0]

    assert analysis["input_summary"]["word_count"] < (
        MIN_STABLE_METRIC_WORD_COUNT
    )
    assert analysis["lexical"]["diversity"]["mattr_method"] == (
        "moving_average"
    )
    assert length_note["note_id"] == "text_length_measurement"
    assert "plain type-token ratio" not in length_note["message"]


def test_long_text_has_no_length_note():
    analysis = analyze_writing(_sentences_with_distinct_words(25))
    student_report = build_student_report(analysis)
    teacher_report = build_teacher_report(analysis)

    assert analysis["input_summary"]["word_count"] >= (
        MIN_STABLE_METRIC_WORD_COUNT
    )
    assert "text_length_measurement" not in _note_ids(student_report)
    assert teacher_report["text_length_note"] is None
    assert teacher_report["method_notes"]["mattr_method"] == (
        "moving_average"
    )


def test_empty_text_length_note_reports_no_mattr_tokens():
    report = build_teacher_report(analyze_writing(""))

    assert report["text_length_note"]["word_count"] == 0
    assert report["text_length_note"]["mattr_method"] == "no_tokens"
    assert report["method_notes"]["minimum_stable_word_count"] == (
        MIN_STABLE_METRIC_WORD_COUNT
    )
