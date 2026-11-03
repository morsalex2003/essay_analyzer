import copy
import json

from src.analysis import analyze_writing
from src.reporting import build_student_report, build_teacher_report


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

    assert len(report["measurement_notes"]) == 1
    assert "not a grammar judgement" in report["measurement_notes"][0][
        "message"
    ]


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
