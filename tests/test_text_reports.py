from src.analysis import analyze_writing
from src.reporting import build_student_report, build_teacher_report
from src.revisions import compare_revisions
from src.text_reports import (
    format_metrics_text,
    format_revision_comparison_text,
    format_student_report_text,
    format_teacher_report_text,
)


SHORT_TEXT = "Technlogy helps. Technlogy changes. Technlogy grows."


def test_metrics_text_skips_optional_sections_that_were_not_computed():
    text = format_metrics_text(analyze_writing(SHORT_TEXT))

    assert text.startswith("Essay analysis")
    assert "Note: fewer than 150 words" in text
    assert "ttr_fallback_below_window" in text
    assert "CEFR Vocabulary:" not in text
    assert "EFLLex" not in text


def test_student_text_shows_evidence_sentences_and_notes():
    report = build_student_report(analyze_writing(SHORT_TEXT))
    text = format_student_report_text(report)

    assert "not a TOEFL, IELTS, or proficiency score" in text
    assert "Check unrecognized words" in text
    assert "technlogy (3x; sentences 1, 2, 3)" in text
    assert "Text is too short for stable measurements" in text


def test_student_text_says_when_nothing_was_flagged():
    report = build_student_report(analyze_writing("Students learn."))

    assert "Nothing was flagged for review." in format_student_report_text(
        report
    )


def test_teacher_text_marks_empty_review_lists():
    report = build_teacher_report(analyze_writing("Students learn."))
    text = format_teacher_report_text(report)

    assert "Unrecognized words:\n  none" in text
    assert "Repeated surface forms:\n  none" in text
    assert "window 50, ttr_fallback_below_window" in text


def test_revision_text_shows_signed_deltas_and_resolved_words():
    comparison = compare_revisions(
        analyze_writing("Students learn. Technlogy helps people."),
        analyze_writing(
            "Students learn because teachers explain ideas. "
            "Technology helps people."
        ),
    )
    text = format_revision_comparison_text(comparison)

    assert "word_count: 5 -> 9 (+4)" in text
    assert "no longer present: technlogy" in text
    assert "not automatically an improvement" in text
