import copy

import pytest

from src.analysis import analyze_writing
from src.reporting import build_student_report, build_teacher_report
from src.revisions import compare_revisions
from src.tasks import create_task_context, validate_task_context


def test_task_context_normalizes_structured_inputs():
    context = create_task_context(
        prompt="  Should universities require community service?  ",
        response_mode="independent_response",
        instructions="  Explain and support your position.  ",
        requirements=[" State a position. ", "Use supporting reasons."],
        task_id=" task_001 ",
    )

    assert context["task_id"] == "task_001"
    assert context["prompt"] == (
        "Should universities require community service?"
    )
    assert context["instructions"] == "Explain and support your position."
    assert context["requirements"] == [
        {
            "requirement_id": "requirement_1",
            "description": "State a position.",
        },
        {
            "requirement_id": "requirement_2",
            "description": "Use supporting reasons.",
        },
    ]


def test_explicit_requirement_ids_are_preserved_and_unique():
    context = create_task_context(
        prompt="Discuss the topic.",
        requirements=[
            {
                "requirement_id": "position",
                "description": "State a position.",
            }
        ],
    )

    assert context["requirements"][0]["requirement_id"] == "position"

    with pytest.raises(ValueError, match="requirement_id values must be unique"):
        create_task_context(
            prompt="Discuss the topic.",
            requirements=[
                {"requirement_id": "reason", "description": "First reason."},
                {"requirement_id": "reason", "description": "Second reason."},
            ],
        )


def test_source_materials_keep_explicit_identity_and_content():
    context = create_task_context(
        prompt="Summarize the relationship between the sources.",
        response_mode="source_based_response",
        source_materials=[
            {
                "source_id": "reading_1",
                "source_type": "reading",
                "title": "Urban transport",
                "content": "Public transport can reduce congestion.",
            }
        ],
    )

    assert context["source_materials"][0]["source_id"] == "reading_1"
    assert context["source_materials"][0]["source_type"] == "reading"
    assert context["source_materials"][0]["content"] == (
        "Public transport can reduce congestion."
    )


def test_source_material_ids_must_be_unique():
    with pytest.raises(ValueError, match="source_id values must be unique"):
        create_task_context(
            prompt="Compare the sources.",
            source_materials=[
                {"source_id": "reading", "content": "First source."},
                {"source_id": "reading", "content": "Second source."},
            ],
        )


def test_invalid_or_noncanonical_task_context_is_rejected():
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        create_task_context(prompt="   ")

    with pytest.raises(ValueError, match="Unsupported response_mode"):
        create_task_context(prompt="Write.", response_mode="toefl_score")

    context = create_task_context(prompt="Write an answer.")
    context["prompt"] = "  Write an answer.  "

    with pytest.raises(ValueError, match="canonical normalized form"):
        validate_task_context(context)


def test_prompt_and_sources_are_not_counted_as_response_text():
    context = create_task_context(
        prompt="This prompt contains many words that must not be counted.",
        response_mode="source_based_response",
        source_materials=[{"content": "A long source also stays separate."}],
    )

    result = analyze_writing("Students learn.", task_context=context)

    assert result["input_summary"]["word_count"] == 2
    assert result["task_context"] == context


def test_analysis_copies_task_context_instead_of_sharing_mutable_data():
    context = create_task_context(
        prompt="Discuss the topic.",
        requirements=["Give one reason."],
    )
    original_context = copy.deepcopy(context)

    result = analyze_writing("Students learn.", task_context=context)
    result["task_context"]["requirements"].append("Added later.")

    assert context == original_context


def test_reports_expose_task_summary_without_claiming_alignment():
    context = create_task_context(
        prompt="Should students study abroad?",
        response_mode="discussion_response",
        source_materials=[{"content": "One student supports the idea."}],
    )
    analysis = analyze_writing("Studying abroad can help.", task_context=context)

    student_report = build_student_report(analysis)
    teacher_report = build_teacher_report(analysis)

    assert student_report["task"]["source_count"] == 1
    assert student_report["task"]["alignment_status"] == "not_evaluated"
    assert teacher_report["task"] == student_report["task"]
    assert "source_materials" not in student_report["task"]


def test_reports_do_not_share_mutable_data_with_analysis():
    context = create_task_context(prompt="Discuss education.")
    analysis = analyze_writing("Students learn.", task_context=context)
    original_analysis = copy.deepcopy(analysis)

    student_report = build_student_report(analysis)
    teacher_report = build_teacher_report(analysis)
    student_report["task"]["requirements"].append("Changed in report.")
    teacher_report["metrics"]["lexical"]["repeated_words"]["new"] = 2

    assert analysis == original_analysis


def test_revision_comparison_requires_the_same_task_context():
    first_task = create_task_context(prompt="Discuss education.")
    second_task = create_task_context(prompt="Discuss technology.")
    original = analyze_writing("Education matters.", task_context=first_task)
    revised = analyze_writing("Technology matters.", task_context=second_task)

    with pytest.raises(ValueError, match="same writing task context"):
        compare_revisions(original, revised)


def test_revision_result_does_not_share_task_context_with_analysis():
    context = create_task_context(prompt="Discuss education.")
    original = analyze_writing("Education matters.", task_context=context)
    revised = analyze_writing("Education matters to students.", task_context=context)
    original_copy = copy.deepcopy(original)

    comparison = compare_revisions(original, revised)
    comparison["task_context"]["requirements"].append("Changed later.")

    assert original == original_copy
