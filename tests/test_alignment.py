import copy
import json

import pytest

from src.alignment import (
    create_alignment_assessment,
    validate_alignment_for_analysis,
)
from src.analysis import analyze_writing
from src.reporting import build_student_report, build_teacher_report
from src.tasks import create_task_context


def _task_context():
    return create_task_context(
        prompt="Should universities require community service?",
        response_mode="independent_response",
        requirements=[
            {
                "requirement_id": "position",
                "description": "State a clear position.",
            },
            {
                "requirement_id": "support",
                "description": "Provide a supporting reason.",
            },
        ],
    )


def test_alignment_assessment_preserves_exact_response_evidence():
    response = "I support community service because it helps local people."
    context = _task_context()
    assessment = create_alignment_assessment(
        response_text=response,
        task_context=context,
        assessor_type="human",
        assessor_id="teacher_01",
        requirement_results=[
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [{"start_char": 0, "end_char": 27}],
            },
            {
                "requirement_id": "support",
                "status": "met",
                "evidence": [{"start_char": 28, "end_char": 58}],
            },
        ],
    )

    assert assessment["requirement_results"][0]["evidence"][0] == {
        "text": "I support community service",
        "start_char": 0,
        "end_char": 27,
    }
    assert assessment["aggregation"] is None
    assert assessment["assessor"]["assessor_type"] == "human"


def test_alignment_rejects_evidence_that_does_not_match_offsets():
    response = "I support community service because it helps local people."

    with pytest.raises(ValueError, match="does not match response offsets"):
        create_alignment_assessment(
            response_text=response,
            task_context=_task_context(),
            assessor_type="model",
            assessor_id="alignment_model",
            requirement_results=[
                {
                    "requirement_id": "position",
                    "status": "met",
                    "evidence": [
                        {
                            "text": "Different text",
                            "start_char": 0,
                            "end_char": 27,
                        }
                    ],
                },
                {
                    "requirement_id": "support",
                    "status": "not_assessed",
                },
            ],
        )


def test_alignment_requires_all_requirements_exactly_once():
    with pytest.raises(ValueError, match="Every task requirement"):
        create_alignment_assessment(
            response_text="I support the idea.",
            task_context=_task_context(),
            assessor_type="human",
            assessor_id="teacher_01",
            requirement_results=[
                {
                    "requirement_id": "position",
                    "status": "met",
                    "evidence": [{"start_char": 0, "end_char": 18}],
                }
            ],
        )


def test_alignment_status_rules_prevent_unsupported_claims():
    context = _task_context()

    with pytest.raises(ValueError, match="met requires response evidence"):
        create_alignment_assessment(
            response_text="I support the idea.",
            task_context=context,
            assessor_type="human",
            assessor_id="teacher_01",
            requirement_results=[
                {"requirement_id": "position", "status": "met"},
                {"requirement_id": "support", "status": "not_assessed"},
            ],
        )

    with pytest.raises(ValueError, match="not_met requires"):
        create_alignment_assessment(
            response_text="I support the idea.",
            task_context=context,
            assessor_type="human",
            assessor_id="teacher_01",
            requirement_results=[
                {
                    "requirement_id": "position",
                    "status": "met",
                    "evidence": [{"start_char": 0, "end_char": 18}],
                },
                {"requirement_id": "support", "status": "not_met"},
            ],
        )


def test_alignment_is_bound_to_exact_task_and_response():
    response = "I support community service because it helps local people."
    context = _task_context()
    assessment = create_alignment_assessment(
        response_text=response,
        task_context=context,
        assessor_type="model_assisted",
        assessor_id="review_workflow",
        assessor_version="1.0",
        requirement_results=[
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [{"start_char": 0, "end_char": 27}],
            },
            {
                "requirement_id": "support",
                "status": "partially_met",
                "evidence": [{"start_char": 28, "end_char": 58}],
                "note": "The reason is present but not developed.",
            },
        ],
    )
    analysis = analyze_writing(response, task_context=context)

    validate_alignment_for_analysis(assessment, analysis)

    other_analysis = analyze_writing(
        "I oppose community service.",
        task_context=context,
    )
    with pytest.raises(ValueError, match="another response"):
        validate_alignment_for_analysis(assessment, other_analysis)


def test_reports_show_assessor_provenance_without_aggregation():
    response = "I support community service because it helps local people."
    context = _task_context()
    analysis = analyze_writing(response, task_context=context)
    assessment = create_alignment_assessment(
        response_text=response,
        task_context=context,
        assessor_type="human",
        assessor_id="teacher_01",
        requirement_results=[
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [{"start_char": 0, "end_char": 27}],
            },
            {
                "requirement_id": "support",
                "status": "met",
                "evidence": [{"start_char": 28, "end_char": 58}],
            },
        ],
    )
    original_assessment = copy.deepcopy(assessment)

    student_report = build_student_report(analysis, assessment)
    teacher_report = build_teacher_report(analysis, assessment)

    assert student_report["task"]["alignment_status"] == "assessed"
    assert student_report["task"]["assessor"] == {
        "assessor_type": "human",
        "assessor_id": "teacher_01",
        "assessor_version": "",
    }
    assert "score" not in student_report["task"]
    assert teacher_report["task"] == student_report["task"]
    json.dumps(student_report)
    assert assessment == original_assessment
