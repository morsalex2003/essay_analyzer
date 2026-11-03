import copy
import json

import pytest

from src.alignment import create_alignment_assessment
from src.analysis import analyze_writing
from src.orchestration import (
    adjudicate_alignment_review,
    create_alignment_review,
    validate_alignment_review_for_analysis,
)
from src.reporting import build_teacher_report
from src.tasks import create_task_context


RESPONSE = "I support community service because it helps local people."


def _context():
    return create_task_context(
        prompt="Should universities require community service?",
        requirements=[
            {"requirement_id": "position", "description": "State a position."},
            {"requirement_id": "support", "description": "Give a reason."},
        ],
    )


def _assessment(assessor_id, support_status="met"):
    support_result = {
        "requirement_id": "support",
        "status": support_status,
    }

    if support_status in ("met", "partially_met"):
        support_result["evidence"] = [{"start_char": 28, "end_char": 58}]
    elif support_status == "not_met":
        support_result["note"] = "No supporting reason was identified."

    return create_alignment_assessment(
        response_text=RESPONSE,
        task_context=_context(),
        assessor_type="model",
        assessor_id=assessor_id,
        assessor_version="1.0",
        requirement_results=[
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [{"start_char": 0, "end_char": 27}],
            },
            support_result,
        ],
    )


def test_agreement_is_provisional_and_not_ground_truth():
    analysis = analyze_writing(RESPONSE, task_context=_context())
    review = create_alignment_review(
        analysis,
        [_assessment("assessor_a"), _assessment("assessor_b")],
    )

    assert review["review_status"] == "provisional_agreement"
    assert review["summary"]["agreement_count"] == 2
    assert review["summary"]["disagreement_count"] == 0
    assert review["requirements"][0]["agreed_status"] == "met"
    assert "not ground truth" in review["interpretation_note"]


def test_disagreement_requires_human_review_without_majority_vote():
    analysis = analyze_writing(RESPONSE, task_context=_context())
    review = create_alignment_review(
        analysis,
        [
            _assessment("assessor_a", "met"),
            _assessment("assessor_b", "partially_met"),
            _assessment("assessor_c", "met"),
        ],
    )
    support_review = review["requirements"][1]

    assert review["review_status"] == "requires_human_review"
    assert review["decision_policy"] == "no_voting_no_automatic_resolution"
    assert support_review["agreement_status"] == "disagreement"
    assert support_review["agreed_status"] is None
    assert support_review["requires_human_review"] is True


def test_review_requires_distinct_compatible_assessors():
    analysis = analyze_writing(RESPONSE, task_context=_context())
    assessment = _assessment("same_assessor")

    with pytest.raises(ValueError, match="distinct assessors"):
        create_alignment_review(analysis, [assessment, assessment])

    another_version = copy.deepcopy(assessment)
    another_version["assessor"]["assessor_version"] = "2.0"
    with pytest.raises(ValueError, match="distinct assessors"):
        create_alignment_review(analysis, [assessment, another_version])

    other_analysis = analyze_writing(
        "I oppose community service.",
        task_context=_context(),
    )
    with pytest.raises(ValueError, match="another response"):
        create_alignment_review(
            other_analysis,
            [_assessment("assessor_a"), _assessment("assessor_b")],
        )


def test_human_adjudication_is_explicit_and_does_not_mutate_review():
    analysis = analyze_writing(RESPONSE, task_context=_context())
    review = create_alignment_review(
        analysis,
        [
            _assessment("assessor_a", "met"),
            _assessment("assessor_b", "partially_met"),
        ],
    )
    original_review = copy.deepcopy(review)

    adjudicated = adjudicate_alignment_review(
        response_text=RESPONSE,
        analysis=analysis,
        review=review,
        adjudicator_id="teacher_01",
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
                "note": "The reason is present but briefly developed.",
            },
        ],
    )

    assert review == original_review
    assert adjudicated["review_status"] == "human_adjudicated"
    assert adjudicated["adjudication"]["assessment"]["assessor"][
        "assessor_type"
    ] == "human"


def test_teacher_report_can_include_unresolved_review():
    analysis = analyze_writing(RESPONSE, task_context=_context())
    review = create_alignment_review(
        analysis,
        [
            _assessment("assessor_a", "met"),
            _assessment("assessor_b", "partially_met"),
        ],
    )

    report = build_teacher_report(analysis, alignment_review=review)

    assert report["alignment_review"]["review_status"] == (
        "requires_human_review"
    )
    json.dumps(report)
    validate_alignment_review_for_analysis(review, analysis)
