from copy import deepcopy

from src.alignment import (
    create_alignment_assessment,
    validate_alignment_for_analysis,
)
from src.tasks import task_context_fingerprint


ALIGNMENT_REVIEW_SCHEMA_VERSION = "1.0"


def _assessor_key(assessment):
    assessor = assessment["assessor"]
    return (
        assessor["assessor_type"]
        + ":"
        + assessor["assessor_id"]
    )


def _results_by_requirement(assessment):
    results = {}

    for result in assessment["requirement_results"]:
        results[result["requirement_id"]] = result

    return results


def _requirement_review(requirement, assessments):
    assessor_results = []
    observed_statuses = []

    for assessment in assessments:
        results_by_requirement = _results_by_requirement(assessment)
        result = results_by_requirement[requirement["requirement_id"]]
        assessor_results.append(
            {
                "assessor": assessment["assessor"],
                "status": result["status"],
                "evidence": result["evidence"],
                "note": result["note"],
            }
        )

        if result["status"] not in observed_statuses:
            observed_statuses.append(result["status"])

    if len(observed_statuses) == 1:
        agreement_status = "agreement"
        agreed_status = observed_statuses[0]
        requires_human_review = False
    else:
        agreement_status = "disagreement"
        agreed_status = None
        requires_human_review = True

    return {
        "requirement_id": requirement["requirement_id"],
        "description": requirement["description"],
        "agreement_status": agreement_status,
        "agreed_status": agreed_status,
        "observed_statuses": observed_statuses,
        "requires_human_review": requires_human_review,
        "assessor_results": assessor_results,
    }


def create_alignment_review(
    analysis: dict[str, object],
    assessments: list[dict[str, object]],
) -> dict[str, object]:
    """Compare independent assessments without voting or choosing a winner."""
    if not isinstance(assessments, list) or len(assessments) < 2:
        raise ValueError("Alignment review requires at least two assessments.")

    assessor_keys = []

    for assessment in assessments:
        validate_alignment_for_analysis(assessment, analysis)
        assessor_key = _assessor_key(assessment)

        if assessor_key in assessor_keys:
            raise ValueError("Alignment review requires distinct assessors.")

        assessor_keys.append(assessor_key)

    task_context = analysis["task_context"]
    requirement_reviews = []
    agreement_count = 0
    disagreement_count = 0

    for requirement in task_context["requirements"]:
        requirement_review = _requirement_review(
            requirement,
            assessments,
        )
        requirement_reviews.append(requirement_review)

        if requirement_review["agreement_status"] == "agreement":
            agreement_count += 1
        else:
            disagreement_count += 1

    if disagreement_count > 0:
        review_status = "requires_human_review"
    else:
        review_status = "provisional_agreement"

    review = {
        "alignment_review_schema_version": ALIGNMENT_REVIEW_SCHEMA_VERSION,
        "analysis_schema_version": analysis["analysis_schema_version"],
        "task_context_sha256": task_context_fingerprint(task_context),
        "response_text_sha256": analysis["input_summary"]["text_sha256"],
        "review_status": review_status,
        "decision_policy": "no_voting_no_automatic_resolution",
        "requirements": requirement_reviews,
        "summary": {
            "assessment_count": len(assessments),
            "requirement_count": len(requirement_reviews),
            "agreement_count": agreement_count,
            "disagreement_count": disagreement_count,
            "requires_human_review": disagreement_count > 0,
        },
        "interpretation_note": (
            "Agreement is provisional evidence, not ground truth. "
            "Disagreement is not resolved by majority vote."
        ),
        "adjudication": None,
    }

    return deepcopy(review)


def validate_alignment_review_for_analysis(review, analysis) -> None:
    if not isinstance(review, dict):
        raise ValueError("alignment_review must be a dictionary.")

    if review.get("alignment_review_schema_version") != (
        ALIGNMENT_REVIEW_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported alignment review schema version.")

    if review.get("analysis_schema_version") != analysis[
        "analysis_schema_version"
    ]:
        raise ValueError("Alignment review uses another analysis schema.")

    task_context = analysis["task_context"]
    if task_context is None:
        raise ValueError("Alignment review requires a task context.")

    if review.get("task_context_sha256") != task_context_fingerprint(
        task_context
    ):
        raise ValueError("Alignment review belongs to another task.")

    if review.get("response_text_sha256") != analysis["input_summary"][
        "text_sha256"
    ]:
        raise ValueError("Alignment review belongs to another response.")

    expected_ids = []
    for requirement in task_context["requirements"]:
        expected_ids.append(requirement["requirement_id"])

    actual_ids = []
    for requirement_review in review.get("requirements", []):
        actual_ids.append(requirement_review.get("requirement_id"))

    if actual_ids != expected_ids:
        raise ValueError("Alignment review requirements do not match the task.")


def adjudicate_alignment_review(
    response_text: str,
    analysis: dict[str, object],
    review: dict[str, object],
    requirement_results,
    adjudicator_id: str,
    adjudicator_version: str = "",
) -> dict[str, object]:
    """Attach an explicit human assessment to an unresolved review."""
    validate_alignment_review_for_analysis(review, analysis)

    if review["adjudication"] is not None:
        raise ValueError("Alignment review has already been adjudicated.")

    adjudicated_assessment = create_alignment_assessment(
        response_text=response_text,
        task_context=analysis["task_context"],
        requirement_results=requirement_results,
        assessor_type="human",
        assessor_id=adjudicator_id,
        assessor_version=adjudicator_version,
    )
    validate_alignment_for_analysis(adjudicated_assessment, analysis)

    adjudicated_review = deepcopy(review)
    adjudicated_review["review_status"] = "human_adjudicated"
    adjudicated_review["adjudication"] = {
        "policy": "explicit_human_adjudication",
        "assessment": adjudicated_assessment,
    }

    return adjudicated_review
