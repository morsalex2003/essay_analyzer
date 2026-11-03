import json
from hashlib import sha256

import pytest

from src.alignment import create_alignment_assessment
from src.alignment_evaluation import (
    create_evaluation_run_record,
    evaluate_alignment_provider_runs,
)
from src.analysis import ANALYSIS_SCHEMA_VERSION
from src.providers import (
    create_provider_spec,
    run_alignment_provider,
)
from src.tasks import create_task_context


FIRST_RESPONSE = (
    "I support free public transport because it reduces traffic."
)
SECOND_RESPONSE = "I support free public transport."


def _dataset(data_role="development"):
    task_context = create_task_context(
        task_id="task_001",
        prompt="Should public transport be free?",
        requirements=[
            {
                "requirement_id": "position",
                "description": "State a position.",
            },
            {
                "requirement_id": "reason",
                "description": "Give one reason.",
            },
        ],
    )
    task_records = [
        {
            "schema_version": "1.0",
            "task_id": "task_001",
            "source_stage": "alignment_evaluation_v1",
            "data_role": data_role,
            "content_rights_status": "owner_provided",
            "task_context": task_context,
        }
    ]
    response_records = []

    for index, response_text in enumerate(
        (FIRST_RESPONSE, SECOND_RESPONSE),
        start=1,
    ):
        response_records.append(
            {
                "schema_version": "1.0",
                "response_id": "response_00" + str(index),
                "task_id": "task_001",
                "participant_id": "participant_00" + str(index),
                "participant_id_is_pseudonymous": True,
                "response_family_id": "family_00" + str(index),
                "source_stage": "alignment_evaluation_v1",
                "data_role": data_role,
                "consent_status": "internal_evaluation_only",
                "original_text": response_text,
                "original_text_sha256": sha256(
                    response_text.encode("utf-8")
                ).hexdigest(),
                "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            }
        )

    first_assessment = create_alignment_assessment(
        response_text=FIRST_RESPONSE,
        task_context=task_context,
        assessor_type="human",
        assessor_id="reference_reviewer",
        requirement_results=[
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [
                    {"start_char": 0, "end_char": len(FIRST_RESPONSE)}
                ],
            },
            {
                "requirement_id": "reason",
                "status": "met",
                "evidence": [
                    {"start_char": 0, "end_char": len(FIRST_RESPONSE)}
                ],
            },
        ],
    )
    second_assessment = create_alignment_assessment(
        response_text=SECOND_RESPONSE,
        task_context=task_context,
        assessor_type="human",
        assessor_id="reference_reviewer",
        requirement_results=[
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [
                    {"start_char": 0, "end_char": len(SECOND_RESPONSE)}
                ],
            },
            {
                "requirement_id": "reason",
                "status": "not_met",
                "note": "The response does not provide a reason.",
            },
        ],
    )
    assessment_records = []

    for index, assessment in enumerate(
        (first_assessment, second_assessment),
        start=1,
    ):
        assessment_records.append(
            {
                "schema_version": "1.0",
                "assessment_id": "assessment_00" + str(index),
                "response_id": "response_00" + str(index),
                "annotation_origin": "human_reference",
                "adjudication_status": "single",
                "reviewer_id_is_pseudonymous": True,
                "alignment_assessment": assessment,
            }
        )

    return task_records, response_records, assessment_records


def _provider_spec(provider_id="provider_a"):
    return create_provider_spec(
        provider_id=provider_id,
        model_name="test-model",
        model_version="1.0",
        execution_environment="local",
        data_handling_policy="local_processing",
    )


def _completed_first_run(provider_id="provider_a"):
    def provider(request, timeout_seconds):
        return {
            "status": "completed",
            "requirement_results": [
                {
                    "requirement_id": "position",
                    "status": "met",
                    "evidence": [
                        {
                            "start_char": 0,
                            "end_char": len(FIRST_RESPONSE),
                        }
                    ],
                },
                {
                    "requirement_id": "reason",
                    "status": "partially_met",
                    "evidence": [
                        {
                            "start_char": 0,
                            "end_char": len(FIRST_RESPONSE),
                        }
                    ],
                },
            ],
        }

    task_context = _dataset()[0][0]["task_context"]
    return run_alignment_provider(
        FIRST_RESPONSE,
        task_context,
        _provider_spec(provider_id),
        provider,
    )


def _abstained_second_run(provider_id="provider_a"):
    def provider(request, timeout_seconds):
        return {
            "status": "abstained",
            "reason_code": "insufficient_context",
        }

    task_context = _dataset()[0][0]["task_context"]
    return run_alignment_provider(
        SECOND_RESPONSE,
        task_context,
        _provider_spec(provider_id),
        provider,
    )


def _run_records():
    return [
        create_evaluation_run_record(
            "response_001",
            _completed_first_run(),
        ),
        create_evaluation_run_record(
            "response_002",
            _abstained_second_run(),
        ),
    ]


def test_evaluation_separates_coverage_from_conditional_agreement():
    tasks, responses, assessments = _dataset()

    result = evaluate_alignment_provider_runs(
        tasks,
        responses,
        assessments,
        _run_records(),
        data_role="development",
    )

    assert result["coverage"] == {
        "response_count": 2,
        "completed_response_count": 1,
        "completion_rate": 0.5,
    }
    assert result["run_status_counts"]["completed"] == 1
    assert result["run_status_counts"]["abstained"] == 1
    status_evaluation = result["status_evaluation"]
    assert status_evaluation["reference_requirement_count"] == 4
    assert status_evaluation["evaluated_requirement_count"] == 2
    assert status_evaluation["exact_status_match_count"] == 1
    assert status_evaluation[
        "conditional_exact_status_agreement_rate"
    ] == 0.5
    assert status_evaluation[
        "conditional_exact_response_agreement_rate"
    ] == 0.0


def test_evaluation_reports_per_status_confusion_and_private_safe_errors():
    tasks, responses, assessments = _dataset()

    result = evaluate_alignment_provider_runs(
        tasks,
        responses,
        assessments,
        _run_records(),
        data_role="development",
    )

    status_evaluation = result["status_evaluation"]
    assert status_evaluation["confusion_matrix"]["met"]["met"] == 1
    assert status_evaluation["confusion_matrix"]["met"][
        "partially_met"
    ] == 1
    assert status_evaluation["per_status"]["met"] == {
        "total_reference_count": 3,
        "evaluated_reference_count": 2,
        "predicted_count": 1,
        "exact_match_count": 1,
        "conditional_precision": 1.0,
        "conditional_recall": 0.5,
    }
    assert status_evaluation["mismatches"] == [
        {
            "response_id": "response_001",
            "requirement_id": "reason",
            "reference_status": "met",
            "predicted_status": "partially_met",
        }
    ]
    serialized_result = json.dumps(result)
    assert FIRST_RESPONSE not in serialized_result
    assert SECOND_RESPONSE not in serialized_result


def test_evidence_span_comparison_is_descriptive_and_conditional():
    tasks, responses, assessments = _dataset()

    result = evaluate_alignment_provider_runs(
        tasks,
        responses,
        assessments,
        _run_records(),
        data_role="development",
    )

    evidence = result["evidence_comparison"]
    assert evidence["comparable_requirement_count"] == 1
    assert evidence["exact_span_match_rate"] == 1.0
    assert evidence["overlapping_span_rate"] == 1.0
    assert "not a semantic correctness score" in evidence[
        "interpretation_note"
    ]


def test_no_completed_runs_produces_undefined_conditional_rates():
    tasks, responses, assessments = _dataset()
    run_records = [
        create_evaluation_run_record(
            "response_001",
            _abstained_second_run(),
        ),
        create_evaluation_run_record(
            "response_002",
            _abstained_second_run(),
        ),
    ]
    run_records[0]["provider_run"]["response_text_sha256"] = (
        responses[0]["original_text_sha256"]
    )
    run_records[0]["provider_run"]["task_context_sha256"] = (
        run_records[1]["provider_run"]["task_context_sha256"]
    )

    result = evaluate_alignment_provider_runs(
        tasks,
        responses,
        assessments,
        run_records,
        data_role="development",
    )

    assert result["coverage"]["completion_rate"] == 0.0
    assert result["status_evaluation"][
        "conditional_exact_status_agreement_rate"
    ] is None
    assert result["evidence_comparison"]["exact_span_match_rate"] is None


def test_evaluation_requires_complete_compatible_single_provider_runs():
    tasks, responses, assessments = _dataset()
    incomplete_runs = _run_records()[:1]

    with pytest.raises(ValueError, match="Every response"):
        evaluate_alignment_provider_runs(
            tasks,
            responses,
            assessments,
            incomplete_runs,
            data_role="development",
        )

    mixed_provider_runs = _run_records()
    mixed_provider_runs[1] = create_evaluation_run_record(
        "response_002",
        _abstained_second_run("provider_b"),
    )
    with pytest.raises(ValueError, match="mix provider"):
        evaluate_alignment_provider_runs(
            tasks,
            responses,
            assessments,
            mixed_provider_runs,
            data_role="development",
        )


def test_sealed_evaluation_requires_explicit_authorization():
    tasks, responses, assessments = _dataset("sealed_evaluation")

    with pytest.raises(ValueError, match="explicit authorization"):
        evaluate_alignment_provider_runs(
            tasks,
            responses,
            assessments,
            _run_records(),
            data_role="sealed_evaluation",
        )

    with pytest.raises(ValueError, match="must be a boolean"):
        evaluate_alignment_provider_runs(
            tasks,
            responses,
            assessments,
            _run_records(),
            data_role="sealed_evaluation",
            sealed_evaluation_authorized="yes",
        )

    result = evaluate_alignment_provider_runs(
        tasks,
        responses,
        assessments,
        _run_records(),
        data_role="sealed_evaluation",
        sealed_evaluation_authorized=True,
    )
    assert result["data_role"] == "sealed_evaluation"


def test_evaluation_rejects_mixed_dataset_roles():
    tasks, responses, assessments = _dataset()
    responses[1]["data_role"] = "sealed_evaluation"

    with pytest.raises(ValueError, match="one data role"):
        evaluate_alignment_provider_runs(
            tasks,
            responses,
            assessments,
            _run_records(),
            data_role="development",
        )
