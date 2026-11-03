from copy import deepcopy
from hashlib import sha256
import json

import pytest

from src.alignment import create_alignment_assessment
from src.alignment_adapter import (
    ALIGNMENT_JSON_ADAPTER_ID,
    ALIGNMENT_JSON_ADAPTER_VERSION,
)
from src.alignment_batch import (
    copy_run_records_from_batch,
    run_frozen_alignment_batch,
    validate_alignment_batch_result,
)
from src.alignment_experiments import (
    create_alignment_experiment_manifest,
    create_provider_configuration,
    evaluate_frozen_alignment_experiment,
)
from src.alignment_prompt import alignment_prompt_template_sha256
from src.analysis import ANALYSIS_SCHEMA_VERSION
from src.providers import create_provider_spec
from src.tasks import create_task_context


FIRST_RESPONSE = "First, public transport reduces traffic."
SECOND_RESPONSE = "Second, public transport helps students."


def _sha256_text(value):
    return sha256(value.encode("utf-8")).hexdigest()


def _dataset():
    task_context = create_task_context(
        task_id="task_001",
        prompt="Give one benefit of public transport.",
        requirements=[
            {
                "requirement_id": "benefit",
                "description": "Give one benefit.",
            }
        ],
    )
    task_records = [
        {
            "schema_version": "1.0",
            "task_id": "task_001",
            "source_stage": "alignment_batch_v1",
            "data_role": "development",
            "content_rights_status": "owner_provided",
            "task_context": task_context,
        }
    ]
    response_records = []
    assessment_records = []

    for index, response_text in enumerate(
        (FIRST_RESPONSE, SECOND_RESPONSE),
        start=1,
    ):
        response_id = "response_00" + str(index)
        response_records.append(
            {
                "schema_version": "1.0",
                "response_id": response_id,
                "task_id": "task_001",
                "participant_id": "participant_00" + str(index),
                "participant_id_is_pseudonymous": True,
                "response_family_id": "family_00" + str(index),
                "source_stage": "alignment_batch_v1",
                "data_role": "development",
                "consent_status": "internal_evaluation_only",
                "original_text": response_text,
                "original_text_sha256": _sha256_text(response_text),
                "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            }
        )
        assessment = create_alignment_assessment(
            response_text=response_text,
            task_context=task_context,
            assessor_type="human",
            assessor_id="human_reviewer",
            requirement_results=[
                {
                    "requirement_id": "benefit",
                    "status": "met",
                    "evidence": [
                        {"start_char": 0, "end_char": len(response_text)}
                    ],
                }
            ],
        )
        assessment_records.append(
            {
                "schema_version": "1.0",
                "assessment_id": "assessment_00" + str(index),
                "response_id": response_id,
                "annotation_origin": "human_reference",
                "adjudication_status": "single",
                "reviewer_id_is_pseudonymous": True,
                "alignment_assessment": assessment,
            }
        )

    return task_records, response_records, assessment_records


def _provider_configuration():
    provider_spec = create_provider_spec(
        provider_id="provider_a",
        model_name="test-model",
        model_version="1.0",
        execution_environment="local",
        data_handling_policy="local_processing",
    )
    return create_provider_configuration(
        provider_spec=provider_spec,
        adapter_id=ALIGNMENT_JSON_ADAPTER_ID,
        adapter_version=ALIGNMENT_JSON_ADAPTER_VERSION,
        prompt_template_sha256=alignment_prompt_template_sha256(),
        temperature=0.0,
        max_output_tokens=500,
        seed=42,
    )


def _source_hashes():
    return {
        "tasks": _sha256_text("tasks file"),
        "responses": _sha256_text("responses file"),
        "assessments": _sha256_text("assessments file"),
    }


def _manifest():
    tasks, responses, assessments = _dataset()
    return create_alignment_experiment_manifest(
        experiment_id="alignment_batch_dev_001",
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=_provider_configuration(),
        data_role="development",
    )


def _completed_json(response_text):
    return json.dumps(
        {
            "status": "completed",
            "requirement_results": [
                {
                    "requirement_id": "benefit",
                    "status": "met",
                    "evidence": [
                        {"start_char": 0, "end_char": len(response_text)}
                    ],
                }
            ],
        }
    )


def test_batch_records_every_response_and_status():
    tasks, responses, assessments = _dataset()

    def completion(prompt_package, timeout_seconds):
        response_text = prompt_package["input_data"]["response_text"]
        if response_text == FIRST_RESPONSE:
            return _completed_json(response_text)

        return "invalid non-JSON output"

    batch = run_frozen_alignment_batch(
        manifest=_manifest(),
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=_provider_configuration(),
        completion_callable=completion,
    )

    assert batch["summary"]["response_count"] == 2
    assert batch["summary"]["recorded_run_count"] == 2
    assert batch["summary"]["run_status_counts"]["completed"] == 1
    assert batch["summary"]["run_status_counts"]["invalid_output"] == 1
    assert [
        record["response_id"] for record in batch["run_records"]
    ] == ["response_001", "response_002"]


def test_batch_prompt_does_not_receive_human_references_or_nlp_metrics():
    tasks, responses, assessments = _dataset()
    received_packages = []

    def completion(prompt_package, timeout_seconds):
        received_packages.append(prompt_package)
        return _completed_json(
            prompt_package["input_data"]["response_text"]
        )

    run_frozen_alignment_batch(
        manifest=_manifest(),
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=_provider_configuration(),
        completion_callable=completion,
    )

    assert len(received_packages) == 2
    serialized_packages = json.dumps(received_packages)
    assert "human_reviewer" not in serialized_packages
    assert "alignment_assessment" not in serialized_packages
    assert "syntax" not in serialized_packages
    assert "lexical" not in serialized_packages


def test_one_failed_response_does_not_remove_later_run_records():
    tasks, responses, assessments = _dataset()
    call_count = {"value": 0}

    def completion(prompt_package, timeout_seconds):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("private provider failure")

        return _completed_json(
            prompt_package["input_data"]["response_text"]
        )

    batch = run_frozen_alignment_batch(
        manifest=_manifest(),
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=_provider_configuration(),
        completion_callable=completion,
    )

    statuses = []
    for record in batch["run_records"]:
        statuses.append(record["provider_run"]["status"])

    assert statuses == ["failed", "completed"]
    assert call_count["value"] == 2


def test_manifest_drift_stops_batch_before_first_completion_call():
    tasks, responses, assessments = _dataset()
    changed_hashes = _source_hashes()
    changed_hashes["responses"] = _sha256_text("changed responses file")
    called = {"value": False}

    def completion(prompt_package, timeout_seconds):
        called["value"] = True
        return _completed_json(FIRST_RESPONSE)

    with pytest.raises(ValueError, match="does not match current inputs"):
        run_frozen_alignment_batch(
            manifest=_manifest(),
            task_records=tasks,
            response_records=responses,
            assessment_records=assessments,
            source_file_sha256=changed_hashes,
            provider_configuration=_provider_configuration(),
            completion_callable=completion,
        )

    assert called["value"] is False


def test_batch_result_can_be_evaluated_against_same_manifest():
    tasks, responses, assessments = _dataset()
    manifest = _manifest()
    provider_configuration = _provider_configuration()

    def completion(prompt_package, timeout_seconds):
        return _completed_json(
            prompt_package["input_data"]["response_text"]
        )

    batch = run_frozen_alignment_batch(
        manifest=manifest,
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=provider_configuration,
        completion_callable=completion,
    )
    evaluation = evaluate_frozen_alignment_experiment(
        manifest=manifest,
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=provider_configuration,
        run_records=batch["run_records"],
    )

    assert evaluation["manifest_sha256"] == manifest["manifest_sha256"]
    assert evaluation["evaluation"]["coverage"]["completion_rate"] == 1.0
    assert evaluation["evaluation"]["status_evaluation"][
        "conditional_exact_status_agreement_rate"
    ] == 1.0


def test_batch_does_not_modify_private_inputs():
    tasks, responses, assessments = _dataset()
    original_inputs = deepcopy((tasks, responses, assessments))

    def completion(prompt_package, timeout_seconds):
        return _completed_json(
            prompt_package["input_data"]["response_text"]
        )

    run_frozen_alignment_batch(
        manifest=_manifest(),
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=_provider_configuration(),
        completion_callable=completion,
    )

    assert (tasks, responses, assessments) == original_inputs


def test_batch_artifact_validation_detects_summary_and_manifest_tampering():
    tasks, responses, assessments = _dataset()
    manifest = _manifest()
    provider_configuration = _provider_configuration()

    def completion(prompt_package, timeout_seconds):
        return _completed_json(
            prompt_package["input_data"]["response_text"]
        )

    batch = run_frozen_alignment_batch(
        manifest=manifest,
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=provider_configuration,
        completion_callable=completion,
    )
    validate_alignment_batch_result(
        batch,
        manifest,
        provider_configuration,
    )

    changed_summary = deepcopy(batch)
    changed_summary["summary"]["run_status_counts"]["completed"] = 1
    with pytest.raises(ValueError, match="summary does not match"):
        validate_alignment_batch_result(
            changed_summary,
            manifest,
            provider_configuration,
        )

    changed_manifest = deepcopy(batch)
    changed_manifest["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another manifest"):
        validate_alignment_batch_result(
            changed_manifest,
            manifest,
            provider_configuration,
        )


def test_copy_run_records_from_batch_returns_independent_copy():
    tasks, responses, assessments = _dataset()
    manifest = _manifest()
    provider_configuration = _provider_configuration()

    def completion(prompt_package, timeout_seconds):
        return _completed_json(
            prompt_package["input_data"]["response_text"]
        )

    batch = run_frozen_alignment_batch(
        manifest=manifest,
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=provider_configuration,
        completion_callable=completion,
    )
    copied_records = copy_run_records_from_batch(
        batch,
        manifest,
        provider_configuration,
    )
    copied_records[0]["response_id"] = "changed"

    assert batch["run_records"][0]["response_id"] == "response_001"
