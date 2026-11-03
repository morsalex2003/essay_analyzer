from copy import deepcopy
from hashlib import sha256
import json

import pytest

from src.alignment import create_alignment_assessment
from src.alignment_adapter import (
    ALIGNMENT_JSON_ADAPTER_ID,
    ALIGNMENT_JSON_ADAPTER_VERSION,
    run_json_alignment_adapter,
)
from src.alignment_evaluation import create_evaluation_run_record
from src.alignment_experiments import (
    create_alignment_experiment_manifest,
    create_provider_configuration,
    evaluate_frozen_alignment_experiment,
    sha256_file,
    validate_alignment_experiment_manifest,
    validate_provider_configuration,
)
from src.analysis import ANALYSIS_SCHEMA_VERSION
from src.alignment_prompt import alignment_prompt_template_sha256
from src.providers import create_provider_spec
from src.tasks import create_task_context


RESPONSE_TEXT = "I support public transport because it reduces traffic."


def _sha256_text(value):
    return sha256(value.encode("utf-8")).hexdigest()


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
            "source_stage": "alignment_experiment_v1",
            "data_role": data_role,
            "content_rights_status": "owner_provided",
            "task_context": task_context,
        }
    ]
    response_records = [
        {
            "schema_version": "1.0",
            "response_id": "response_001",
            "task_id": "task_001",
            "participant_id": "participant_001",
            "participant_id_is_pseudonymous": True,
            "response_family_id": "family_001",
            "source_stage": "alignment_experiment_v1",
            "data_role": data_role,
            "consent_status": "internal_evaluation_only",
            "original_text": RESPONSE_TEXT,
            "original_text_sha256": _sha256_text(RESPONSE_TEXT),
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        }
    ]
    assessment = create_alignment_assessment(
        response_text=RESPONSE_TEXT,
        task_context=task_context,
        assessor_type="human",
        assessor_id="reviewer_001",
        requirement_results=[
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [
                    {"start_char": 0, "end_char": len(RESPONSE_TEXT)}
                ],
            },
            {
                "requirement_id": "reason",
                "status": "met",
                "evidence": [
                    {"start_char": 0, "end_char": len(RESPONSE_TEXT)}
                ],
            },
        ],
    )
    assessment_records = [
        {
            "schema_version": "1.0",
            "assessment_id": "assessment_001",
            "response_id": "response_001",
            "annotation_origin": "human_reference",
            "adjudication_status": "single",
            "reviewer_id_is_pseudonymous": True,
            "alignment_assessment": assessment,
        }
    ]
    return task_records, response_records, assessment_records


def _provider_configuration(prompt_text=None):
    provider_spec = create_provider_spec(
        provider_id="provider_a",
        model_name="test-model",
        model_version="1.0",
        execution_environment="local",
        data_handling_policy="local_processing",
        timeout_seconds=20,
    )
    return create_provider_configuration(
        provider_spec=provider_spec,
        adapter_id=ALIGNMENT_JSON_ADAPTER_ID,
        adapter_version=ALIGNMENT_JSON_ADAPTER_VERSION,
        prompt_template_sha256=(
            alignment_prompt_template_sha256()
            if prompt_text is None
            else _sha256_text(prompt_text)
        ),
        temperature=0.0,
        max_output_tokens=800,
        seed=42,
    )


def _source_hashes():
    return {
        "tasks": _sha256_text("exact tasks file bytes"),
        "responses": _sha256_text("exact responses file bytes"),
        "assessments": _sha256_text("exact assessments file bytes"),
    }


def _manifest(data_role="development", sealed_authorized=False):
    tasks, responses, assessments = _dataset(data_role)
    return create_alignment_experiment_manifest(
        experiment_id="alignment_dev_001",
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=_provider_configuration(),
        data_role=data_role,
        sealed_evaluation_authorized=sealed_authorized,
    )


def _completed_run_record(provider_configuration=None):
    if provider_configuration is None:
        provider_configuration = _provider_configuration()

    def provider(request, timeout_seconds):
        return {
            "status": "completed",
            "requirement_results": [
                {
                    "requirement_id": "position",
                    "status": "met",
                    "evidence": [
                        {"start_char": 0, "end_char": len(RESPONSE_TEXT)}
                    ],
                },
                {
                    "requirement_id": "reason",
                    "status": "met",
                    "evidence": [
                        {"start_char": 0, "end_char": len(RESPONSE_TEXT)}
                    ],
                },
            ],
        }

    task_context = _dataset()[0][0]["task_context"]
    def completion(prompt_package, timeout_seconds):
        return json.dumps(provider(prompt_package, timeout_seconds))

    provider_run = run_json_alignment_adapter(
        response_text=RESPONSE_TEXT,
        task_context=task_context,
        provider_configuration=provider_configuration,
        completion_callable=completion,
    )
    return create_evaluation_run_record("response_001", provider_run)


def test_provider_configuration_is_strict_and_contains_no_prompt():
    configuration = _provider_configuration()

    validate_provider_configuration(configuration)
    assert configuration["adapter_id"] == ALIGNMENT_JSON_ADAPTER_ID
    assert configuration["generation_parameters"]["temperature"] == 0.0
    assert "frozen prompt template" not in json.dumps(configuration)

    contaminated = deepcopy(configuration)
    contaminated["api_key"] = "secret"
    with pytest.raises(ValueError, match="fields do not match"):
        validate_provider_configuration(contaminated)


def test_manifest_is_deterministic_and_excludes_private_text():
    first_manifest = _manifest()
    second_manifest = _manifest()

    assert first_manifest == second_manifest
    assert first_manifest["freeze_status"] == "configuration_frozen"
    assert first_manifest["dataset"]["summary"]["response_count"] == 1
    assert len(first_manifest["manifest_sha256"]) == 64
    serialized_manifest = json.dumps(first_manifest)
    assert RESPONSE_TEXT not in serialized_manifest
    assert "Should public transport be free?" not in serialized_manifest
    assert "participant_001" not in serialized_manifest


def test_manifest_validation_detects_dataset_file_and_prompt_drift():
    tasks, responses, assessments = _dataset()
    manifest = _manifest()

    validate_alignment_experiment_manifest(
        manifest=manifest,
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=_provider_configuration(),
    )

    changed_hashes = _source_hashes()
    changed_hashes["responses"] = _sha256_text("changed file bytes")
    with pytest.raises(ValueError, match="does not match current inputs"):
        validate_alignment_experiment_manifest(
            manifest=manifest,
            task_records=tasks,
            response_records=responses,
            assessment_records=assessments,
            source_file_sha256=changed_hashes,
            provider_configuration=_provider_configuration(),
        )

    with pytest.raises(ValueError, match="does not match current inputs"):
        validate_alignment_experiment_manifest(
            manifest=manifest,
            task_records=tasks,
            response_records=responses,
            assessment_records=assessments,
            source_file_sha256=_source_hashes(),
            provider_configuration=_provider_configuration("new prompt"),
        )


def test_manifest_validation_detects_manifest_tampering():
    tasks, responses, assessments = _dataset()
    manifest = _manifest()
    manifest["evaluation_policy"]["status_aggregation"] = "macro_score"

    with pytest.raises(ValueError, match="does not match current inputs"):
        validate_alignment_experiment_manifest(
            manifest=manifest,
            task_records=tasks,
            response_records=responses,
            assessment_records=assessments,
            source_file_sha256=_source_hashes(),
            provider_configuration=_provider_configuration(),
        )


def test_frozen_evaluation_binds_result_to_manifest_and_provider():
    tasks, responses, assessments = _dataset()
    provider_configuration = _provider_configuration()
    manifest = _manifest()

    result = evaluate_frozen_alignment_experiment(
        manifest=manifest,
        task_records=tasks,
        response_records=responses,
        assessment_records=assessments,
        source_file_sha256=_source_hashes(),
        provider_configuration=provider_configuration,
        run_records=[_completed_run_record(provider_configuration)],
    )

    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert result["experiment_id"] == "alignment_dev_001"
    assert result["evaluation"]["coverage"]["completion_rate"] == 1.0
    assert result["evaluation"]["status_evaluation"][
        "conditional_exact_status_agreement_rate"
    ] == 1.0

    other_configuration = deepcopy(provider_configuration)
    other_configuration["provider_spec"] = create_provider_spec(
        provider_id="provider_b",
        model_name="test-model",
        model_version="1.0",
        execution_environment="local",
        data_handling_policy="local_processing",
        timeout_seconds=20,
    )
    with pytest.raises(ValueError, match="do not match the frozen"):
        evaluate_frozen_alignment_experiment(
            manifest=manifest,
            task_records=tasks,
            response_records=responses,
            assessment_records=assessments,
            source_file_sha256=_source_hashes(),
            provider_configuration=provider_configuration,
            run_records=[_completed_run_record(other_configuration)],
        )


def test_sealed_manifest_requires_explicit_boolean_authorization():
    with pytest.raises(ValueError, match="explicit authorization"):
        _manifest("sealed_evaluation")

    sealed_manifest = _manifest("sealed_evaluation", sealed_authorized=True)
    assert sealed_manifest["data_role"] == "sealed_evaluation"
    assert sealed_manifest["sealed_evaluation_authorized_at_freeze"] is True

    tasks, responses, assessments = _dataset("sealed_evaluation")
    with pytest.raises(ValueError, match="must be a boolean"):
        create_alignment_experiment_manifest(
            experiment_id="sealed_001",
            task_records=tasks,
            response_records=responses,
            assessment_records=assessments,
            source_file_sha256=_source_hashes(),
            provider_configuration=_provider_configuration(),
            data_role="sealed_evaluation",
            sealed_evaluation_authorized="yes",
        )


def test_development_manifest_rejects_irrelevant_sealed_authorization():
    tasks, responses, assessments = _dataset()

    with pytest.raises(ValueError, match="development data"):
        create_alignment_experiment_manifest(
            experiment_id="development_001",
            task_records=tasks,
            response_records=responses,
            assessment_records=assessments,
            source_file_sha256=_source_hashes(),
            provider_configuration=_provider_configuration(),
            data_role="development",
            sealed_evaluation_authorized=True,
        )


def test_sha256_file_hashes_exact_bytes(tmp_path):
    source_path = tmp_path / "records.jsonl"
    source_path.write_bytes(b'{"record": 1}\n')

    assert sha256_file(source_path) == sha256(
        b'{"record": 1}\n'
    ).hexdigest()
