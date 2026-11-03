from copy import deepcopy
from hashlib import sha256
import json

from src.alignment import ALIGNMENT_ASSESSMENT_SCHEMA_VERSION
from src.alignment_dataset import (
    ALIGNMENT_DATASET_SCHEMA_VERSION,
    DATA_ROLES,
    validate_alignment_dataset,
)
from src.alignment_evaluation import (
    ALIGNMENT_EVALUATION_SCHEMA_VERSION,
    evaluate_alignment_provider_runs,
)
from src.alignment_pilot import ANNOTATION_GUIDELINE_VERSION
from src.alignment_prompt import (
    ALIGNMENT_PROMPT_PACKAGE_SCHEMA_VERSION,
    ALIGNMENT_PROMPT_TEMPLATE_VERSION,
)
from src.analysis import ANALYSIS_SCHEMA_VERSION
from src.providers import (
    PROVIDER_REQUEST_SCHEMA_VERSION,
    PROVIDER_RUN_SCHEMA_VERSION,
    PROVIDER_SPEC_SCHEMA_VERSION,
    validate_provider_spec,
)
from src.tasks import TASK_CONTEXT_SCHEMA_VERSION


EXPERIMENT_MANIFEST_SCHEMA_VERSION = "1.0"
PROVIDER_CONFIGURATION_SCHEMA_VERSION = "1.0"
ALIGNMENT_BATCH_SCHEMA_VERSION = "1.0"


def sha256_file(path) -> str:
    """Return the SHA-256 of exact file bytes without loading the whole file."""
    digest = sha256()

    with path.open("rb") as input_file:
        while True:
            data_chunk = input_file.read(65536)
            if data_chunk == b"":
                break

            digest.update(data_chunk)

    return digest.hexdigest()


def _required_identifier(value, field_name):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(field_name + " must be a non-empty string.")

    normalized_value = value.strip()
    for character in normalized_value:
        if not (
            character.isascii()
            and (character.isalnum() or character in ("_", "-"))
        ):
            raise ValueError(
                field_name
                + " may contain only ASCII letters, digits, _ and -."
            )

    return normalized_value


def _required_text(value, field_name):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(field_name + " must be a non-empty string.")

    return value.strip()


def _validate_sha256(value, field_name):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(field_name + " must be a lowercase SHA-256 string.")

    allowed_characters = "0123456789abcdef"
    for character in value:
        if character not in allowed_characters:
            raise ValueError(
                field_name + " must be a lowercase SHA-256 string."
            )

    return value


def _optional_number(value, field_name, minimum=None, maximum=None):
    if value is None:
        return None

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(field_name + " must be a number or null.")

    normalized_value = float(value)
    if minimum is not None and normalized_value < minimum:
        raise ValueError(field_name + " is below the allowed minimum.")

    if maximum is not None and normalized_value > maximum:
        raise ValueError(field_name + " exceeds the allowed maximum.")

    return normalized_value


def _optional_integer(value, field_name, positive=False):
    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(field_name + " must be an integer or null.")

    if positive and value <= 0:
        raise ValueError(field_name + " must be positive.")

    return value


def _optional_text(value, field_name):
    if value is None:
        return None

    return _required_text(value, field_name)


def create_provider_configuration(
    provider_spec: dict[str, object],
    adapter_id: str,
    adapter_version: str,
    prompt_template_sha256: str,
    temperature=None,
    top_p=None,
    max_output_tokens=None,
    seed=None,
    reasoning_effort=None,
) -> dict[str, object]:
    """Freeze provider settings without storing a prompt, text, or secret."""
    validate_provider_spec(provider_spec)

    return {
        "provider_configuration_schema_version": (
            PROVIDER_CONFIGURATION_SCHEMA_VERSION
        ),
        "provider_spec": deepcopy(provider_spec),
        "adapter_id": _required_identifier(adapter_id, "adapter_id"),
        "adapter_version": _required_text(
            adapter_version,
            "adapter_version",
        ),
        "prompt_template_sha256": _validate_sha256(
            prompt_template_sha256,
            "prompt_template_sha256",
        ),
        "generation_parameters": {
            "temperature": _optional_number(
                temperature,
                "temperature",
                minimum=0.0,
            ),
            "top_p": _optional_number(
                top_p,
                "top_p",
                minimum=0.0,
                maximum=1.0,
            ),
            "max_output_tokens": _optional_integer(
                max_output_tokens,
                "max_output_tokens",
                positive=True,
            ),
            "seed": _optional_integer(seed, "seed"),
            "reasoning_effort": _optional_text(
                reasoning_effort,
                "reasoning_effort",
            ),
        },
    }


def validate_provider_configuration(provider_configuration) -> None:
    if not isinstance(provider_configuration, dict):
        raise ValueError("provider_configuration must be a dictionary.")

    expected_fields = (
        "provider_configuration_schema_version",
        "provider_spec",
        "adapter_id",
        "adapter_version",
        "prompt_template_sha256",
        "generation_parameters",
    )
    if sorted(provider_configuration.keys()) != sorted(expected_fields):
        raise ValueError("Provider configuration fields do not match schema.")

    generation_parameters = provider_configuration["generation_parameters"]
    if not isinstance(generation_parameters, dict):
        raise ValueError("generation_parameters must be a dictionary.")

    expected_parameter_fields = (
        "temperature",
        "top_p",
        "max_output_tokens",
        "seed",
        "reasoning_effort",
    )
    if sorted(generation_parameters.keys()) != sorted(
        expected_parameter_fields
    ):
        raise ValueError("Generation parameter fields do not match schema.")

    normalized = create_provider_configuration(
        provider_spec=provider_configuration["provider_spec"],
        adapter_id=provider_configuration["adapter_id"],
        adapter_version=provider_configuration["adapter_version"],
        prompt_template_sha256=provider_configuration[
            "prompt_template_sha256"
        ],
        temperature=generation_parameters["temperature"],
        top_p=generation_parameters["top_p"],
        max_output_tokens=generation_parameters["max_output_tokens"],
        seed=generation_parameters["seed"],
        reasoning_effort=generation_parameters["reasoning_effort"],
    )

    if normalized != provider_configuration:
        raise ValueError("Provider configuration is not canonical.")


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value):
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_records_sha256(records, id_field):
    records_by_id = {}

    for record in records:
        records_by_id[record[id_field]] = record

    sorted_records = []
    for record_id in sorted(records_by_id.keys()):
        sorted_records.append(records_by_id[record_id])

    return _sha256_json(sorted_records)


def _validate_source_file_hashes(source_file_sha256):
    if not isinstance(source_file_sha256, dict):
        raise ValueError("source_file_sha256 must be a dictionary.")

    expected_fields = ("tasks", "responses", "assessments")
    if sorted(source_file_sha256.keys()) != sorted(expected_fields):
        raise ValueError("Source file hash fields do not match schema.")

    normalized_hashes = {}
    for field_name in expected_fields:
        normalized_hashes[field_name] = _validate_sha256(
            source_file_sha256[field_name],
            field_name + " source file hash",
        )

    return normalized_hashes


def _require_role_pure_records(task_records, response_records, data_role):
    for task_record in task_records:
        if task_record.get("data_role") != data_role:
            raise ValueError("Experiment inputs must contain one data role.")

    for response_record in response_records:
        if response_record.get("data_role") != data_role:
            raise ValueError("Experiment inputs must contain one data role.")


def _protocol_versions():
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "task_context_schema_version": TASK_CONTEXT_SCHEMA_VERSION,
        "alignment_assessment_schema_version": (
            ALIGNMENT_ASSESSMENT_SCHEMA_VERSION
        ),
        "alignment_dataset_schema_version": (
            ALIGNMENT_DATASET_SCHEMA_VERSION
        ),
        "annotation_guideline_version": ANNOTATION_GUIDELINE_VERSION,
        "provider_spec_schema_version": PROVIDER_SPEC_SCHEMA_VERSION,
        "provider_request_schema_version": PROVIDER_REQUEST_SCHEMA_VERSION,
        "provider_run_schema_version": PROVIDER_RUN_SCHEMA_VERSION,
        "alignment_evaluation_schema_version": (
            ALIGNMENT_EVALUATION_SCHEMA_VERSION
        ),
        "alignment_batch_schema_version": ALIGNMENT_BATCH_SCHEMA_VERSION,
        "alignment_prompt_package_schema_version": (
            ALIGNMENT_PROMPT_PACKAGE_SCHEMA_VERSION
        ),
        "alignment_prompt_template_version": (
            ALIGNMENT_PROMPT_TEMPLATE_VERSION
        ),
    }


def _evaluation_policy():
    return {
        "evaluation_unit": "requirement",
        "coverage_reported_separately": True,
        "agreement_denominator": "completed_valid_runs_only",
        "runtime_abstention_is_not_not_assessed": True,
        "status_aggregation": "none",
        "evidence_comparison": (
            "exact_and_overlap_descriptive_not_semantic_correctness"
        ),
        "development_and_sealed_combined": False,
    }


def create_alignment_experiment_manifest(
    experiment_id: str,
    task_records,
    response_records,
    assessment_records,
    source_file_sha256,
    provider_configuration,
    data_role: str,
    sealed_evaluation_authorized: bool = False,
) -> dict[str, object]:
    """Create a deterministic freeze manifest without learner content."""
    normalized_experiment_id = _required_identifier(
        experiment_id,
        "experiment_id",
    )

    if data_role not in DATA_ROLES:
        raise ValueError("Unsupported experiment data_role.")

    if not isinstance(sealed_evaluation_authorized, bool):
        raise ValueError("sealed_evaluation_authorized must be a boolean.")

    if (
        data_role == "sealed_evaluation"
        and sealed_evaluation_authorized is not True
    ):
        raise ValueError("Sealed manifest requires explicit authorization.")

    if data_role == "development" and sealed_evaluation_authorized is True:
        raise ValueError(
            "Sealed authorization cannot be applied to development data."
        )

    if not isinstance(task_records, list):
        raise ValueError("task_records must be a list.")
    if not isinstance(response_records, list) or len(response_records) == 0:
        raise ValueError("response_records must be a non-empty list.")
    if not isinstance(assessment_records, list):
        raise ValueError("assessment_records must be a list.")

    _require_role_pure_records(task_records, response_records, data_role)
    dataset_summary = validate_alignment_dataset(
        task_records,
        response_records,
        assessment_records,
    )
    validate_provider_configuration(provider_configuration)
    normalized_source_hashes = _validate_source_file_hashes(
        source_file_sha256
    )
    content_hashes = {
        "tasks": _canonical_records_sha256(task_records, "task_id"),
        "responses": _canonical_records_sha256(
            response_records,
            "response_id",
        ),
        "assessments": _canonical_records_sha256(
            assessment_records,
            "assessment_id",
        ),
    }
    combined_dataset_sha256 = _sha256_json(content_hashes)
    manifest = {
        "experiment_manifest_schema_version": (
            EXPERIMENT_MANIFEST_SCHEMA_VERSION
        ),
        "experiment_id": normalized_experiment_id,
        "freeze_status": "configuration_frozen",
        "data_role": data_role,
        "sealed_evaluation_authorized_at_freeze": (
            sealed_evaluation_authorized
        ),
        "protocol_versions": _protocol_versions(),
        "evaluation_policy": _evaluation_policy(),
        "provider_configuration": deepcopy(provider_configuration),
        "dataset": {
            "source_file_sha256": normalized_source_hashes,
            "canonical_content_sha256": content_hashes,
            "combined_canonical_sha256": combined_dataset_sha256,
            "summary": dataset_summary,
        },
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return manifest


def validate_alignment_experiment_manifest(
    manifest,
    task_records,
    response_records,
    assessment_records,
    source_file_sha256,
    provider_configuration,
    sealed_evaluation_authorized: bool = False,
) -> None:
    """Detect drift between a frozen manifest and current experiment inputs."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dictionary.")

    expected_manifest = create_alignment_experiment_manifest(
        experiment_id=manifest.get("experiment_id"),
        task_records=task_records,
        response_records=response_records,
        assessment_records=assessment_records,
        source_file_sha256=source_file_sha256,
        provider_configuration=provider_configuration,
        data_role=manifest.get("data_role"),
        sealed_evaluation_authorized=sealed_evaluation_authorized,
    )

    if manifest != expected_manifest:
        raise ValueError("Experiment manifest does not match current inputs.")


def evaluate_frozen_alignment_experiment(
    manifest,
    task_records,
    response_records,
    assessment_records,
    source_file_sha256,
    provider_configuration,
    run_records,
    sealed_evaluation_authorized: bool = False,
) -> dict[str, object]:
    """Evaluate runs only after verifying the complete frozen experiment."""
    validate_alignment_experiment_manifest(
        manifest=manifest,
        task_records=task_records,
        response_records=response_records,
        assessment_records=assessment_records,
        source_file_sha256=source_file_sha256,
        provider_configuration=provider_configuration,
        sealed_evaluation_authorized=sealed_evaluation_authorized,
    )
    evaluation = evaluate_alignment_provider_runs(
        task_records=task_records,
        response_records=response_records,
        assessment_records=assessment_records,
        run_records=run_records,
        data_role=manifest["data_role"],
        sealed_evaluation_authorized=sealed_evaluation_authorized,
    )

    if evaluation["provider"] != provider_configuration["provider_spec"]:
        raise ValueError("Provider runs do not match the frozen configuration.")

    expected_adapter = {
        "adapter_id": provider_configuration["adapter_id"],
        "adapter_version": provider_configuration["adapter_version"],
        "prompt_template_version": ALIGNMENT_PROMPT_TEMPLATE_VERSION,
        "prompt_template_sha256": provider_configuration[
            "prompt_template_sha256"
        ],
    }
    for run_record in run_records:
        if run_record["provider_run"].get("adapter") != expected_adapter:
            raise ValueError(
                "Provider runs do not match the frozen adapter configuration."
            )

    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "experiment_id": manifest["experiment_id"],
        "evaluation": evaluation,
    }
