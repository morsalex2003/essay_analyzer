from copy import deepcopy

from src.alignment_adapter import (
    run_json_alignment_adapter,
    validate_json_adapter_configuration,
)
from src.alignment_evaluation import create_evaluation_run_record
from src.alignment_experiments import (
    ALIGNMENT_BATCH_SCHEMA_VERSION,
    validate_alignment_experiment_manifest,
)
from src.providers import PROVIDER_RUN_STATUSES


ALIGNMENT_BATCH_PRIVACY_NOTE = (
    "Completed assessments contain learner evidence quotations; "
    "this batch is private and local-only."
)


def _tasks_by_id(task_records):
    tasks = {}

    for task_record in task_records:
        tasks[task_record["task_id"]] = task_record

    return tasks


def run_frozen_alignment_batch(
    manifest,
    task_records,
    response_records,
    assessment_records,
    source_file_sha256,
    provider_configuration,
    completion_callable,
    sealed_evaluation_authorized: bool = False,
) -> dict[str, object]:
    """Run one frozen adapter over every response and preserve every outcome."""
    validate_alignment_experiment_manifest(
        manifest=manifest,
        task_records=task_records,
        response_records=response_records,
        assessment_records=assessment_records,
        source_file_sha256=source_file_sha256,
        provider_configuration=provider_configuration,
        sealed_evaluation_authorized=sealed_evaluation_authorized,
    )
    validate_json_adapter_configuration(provider_configuration)

    if not callable(completion_callable):
        raise ValueError("completion_callable must be callable.")

    tasks_by_id = _tasks_by_id(task_records)
    run_records = []
    run_status_counts = {}

    for run_status in PROVIDER_RUN_STATUSES:
        run_status_counts[run_status] = 0

    for response_record in response_records:
        task_record = tasks_by_id[response_record["task_id"]]
        provider_run = run_json_alignment_adapter(
            response_text=response_record["original_text"],
            task_context=task_record["task_context"],
            provider_configuration=provider_configuration,
            completion_callable=completion_callable,
        )
        run_status_counts[provider_run["status"]] += 1
        run_records.append(
            create_evaluation_run_record(
                response_record["response_id"],
                provider_run,
            )
        )

    return {
        "alignment_batch_schema_version": ALIGNMENT_BATCH_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "data_role": manifest["data_role"],
        "provider_configuration": deepcopy(provider_configuration),
        "run_records": run_records,
        "summary": {
            "response_count": len(response_records),
            "recorded_run_count": len(run_records),
            "run_status_counts": run_status_counts,
        },
        "privacy_note": ALIGNMENT_BATCH_PRIVACY_NOTE,
    }


def validate_alignment_batch_result(
    batch_result,
    manifest,
    provider_configuration,
) -> None:
    """Validate batch provenance and its aggregate run-status summary."""
    if not isinstance(batch_result, dict):
        raise ValueError("batch_result must be a dictionary.")

    expected_fields = (
        "alignment_batch_schema_version",
        "experiment_id",
        "manifest_sha256",
        "data_role",
        "provider_configuration",
        "run_records",
        "summary",
        "privacy_note",
    )
    if sorted(batch_result.keys()) != sorted(expected_fields):
        raise ValueError("Alignment batch fields do not match schema.")

    if batch_result["alignment_batch_schema_version"] != (
        ALIGNMENT_BATCH_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported alignment batch schema version.")

    if batch_result["experiment_id"] != manifest.get("experiment_id"):
        raise ValueError("Alignment batch belongs to another experiment.")

    if batch_result["manifest_sha256"] != manifest.get("manifest_sha256"):
        raise ValueError("Alignment batch belongs to another manifest.")

    if batch_result["data_role"] != manifest.get("data_role"):
        raise ValueError("Alignment batch uses another data role.")

    if batch_result["provider_configuration"] != provider_configuration:
        raise ValueError("Alignment batch uses another provider configuration.")

    if batch_result["privacy_note"] != ALIGNMENT_BATCH_PRIVACY_NOTE:
        raise ValueError("Alignment batch privacy note changed.")

    run_records = batch_result["run_records"]
    if not isinstance(run_records, list):
        raise ValueError("Alignment batch run_records must be a list.")

    response_ids = []
    actual_status_counts = {}
    for run_status in PROVIDER_RUN_STATUSES:
        actual_status_counts[run_status] = 0

    for run_record in run_records:
        if not isinstance(run_record, dict):
            raise ValueError("Each batch run record must be a dictionary.")

        response_id = run_record.get("response_id")
        if not isinstance(response_id, str) or response_id.strip() == "":
            raise ValueError("Batch response_id must be a non-empty string.")

        if response_id in response_ids:
            raise ValueError("Alignment batch has duplicate response_id.")

        response_ids.append(response_id)
        provider_run = run_record.get("provider_run")
        if not isinstance(provider_run, dict):
            raise ValueError("Batch provider_run must be a dictionary.")

        run_status = provider_run.get("status")
        if run_status not in PROVIDER_RUN_STATUSES:
            raise ValueError("Batch provider run has unsupported status.")

        actual_status_counts[run_status] += 1

    expected_summary = {
        "response_count": len(run_records),
        "recorded_run_count": len(run_records),
        "run_status_counts": actual_status_counts,
    }
    if batch_result["summary"] != expected_summary:
        raise ValueError("Alignment batch summary does not match run records.")


def copy_run_records_from_batch(
    batch_result,
    manifest,
    provider_configuration,
):
    validate_alignment_batch_result(
        batch_result,
        manifest,
        provider_configuration,
    )
    return deepcopy(batch_result["run_records"])
