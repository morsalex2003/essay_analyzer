"""Convert private task-alignment intake files into canonical dataset records.

The intake format is a simpler hand-written form of the canonical private
JSONL records validated by ``src.alignment_dataset``. The importer only adds
canonical metadata and hashes; it never corrects or trims learner text.
"""

from hashlib import sha256

from src.alignment_dataset import (
    ALIGNMENT_DATASET_SCHEMA_VERSION,
    CONSENT_STATUSES,
    CONTENT_RIGHTS_STATUSES,
    validate_alignment_source_records,
)
from src.tasks import create_task_context


INTAKE_FORMAT_VERSION = "1.0"
INTAKE_SOURCE_STAGE = "alignment_development_pilot_v1"
INTAKE_DATA_ROLE = "development"

INTAKE_TASK_FIELDS = (
    "task_id",
    "response_mode",
    "prompt",
    "instructions",
    "source_materials",
    "requirements",
    "content_rights_status",
)
REQUIRED_INTAKE_TASK_FIELDS = (
    "task_id",
    "response_mode",
    "prompt",
    "requirements",
    "content_rights_status",
)

INTAKE_RESPONSE_FIELDS = (
    "response_id",
    "task_id",
    "participant_id",
    "response_family_id",
    "original_text",
    "consent_status",
)


def _check_record_fields(record, allowed_fields, required_fields, record_name):
    """Reject missing and unexpected fields.

    Unexpected fields are rejected so that accidental identifiers such as a
    name or email column cannot silently travel into canonical records.
    Error messages contain field names only, never field values.
    """
    if not isinstance(record, dict):
        raise ValueError(record_name + " must be a JSON object.")

    for field_name in required_fields:
        if field_name not in record:
            raise ValueError(record_name + " is missing field: " + field_name)

    for field_name in record:
        if field_name not in allowed_fields:
            raise ValueError(
                record_name + " has unexpected field: " + str(field_name)
            )


def _record_label(record, id_field, record_name, record_number):
    record_id = record.get(id_field) if isinstance(record, dict) else None

    if isinstance(record_id, str) and record_id.strip() != "":
        return record_name + " " + record_id

    return record_name + " #" + str(record_number)


def convert_intake_tasks(intake_tasks) -> list[dict[str, object]]:
    """Return canonical development task records for intake task objects."""
    if not isinstance(intake_tasks, list):
        raise ValueError("Intake tasks must be a JSON list.")

    task_records = []

    for record_number, intake_task in enumerate(intake_tasks, start=1):
        label = _record_label(intake_task, "task_id", "Task", record_number)
        _check_record_fields(
            intake_task,
            INTAKE_TASK_FIELDS,
            REQUIRED_INTAKE_TASK_FIELDS,
            label,
        )

        requirements = intake_task["requirements"]
        if not isinstance(requirements, list) or len(requirements) == 0:
            raise ValueError(label + " must have at least one requirement.")

        if intake_task["content_rights_status"] not in CONTENT_RIGHTS_STATUSES:
            raise ValueError(label + " has unsupported content_rights_status.")

        task_context = create_task_context(
            prompt=intake_task["prompt"],
            response_mode=intake_task["response_mode"],
            instructions=intake_task.get("instructions", ""),
            source_materials=intake_task.get("source_materials", []),
            requirements=requirements,
            task_id=intake_task["task_id"],
        )

        # create_task_context() silently drops requirements whose description
        # repeats an earlier one. An intake requirement ID must not disappear.
        if len(task_context["requirements"]) != len(requirements):
            raise ValueError(label + " has duplicate requirement descriptions.")

        if (
            task_context["response_mode"] == "source_based_response"
            and len(task_context["source_materials"]) == 0
        ):
            raise ValueError(
                label + " is source-based but has no source_materials."
            )

        task_records.append(
            {
                "schema_version": ALIGNMENT_DATASET_SCHEMA_VERSION,
                "task_id": task_context["task_id"],
                "source_stage": INTAKE_SOURCE_STAGE,
                "data_role": INTAKE_DATA_ROLE,
                "content_rights_status": intake_task["content_rights_status"],
                "task_context": task_context,
            }
        )

    return task_records


def convert_intake_responses(
    intake_responses,
    analysis_schema_version: str,
) -> list[dict[str, object]]:
    """Return canonical development response records.

    ``original_text`` is copied exactly: no trimming, spelling correction, or
    newline normalization is applied before hashing.
    """
    if not isinstance(intake_responses, list):
        raise ValueError("Intake responses must be a list.")

    response_records = []

    for record_number, intake_response in enumerate(intake_responses, start=1):
        label = _record_label(
            intake_response,
            "response_id",
            "Response",
            record_number,
        )
        _check_record_fields(
            intake_response,
            INTAKE_RESPONSE_FIELDS,
            INTAKE_RESPONSE_FIELDS,
            label,
        )

        if intake_response["consent_status"] not in CONSENT_STATUSES:
            raise ValueError(label + " has unsupported consent_status.")

        original_text = intake_response["original_text"]
        if not isinstance(original_text, str) or original_text.strip() == "":
            raise ValueError(label + " has empty original_text.")

        text_hash = sha256(original_text.encode("utf-8")).hexdigest()

        response_records.append(
            {
                "schema_version": ALIGNMENT_DATASET_SCHEMA_VERSION,
                "response_id": intake_response["response_id"],
                "task_id": intake_response["task_id"],
                "participant_id": intake_response["participant_id"],
                "participant_id_is_pseudonymous": True,
                "response_family_id": intake_response["response_family_id"],
                "source_stage": INTAKE_SOURCE_STAGE,
                "data_role": INTAKE_DATA_ROLE,
                "consent_status": intake_response["consent_status"],
                "original_text": original_text,
                "original_text_sha256": text_hash,
                "analysis_schema_version": analysis_schema_version,
            }
        )

    return response_records


def _count_values(records, field_name):
    counts = {}

    for record in records:
        value = record[field_name]
        counts[value] = counts.get(value, 0) + 1

    return counts


def import_alignment_intake(
    intake_tasks,
    intake_responses,
    analysis_schema_version: str,
):
    """Convert and validate intake data.

    Returns canonical task records, canonical response records, and an
    aggregate summary that contains no prompt, source, or learner text.
    """
    task_records = convert_intake_tasks(intake_tasks)
    response_records = convert_intake_responses(
        intake_responses,
        analysis_schema_version,
    )
    validate_alignment_source_records(task_records, response_records)

    response_mode_counts = {}
    for task_record in task_records:
        response_mode = task_record["task_context"]["response_mode"]
        response_mode_counts[response_mode] = (
            response_mode_counts.get(response_mode, 0) + 1
        )

    participant_ids = []
    family_ids = []
    for response_record in response_records:
        if response_record["participant_id"] not in participant_ids:
            participant_ids.append(response_record["participant_id"])
        if response_record["response_family_id"] not in family_ids:
            family_ids.append(response_record["response_family_id"])

    summary = {
        "intake_format_version": INTAKE_FORMAT_VERSION,
        "source_stage": INTAKE_SOURCE_STAGE,
        "data_role": INTAKE_DATA_ROLE,
        "task_count": len(task_records),
        "response_count": len(response_records),
        "participant_count": len(participant_ids),
        "response_family_count": len(family_ids),
        "task_response_mode_counts": response_mode_counts,
        "content_rights_status_counts": _count_values(
            task_records,
            "content_rights_status",
        ),
        "consent_status_counts": _count_values(
            response_records,
            "consent_status",
        ),
    }

    return task_records, response_records, summary
