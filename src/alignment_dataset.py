from hashlib import sha256

from src.alignment import (
    create_alignment_assessment,
    validate_alignment_for_analysis,
)
from src.tasks import validate_task_context


ALIGNMENT_DATASET_SCHEMA_VERSION = "1.0"

DATA_ROLES = (
    "development",
    "sealed_evaluation",
)

CONSENT_STATUSES = (
    "research_use_confirmed",
    "educational_use_confirmed",
    "internal_evaluation_only",
)

ANNOTATION_ORIGINS = (
    "human_reference",
    "model_assisted_human_adjudication",
)

CONTENT_RIGHTS_STATUSES = (
    "owner_provided",
    "licensed_for_evaluation",
    "permission_confirmed",
    "restricted_local_only",
)


def _required_fields(record, required_fields, record_name):
    missing_fields = []

    for field_name in required_fields:
        if field_name not in record:
            missing_fields.append(field_name)

    if len(missing_fields) > 0:
        raise ValueError(
            record_name
            + " is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )


def _require_unique(record_id, seen_ids, field_name):
    if not isinstance(record_id, str) or record_id.strip() == "":
        raise ValueError(field_name + " must be a non-empty string.")

    if record_id in seen_ids:
        raise ValueError("Duplicate " + field_name + ": " + record_id)

    seen_ids.append(record_id)


def _validate_task_records(task_records):
    tasks_by_id = {}
    seen_task_ids = []

    for record in task_records:
        _required_fields(
            record,
            (
                "schema_version",
                "task_id",
                "source_stage",
                "data_role",
                "content_rights_status",
                "task_context",
            ),
            "Task record",
        )

        if record["schema_version"] != ALIGNMENT_DATASET_SCHEMA_VERSION:
            raise ValueError("Unsupported task record schema version.")

        task_id = record["task_id"]
        _require_unique(task_id, seen_task_ids, "task_id")
        validate_task_context(record["task_context"])

        if record["task_context"]["task_id"] != task_id:
            raise ValueError("Task record ID does not match task context.")

        if record["data_role"] not in DATA_ROLES:
            raise ValueError("Unsupported task data_role.")

        if record["content_rights_status"] not in CONTENT_RIGHTS_STATUSES:
            raise ValueError("Unsupported content_rights_status.")

        if (
            not isinstance(record["source_stage"], str)
            or record["source_stage"].strip() == ""
        ):
            raise ValueError("Task source_stage must be a non-empty string.")

        tasks_by_id[task_id] = record

    return tasks_by_id


def _validate_response_records(response_records, tasks_by_id):
    responses_by_id = {}
    seen_response_ids = []
    participant_roles = {}
    family_roles = {}

    for record in response_records:
        _required_fields(
            record,
            (
                "schema_version",
                "response_id",
                "task_id",
                "participant_id",
                "participant_id_is_pseudonymous",
                "response_family_id",
                "source_stage",
                "data_role",
                "consent_status",
                "original_text",
                "original_text_sha256",
                "analysis_schema_version",
            ),
            "Response record",
        )

        if record["schema_version"] != ALIGNMENT_DATASET_SCHEMA_VERSION:
            raise ValueError("Unsupported response record schema version.")

        response_id = record["response_id"]
        _require_unique(response_id, seen_response_ids, "response_id")

        if record["task_id"] not in tasks_by_id:
            raise ValueError("Response references unknown task_id.")

        if record["data_role"] not in DATA_ROLES:
            raise ValueError("Unsupported response data_role.")

        if record["consent_status"] not in CONSENT_STATUSES:
            raise ValueError("Unsupported consent_status.")

        if record["participant_id_is_pseudonymous"] is not True:
            raise ValueError("participant_id must be explicitly pseudonymous.")

        for field_name in (
            "participant_id",
            "response_family_id",
            "source_stage",
            "analysis_schema_version",
        ):
            field_value = record[field_name]
            if not isinstance(field_value, str) or field_value.strip() == "":
                raise ValueError(field_name + " must be a non-empty string.")

        original_text = record["original_text"]
        if not isinstance(original_text, str) or original_text == "":
            raise ValueError("original_text must be a non-empty string.")

        expected_hash = sha256(original_text.encode("utf-8")).hexdigest()
        if record["original_text_sha256"] != expected_hash:
            raise ValueError("Response text SHA-256 does not match text.")

        participant_id = record["participant_id"]
        existing_role = participant_roles.get(participant_id)
        if existing_role is not None and existing_role != record["data_role"]:
            raise ValueError("Participant crosses development/evaluation roles.")
        participant_roles[participant_id] = record["data_role"]

        family_id = record["response_family_id"]
        existing_family_role = family_roles.get(family_id)
        if (
            existing_family_role is not None
            and existing_family_role != record["data_role"]
        ):
            raise ValueError("Response family crosses dataset roles.")
        family_roles[family_id] = record["data_role"]

        task_role = tasks_by_id[record["task_id"]]["data_role"]
        if task_role != record["data_role"]:
            raise ValueError("Task and response data roles must match.")

        responses_by_id[response_id] = record

    return responses_by_id


def _minimal_analysis(task_context, response_record):
    return {
        "analysis_schema_version": response_record["analysis_schema_version"],
        "task_context": task_context,
        "input_summary": {
            "text_sha256": response_record["original_text_sha256"],
            "character_count": len(response_record["original_text"]),
        },
    }


def _validate_assessment_records(
    assessment_records,
    tasks_by_id,
    responses_by_id,
):
    seen_assessment_ids = []
    assessed_response_ids = []
    origin_counts = {}
    status_counts = {}

    for record in assessment_records:
        _required_fields(
            record,
            (
                "schema_version",
                "assessment_id",
                "response_id",
                "annotation_origin",
                "adjudication_status",
                "reviewer_id_is_pseudonymous",
                "alignment_assessment",
            ),
            "Assessment record",
        )

        if record["schema_version"] != ALIGNMENT_DATASET_SCHEMA_VERSION:
            raise ValueError("Unsupported assessment record schema version.")

        _require_unique(
            record["assessment_id"],
            seen_assessment_ids,
            "assessment_id",
        )

        response_id = record["response_id"]
        if response_id not in responses_by_id:
            raise ValueError("Assessment references unknown response_id.")

        if response_id in assessed_response_ids:
            raise ValueError("Each response must have one reference assessment.")

        origin = record["annotation_origin"]
        if origin not in ANNOTATION_ORIGINS:
            raise ValueError("Unsupported annotation_origin.")

        if record["adjudication_status"] not in ("single", "adjudicated"):
            raise ValueError("Unsupported adjudication_status.")

        if record["reviewer_id_is_pseudonymous"] is not True:
            raise ValueError("Reviewer identity must be pseudonymous.")

        if (
            origin == "model_assisted_human_adjudication"
            and record["adjudication_status"] != "adjudicated"
        ):
            raise ValueError(
                "Model-assisted reference requires adjudicated status."
            )

        response_record = responses_by_id[response_id]
        task_record = tasks_by_id[response_record["task_id"]]
        analysis = _minimal_analysis(
            task_record["task_context"],
            response_record,
        )
        alignment_assessment = record["alignment_assessment"]
        validate_alignment_for_analysis(alignment_assessment, analysis)

        assessor = alignment_assessment["assessor"]
        if assessor["assessor_type"] != "human":
            raise ValueError("Reference assessment must have a human assessor.")

        canonical_assessment = create_alignment_assessment(
            response_text=response_record["original_text"],
            task_context=task_record["task_context"],
            requirement_results=alignment_assessment[
                "requirement_results"
            ],
            assessor_type=assessor["assessor_type"],
            assessor_id=assessor["assessor_id"],
            assessor_version=assessor["assessor_version"],
        )

        if canonical_assessment != alignment_assessment:
            raise ValueError("Reference assessment is not canonical.")

        assessed_response_ids.append(response_id)
        origin_counts[origin] = origin_counts.get(origin, 0) + 1

        for requirement_result in alignment_assessment[
            "requirement_results"
        ]:
            status = requirement_result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

    if len(assessed_response_ids) != len(responses_by_id):
        raise ValueError("Every response must have one reference assessment.")

    return origin_counts, status_counts


def validate_alignment_dataset(
    task_records,
    response_records,
    assessment_records,
) -> dict[str, object]:
    """Validate a private alignment dataset and return aggregate counts only."""
    if not isinstance(task_records, list):
        raise ValueError("task_records must be a list.")
    if not isinstance(response_records, list):
        raise ValueError("response_records must be a list.")
    if not isinstance(assessment_records, list):
        raise ValueError("assessment_records must be a list.")

    tasks_by_id = _validate_task_records(task_records)
    responses_by_id = _validate_response_records(
        response_records,
        tasks_by_id,
    )
    origin_counts, status_counts = _validate_assessment_records(
        assessment_records,
        tasks_by_id,
        responses_by_id,
    )
    role_counts = {}
    participant_ids = []

    for response in response_records:
        role = response["data_role"]
        role_counts[role] = role_counts.get(role, 0) + 1

        participant_id = response["participant_id"]
        if participant_id not in participant_ids:
            participant_ids.append(participant_id)

    return {
        "schema_version": ALIGNMENT_DATASET_SCHEMA_VERSION,
        "task_count": len(task_records),
        "response_count": len(response_records),
        "assessment_count": len(assessment_records),
        "participant_count": len(participant_ids),
        "data_role_counts": role_counts,
        "annotation_origin_counts": origin_counts,
        "requirement_status_counts": status_counts,
    }


def validate_alignment_source_records(
    task_records,
    response_records,
) -> dict[str, object]:
    """Validate private task/response inputs before annotation starts."""
    if not isinstance(task_records, list):
        raise ValueError("task_records must be a list.")
    if not isinstance(response_records, list):
        raise ValueError("response_records must be a list.")

    tasks_by_id = _validate_task_records(task_records)
    _validate_response_records(response_records, tasks_by_id)
    development_count = 0
    sealed_count = 0

    for response in response_records:
        if response["data_role"] == "development":
            development_count += 1
        else:
            sealed_count += 1

    return {
        "schema_version": ALIGNMENT_DATASET_SCHEMA_VERSION,
        "task_count": len(task_records),
        "response_count": len(response_records),
        "development_response_count": development_count,
        "sealed_evaluation_response_count": sealed_count,
    }
