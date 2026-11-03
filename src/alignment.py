from copy import deepcopy
from hashlib import sha256

from src.tasks import task_context_fingerprint, validate_task_context


ALIGNMENT_ASSESSMENT_SCHEMA_VERSION = "1.0"

ALIGNMENT_STATUSES = (
    "met",
    "partially_met",
    "not_met",
    "not_assessed",
)

ASSESSOR_TYPES = (
    "human",
    "model",
    "model_assisted",
)


def _required_text(value, field_name):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(field_name + " must be a non-empty string.")

    return value.strip()


def _optional_text(value, field_name):
    if value is None:
        return ""

    if not isinstance(value, str):
        raise ValueError(field_name + " must be a string.")

    return value.strip()


def _normalize_evidence(response_text, evidence):
    normalized_evidence = []

    if evidence is None:
        return normalized_evidence

    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list.")

    for evidence_item in evidence:
        if not isinstance(evidence_item, dict):
            raise ValueError("Each evidence item must be a dictionary.")

        start_char = evidence_item.get("start_char")
        end_char = evidence_item.get("end_char")

        if (
            not isinstance(start_char, int)
            or isinstance(start_char, bool)
            or not isinstance(end_char, int)
            or isinstance(end_char, bool)
        ):
            raise ValueError("Evidence offsets must be integers.")

        if start_char < 0 or end_char <= start_char:
            raise ValueError("Evidence offsets are invalid.")

        if end_char > len(response_text):
            raise ValueError("Evidence offsets exceed the response text.")

        evidence_text = response_text[start_char:end_char]
        supplied_text = evidence_item.get("text")

        if supplied_text is not None and supplied_text != evidence_text:
            raise ValueError("Evidence text does not match response offsets.")

        normalized_evidence.append(
            {
                "text": evidence_text,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return normalized_evidence


def _requirement_lookup(task_context):
    requirements_by_id = {}

    for requirement in task_context["requirements"]:
        requirements_by_id[requirement["requirement_id"]] = requirement

    return requirements_by_id


def _normalize_requirement_results(
    response_text,
    task_context,
    requirement_results,
):
    if not isinstance(requirement_results, list):
        raise ValueError("requirement_results must be a list.")

    requirements_by_id = _requirement_lookup(task_context)
    normalized_results = []
    assessed_ids = []

    for result in requirement_results:
        if not isinstance(result, dict):
            raise ValueError("Each requirement result must be a dictionary.")

        requirement_id = _required_text(
            result.get("requirement_id"),
            "requirement_id",
        )

        if requirement_id not in requirements_by_id:
            raise ValueError("Unknown requirement_id: " + requirement_id)

        if requirement_id in assessed_ids:
            raise ValueError("Each requirement must be assessed exactly once.")

        status = result.get("status")
        if status not in ALIGNMENT_STATUSES:
            raise ValueError("Unsupported alignment status: " + str(status))

        evidence = _normalize_evidence(
            response_text,
            result.get("evidence", []),
        )
        note = _optional_text(result.get("note", ""), "note")

        if status in ("met", "partially_met") and len(evidence) == 0:
            raise ValueError(status + " requires response evidence.")

        if status == "not_met" and note == "":
            raise ValueError("not_met requires an explanatory note.")

        if status == "not_assessed" and len(evidence) > 0:
            raise ValueError("not_assessed cannot contain response evidence.")

        requirement = requirements_by_id[requirement_id]
        normalized_results.append(
            {
                "requirement_id": requirement_id,
                "description": requirement["description"],
                "status": status,
                "evidence": evidence,
                "note": note,
            }
        )
        assessed_ids.append(requirement_id)

    if len(assessed_ids) != len(requirements_by_id):
        raise ValueError("Every task requirement must be assessed exactly once.")

    return normalized_results


def create_alignment_assessment(
    response_text: str,
    task_context: dict[str, object],
    requirement_results,
    assessor_type: str,
    assessor_id: str,
    assessor_version: str = "",
) -> dict[str, object]:
    """Create an evidence-grounded assessment supplied by an explicit assessor."""
    if not isinstance(response_text, str):
        raise ValueError("response_text must be a string.")

    validate_task_context(task_context)

    if assessor_type not in ASSESSOR_TYPES:
        raise ValueError("Unsupported assessor_type: " + str(assessor_type))

    normalized_results = _normalize_requirement_results(
        response_text,
        task_context,
        requirement_results,
    )

    return {
        "alignment_assessment_schema_version": (
            ALIGNMENT_ASSESSMENT_SCHEMA_VERSION
        ),
        "task_context_sha256": task_context_fingerprint(task_context),
        "response_text_sha256": sha256(
            response_text.encode("utf-8")
        ).hexdigest(),
        "response_character_count": len(response_text),
        "assessor": {
            "assessor_type": assessor_type,
            "assessor_id": _required_text(assessor_id, "assessor_id"),
            "assessor_version": _optional_text(
                assessor_version,
                "assessor_version",
            ),
        },
        "requirement_results": normalized_results,
        "aggregation": None,
        "aggregation_note": (
            "Requirement statuses are not combined into a score."
        ),
    }


def validate_alignment_for_analysis(alignment_assessment, analysis) -> None:
    if not isinstance(alignment_assessment, dict):
        raise ValueError("alignment_assessment must be a dictionary.")

    version = alignment_assessment.get(
        "alignment_assessment_schema_version"
    )
    if version != ALIGNMENT_ASSESSMENT_SCHEMA_VERSION:
        raise ValueError("Unsupported alignment assessment schema version.")

    task_context = analysis["task_context"]
    if task_context is None:
        raise ValueError("Alignment assessment requires a task context.")

    expected_task_hash = task_context_fingerprint(task_context)
    if alignment_assessment.get("task_context_sha256") != expected_task_hash:
        raise ValueError("Alignment assessment belongs to another task.")

    input_summary = analysis["input_summary"]
    if alignment_assessment.get("response_text_sha256") != input_summary.get(
        "text_sha256"
    ):
        raise ValueError("Alignment assessment belongs to another response.")

    if alignment_assessment.get("response_character_count") != input_summary[
        "character_count"
    ]:
        raise ValueError("Alignment response length does not match analysis.")

    expected_ids = []
    for requirement in task_context["requirements"]:
        expected_ids.append(requirement["requirement_id"])

    actual_ids = []
    for result in alignment_assessment.get("requirement_results", []):
        actual_ids.append(result.get("requirement_id"))

    if actual_ids != expected_ids:
        raise ValueError("Alignment requirements do not match task context.")


def copy_alignment_for_analysis(alignment_assessment, analysis):
    if alignment_assessment is None:
        return None

    validate_alignment_for_analysis(alignment_assessment, analysis)
    return deepcopy(alignment_assessment)
