from copy import deepcopy
from hashlib import sha256
import json


TASK_CONTEXT_SCHEMA_VERSION = "1.0"

RESPONSE_MODES = (
    "general_response",
    "independent_response",
    "source_based_response",
    "discussion_response",
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


def _normalize_sources(source_materials):
    normalized_sources = []
    source_ids = []

    if source_materials is None:
        return normalized_sources

    if not isinstance(source_materials, list):
        raise ValueError("source_materials must be a list.")

    for source_index, source in enumerate(source_materials, start=1):
        if not isinstance(source, dict):
            raise ValueError("Each source material must be a dictionary.")

        source_id = source.get("source_id", "source_" + str(source_index))
        source_type = source.get("source_type", "text")
        title = source.get("title", "")
        content = source.get("content")

        normalized_source_id = _required_text(source_id, "source_id")

        if normalized_source_id in source_ids:
            raise ValueError("source_id values must be unique.")

        source_ids.append(normalized_source_id)
        normalized_sources.append(
            {
                "source_id": normalized_source_id,
                "source_type": _required_text(source_type, "source_type"),
                "title": _optional_text(title, "title"),
                "content": _required_text(content, "source content"),
            }
        )

    return normalized_sources


def _normalize_requirements(requirements):
    normalized_requirements = []
    requirement_ids = []
    requirement_descriptions = []

    if requirements is None:
        return normalized_requirements

    if not isinstance(requirements, list):
        raise ValueError("requirements must be a list.")

    for requirement_index, requirement in enumerate(requirements, start=1):
        default_id = "requirement_" + str(requirement_index)

        if isinstance(requirement, str):
            requirement_id = default_id
            description = requirement
        elif isinstance(requirement, dict):
            requirement_id = requirement.get("requirement_id", default_id)
            description = requirement.get("description")
        else:
            raise ValueError(
                "Each requirement must be a string or dictionary."
            )

        normalized_id = _required_text(requirement_id, "requirement_id")
        normalized_description = _required_text(
            description,
            "requirement description",
        )

        if normalized_id in requirement_ids:
            raise ValueError("requirement_id values must be unique.")

        if normalized_description in requirement_descriptions:
            continue

        requirement_ids.append(normalized_id)
        requirement_descriptions.append(normalized_description)
        normalized_requirements.append(
            {
                "requirement_id": normalized_id,
                "description": normalized_description,
            }
        )

    return normalized_requirements


def create_task_context(
    prompt: str,
    response_mode: str = "general_response",
    instructions: str = "",
    source_materials=None,
    requirements=None,
    task_id: str | None = None,
) -> dict[str, object]:
    """Create a validated writing-task context without analyzing its content."""
    if not isinstance(response_mode, str) or response_mode not in RESPONSE_MODES:
        raise ValueError("Unsupported response_mode: " + str(response_mode))

    normalized_task_id = None
    if task_id is not None:
        normalized_task_id = _required_text(task_id, "task_id")

    return {
        "task_context_schema_version": TASK_CONTEXT_SCHEMA_VERSION,
        "task_id": normalized_task_id,
        "response_mode": response_mode,
        "prompt": _required_text(prompt, "prompt"),
        "instructions": _optional_text(instructions, "instructions"),
        "source_materials": _normalize_sources(source_materials),
        "requirements": _normalize_requirements(requirements),
    }


def validate_task_context(task_context: dict[str, object]) -> None:
    """Validate a task context supplied to the application boundary."""
    if not isinstance(task_context, dict):
        raise ValueError("task_context must be a dictionary.")

    version = task_context.get("task_context_schema_version")
    if version != TASK_CONTEXT_SCHEMA_VERSION:
        raise ValueError("Unsupported task context schema version.")

    normalized = create_task_context(
        prompt=task_context.get("prompt"),
        response_mode=task_context.get("response_mode"),
        instructions=task_context.get("instructions", ""),
        source_materials=task_context.get("source_materials", []),
        requirements=task_context.get("requirements", []),
        task_id=task_context.get("task_id"),
    )

    if normalized != task_context:
        raise ValueError("task_context is not in canonical normalized form.")


def copy_task_context(task_context):
    if task_context is None:
        return None

    validate_task_context(task_context)
    return deepcopy(task_context)


def task_context_fingerprint(task_context: dict[str, object]) -> str:
    validate_task_context(task_context)
    serialized_context = json.dumps(
        task_context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized_context.encode("utf-8")).hexdigest()
