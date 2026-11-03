from copy import deepcopy
from hashlib import sha256

from src.alignment import create_alignment_assessment
from src.alignment_dataset import validate_alignment_source_records
from src.orchestration import (
    adjudicate_alignment_review,
    create_alignment_review,
)
from src.tasks import validate_task_context


ANNOTATION_WORKLIST_SCHEMA_VERSION = "1.0"
ANNOTATION_GUIDELINE_VERSION = "1.0"
ANNOTATION_REVIEW_BATCH_SCHEMA_VERSION = "1.0"

WORKLIST_FIELDS = (
    "worklist_schema_version",
    "annotation_guideline_version",
    "analysis_schema_version",
    "reviewer_id",
    "reviewer_id_is_pseudonymous",
    "response_id",
    "task_id",
    "data_role",
    "source_stage",
    "task_context",
    "response_text",
    "requirement_results",
    "review_status",
)


def validate_reviewer_id(reviewer_id: str) -> str:
    if not isinstance(reviewer_id, str) or reviewer_id.strip() == "":
        raise ValueError("reviewer_id must be a non-empty string.")

    normalized_id = reviewer_id.strip()

    for character in normalized_id:
        if not (
            character.isascii()
            and (character.isalnum() or character in ("_", "-"))
        ):
            raise ValueError(
                "reviewer_id may contain only ASCII letters, digits, _ and -."
            )

    return normalized_id


def _tasks_by_id(task_records):
    lookup = {}

    for record in task_records:
        lookup[record["task_id"]] = record

    return lookup


def _blank_requirement_results(task_context):
    requirement_results = []

    for requirement in task_context["requirements"]:
        requirement_results.append(
            {
                "requirement_id": requirement["requirement_id"],
                "description": requirement["description"],
                "status": None,
                "evidence": [],
                "note": "",
            }
        )

    return requirement_results


def _required_text(value, field_name):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(field_name + " must be a non-empty string.")

    return value.strip()


def _validate_worklist_fields(record):
    if not isinstance(record, dict):
        raise ValueError("Each worklist item must be a dictionary.")

    actual_fields = sorted(record.keys())
    expected_fields = sorted(WORKLIST_FIELDS)

    if actual_fields != expected_fields:
        raise ValueError("Worklist item fields do not match the schema.")


def _analysis_binding(record):
    response_text = record["response_text"]

    return {
        "analysis_schema_version": record["analysis_schema_version"],
        "task_context": record["task_context"],
        "input_summary": {
            "text_sha256": sha256(
                response_text.encode("utf-8")
            ).hexdigest(),
            "character_count": len(response_text),
        },
    }


def _validate_completed_result_fields(requirement_results, task_context):
    if not isinstance(requirement_results, list):
        raise ValueError("requirement_results must be a list.")

    requirements = task_context["requirements"]
    if len(requirement_results) != len(requirements):
        raise ValueError("Every task requirement must be assessed once.")

    expected_fields = sorted(
        ("requirement_id", "description", "status", "evidence", "note")
    )

    for index, result in enumerate(requirement_results):
        if not isinstance(result, dict):
            raise ValueError("Each requirement result must be a dictionary.")

        if sorted(result.keys()) != expected_fields:
            raise ValueError("Requirement result fields do not match schema.")

        requirement = requirements[index]
        if result["requirement_id"] != requirement["requirement_id"]:
            raise ValueError("Requirement results are in the wrong order.")

        if result["description"] != requirement["description"]:
            raise ValueError("Requirement description does not match task.")

        evidence = result["evidence"]
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list.")

        for evidence_item in evidence:
            if not isinstance(evidence_item, dict):
                raise ValueError("Each evidence item must be a dictionary.")

            allowed_fields = ("start_char", "end_char", "text")
            for field_name in evidence_item:
                if field_name not in allowed_fields:
                    raise ValueError(
                        "Evidence item contains an unsupported field."
                    )


def _canonical_completed_item(record, expected_reviewer_id=None):
    _validate_worklist_fields(record)

    if record["worklist_schema_version"] != (
        ANNOTATION_WORKLIST_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported annotation worklist schema version.")

    if record["annotation_guideline_version"] != (
        ANNOTATION_GUIDELINE_VERSION
    ):
        raise ValueError("Unsupported annotation guideline version.")

    reviewer_id = validate_reviewer_id(record["reviewer_id"])
    if expected_reviewer_id is not None and reviewer_id != (
        expected_reviewer_id
    ):
        raise ValueError("One worklist cannot contain multiple reviewers.")

    if record["reviewer_id_is_pseudonymous"] is not True:
        raise ValueError("Reviewer identity must be pseudonymous.")

    if record["data_role"] != "development":
        raise ValueError("Completed pilot worklists must be development data.")

    _required_text(record["response_id"], "response_id")
    _required_text(record["task_id"], "task_id")
    _required_text(record["source_stage"], "source_stage")
    _required_text(
        record["analysis_schema_version"],
        "analysis_schema_version",
    )

    response_text = record["response_text"]
    if not isinstance(response_text, str) or response_text == "":
        raise ValueError("response_text must be a non-empty string.")

    task_context = record["task_context"]
    validate_task_context(task_context)

    if task_context["task_id"] != record["task_id"]:
        raise ValueError("Worklist task_id does not match task context.")

    if record["review_status"] != "completed":
        raise ValueError("Completed worklist items require completed status.")

    _validate_completed_result_fields(
        record["requirement_results"],
        task_context,
    )
    assessment = create_alignment_assessment(
        response_text=response_text,
        task_context=task_context,
        requirement_results=record["requirement_results"],
        assessor_type="human",
        assessor_id=reviewer_id,
        assessor_version=record["annotation_guideline_version"],
    )

    return {
        "response_id": record["response_id"],
        "task_id": record["task_id"],
        "source_stage": record["source_stage"],
        "analysis_schema_version": record["analysis_schema_version"],
        "task_context": deepcopy(task_context),
        "response_text": response_text,
        "assessment": assessment,
    }


def _canonical_completed_worklist(worklist):
    if not isinstance(worklist, list) or len(worklist) == 0:
        raise ValueError("Completed worklist must be a non-empty list.")

    first_record = worklist[0]
    if not isinstance(first_record, dict):
        raise ValueError("Each worklist item must be a dictionary.")

    reviewer_id = validate_reviewer_id(first_record.get("reviewer_id"))
    response_ids = []
    canonical_items = []

    for record in worklist:
        canonical_item = _canonical_completed_item(record, reviewer_id)
        response_id = canonical_item["response_id"]

        if response_id in response_ids:
            raise ValueError("Completed worklist has duplicate response_id.")

        response_ids.append(response_id)
        canonical_items.append(canonical_item)

    return reviewer_id, canonical_items


def _items_by_response(canonical_items):
    items_by_response = {}

    for item in canonical_items:
        items_by_response[item["response_id"]] = item

    return items_by_response


def _require_same_response(first_item, other_item):
    fields = (
        "task_id",
        "source_stage",
        "analysis_schema_version",
        "task_context",
        "response_text",
    )

    for field_name in fields:
        if first_item[field_name] != other_item[field_name]:
            raise ValueError(
                "Reviewer worklists contain incompatible response data."
            )


def _disagreement_ids(review):
    requirement_ids = []

    for requirement in review["requirements"]:
        if requirement["requires_human_review"]:
            requirement_ids.append(requirement["requirement_id"])

    return requirement_ids


def create_development_annotation_worklist(
    task_records,
    response_records,
    reviewer_id: str,
) -> list[dict[str, object]]:
    """Create a private blind worklist without model or parser outputs."""
    validate_alignment_source_records(task_records, response_records)
    normalized_reviewer_id = validate_reviewer_id(reviewer_id)
    tasks_by_id = _tasks_by_id(task_records)
    worklist = []

    for response in response_records:
        if response["data_role"] != "development":
            raise ValueError(
                "Development annotation worklists cannot include sealed data."
            )

        task_record = tasks_by_id[response["task_id"]]
        task_context = task_record["task_context"]
        worklist.append(
            {
                "worklist_schema_version": (
                    ANNOTATION_WORKLIST_SCHEMA_VERSION
                ),
                "annotation_guideline_version": (
                    ANNOTATION_GUIDELINE_VERSION
                ),
                "analysis_schema_version": response[
                    "analysis_schema_version"
                ],
                "reviewer_id": normalized_reviewer_id,
                "reviewer_id_is_pseudonymous": True,
                "response_id": response["response_id"],
                "task_id": response["task_id"],
                "data_role": response["data_role"],
                "source_stage": response["source_stage"],
                "task_context": deepcopy(task_context),
                "response_text": response["original_text"],
                "requirement_results": _blank_requirement_results(
                    task_context
                ),
                "review_status": "not_started",
            }
        )

    return worklist


def create_independent_reviewer_worklists(
    task_records,
    response_records,
    reviewer_ids,
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(reviewer_ids, list) or len(reviewer_ids) < 2:
        raise ValueError("At least two independent reviewer IDs are required.")

    normalized_ids = []
    worklists = {}

    for reviewer_id in reviewer_ids:
        normalized_id = validate_reviewer_id(reviewer_id)

        if normalized_id in normalized_ids:
            raise ValueError("Reviewer IDs must be unique.")

        normalized_ids.append(normalized_id)
        worklists[normalized_id] = create_development_annotation_worklist(
            task_records,
            response_records,
            normalized_id,
        )

    return worklists


def validate_completed_annotation_worklist(
    worklist,
) -> dict[str, object]:
    """Validate one private human worklist and return aggregate metadata."""
    reviewer_id, canonical_items = _canonical_completed_worklist(worklist)

    return {
        "worklist_schema_version": ANNOTATION_WORKLIST_SCHEMA_VERSION,
        "annotation_guideline_version": ANNOTATION_GUIDELINE_VERSION,
        "reviewer_id": reviewer_id,
        "completed_response_count": len(canonical_items),
    }


def create_annotation_review_batch(
    completed_worklists,
) -> dict[str, object]:
    """Compare independent human worklists without automatic resolution."""
    if not isinstance(completed_worklists, list) or len(
        completed_worklists
    ) < 2:
        raise ValueError("At least two completed worklists are required.")

    reviewer_ids = []
    canonical_worklists = []

    for worklist in completed_worklists:
        reviewer_id, canonical_items = _canonical_completed_worklist(worklist)

        if reviewer_id in reviewer_ids:
            raise ValueError("Completed worklists require distinct reviewers.")

        reviewer_ids.append(reviewer_id)
        canonical_worklists.append(canonical_items)

    first_worklist = canonical_worklists[0]
    expected_response_ids = []

    for item in first_worklist:
        expected_response_ids.append(item["response_id"])

    response_lookups = []
    for canonical_items in canonical_worklists:
        lookup = _items_by_response(canonical_items)

        if sorted(lookup.keys()) != sorted(expected_response_ids):
            raise ValueError(
                "Completed worklists must cover the same responses."
            )

        response_lookups.append(lookup)

    response_reviews = []
    adjudication_queue = []
    agreement_count = 0
    disagreement_count = 0

    for response_id in expected_response_ids:
        first_item = response_lookups[0][response_id]
        assessments = []

        for lookup in response_lookups:
            reviewer_item = lookup[response_id]
            _require_same_response(first_item, reviewer_item)
            assessments.append(reviewer_item["assessment"])

        review = create_alignment_review(
            _analysis_binding(first_item),
            assessments,
        )
        disagreement_ids = _disagreement_ids(review)
        response_review = {
            "response_id": response_id,
            "task_id": first_item["task_id"],
            "review": review,
            "disagreement_requirement_ids": disagreement_ids,
        }
        response_reviews.append(response_review)
        agreement_count += review["summary"]["agreement_count"]
        disagreement_count += review["summary"]["disagreement_count"]

        if len(disagreement_ids) > 0:
            adjudication_queue.append(
                {
                    "response_id": response_id,
                    "task_id": first_item["task_id"],
                    "source_stage": first_item["source_stage"],
                    "analysis_schema_version": first_item[
                        "analysis_schema_version"
                    ],
                    "task_context": deepcopy(first_item["task_context"]),
                    "response_text": first_item["response_text"],
                    "review": review,
                    "disagreement_requirement_ids": disagreement_ids,
                    "adjudication_requirement_results": (
                        _blank_requirement_results(
                            first_item["task_context"]
                        )
                    ),
                    "adjudication_status": "not_started",
                }
            )

    return {
        "review_batch_schema_version": (
            ANNOTATION_REVIEW_BATCH_SCHEMA_VERSION
        ),
        "annotation_guideline_version": ANNOTATION_GUIDELINE_VERSION,
        "decision_policy": "no_voting_human_adjudication",
        "reviewer_ids": reviewer_ids,
        "response_reviews": response_reviews,
        "adjudication_queue": adjudication_queue,
        "summary": {
            "reviewer_count": len(reviewer_ids),
            "response_count": len(response_reviews),
            "requirement_agreement_count": agreement_count,
            "requirement_disagreement_count": disagreement_count,
            "responses_requiring_adjudication": len(adjudication_queue),
        },
    }


def complete_annotation_adjudication(
    queue_item,
    requirement_results,
    adjudicator_id: str,
) -> dict[str, object]:
    """Complete one queued disagreement with an explicit human decision."""
    if not isinstance(queue_item, dict):
        raise ValueError("queue_item must be a dictionary.")

    if queue_item.get("adjudication_status") != "not_started":
        raise ValueError("Queue item is not awaiting adjudication.")

    if len(queue_item.get("disagreement_requirement_ids", [])) == 0:
        raise ValueError("Queue item does not contain a disagreement.")

    normalized_adjudicator_id = validate_reviewer_id(adjudicator_id)
    analysis = _analysis_binding(queue_item)
    adjudicated_review = adjudicate_alignment_review(
        response_text=queue_item["response_text"],
        analysis=analysis,
        review=queue_item["review"],
        requirement_results=requirement_results,
        adjudicator_id=normalized_adjudicator_id,
        adjudicator_version=ANNOTATION_GUIDELINE_VERSION,
    )

    return {
        "response_id": queue_item["response_id"],
        "task_id": queue_item["task_id"],
        "adjudication_status": "completed",
        "review": adjudicated_review,
    }
