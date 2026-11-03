from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from src.alignment_dataset import validate_alignment_source_records
from src.alignment_intake import (
    INTAKE_DATA_ROLE,
    INTAKE_SOURCE_STAGE,
    convert_intake_responses,
    convert_intake_tasks,
    import_alignment_intake,
)
from src.alignment_readiness import (
    assess_development_pilot_readiness,
    create_readiness_declarations,
)


# Literal value keeps these tests independent of the spaCy-loading
# src.analysis module; the CLI passes src.analysis.ANALYSIS_SCHEMA_VERSION.
TEST_ANALYSIS_SCHEMA_VERSION = "1.0"


def _intake_tasks():
    return [
        {
            "task_id": "task_001",
            "response_mode": "independent_response",
            "prompt": "Synthetic prompt one.",
            "instructions": "Synthetic instructions.",
            "source_materials": [],
            "requirements": [
                {"requirement_id": "req_001", "description": "Position."},
                {"requirement_id": "req_002", "description": "Reasons."},
            ],
            "content_rights_status": "owner_provided",
        },
        {
            "task_id": "task_002",
            "response_mode": "source_based_response",
            "prompt": "Synthetic prompt two.",
            "source_materials": [
                {
                    "source_id": "source_001",
                    "source_type": "text",
                    "title": "Synthetic source",
                    "content": "Synthetic source content.",
                }
            ],
            "requirements": [
                {"requirement_id": "req_001", "description": "Use source."},
            ],
            "content_rights_status": "owner_provided",
        },
        {
            "task_id": "task_003",
            "response_mode": "discussion_response",
            "prompt": "Synthetic prompt three.",
            "requirements": [
                {"requirement_id": "req_001", "description": "Respond."},
            ],
            "content_rights_status": "permission_confirmed",
        },
    ]


def _intake_responses():
    responses = []

    for response_index in range(12):
        task_number = (response_index % 3) + 1
        participant_number = (response_index % 6) + 1
        responses.append(
            {
                "response_id": "response_" + str(response_index + 1),
                "task_id": "task_00" + str(task_number),
                "participant_id": "participant_" + str(participant_number),
                "response_family_id": "family_" + str(response_index + 1),
                "original_text": (
                    "Synthetic response " + str(response_index + 1) + "."
                ),
                "consent_status": "internal_evaluation_only",
            }
        )

    return responses


def test_import_produces_records_accepted_by_canonical_validator():
    task_records, response_records, summary = import_alignment_intake(
        _intake_tasks(),
        _intake_responses(),
        TEST_ANALYSIS_SCHEMA_VERSION,
    )

    validation = validate_alignment_source_records(
        task_records,
        response_records,
    )

    assert validation["task_count"] == 3
    assert validation["development_response_count"] == 12
    assert validation["sealed_evaluation_response_count"] == 0
    assert summary["task_count"] == 3
    assert summary["response_count"] == 12
    assert summary["participant_count"] == 6
    assert summary["response_family_count"] == 12
    assert summary["task_response_mode_counts"] == {
        "independent_response": 1,
        "source_based_response": 1,
        "discussion_response": 1,
    }
    assert summary["consent_status_counts"] == {"internal_evaluation_only": 12}


def test_imported_records_meet_development_pilot_plan():
    task_records, response_records, _ = import_alignment_intake(
        _intake_tasks(),
        _intake_responses(),
        TEST_ANALYSIS_SCHEMA_VERSION,
    )
    declarations = create_readiness_declarations(
        direct_identifiers_removed=True,
        consent_scope_reviewed=True,
        two_independent_reviewers_confirmed=True,
        separate_adjudicator_confirmed=True,
    )

    result = assess_development_pilot_readiness(
        task_records=task_records,
        response_records=response_records,
        declarations=declarations,
    )

    assert result["operational_status"] == "meets_development_pilot_plan"


def test_summary_contains_no_task_or_learner_text():
    _, _, summary = import_alignment_intake(
        _intake_tasks(),
        _intake_responses(),
        TEST_ANALYSIS_SCHEMA_VERSION,
    )

    serialized_summary = str(summary)

    assert "Synthetic response" not in serialized_summary
    assert "Synthetic prompt" not in serialized_summary
    assert "Synthetic source content" not in serialized_summary


def test_original_text_is_copied_exactly_and_hashed():
    intake_responses = _intake_responses()
    exact_text = "  i thinks  that zoos is bad\n\nbecause animals  "
    intake_responses[0]["original_text"] = exact_text

    response_records = convert_intake_responses(
        intake_responses,
        TEST_ANALYSIS_SCHEMA_VERSION,
    )

    first_record = response_records[0]
    expected_hash = sha256(exact_text.encode("utf-8")).hexdigest()
    assert first_record["original_text"] == exact_text
    assert first_record["original_text_sha256"] == expected_hash
    assert first_record["participant_id_is_pseudonymous"] is True
    assert first_record["data_role"] == INTAKE_DATA_ROLE
    assert first_record["source_stage"] == INTAKE_SOURCE_STAGE


def test_all_records_are_development_only():
    task_records, response_records, _ = import_alignment_intake(
        _intake_tasks(),
        _intake_responses(),
        TEST_ANALYSIS_SCHEMA_VERSION,
    )

    for record in task_records + response_records:
        assert record["data_role"] == "development"


def test_task_context_is_canonical_and_keeps_task_id():
    task_records = convert_intake_tasks(_intake_tasks())

    assert task_records[0]["task_id"] == "task_001"
    assert task_records[0]["task_context"]["task_id"] == "task_001"
    assert task_records[1]["task_context"]["source_materials"][0][
        "source_id"
    ] == "source_001"


def test_unexpected_response_field_is_rejected():
    intake_responses = _intake_responses()
    intake_responses[0]["email"] = "student@example.com"

    with pytest.raises(ValueError, match="unexpected field: email"):
        convert_intake_responses(
            intake_responses,
            TEST_ANALYSIS_SCHEMA_VERSION,
        )


def test_unexpected_task_field_is_rejected():
    intake_tasks = _intake_tasks()
    intake_tasks[0]["teacher_name"] = "Synthetic Teacher"

    with pytest.raises(ValueError, match="unexpected field: teacher_name"):
        convert_intake_tasks(intake_tasks)


def test_missing_response_field_is_rejected():
    intake_responses = _intake_responses()
    del intake_responses[0]["consent_status"]

    with pytest.raises(ValueError, match="missing field: consent_status"):
        convert_intake_responses(
            intake_responses,
            TEST_ANALYSIS_SCHEMA_VERSION,
        )


def test_unsupported_consent_status_is_rejected():
    intake_responses = _intake_responses()
    intake_responses[0]["consent_status"] = "probably_fine"

    with pytest.raises(ValueError, match="consent_status"):
        convert_intake_responses(
            intake_responses,
            TEST_ANALYSIS_SCHEMA_VERSION,
        )


def test_unsupported_content_rights_status_is_rejected():
    intake_tasks = _intake_tasks()
    intake_tasks[0]["content_rights_status"] = "found_online"

    with pytest.raises(ValueError, match="content_rights_status"):
        convert_intake_tasks(intake_tasks)


def test_task_without_requirements_is_rejected():
    intake_tasks = _intake_tasks()
    intake_tasks[0]["requirements"] = []

    with pytest.raises(ValueError, match="at least one requirement"):
        convert_intake_tasks(intake_tasks)


def test_duplicate_requirement_description_is_not_silently_dropped():
    intake_tasks = _intake_tasks()
    intake_tasks[0]["requirements"] = [
        {"requirement_id": "req_001", "description": "Same."},
        {"requirement_id": "req_002", "description": "Same."},
    ]

    with pytest.raises(ValueError, match="duplicate requirement"):
        convert_intake_tasks(intake_tasks)


def test_source_based_task_without_sources_is_rejected():
    intake_tasks = _intake_tasks()
    intake_tasks[1]["source_materials"] = []

    with pytest.raises(ValueError, match="no source_materials"):
        convert_intake_tasks(intake_tasks)


def test_empty_response_text_is_rejected():
    intake_responses = _intake_responses()
    intake_responses[0]["original_text"] = "   "

    with pytest.raises(ValueError, match="empty original_text"):
        convert_intake_responses(
            intake_responses,
            TEST_ANALYSIS_SCHEMA_VERSION,
        )


def test_response_with_unknown_task_is_rejected():
    intake_responses = _intake_responses()
    intake_responses[0]["task_id"] = "task_999"

    with pytest.raises(ValueError, match="unknown task_id"):
        import_alignment_intake(
            _intake_tasks(),
            intake_responses,
            TEST_ANALYSIS_SCHEMA_VERSION,
        )


def test_duplicate_response_id_is_rejected():
    intake_responses = _intake_responses()
    intake_responses[1]["response_id"] = intake_responses[0]["response_id"]

    with pytest.raises(ValueError):
        import_alignment_intake(
            _intake_tasks(),
            intake_responses,
            TEST_ANALYSIS_SCHEMA_VERSION,
        )


def test_error_messages_do_not_contain_learner_text():
    intake_responses = _intake_responses()
    secret_text = "Private learner sentence that must not leak."
    intake_responses[0]["original_text"] = secret_text
    intake_responses[0]["consent_status"] = "unknown"

    with pytest.raises(ValueError) as error_info:
        convert_intake_responses(
            intake_responses,
            TEST_ANALYSIS_SCHEMA_VERSION,
        )

    assert secret_text not in str(error_info.value)


def test_input_records_are_not_mutated():
    intake_tasks = _intake_tasks()
    intake_responses = _intake_responses()
    original_tasks = deepcopy(intake_tasks)
    original_responses = deepcopy(intake_responses)

    import_alignment_intake(
        intake_tasks,
        intake_responses,
        TEST_ANALYSIS_SCHEMA_VERSION,
    )

    assert intake_tasks == original_tasks
    assert intake_responses == original_responses


def test_tracked_intake_template_meets_development_pilot_plan():
    # The template is synthetic example data, not learner text. Keeping it
    # passing here guarantees that copying it into local/intake/ works.
    template_directory = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "alignment_evaluation"
        / "intake_template"
    )
    intake_tasks = json.loads(
        (template_directory / "tasks.json").read_text(encoding="utf-8")
    )
    intake_responses = [
        json.loads(line)
        for line in (template_directory / "responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() != ""
    ]

    task_records, response_records, _ = import_alignment_intake(
        intake_tasks,
        intake_responses,
        TEST_ANALYSIS_SCHEMA_VERSION,
    )
    result = assess_development_pilot_readiness(
        task_records=task_records,
        response_records=response_records,
        declarations=create_readiness_declarations(True, True, True, True),
    )

    assert result["operational_status"] == "meets_development_pilot_plan"
    assert result["blockers"] == []
    assert result["warnings"] == []
