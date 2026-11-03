from copy import deepcopy
from hashlib import sha256
import json

import pytest

from src.alignment_readiness import (
    assess_development_pilot_readiness,
    create_readiness_declarations,
)
from src.analysis import ANALYSIS_SCHEMA_VERSION
from src.tasks import create_task_context


def _sha256_text(value):
    return sha256(value.encode("utf-8")).hexdigest()


def _declarations(all_confirmed=True):
    return create_readiness_declarations(
        direct_identifiers_removed=all_confirmed,
        consent_scope_reviewed=all_confirmed,
        two_independent_reviewers_confirmed=all_confirmed,
        separate_adjudicator_confirmed=all_confirmed,
    )


def _pilot_records(data_role="development"):
    task_contexts = [
        create_task_context(
            task_id="task_001",
            prompt="State a position and give one reason.",
            response_mode="independent_response",
            requirements=[
                {"requirement_id": "position", "description": "Position."},
                {"requirement_id": "reason", "description": "One reason."},
            ],
        ),
        create_task_context(
            task_id="task_002",
            prompt="Use the source to explain one benefit.",
            response_mode="source_based_response",
            source_materials=[
                {
                    "source_id": "source_001",
                    "source_type": "text",
                    "title": "Synthetic source",
                    "content": "Public transport can reduce traffic.",
                }
            ],
            requirements=[
                {"requirement_id": "benefit", "description": "A benefit."},
                {
                    "requirement_id": "source_use",
                    "description": "Use source information.",
                },
            ],
        ),
        create_task_context(
            task_id="task_003",
            prompt="Respond to a class discussion.",
            response_mode="discussion_response",
            requirements=[
                {"requirement_id": "response", "description": "Respond."},
                {
                    "requirement_id": "support",
                    "description": "Support the response.",
                },
            ],
        ),
    ]
    task_records = []

    for task_context in task_contexts:
        task_records.append(
            {
                "schema_version": "1.0",
                "task_id": task_context["task_id"],
                "source_stage": "alignment_development_pilot_v1",
                "data_role": data_role,
                "content_rights_status": "owner_provided",
                "task_context": task_context,
            }
        )

    response_records = []
    for response_index in range(12):
        task_number = (response_index % 3) + 1
        participant_number = (response_index % 6) + 1
        response_text = (
            "Synthetic private response " + str(response_index + 1) + "."
        )
        response_records.append(
            {
                "schema_version": "1.0",
                "response_id": "response_" + str(response_index + 1),
                "task_id": "task_00" + str(task_number),
                "participant_id": (
                    "participant_" + str(participant_number)
                ),
                "participant_id_is_pseudonymous": True,
                "response_family_id": "family_" + str(response_index + 1),
                "source_stage": "alignment_development_pilot_v1",
                "data_role": data_role,
                "consent_status": "internal_evaluation_only",
                "original_text": response_text,
                "original_text_sha256": _sha256_text(response_text),
                "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            }
        )

    return task_records, response_records


def test_ready_pilot_meets_operational_plan_without_exposing_text():
    tasks, responses = _pilot_records()

    result = assess_development_pilot_readiness(
        tasks,
        responses,
        _declarations(),
    )

    assert result["operational_status"] == (
        "meets_development_pilot_plan"
    )
    assert result["observed_counts"]["task_count"] == 3
    assert result["observed_counts"]["response_count"] == 12
    assert result["observed_counts"]["participant_count"] == 6
    assert result["target_checks"]["minimum_response_modes_met"] is True
    assert result["blockers"] == []
    assert result["warnings"] == []
    serialized_result = json.dumps(result)
    assert "Synthetic private response" not in serialized_result
    assert "participant_1" not in serialized_result


def test_unconfirmed_manual_safeguards_block_annotation_start():
    tasks, responses = _pilot_records()

    result = assess_development_pilot_readiness(
        tasks,
        responses,
        _declarations(all_confirmed=False),
    )

    assert result["operational_status"] == "blocked"
    assert sorted(result["blockers"]) == sorted(
        (
            "direct_identifiers_removed",
            "consent_scope_reviewed",
            "two_independent_reviewers_confirmed",
            "separate_adjudicator_confirmed",
        )
    )


def test_small_valid_dataset_is_below_plan_not_invalid():
    tasks, responses = _pilot_records()

    result = assess_development_pilot_readiness(
        tasks[:1],
        responses[:1],
        _declarations(),
    )

    assert result["operational_status"] == "below_plan"
    assert result["blockers"] == []
    assert "minimum_responses_met" in result["warnings"]
    assert "minimum_tasks_met" in result["warnings"]
    assert "statistical power analysis" in result["interpretation_note"]


def test_task_without_explicit_requirements_is_blocking():
    task_context = create_task_context(
        task_id="task_001",
        prompt="Write a response.",
        requirements=[],
    )
    tasks = [
        {
            "schema_version": "1.0",
            "task_id": "task_001",
            "source_stage": "alignment_development_pilot_v1",
            "data_role": "development",
            "content_rights_status": "owner_provided",
            "task_context": task_context,
        }
    ]
    response_text = "Synthetic response."
    responses = [
        {
            "schema_version": "1.0",
            "response_id": "response_001",
            "task_id": "task_001",
            "participant_id": "participant_001",
            "participant_id_is_pseudonymous": True,
            "response_family_id": "family_001",
            "source_stage": "alignment_development_pilot_v1",
            "data_role": "development",
            "consent_status": "internal_evaluation_only",
            "original_text": response_text,
            "original_text_sha256": _sha256_text(response_text),
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        }
    ]

    result = assess_development_pilot_readiness(
        tasks,
        responses,
        _declarations(),
    )

    assert result["operational_status"] == "blocked"
    assert "task_without_explicit_requirements" in result["blockers"]


def test_readiness_audit_rejects_sealed_records():
    tasks, responses = _pilot_records(data_role="sealed_evaluation")

    with pytest.raises(ValueError, match="development data only"):
        assess_development_pilot_readiness(
            tasks,
            responses,
            _declarations(),
        )


def test_readiness_declarations_require_real_booleans():
    with pytest.raises(ValueError, match="must be a boolean"):
        create_readiness_declarations(
            direct_identifiers_removed="yes",
            consent_scope_reviewed=True,
            two_independent_reviewers_confirmed=True,
            separate_adjudicator_confirmed=True,
        )


def test_readiness_audit_does_not_modify_private_records():
    tasks, responses = _pilot_records()
    original_records = deepcopy((tasks, responses))

    assess_development_pilot_readiness(
        tasks,
        responses,
        _declarations(),
    )

    assert (tasks, responses) == original_records
