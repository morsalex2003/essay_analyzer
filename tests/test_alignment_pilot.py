from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import subprocess

import pytest

from scripts.prepare_alignment_annotation_pilot import write_jsonl
from src.alignment_dataset import validate_alignment_source_records
from src.alignment_pilot import (
    complete_annotation_adjudication,
    create_annotation_review_batch,
    create_development_annotation_worklist,
    create_independent_reviewer_worklists,
    validate_completed_annotation_worklist,
    validate_reviewer_id,
)
from src.analysis import ANALYSIS_SCHEMA_VERSION
from src.tasks import create_task_context


RESPONSE_TEXT = "Public transport should be free because it reduces traffic."


def _source_records(data_role="development"):
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
            "source_stage": "alignment_pilot_v1",
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
            "source_stage": "alignment_pilot_v1",
            "data_role": data_role,
            "consent_status": "internal_evaluation_only",
            "original_text": RESPONSE_TEXT,
            "original_text_sha256": sha256(
                RESPONSE_TEXT.encode("utf-8")
            ).hexdigest(),
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        }
    ]
    return task_records, response_records


def _completed_worklist(reviewer_id, reason_status="met"):
    task_records, response_records = _source_records()
    worklist = create_development_annotation_worklist(
        task_records,
        response_records,
        reviewer_id,
    )
    item = worklist[0]
    item["review_status"] = "completed"

    for result in item["requirement_results"]:
        result["status"] = "met"
        result["evidence"] = [
            {"start_char": 0, "end_char": len(RESPONSE_TEXT)}
        ]

    reason_result = item["requirement_results"][1]
    reason_result["status"] = reason_status

    if reason_status == "not_met":
        reason_result["evidence"] = []
        reason_result["note"] = "No reason was identified."
    elif reason_status == "not_assessed":
        reason_result["evidence"] = []

    return worklist


def test_source_records_can_be_validated_before_annotation():
    task_records, response_records = _source_records()

    summary = validate_alignment_source_records(
        task_records,
        response_records,
    )

    assert summary == {
        "schema_version": "1.0",
        "task_count": 1,
        "response_count": 1,
        "development_response_count": 1,
        "sealed_evaluation_response_count": 0,
    }


def test_worklist_contains_only_blind_annotation_inputs():
    task_records, response_records = _source_records()

    worklist = create_development_annotation_worklist(
        task_records,
        response_records,
        "reviewer_01",
    )

    assert len(worklist) == 1
    item = worklist[0]
    assert item["response_text"] == RESPONSE_TEXT
    assert item["task_context"] == task_records[0]["task_context"]
    assert item["reviewer_id"] == "reviewer_01"
    assert item["review_status"] == "not_started"
    assert item["requirement_results"] == [
        {
            "requirement_id": "position",
            "description": "State a position.",
            "status": None,
            "evidence": [],
            "note": "",
        },
        {
            "requirement_id": "reason",
            "description": "Give one reason.",
            "status": None,
            "evidence": [],
            "note": "",
        },
    ]

    forbidden_fields = (
        "alignment_assessment",
        "analysis",
        "cefr",
        "lexical",
        "model_prediction",
        "orchestration_review",
        "provider_run",
        "syntax",
    )
    for field_name in forbidden_fields:
        assert field_name not in item


def test_reviewer_worklists_are_independent_deep_copies():
    task_records, response_records = _source_records()

    worklists = create_independent_reviewer_worklists(
        task_records,
        response_records,
        ["reviewer_01", "reviewer_02"],
    )

    first_item = worklists["reviewer_01"][0]
    second_item = worklists["reviewer_02"][0]
    first_item["requirement_results"][0]["status"] = "met"
    first_item["task_context"]["prompt"] = "Changed locally"

    assert second_item["requirement_results"][0]["status"] is None
    assert second_item["task_context"]["prompt"] == (
        "Should public transport be free?"
    )
    assert task_records[0]["task_context"]["prompt"] == (
        "Should public transport be free?"
    )


def test_worklist_rejects_sealed_evaluation_records():
    task_records, response_records = _source_records(
        data_role="sealed_evaluation"
    )

    with pytest.raises(ValueError, match="cannot include sealed data"):
        create_development_annotation_worklist(
            task_records,
            response_records,
            "reviewer_01",
        )


def test_at_least_two_unique_safe_reviewer_ids_are_required():
    task_records, response_records = _source_records()

    with pytest.raises(ValueError, match="At least two"):
        create_independent_reviewer_worklists(
            task_records,
            response_records,
            ["reviewer_01"],
        )

    with pytest.raises(ValueError, match="must be unique"):
        create_independent_reviewer_worklists(
            task_records,
            response_records,
            ["reviewer_01", "reviewer_01"],
        )

    with pytest.raises(ValueError, match="ASCII letters"):
        validate_reviewer_id("../reviewer")


def test_private_worklist_directory_is_ignored_by_git():
    # The local/ directory is where private pilot data is meant to live, so
    # it may legitimately exist. What matters is that Git never tracks it.
    # Tests themselves write only to pytest's tmp_path.
    repository_root = Path(__file__).resolve().parents[1]
    gitignore = (repository_root / ".gitignore").read_text(encoding="utf-8")
    private_file = (
        repository_root
        / "data/alignment_evaluation/local/reviewer_01_worklist_v1.jsonl"
    )

    assert "data/alignment_evaluation/local/" in gitignore

    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(private_file)],
            cwd=repository_root,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git is not available")

    if result.returncode == 128:
        pytest.skip("not running inside a Git repository")

    assert result.returncode == 0


def test_source_validation_does_not_modify_records():
    task_records, response_records = _source_records()
    original_tasks = deepcopy(task_records)
    original_responses = deepcopy(response_records)

    validate_alignment_source_records(task_records, response_records)

    assert task_records == original_tasks
    assert response_records == original_responses


def test_completed_worklist_is_validated_through_alignment_contract():
    worklist = _completed_worklist("reviewer_01")

    summary = validate_completed_annotation_worklist(worklist)

    assert summary == {
        "worklist_schema_version": "1.0",
        "annotation_guideline_version": "1.0",
        "reviewer_id": "reviewer_01",
        "completed_response_count": 1,
    }

    worklist[0]["requirement_results"][0]["evidence"] = []
    with pytest.raises(ValueError, match="requires response evidence"):
        validate_completed_annotation_worklist(worklist)


def test_incomplete_or_contaminated_worklist_is_rejected():
    task_records, response_records = _source_records()
    incomplete = create_development_annotation_worklist(
        task_records,
        response_records,
        "reviewer_01",
    )

    with pytest.raises(ValueError, match="completed status"):
        validate_completed_annotation_worklist(incomplete)

    contaminated = _completed_worklist("reviewer_01")
    contaminated[0]["model_prediction"] = "met"
    with pytest.raises(ValueError, match="fields do not match"):
        validate_completed_annotation_worklist(contaminated)

    contaminated = _completed_worklist("reviewer_01")
    contaminated[0]["requirement_results"][0][
        "model_prediction"
    ] = "met"
    with pytest.raises(ValueError, match="fields do not match"):
        validate_completed_annotation_worklist(contaminated)


def test_review_batch_preserves_agreement_and_queues_disagreement():
    first_worklist = _completed_worklist("reviewer_01", "met")
    second_worklist = _completed_worklist(
        "reviewer_02",
        "partially_met",
    )

    review_batch = create_annotation_review_batch(
        [first_worklist, second_worklist]
    )

    assert review_batch["decision_policy"] == (
        "no_voting_human_adjudication"
    )
    assert review_batch["summary"] == {
        "reviewer_count": 2,
        "response_count": 1,
        "requirement_agreement_count": 1,
        "requirement_disagreement_count": 1,
        "responses_requiring_adjudication": 1,
    }
    assert len(review_batch["adjudication_queue"]) == 1
    queue_item = review_batch["adjudication_queue"][0]
    assert queue_item["disagreement_requirement_ids"] == ["reason"]
    assert queue_item["adjudication_status"] == "not_started"
    assert queue_item["review"]["requirements"][1][
        "agreed_status"
    ] is None
    assert len(
        queue_item["review"]["requirements"][1]["assessor_results"]
    ) == 2


def test_review_batch_requires_distinct_compatible_worklists():
    first_worklist = _completed_worklist("reviewer_01")
    duplicate_reviewer = _completed_worklist("reviewer_01")

    with pytest.raises(ValueError, match="distinct reviewers"):
        create_annotation_review_batch(
            [first_worklist, duplicate_reviewer]
        )

    second_worklist = _completed_worklist("reviewer_02")
    second_worklist[0]["response_id"] = "different_response"
    with pytest.raises(ValueError, match="same responses"):
        create_annotation_review_batch(
            [first_worklist, second_worklist]
        )

    second_worklist = _completed_worklist("reviewer_02")
    second_worklist[0]["task_context"]["prompt"] = "Another prompt"
    with pytest.raises(ValueError, match="incompatible response data"):
        create_annotation_review_batch(
            [first_worklist, second_worklist]
        )


def test_human_adjudication_completes_full_canonical_assessment():
    first_worklist = _completed_worklist("reviewer_01", "met")
    second_worklist = _completed_worklist(
        "reviewer_02",
        "partially_met",
    )
    review_batch = create_annotation_review_batch(
        [first_worklist, second_worklist]
    )
    queue_item = review_batch["adjudication_queue"][0]
    original_queue_item = deepcopy(queue_item)

    adjudicated = complete_annotation_adjudication(
        queue_item=queue_item,
        adjudicator_id="adjudicator_01",
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
                "status": "partially_met",
                "evidence": [
                    {"start_char": 0, "end_char": len(RESPONSE_TEXT)}
                ],
                "note": "The reason is present but brief.",
            },
        ],
    )

    assert queue_item == original_queue_item
    assert adjudicated["adjudication_status"] == "completed"
    assert adjudicated["review"]["review_status"] == "human_adjudicated"
    assessment = adjudicated["review"]["adjudication"]["assessment"]
    assert assessment["assessor"] == {
        "assessor_type": "human",
        "assessor_id": "adjudicator_01",
        "assessor_version": "1.0",
    }


def test_private_worklist_writer_refuses_to_overwrite(tmp_path):
    output_path = tmp_path / "reviewer_worklist.jsonl"
    write_jsonl(output_path, [{"private": "synthetic test record"}])

    with pytest.raises(FileExistsError):
        write_jsonl(output_path, [{"private": "replacement"}])
