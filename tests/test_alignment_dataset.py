from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from src.alignment import create_alignment_assessment
from src.alignment_dataset import validate_alignment_dataset
from src.analysis import ANALYSIS_SCHEMA_VERSION
from src.tasks import create_task_context


RESPONSE_TEXT = (
    "I support community service because it helps local people."
)


def _dataset():
    task_context = create_task_context(
        task_id="task_001",
        prompt="Should universities require community service?",
        requirements=[
            {"requirement_id": "position", "description": "State a position."},
            {"requirement_id": "support", "description": "Give a reason."},
        ],
    )
    task_records = [
        {
            "schema_version": "1.0",
            "task_id": "task_001",
            "source_stage": "alignment_pilot_v1",
            "data_role": "development",
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
            "data_role": "development",
            "consent_status": "internal_evaluation_only",
            "original_text": RESPONSE_TEXT,
            "original_text_sha256": sha256(
                RESPONSE_TEXT.encode("utf-8")
            ).hexdigest(),
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        }
    ]
    alignment_assessment = create_alignment_assessment(
        response_text=RESPONSE_TEXT,
        task_context=task_context,
        assessor_type="human",
        assessor_id="human_reviewer_001",
        requirement_results=[
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [{"start_char": 0, "end_char": 27}],
            },
            {
                "requirement_id": "support",
                "status": "met",
                "evidence": [{"start_char": 28, "end_char": 58}],
            },
        ],
    )
    assessment_records = [
        {
            "schema_version": "1.0",
            "assessment_id": "assessment_001",
            "response_id": "response_001",
            "annotation_origin": "human_reference",
            "adjudication_status": "single",
            "reviewer_id_is_pseudonymous": True,
            "alignment_assessment": alignment_assessment,
        }
    ]
    return task_records, response_records, assessment_records


def test_alignment_dataset_validates_and_returns_aggregate_counts():
    tasks, responses, assessments = _dataset()

    summary = validate_alignment_dataset(tasks, responses, assessments)

    assert summary == {
        "schema_version": "1.0",
        "task_count": 1,
        "response_count": 1,
        "assessment_count": 1,
        "participant_count": 1,
        "data_role_counts": {"development": 1},
        "annotation_origin_counts": {"human_reference": 1},
        "requirement_status_counts": {"met": 2},
    }


def test_response_text_hash_must_match_exact_private_text():
    tasks, responses, assessments = _dataset()
    responses[0]["original_text_sha256"] = "wrong"

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        validate_alignment_dataset(tasks, responses, assessments)


def test_participant_and_revision_family_cannot_cross_roles():
    tasks, responses, assessments = _dataset()
    second_response = deepcopy(responses[0])
    second_response["response_id"] = "response_002"
    second_response["data_role"] = "sealed_evaluation"
    responses.append(second_response)

    with pytest.raises(ValueError, match="Participant crosses"):
        validate_alignment_dataset(tasks, responses, assessments)

    second_response["participant_id"] = "participant_002"
    with pytest.raises(ValueError, match="Response family crosses"):
        validate_alignment_dataset(tasks, responses, assessments)


def test_reference_assessment_must_be_human_and_cover_each_response():
    tasks, responses, assessments = _dataset()
    model_assessment = deepcopy(assessments[0]["alignment_assessment"])
    model_assessment["assessor"]["assessor_type"] = "model"
    assessments[0]["alignment_assessment"] = model_assessment

    with pytest.raises(ValueError, match="human assessor"):
        validate_alignment_dataset(tasks, responses, assessments)

    with pytest.raises(ValueError, match="Every response"):
        validate_alignment_dataset(tasks, responses, [])


def test_manually_corrupted_reference_assessment_is_rejected():
    tasks, responses, assessments = _dataset()
    assessments[0]["alignment_assessment"]["requirement_results"][0][
        "status"
    ] = "invented_status"

    with pytest.raises(ValueError, match="Unsupported alignment status"):
        validate_alignment_dataset(tasks, responses, assessments)


def test_reviewer_identity_and_content_rights_must_be_explicit():
    tasks, responses, assessments = _dataset()
    assessments[0]["reviewer_id_is_pseudonymous"] = False

    with pytest.raises(ValueError, match="Reviewer identity"):
        validate_alignment_dataset(tasks, responses, assessments)

    tasks, responses, assessments = _dataset()
    tasks[0]["content_rights_status"] = "probably_public"

    with pytest.raises(ValueError, match="content_rights_status"):
        validate_alignment_dataset(tasks, responses, assessments)


def test_task_record_and_context_ids_must_match():
    tasks, responses, assessments = _dataset()
    tasks[0]["task_id"] = "different_task"

    with pytest.raises(ValueError, match="does not match task context"):
        validate_alignment_dataset(tasks, responses, assessments)


def test_private_alignment_jsonl_paths_are_gitignored_and_absent():
    repository_root = Path(__file__).resolve().parents[1]
    gitignore = (repository_root / ".gitignore").read_text(encoding="utf-8")
    private_paths = (
        "data/alignment_evaluation/tasks_v1.jsonl",
        "data/alignment_evaluation/responses_v1.jsonl",
        "data/alignment_evaluation/assessments_v1.jsonl",
    )

    for private_path in private_paths:
        assert private_path in gitignore
        assert not (repository_root / private_path).exists()
