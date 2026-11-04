import json

import pytest

from src import reliability_development_set as development_set
from src.nlp.pipeline import nlp
from src.reliability_development_set import (
    build_development_manifest,
    essay_context_mismatches,
)


PROVENANCE = {
    "spacy_version": "3.8.13",
    "spacy_model_name": "en_core_web_sm",
    "spacy_model_version": "3.8.0",
    "clause_detector_version": "2.1",
    "reliability_version": "1.2",
}


def _essay(essay_id, student_id, text="The cat slept. The dog barked."):
    return {
        "essay_id": essay_id,
        "student_id": student_id,
        "source_stage": "synthetic_stage",
        "data_status": "development",
        "original_text": text,
        **PROVENANCE,
    }


def _annotation(essay_id, student_id, index, label):
    return {
        "sentence_id": f"{essay_id}_s{index}",
        "essay_id": essay_id,
        "student_id": student_id,
        "sentence_index": index,
        "original_text": "Synthetic sentence.",
        "manual_label": label,
        "annotation_guideline_version": "1.2",
        "annotation_origin": "model_assisted_manual_review",
        "annotator_id": "synthetic_annotator",
        "adjudication_status": "single",
        **PROVENANCE,
    }


def _batch(batch_id, student_ids, labels=("RELIABLE", "UNRELIABLE")):
    essays = []
    annotations = []
    for student_id in student_ids:
        essay_id = f"{batch_id}_{student_id}"
        essays.append(_essay(essay_id, student_id))
        for index, label in enumerate(labels, start=1):
            annotations.append(_annotation(essay_id, student_id, index, label))

    return {
        "batch_id": batch_id,
        "role_assigned_at": "2026-01-01",
        "task_types": ["unrecorded"],
        "duplicate_audit_status": "complete",
        "essays": essays,
        "annotations": annotations,
    }


def _manifest(batches, holdout_students=frozenset()):
    return build_development_manifest(
        batches,
        set(holdout_students),
        exclusions=[],
        known_limitations=[],
    )


def test_manifest_counts_and_binary_targets():
    manifest = _manifest(
        [
            _batch("batch_a", ["s1", "s2"]),
            _batch(
                "batch_b",
                ["s3"],
                labels=("MOSTLY_RELIABLE", "NOT_ANALYZABLE"),
            ),
        ]
    )

    assert manifest["totals"]["counts"] == {
        "students": 3,
        "essays": 3,
        "sentences": 6,
    }
    assert manifest["totals"]["binary_counts"] == {
        "usable": 3,
        "unreliable": 2,
        "excluded_not_analyzable": 1,
    }
    assert [record["student_id"] for record in manifest["students"]] == [
        "s1",
        "s2",
        "s3",
    ]
    assert manifest["holdout_opened"] is False
    assert manifest["batches"][0]["provenance"][
        "sentence_segmentation_version"
    ] == "1"


def test_manifest_exposes_no_private_record_fields():
    batch = _batch("batch_a", ["s1"])
    manifest_text = json.dumps(_manifest([batch]))

    for record in batch["annotations"]:
        assert record["sentence_id"] not in manifest_text
        assert record["essay_id"] not in manifest_text
    assert "original_text" not in manifest_text


def test_holdout_student_is_rejected():
    with pytest.raises(ValueError, match="holdout"):
        _manifest([_batch("batch_a", ["s1"])], holdout_students={"s1"})


def test_student_in_two_batches_is_rejected():
    with pytest.raises(ValueError, match="two batches"):
        _manifest([_batch("batch_a", ["s1"]), _batch("batch_b", ["s1"])])


def test_unlabelled_sentence_is_rejected():
    batch = _batch("batch_a", ["s1"])
    batch["annotations"][0]["manual_label"] = None

    with pytest.raises(ValueError, match="no final manual label"):
        _manifest([batch])


def test_mixed_clause_detector_versions_are_rejected():
    batch = _batch("batch_a", ["s1"])
    batch["annotations"][0]["clause_detector_version"] = "2.2"

    with pytest.raises(ValueError, match="clause_detector_version"):
        _manifest([batch])


def test_non_development_essay_is_rejected():
    batch = _batch("batch_a", ["s1"])
    batch["essays"][0]["data_status"] = "untouched_holdout"

    with pytest.raises(ValueError, match="not development"):
        _manifest([batch])


def test_essay_context_mismatches_follow_essay_segmentation():
    essay = _essay("essay_1", "s1")
    first = _annotation("essay_1", "s1", 1, "RELIABLE")
    first["original_text"] = "The cat slept. "
    second = _annotation("essay_1", "s1", 2, "RELIABLE")
    second["original_text"] = "The cat slept."

    assert essay_context_mismatches([essay], [first, second], nlp) == [
        "essay_1_s2"
    ]


def test_lock_detects_changed_source_file(tmp_path, monkeypatch):
    (tmp_path / "essays.jsonl").write_text('{"a": 1}\n')
    (tmp_path / "labels.jsonl").write_text('{"label": "RELIABLE"}\n')
    monkeypatch.setattr(
        development_set,
        "BATCH_DEFINITIONS",
        ({"essays_path": "essays.jsonl", "annotations_path": "labels.jsonl"},),
    )
    lock = development_set.build_lock(tmp_path)

    assert development_set.lock_mismatches(lock, tmp_path) == []
    assert lock["files"]["labels.jsonl"]["records"] == 1

    (tmp_path / "labels.jsonl").write_text('{"label": "UNRELIABLE"}\n')

    assert development_set.lock_mismatches(lock, tmp_path) == [
        "labels.jsonl"
    ]
