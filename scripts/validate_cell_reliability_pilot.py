"""Validate CELL Reliability Pilot v1 privacy and batch integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_DIRECTORY = (
    PROJECT_ROOT / "data/syntax_reliability/external/cell"
)
LOCAL_DIRECTORY = CELL_DIRECTORY / "local/pilot_v1"

sys.path.insert(0, str(PROJECT_ROOT))

from src.cell_annotation import merge_progress  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    split = json.loads((CELL_DIRECTORY / "split_v1.json").read_text())
    report = json.loads((CELL_DIRECTORY / "pilot_v1_report.json").read_text())
    schema = json.loads(
        (LOCAL_DIRECTORY / "annotation_schema.json").read_text()
    )
    essays = load_jsonl(LOCAL_DIRECTORY / "essays.jsonl")
    annotations = load_jsonl(
        LOCAL_DIRECTORY / "annotations_template.jsonl"
    )
    progress_path = LOCAL_DIRECTORY / "annotations_v1.jsonl"
    if progress_path.exists():
        # The labelled ledger replaces the template for label checks, after
        # confirming that it changes nothing but annotation fields.
        annotations = merge_progress(annotations, load_jsonl(progress_path))

    development_students = set(split["students"]["development"])
    holdout_students = set(split["students"]["untouched_holdout"])
    essay_ids = {essay["essay_id"] for essay in essays}
    student_ids = {essay["student_id"] for essay in essays}
    sentence_ids = {annotation["sentence_id"] for annotation in annotations}

    assert len(essay_ids) == len(essays) == report["counts"]["essays"]
    assert len(student_ids) == report["counts"]["students"]
    assert len(sentence_ids) == len(annotations) == report["counts"]["sentences"]
    assert student_ids <= development_students
    assert student_ids.isdisjoint(holdout_students)
    assert report["holdout_opened"] is False
    assert report["privacy"]["raw_git_status"] == "ignored_untracked"

    report_text = json.dumps(report, sort_keys=True)
    for essay in essays:
        original_text = essay["original_text"]
        assert hashlib.sha256(original_text.encode()).hexdigest() == (
            essay["original_text_sha256"]
        )
        assert original_text not in report_text

    for annotation in annotations:
        original_text = annotation["original_text"]
        assert annotation["essay_id"] in essay_ids
        assert annotation["student_id"] in student_ids
        manual_label = annotation["manual_label"]
        if manual_label is None:
            assert annotation["usable_binary_label"] is None
        else:
            assert annotation["annotation_origin"] in (
                "human_manual_review",
                "model_assisted_manual_review",
            )
            assert manual_label in schema["allowed_manual_labels"]
            assert annotation["usable_binary_label"] == (
                schema["binary_mapping"][manual_label]
            )
            if manual_label == "NOT_ANALYZABLE":
                assert annotation["abstention_status"] not in (None, "none")
            else:
                assert annotation["abstention_status"] == "none"
            assert annotation["annotator_id"]
        assert hashlib.sha256(original_text.encode()).hexdigest() == (
            annotation["original_text_sha256"]
        )
        assert original_text not in report_text

    assert schema["allowed_manual_labels"] == [
        "RELIABLE",
        "MOSTLY_RELIABLE",
        "UNRELIABLE",
        "NOT_ANALYZABLE",
    ]
    labelled_count = sum(
        annotation["manual_label"] is not None for annotation in annotations
    )
    print(
        "CELL pilot valid: "
        f"{len(student_ids)} students, {len(essays)} essays, "
        f"{len(annotations)} sentences ({labelled_count} labelled); "
        "holdout unopened; privacy passed"
    )


if __name__ == "__main__":
    main()
