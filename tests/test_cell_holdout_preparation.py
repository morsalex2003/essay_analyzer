import json

import pytest

from scripts.apply_cell_annotation_labels import apply_label_rows
from scripts.build_cell_annotation_packet import (
    DATASET_README,
    GUIDELINE_END,
    GUIDELINE_START,
    guideline_text,
    instructions_text,
    sentence_sheet,
)
from scripts.review_cell_annotation_labels import (
    apply_review_rows,
    review_positions,
    review_rows,
)
from src.cell_annotation import SentenceParser, load_annotation_enums
from src.cell_pilot import select_pilot_students
from src.nlp.pipeline import nlp
from tests.test_cell_pilot import make_record


SCHEMA = json.loads(open("data/syntax_reliability/schema_v1.json").read())
ENUMS = load_annotation_enums(SCHEMA)
BINARY_MAPPING = {
    "RELIABLE": True,
    "MOSTLY_RELIABLE": True,
    "UNRELIABLE": False,
    "NOT_ANALYZABLE": None,
}
ESSAY_TEXT = "The students wrote essays. Teachers who care help them."


def _roles_and_records():
    records = []
    roles = {}
    for index in range(12):
        student_id = f"s{index:05d}"
        roles[student_id] = (
            "untouched_holdout" if index < 4 else "development"
        )
        records.append(make_record(student_id, "2018_01", "engle102f", index))
    return roles, records


def test_holdout_selection_uses_only_holdout_students():
    roles, records = _roles_and_records()

    selected = select_pilot_students(
        records, roles, student_count=3, role="untouched_holdout"
    )

    assert len(selected) == 3
    assert all(roles[student] == "untouched_holdout" for student in selected)


def test_holdout_selection_cannot_exceed_holdout_size():
    roles, records = _roles_and_records()

    with pytest.raises(ValueError, match="untouched_holdout"):
        select_pilot_students(
            records, roles, student_count=5, role="untouched_holdout"
        )


def _template():
    sentences = list(nlp(ESSAY_TEXT).sents)
    return [
        {
            "sentence_id": f"essay_1:s{index:04d}",
            "essay_id": "essay_1",
            "student_id": "s00001",
            "sentence_index": index,
            "original_text": sentence.text,
            "clause_detector_version": "2.1",
            "manual_label": None,
            "adjudication_status": "single",
        }
        for index, sentence in enumerate(sentences, start=1)
    ]


def test_sentence_sheet_is_blind():
    records = _template()
    parser = SentenceParser(
        nlp, {"essay_1": {"essay_id": "essay_1", "original_text": ESSAY_TEXT}}
    )

    sheet = sentence_sheet(records, parser)

    assert "=== 1 / 2   id essay_1:s0001" in sheet
    assert "finite clause heads" in sheet
    for leaked in ("reliab", "flag", "medium", "probab", "warning"):
        assert leaked not in sheet.lower()


def test_guideline_and_instructions():
    guideline = guideline_text(DATASET_README.read_text())
    instructions = instructions_text(300, ENUMS)

    assert guideline.startswith(GUIDELINE_START)
    assert GUIDELINE_END not in guideline
    assert "300 sentences" in instructions
    assert "finite_inventory" in instructions


def _row(position, sentence_id, label, failures="-", abstention="none"):
    return {
        "position": str(position),
        "sentence_id": sentence_id,
        "label": label,
        "failure_types": failures,
        "abstention_status": abstention,
        "note": "",
    }


def test_label_rows_are_applied_with_validation():
    records = _template()
    rows = [
        _row(1, "essay_1:s0001", "RELIABLE"),
        _row(2, "essay_1:s0002", "UNRELIABLE", "finite_inventory"),
    ]

    applied = apply_label_rows(
        records, rows, "sealed_test", BINARY_MAPPING, ENUMS
    )

    assert applied == 2
    assert records[0]["usable_binary_label"] is True
    assert records[1]["manual_failure_types"] == ["finite_inventory"]
    assert records[1]["annotator_id"] == "sealed_test"
    assert records[1]["annotation_origin"] == "model_assisted_manual_review"


def test_label_rows_refuse_mismatch_and_overwrite():
    records = _template()

    with pytest.raises(ValueError, match="mismatch"):
        apply_label_rows(
            records,
            [_row(1, "essay_1:s0002", "RELIABLE")],
            "sealed_test",
            BINARY_MAPPING,
            ENUMS,
        )

    apply_label_rows(
        records,
        [_row(1, "essay_1:s0001", "RELIABLE")],
        "sealed_test",
        BINARY_MAPPING,
        ENUMS,
    )
    with pytest.raises(ValueError, match="already labelled"):
        apply_label_rows(
            records,
            [_row(1, "essay_1:s0001", "UNRELIABLE")],
            "sealed_test",
            BINARY_MAPPING,
            ENUMS,
        )


def _labelled(labels):
    records = []
    for position, label in enumerate(labels, start=1):
        records.append(
            {
                "sentence_id": f"essay_1:s{position:04d}",
                "essay_id": "essay_1",
                "manual_label": label,
                "usable_binary_label": BINARY_MAPPING[label],
                "manual_failure_types": [],
                "abstention_status": "none",
                "annotation_notes": None,
                "annotation_origin": "model_assisted_manual_review",
                "annotator_id": "sealed_test",
                "adjudication_status": "single",
            }
        )
    return records


def test_review_takes_all_non_reliable_and_a_seeded_sample():
    labels = ["RELIABLE"] * 10 + ["UNRELIABLE", "MOSTLY_RELIABLE"]
    records = _labelled(labels)

    first = review_positions(records, reliable_sample_size=3, seed=7)
    second = review_positions(records, reliable_sample_size=3, seed=7)

    assert first == second
    assert {11, 12} <= set(first)
    assert len(first) == 5


def test_review_refuses_unlabelled_batch():
    records = _labelled(["RELIABLE"])
    records[0]["manual_label"] = None

    with pytest.raises(ValueError, match="labelled"):
        review_positions(records, 1, 7)


def test_review_confirms_and_changes_labels():
    records = _labelled(["RELIABLE", "UNRELIABLE"])
    rows = review_rows(records, [1, 2])
    rows[0]["decision"] = "change"
    rows[0]["new_label"] = "MOSTLY_RELIABLE"
    rows[0]["new_failure_types"] = "major_dependency_structure"
    rows[1]["decision"] = "confirm"
    rows = [{key: str(value) for key, value in row.items()} for row in rows]

    counts = apply_review_rows(
        records, rows, "owner", "2026-11-05", BINARY_MAPPING, ENUMS
    )

    assert counts == {"confirm": 1, "change": 1}
    assert records[0]["manual_label"] == "MOSTLY_RELIABLE"
    assert records[0]["pre_adjudication_label"] == "RELIABLE"
    assert records[1]["adjudication_status"] == "adjudicated"
    assert records[1]["adjudicator_id"] == "owner"


def test_review_rejects_unknown_decision():
    records = _labelled(["RELIABLE"])
    rows = [
        {key: str(value) for key, value in row.items()}
        for row in review_rows(records, [1])
    ]

    with pytest.raises(ValueError, match="confirm or change"):
        apply_review_rows(
            records, rows, "owner", "2026-11-05", BINARY_MAPPING, ENUMS
        )
