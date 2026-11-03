import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.cell_annotation import (
    SentenceParser,
    create_annotated_record,
    first_pending_position,
    format_sentence_view,
    label_counts,
    load_annotation_enums,
    load_jsonl,
    merge_progress,
    run_annotation_session,
    write_jsonl_atomically,
)
from src.nlp.pipeline import nlp


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENUMS = load_annotation_enums(
    json.loads(
        (REPOSITORY_ROOT / "data/syntax_reliability/schema_v1.json")
        .read_text(encoding="utf-8")
    )
)
BINARY_MAPPING = {
    "RELIABLE": True,
    "MOSTLY_RELIABLE": True,
    "UNRELIABLE": False,
    "NOT_ANALYZABLE": None,
}

# Synthetic text written for these tests, not learner data.
ESSAY_TEXT = (
    "Students learn when teachers explain. "
    "I think that homework helps. "
    "Because the bus."
)
SENTENCES = (
    "Students learn when teachers explain.",
    "I think that homework helps.",
    "Because the bus.",
)


def _template_records():
    records = []

    for sentence_index, sentence_text in enumerate(SENTENCES, start=1):
        records.append(
            {
                "sentence_id": "essay_1:" + str(sentence_index),
                "essay_id": "essay_1",
                "sentence_index": sentence_index,
                "original_text": sentence_text,
                "original_text_sha256": sha256(
                    sentence_text.encode("utf-8")
                ).hexdigest(),
                "manual_label": None,
                "usable_binary_label": None,
                "abstention_status": None,
                "manual_failure_types": [],
                "annotation_notes": None,
                "annotator_id": None,
                "adjudication_status": "single",
                "annotation_origin": "model_assisted_manual_review",
            }
        )

    return records


def _parser():
    return SentenceParser(
        nlp,
        {"essay_1": {"original_text": ESSAY_TEXT}},
    )


def _annotate(record, label, failure_types=None, abstention="none"):
    return create_annotated_record(
        record,
        label,
        failure_types or [],
        abstention,
        None,
        "annotator_01",
        BINARY_MAPPING,
        ENUMS,
    )


def _scripted_input(answers):
    remaining_answers = list(answers)

    def input_function(prompt):
        return remaining_answers.pop(0)

    return input_function


def test_annotated_record_sets_binary_label_and_keeps_template():
    template = _template_records()[1]
    record = _annotate(template, "UNRELIABLE", ["matrix_hierarchy"])

    assert record["usable_binary_label"] is False
    assert record["manual_failure_types"] == ["matrix_hierarchy"]
    assert record["abstention_status"] == "none"
    assert record["annotator_id"] == "annotator_01"
    assert record["annotation_origin"] == "human_manual_review"
    assert template["manual_label"] is None
    assert template["annotation_origin"] == "model_assisted_manual_review"


def test_not_analyzable_needs_reason_and_records_failure_type():
    template = _template_records()[2]

    with pytest.raises(ValueError, match="abstention reason"):
        _annotate(template, "NOT_ANALYZABLE")

    record = _annotate(template, "NOT_ANALYZABLE", abstention="source_fragment")

    assert record["usable_binary_label"] is None
    assert record["manual_failure_types"] == ["source_fragment"]


def test_analyzable_label_rejects_abstention_and_unknown_failure_type():
    template = _template_records()[0]

    with pytest.raises(ValueError, match="abstention_status none"):
        _annotate(template, "RELIABLE", abstention="other")

    with pytest.raises(ValueError, match="Unsupported failure type"):
        _annotate(template, "UNRELIABLE", ["made_up"])


def test_merge_progress_rejects_changed_frozen_fields():
    template = _template_records()
    progress = [dict(record) for record in template]
    progress[0] = _annotate(progress[0], "RELIABLE")

    assert merge_progress(template, progress)[0]["manual_label"] == (
        "RELIABLE"
    )

    progress[1]["original_text"] = "Changed text."
    with pytest.raises(ValueError, match="frozen field changed"):
        merge_progress(template, progress)


def test_atomic_write_round_trips(tmp_path):
    path = tmp_path / "annotations_v1.jsonl"
    records = _template_records()

    write_jsonl_atomically(path, records)

    assert load_jsonl(path) == records
    assert not (tmp_path / "annotations_v1.jsonl.tmp").exists()


def test_sentence_view_shows_context_parse_and_clauses():
    parser = _parser()
    record = _template_records()[1]
    view = format_sentence_view(
        record,
        parser.sentence(record),
        parser.sentences("essay_1"),
    )

    assert "before: Students learn when teachers explain." in view
    assert ">> I think that homework helps." in view
    assert "after:  Because the bus." in view
    assert "ROOT" in view
    assert "finite clause heads:" in view
    assert "reliability" not in view.lower()


def test_session_saves_after_each_label_and_resumes_at_first_pending():
    records = _template_records()
    saved_snapshots = []
    answers = [
        "r", "",                     # sentence 1: RELIABLE, no note
        "u", "3,1", "parse note",    # sentence 2: UNRELIABLE with types
        "q",                         # stop before sentence 3
    ]

    saved_count = run_annotation_session(
        records,
        _parser(),
        "annotator_01",
        BINARY_MAPPING,
        ENUMS,
        lambda current: saved_snapshots.append(label_counts(current)),
        input_function=_scripted_input(answers),
        print_function=lambda *parts: None,
    )

    assert saved_count == 2
    assert len(saved_snapshots) == 2
    assert records[1]["manual_failure_types"] == [
        ENUMS["manual_failure_type"][2],
        ENUMS["manual_failure_type"][0],
    ]
    assert records[1]["annotation_notes"] == "parse note"
    assert first_pending_position(records) == 2


def test_session_skip_leaves_sentence_pending_and_back_relabels():
    records = _template_records()
    answers = [
        "s",            # skip sentence 1
        "r", "",        # sentence 2: RELIABLE
        "b",            # back to sentence 2
        "m", "", "",    # relabel sentence 2 as MOSTLY_RELIABLE
        "q",
    ]

    run_annotation_session(
        records,
        _parser(),
        "annotator_01",
        BINARY_MAPPING,
        ENUMS,
        lambda current: None,
        input_function=_scripted_input(answers),
        print_function=lambda *parts: None,
    )

    assert records[0]["manual_label"] is None
    assert records[1]["manual_label"] == "MOSTLY_RELIABLE"


def test_schema_without_human_origin_is_rejected():
    enums_without_human_origin = dict(ENUMS)
    enums_without_human_origin["annotation_origin"] = [
        "model_assisted_manual_review"
    ]

    with pytest.raises(ValueError, match="human_manual_review"):
        create_annotated_record(
            _template_records()[0],
            "RELIABLE",
            [],
            "none",
            None,
            "annotator_01",
            BINARY_MAPPING,
            enums_without_human_origin,
        )


def test_merge_progress_accepts_human_origin_on_labelled_records():
    template = _template_records()
    progress = [dict(record) for record in template]
    progress[0] = _annotate(progress[0], "RELIABLE")

    merged = merge_progress(template, progress)

    assert merged[0]["annotation_origin"] == "human_manual_review"
    assert merged[1]["annotation_origin"] == "model_assisted_manual_review"


def test_model_assisted_origin_is_recorded_when_requested():
    record = create_annotated_record(
        _template_records()[0],
        "RELIABLE",
        [],
        "none",
        None,
        "owner_model_assisted_v1",
        BINARY_MAPPING,
        ENUMS,
        annotation_origin="model_assisted_manual_review",
    )

    assert record["annotation_origin"] == "model_assisted_manual_review"


def test_unknown_annotation_origin_is_rejected():
    with pytest.raises(ValueError, match="does not allow"):
        create_annotated_record(
            _template_records()[0],
            "RELIABLE",
            [],
            "none",
            None,
            "annotator_01",
            BINARY_MAPPING,
            ENUMS,
            annotation_origin="guessed",
        )


def test_session_uses_one_origin_for_every_label():
    records = _template_records()

    run_annotation_session(
        records,
        _parser(),
        "owner_model_assisted_v1",
        BINARY_MAPPING,
        ENUMS,
        lambda current: None,
        input_function=_scripted_input(["r", "", "m", "", "", "q"]),
        print_function=lambda *parts: None,
        annotation_origin="model_assisted_manual_review",
    )

    assert records[0]["annotation_origin"] == "model_assisted_manual_review"
    assert records[1]["annotation_origin"] == "model_assisted_manual_review"
    assert records[2]["annotation_origin"] == "model_assisted_manual_review"
    assert records[2]["manual_label"] is None
