import json

import pytest

from src.analysis import analyze_writing
from src.cell_annotation import SentenceParser
from src.cell_pilot_evaluation import (
    evaluate_pilot,
    heuristic_verdicts,
    wilson_interval,
)
from src.nlp.pipeline import nlp


def _record(position, label, failure_types=None, structures=None):
    return {
        "sentence_id": "essay_1:" + str(position),
        "essay_id": "essay_1",
        "student_id": "student_1",
        "sentence_index": position,
        "original_text": "Synthetic sentence.",
        "manual_label": label,
        "manual_failure_types": failure_types or [],
        "selection_structures": structures or [],
        "annotation_origin": "human_manual_review",
    }


def _ledger_and_verdicts():
    # (manual label, heuristic verdict) pairs for synthetic sentences.
    pairs = [
        ("UNRELIABLE", "low"),
        ("UNRELIABLE", "medium"),
        ("UNRELIABLE", "high"),
        ("RELIABLE", "high"),
        ("RELIABLE", "high"),
        ("MOSTLY_RELIABLE", "medium"),
        ("NOT_ANALYZABLE", "not_analyzable"),
        ("RELIABLE", "not_analyzable"),
        (None, "high"),
    ]
    records = []
    verdicts = {}

    for position, (label, verdict) in enumerate(pairs, start=1):
        failure_types = None
        if label == "UNRELIABLE":
            failure_types = ["matrix_hierarchy"]
        if position == 2:
            failure_types = ["finite_inventory", "matrix_hierarchy"]
        record = _record(
            position,
            label,
            failure_types,
            ["coordination"] if position <= 2 else ["relative_clause"],
        )
        records.append(record)
        verdicts[record["sentence_id"]] = verdict

    return records, verdicts


def test_warning_operating_point_matches_hand_counts():
    records, verdicts = _ledger_and_verdicts()
    report = evaluate_pilot(records, verdicts)
    warning = report["binary_task"]["operating_points"]["warning"]

    # Scored: 3 UNRELIABLE and 3 usable; NA label and heuristic abstention
    # on an analyzable label are excluded.
    assert report["binary_task"]["scored_sentences"] == 6
    assert report["binary_task"]["heuristic_abstained_on_analyzable"] == 1
    assert warning["true_positive"] == 2
    assert warning["false_negative"] == 1
    assert warning["false_positive"] == 1
    assert warning["true_negative"] == 2
    assert warning["recall"] == pytest.approx(0.667, abs=0.001)


def test_low_only_operating_point_is_stricter():
    records, verdicts = _ledger_and_verdicts()
    low_only = evaluate_pilot(records, verdicts)["binary_task"][
        "operating_points"
    ]["low_only"]

    assert low_only["true_positive"] == 1
    assert low_only["false_positive"] == 0


def test_report_marks_partial_coverage_and_error_positions():
    records, verdicts = _ledger_and_verdicts()
    report = evaluate_pilot(records, verdicts)

    assert report["status"] == "partial"
    assert report["coverage"]["labelled_sentences"] == 8
    assert report["coverage"]["pending_sentences"] == 1
    assert report["warning_errors_by_position"] == {
        "missed_unreliable": [3],
        "false_warning": [6],
    }
    assert report["label_by_heuristic"]["NOT_ANALYZABLE"][
        "not_analyzable"
    ] == 1


def test_recall_breakdowns_count_multi_valued_groups():
    records, verdicts = _ledger_and_verdicts()
    report = evaluate_pilot(records, verdicts)

    assert report["warning_recall_by_failure_type"]["matrix_hierarchy"] == {
        "unreliable": 3,
        "caught": 2,
        "recall": 0.667,
    }
    assert report["warning_recall_by_failure_type"]["finite_inventory"][
        "caught"
    ] == 1
    assert report["warning_recall_by_selection_structure"]["coordination"][
        "recall"
    ] == 1.0


def test_report_without_labels_has_no_metrics():
    records = [_record(1, None)]
    report = evaluate_pilot(records, {})

    assert report["status"] == "no_labels"
    assert report["binary_task"]["operating_points"] == {}


def test_report_contains_no_learner_text():
    records, verdicts = _ledger_and_verdicts()
    report_text = json.dumps(evaluate_pilot(records, verdicts))

    assert "Synthetic sentence." not in report_text


def test_wilson_interval_bounds():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(8, 10)
    assert 0.0 < low < 0.8 < high < 1.0


def test_heuristic_verdicts_match_analyze_writing_in_essay_context():
    # Synthetic essay; verdicts must equal the production per-sentence ones.
    essay_text = (
        "Students learn when teachers explain. "
        "For example, to help with the difficult question on the class "
        "or do the science article."
    )
    parser = SentenceParser(nlp, {"essay_1": {"original_text": essay_text}})
    records = []
    for index, sentence in enumerate(parser.sentences("essay_1"), start=1):
        record = _record(index, None)
        record["original_text"] = sentence.text
        records.append(record)

    verdicts = heuristic_verdicts(records, parser)
    production = analyze_writing(essay_text)["syntax_reliability"]["sentences"]

    for record, sentence in zip(records, production):
        assert verdicts[record["sentence_id"]] == sentence["reliability"]
