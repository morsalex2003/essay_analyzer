import json

import pytest

import scripts.evaluate_cell_holdout as holdout_script
from src.cell_holdout_evaluation import (
    evaluate_holdout,
    exact_mcnemar_p_value,
    candidate_decision,
    paired_comparison,
)


def test_exact_mcnemar_values():
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(6, 0) == pytest.approx(2 / 64)
    assert exact_mcnemar_p_value(5, 0) == pytest.approx(2 / 32)
    assert exact_mcnemar_p_value(3, 3) == 1.0


def _comparison(gained, lost, added, removed):
    targets = [1] * (gained + lost) + [0] * (added + removed)
    candidate = [1] * gained + [0] * lost + [1] * added + [0] * removed
    reference = [0] * gained + [1] * lost + [0] * added + [1] * removed
    return paired_comparison(targets, candidate, reference)


def test_decision_rule():
    def decide(comparison):
        return candidate_decision(comparison, "adopt_v1_3")

    assert decide(_comparison(6, 0, 0, 0)) == "adopt_v1_3"
    assert decide(_comparison(3, 0, 0, 0)) == "inconclusive"
    assert decide(_comparison(6, 0, 6, 0)) == "negative"
    assert decide(_comparison(0, 2, 0, 0)) == "negative"


def _record(index, label, status="single", failures=()):
    return {
        "sentence_id": f"s{index}",
        "manual_label": label,
        "adjudication_status": status,
        "manual_failure_types": list(failures),
    }


def test_evaluate_holdout_uses_reference_analysis_set():
    records = [
        _record(1, "UNRELIABLE", "adjudicated", ["finite_inventory"]),
        _record(2, "UNRELIABLE", failures=["dependent_inventory"]),
        _record(3, "RELIABLE"),
        _record(4, "MOSTLY_RELIABLE", "adjudicated"),
        _record(5, "NOT_ANALYZABLE"),
        _record(6, "RELIABLE"),
    ]
    verdicts = {
        "1.2": {"s1": "low", "s2": "high", "s3": "high", "s4": "medium",
                "s5": "not_analyzable", "s6": "not_analyzable"},
        "1.3": {"s1": "low", "s2": "medium", "s3": "high", "s4": "medium",
                "s5": "not_analyzable", "s6": "not_analyzable"},
    }
    model_b = {"s1": 1, "s2": 0, "s3": 1, "s4": 0, "s6": 0}

    report = evaluate_holdout(records, verdicts, model_b)

    assert report["counts"] == {
        "sentences": 6,
        "not_analyzable_labels": 1,
        "reference_abstentions": 1,
        "analysis_set": 4,
        "unreliable": 2,
        "usable": 2,
    }
    assert report["metrics"]["v1.2"]["recall"] == 0.5
    assert report["metrics"]["v1.3"]["recall"] == 1.0
    assert report["paired_comparisons"]["v1.3_vs_v1.2"]["unreliable"][
        "candidate_only_warns"
    ] == 1
    assert report["recall_by_failure_type"]["v1.3"]["dependent_inventory"] == {
        "unreliable": 1,
        "caught": 1,
    }
    assert report["v1_3_decision"] == "inconclusive"
    assert "text" not in json.dumps(report)


def test_v2_compares_v14_with_v13_and_every_system_with_v12():
    records = [
        _record(1, "UNRELIABLE", failures=["sentence_segmentation"]),
        _record(2, "UNRELIABLE", failures=["finite_inventory"]),
        _record(3, "RELIABLE"),
    ]
    verdicts = {
        "1.2": {"s1": "high", "s2": "high", "s3": "high"},
        "1.3": {"s1": "high", "s2": "medium", "s3": "high"},
        "1.4": {"s1": "medium", "s2": "medium", "s3": "high"},
    }
    model_b = {"s1": 0, "s2": 1, "s3": 0}

    report = evaluate_holdout(
        records,
        verdicts,
        model_b,
        candidates=("1.3", "1.4"),
        decision_pair=("1.4", "1.3"),
        positive_outcome="confirm_v1_4",
    )

    assert list(report["paired_comparisons"]) == [
        "v1.3_vs_v1.2",
        "v1.4_vs_v1.2",
        "model_b_vs_v1.2",
        "v1.4_vs_v1.3",
    ]
    assert report["paired_comparisons"]["v1.4_vs_v1.3"]["unreliable"] == {
        "candidate_only_warns": 1,
        "reference_only_warns": 0,
        "exact_mcnemar_p": 1.0,
    }
    assert report["metrics"]["v1.4"]["recall"] == 1.0
    assert report["v1_4_decision"] == "inconclusive"
    assert "v1_3_decision" not in report


def test_v13_abstaining_alone_is_an_error():
    records = [_record(1, "UNRELIABLE"), _record(2, "RELIABLE")]
    verdicts = {
        "1.2": {"s1": "low", "s2": "high"},
        "1.3": {"s1": "not_analyzable", "s2": "high"},
    }

    with pytest.raises(ValueError, match="abstained"):
        evaluate_holdout(records, verdicts, {"s1": 1, "s2": 0})


@pytest.fixture
def batch(tmp_path, monkeypatch):
    """A labelled, reviewed holdout v2 batch in a temporary CELL folder."""
    directory = tmp_path / "local/holdout_v2"
    template = [
        {"sentence_id": "s1", "manual_label": None,
         "adjudication_status": "single"},
        {"sentence_id": "s2", "manual_label": None,
         "adjudication_status": "single"},
    ]
    ledger = [
        dict(template[0], manual_label="UNRELIABLE",
             adjudication_status="adjudicated"),
        dict(template[1], manual_label="RELIABLE"),
    ]
    (directory / "packet").mkdir(parents=True)
    (directory / "annotations_template.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in template)
    )
    (directory / "annotations_v1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ledger)
    )
    (directory / "packet/review.tsv").write_text(
        "position\tsentence_id\n1\ts1\n"
    )
    split = {
        "holdout_opened_at": "2026-11-06",
        "holdout_open_reason": "holdout v1",
        "holdout_openings": [
            {
                "batch_id": "cell_reliability_holdout_v1",
                "opened_at": "2026-11-06",
                "reason": "holdout v1",
            }
        ],
    }
    (tmp_path / "split.json").write_text(json.dumps(split))

    monkeypatch.setattr(holdout_script, "CELL_DIRECTORY", tmp_path)
    monkeypatch.setattr(holdout_script, "SPLIT_PATH", tmp_path / "split.json")
    return directory


def test_opening_requires_label_lock(batch):
    with pytest.raises(SystemExit, match="Lock"):
        holdout_script.record_opening("v2", "final evaluation")


def test_lock_then_open_once(batch):
    holdout_script.lock_labels("v2")
    holdout_script.record_opening("v2", "final evaluation")

    lock = json.loads((batch / "labels_lock.json").read_text())
    split = json.loads((batch.parents[1] / "split.json").read_text())
    assert lock["batch_id"] == "cell_reliability_holdout_v2"
    assert lock["records"] == 2 and lock["reviewed"] == 1
    assert [opening["batch_id"] for opening in split["holdout_openings"]] == [
        "cell_reliability_holdout_v1",
        "cell_reliability_holdout_v2",
    ]
    assert split["holdout_openings"][1]["reason"] == "final evaluation"
    # The first opening of the holdout as a whole is kept.
    assert split["holdout_open_reason"] == "holdout v1"
    with pytest.raises(SystemExit, match="already recorded"):
        holdout_script.record_opening("v2", "again")
    with pytest.raises(FileExistsError):
        holdout_script.lock_labels("v2")


def test_lock_refuses_incomplete_review(batch):
    (batch / "packet/review.tsv").write_text(
        "position\tsentence_id\n2\ts2\n"
    )

    with pytest.raises(SystemExit, match="incomplete"):
        holdout_script.lock_labels("v2")


def test_opening_record_is_found_per_batch():
    split = {
        "holdout_openings": [
            {"batch_id": "cell_reliability_holdout_v1", "opened_at": "d"}
        ]
    }

    assert holdout_script.opening_record(
        split, "cell_reliability_holdout_v1"
    ) == {"batch_id": "cell_reliability_holdout_v1", "opened_at": "d"}
    assert holdout_script.opening_record(
        split, "cell_reliability_holdout_v2"
    ) is None
    assert holdout_script.opening_record({}, "any") is None
