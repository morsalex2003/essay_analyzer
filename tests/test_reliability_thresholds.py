import numpy as np
import pytest

from src.syntax import reliability_thresholds
from src.syntax.reliability_features import ALL_FEATURE_NAMES
from src.syntax.reliability_thresholds import (
    NO_WARNING_THRESHOLD,
    heuristic_warning_precision,
    nested_threshold_predictions,
    select_fold_threshold,
    select_max_f1_threshold,
    select_precision_floor_threshold,
)


TARGETS = np.asarray([1, 1, 0, 1, 0, 0])
PROBABILITIES = np.asarray([0.9, 0.8, 0.7, 0.6, 0.3, 0.1])


def test_max_f1_threshold_picks_best_cut():
    # Cut at 0.6: three true positives, one false positive, F1 = 6/7.
    assert select_max_f1_threshold(TARGETS, PROBABILITIES) == 0.6


def test_max_f1_ties_prefer_fewer_warnings():
    targets = np.asarray([1, 0, 0, 1])
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.6])

    # 0.9 (P 1, R 1/2) and 0.6 (P 1/2, R 1) both give F1 2/3; the higher
    # threshold warns less.
    assert select_max_f1_threshold(targets, probabilities) == 0.9


def test_precision_floor_takes_lowest_threshold_reaching_floor():
    assert select_precision_floor_threshold(TARGETS, PROBABILITIES, 1.0) == 0.8
    assert select_precision_floor_threshold(TARGETS, PROBABILITIES, 0.75) == (
        0.6
    )


def test_unreachable_precision_floor_gives_no_warnings():
    targets = np.asarray([0, 1])
    probabilities = np.asarray([0.9, 0.1])

    threshold = select_precision_floor_threshold(targets, probabilities, 0.9)

    assert threshold == NO_WARNING_THRESHOLD
    assert not np.any(probabilities >= threshold)


def test_heuristic_precision_ignores_abstentions_and_high_verdicts():
    observations = [
        {"target": 1, "frozen_reliability": "low"},
        {"target": 0, "frozen_reliability": "medium"},
        {"target": 1, "frozen_reliability": "medium"},
        {"target": 1, "frozen_reliability": "not_analyzable"},
        {"target": 0, "frozen_reliability": "high"},
    ]

    assert heuristic_warning_precision(observations) == pytest.approx(2 / 3)


def test_unknown_rule_is_rejected(monkeypatch):
    monkeypatch.setattr(
        reliability_thresholds,
        "inner_oof_probabilities",
        lambda observations, feature_names: PROBABILITIES,
    )
    observations = [{"target": int(target)} for target in TARGETS]

    with pytest.raises(ValueError, match="Unknown threshold rule"):
        select_fold_threshold(observations, [], "best_on_validation")


def _observations(count: int) -> list[dict]:
    random = np.random.default_rng(7)
    observations = []
    for index in range(count):
        target = int(index % 3 == 0)
        observations.append(
            {
                "sentence_id": f"sentence_{index}",
                "essay_id": f"essay_{index // 2}",
                "student_id": f"student_{index % 12}",
                "manual_label": "UNRELIABLE" if target else "RELIABLE",
                "target": target,
                "features": {
                    name: float(random.normal() + 2 * target)
                    for name in ALL_FEATURE_NAMES
                },
                "frozen_reliability": "low" if target else "high",
            }
        )
    return observations


def test_thresholds_never_see_outer_validation_labels(monkeypatch):
    observations = _observations(72)
    seen_training_sizes = []
    original = reliability_thresholds.select_fold_threshold

    def recording_select(training_observations, feature_names, rule):
        seen_training_sizes.append(len(training_observations))
        return original(training_observations, feature_names, rule)

    monkeypatch.setattr(
        reliability_thresholds, "select_fold_threshold", recording_select
    )
    validation_indexes = np.arange(0, 24)
    training_indexes = np.arange(24, 72)
    splits = [
        (training_indexes, validation_indexes),
        (validation_indexes, training_indexes),
    ]

    predictions, probabilities, thresholds = nested_threshold_predictions(
        observations, ALL_FEATURE_NAMES, splits, "max_f1"
    )

    assert seen_training_sizes == [48, 24]
    assert len(thresholds) == 2
    assert predictions.shape == probabilities.shape == (72,)
    assert set(predictions.tolist()) <= {0, 1}
