"""Choose Reliability ML decision thresholds inside training folds only.

For each outer fold, the threshold is chosen from inner out-of-fold
probabilities computed on that fold's training students, then applied
unchanged to the outer validation students. The outer validation labels
never influence the threshold.

Two rules were fixed before the first run:

- ``max_f1``: the threshold with the highest inner F1;
- ``heuristic_precision``: the lowest threshold whose inner precision is at
  least the frozen v1.2 warning precision on the same training sentences,
  i.e. the most recall available without being less precise than v1.2.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.syntax.reliability_ml import (
    POSITIVE_CLASS,
    ReliabilityObservation,
    observation_targets,
    run_grouped_oof,
    select_grouped_splits,
)


THRESHOLD_RULES = ("max_f1", "heuristic_precision")
WARNING_CATEGORIES = ("medium", "low")

# Returned when no threshold reaches a precision floor: no warnings at all.
NO_WARNING_THRESHOLD = float("inf")


def _precision_recall_at(
    targets: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    threshold: float,
) -> tuple[float, float]:
    predicted = probabilities >= threshold
    true_positive = int(np.sum(predicted & (targets == POSITIVE_CLASS)))
    predicted_positive = int(np.sum(predicted))
    actual_positive = int(np.sum(targets == POSITIVE_CLASS))
    precision = (
        true_positive / predicted_positive if predicted_positive else 0.0
    )
    recall = true_positive / actual_positive if actual_positive else 0.0
    return precision, recall


def select_max_f1_threshold(
    targets: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    """Return the candidate threshold with the highest F1.

    Candidates are the observed probabilities. Ties go to the higher
    threshold, which gives fewer warnings for the same F1.
    """

    best_threshold = NO_WARNING_THRESHOLD
    best_f1 = -1.0

    for threshold in sorted(set(probabilities.tolist()), reverse=True):
        precision, recall = _precision_recall_at(
            targets, probabilities, threshold
        )
        total = precision + recall
        f1 = 2 * precision * recall / total if total else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1

    return best_threshold


def select_precision_floor_threshold(
    targets: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    minimum_precision: float,
) -> float:
    """Return the lowest threshold whose precision reaches the floor."""

    selected = NO_WARNING_THRESHOLD

    for threshold in sorted(set(probabilities.tolist()), reverse=True):
        precision, _ = _precision_recall_at(targets, probabilities, threshold)
        if precision >= minimum_precision:
            selected = threshold

    return selected


def heuristic_warning_precision(
    observations: Sequence[ReliabilityObservation],
) -> float:
    """Precision of frozen v1.2 warnings, ignoring its abstentions."""

    true_positive = 0
    predicted_positive = 0

    for observation in observations:
        if observation["frozen_reliability"] in WARNING_CATEGORIES:
            predicted_positive += 1
            true_positive += observation["target"] == POSITIVE_CLASS

    return true_positive / predicted_positive if predicted_positive else 0.0


def inner_oof_probabilities(
    observations: Sequence[ReliabilityObservation],
    feature_names: Sequence[str],
    group_field: str = "student_id",
) -> NDArray[np.float64]:
    """Out-of-fold probabilities inside one training set, grouped."""

    targets = observation_targets(observations)
    groups = np.asarray(
        [observation[group_field] for observation in observations],
        dtype=str,
    )
    _, splits = select_grouped_splits(targets, groups)
    result = run_grouped_oof(observations, feature_names, splits)
    return np.asarray(
        [record["unreliable_probability"] for record in result["oof_records"]]
    )


def select_fold_threshold(
    training_observations: Sequence[ReliabilityObservation],
    feature_names: Sequence[str],
    rule: str,
) -> float:
    """Choose one outer fold's threshold from its training students only."""

    targets = observation_targets(training_observations)
    probabilities = inner_oof_probabilities(
        training_observations, feature_names
    )

    if rule == "max_f1":
        return select_max_f1_threshold(targets, probabilities)

    if rule == "heuristic_precision":
        return select_precision_floor_threshold(
            targets,
            probabilities,
            heuristic_warning_precision(training_observations),
        )

    raise ValueError(f"Unknown threshold rule: {rule}")


def nested_threshold_predictions(
    observations: Sequence[ReliabilityObservation],
    feature_names: Sequence[str],
    outer_splits: Sequence[tuple[NDArray[np.int64], NDArray[np.int64]]],
    rule: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], list[float]]:
    """Return outer predictions, outer probabilities, and fold thresholds."""

    outer = run_grouped_oof(observations, feature_names, outer_splits)
    probabilities = np.asarray(
        [record["unreliable_probability"] for record in outer["oof_records"]]
    )
    predictions = np.full(len(observations), -1, dtype=int)
    thresholds = []

    for train_indexes, validation_indexes in outer_splits:
        training_observations = [observations[i] for i in train_indexes]
        threshold = select_fold_threshold(
            training_observations, feature_names, rule
        )
        thresholds.append(threshold)
        predictions[validation_indexes] = (
            probabilities[validation_indexes] >= threshold
        ).astype(int)

    if np.any(predictions < 0):
        raise ValueError("Some observations have no outer prediction")

    return predictions, probabilities, thresholds
