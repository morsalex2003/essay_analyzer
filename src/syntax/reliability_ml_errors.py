from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from src.syntax.reliability_features import HEURISTIC_SIGNAL_NAMES
from src.syntax.reliability_ml import ReliabilityObservation


UNRELIABLE_GROUP_NAMES = (
    "caught_by_both",
    "model_b_only",
    "frozen_only",
    "missed_by_both",
)

USABLE_GROUP_NAMES = (
    "accepted_by_both",
    "model_b_false_positive_only",
    "frozen_false_positive_only",
    "false_positive_by_both",
)


def _records_by_sentence_id(
    oof_records: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    records_by_id = {}

    for record in oof_records:
        sentence_id = record["sentence_id"]

        if sentence_id in records_by_id:
            raise ValueError(f"Duplicate OOF sentence ID: {sentence_id}")

        records_by_id[sentence_id] = record

    return records_by_id


def combine_system_predictions(
    observations: Sequence[ReliabilityObservation],
    model_a_oof_records: Sequence[dict[str, object]],
    model_b_oof_records: Sequence[dict[str, object]],
    frozen_predictions: NDArray[np.int64],
) -> list[dict[str, object]]:
    """Combine prediction metadata without retaining learner-written text."""
    if len(frozen_predictions) != len(observations):
        raise ValueError("Frozen prediction count does not match observations")

    model_a_by_id = _records_by_sentence_id(model_a_oof_records)
    model_b_by_id = _records_by_sentence_id(model_b_oof_records)
    comparison_records = []

    for observation_index, observation in enumerate(observations):
        sentence_id = observation["sentence_id"]
        model_a_record = model_a_by_id.get(sentence_id)
        model_b_record = model_b_by_id.get(sentence_id)

        if model_a_record is None or model_b_record is None:
            raise ValueError(f"Missing OOF prediction for {sentence_id}")

        if model_a_record["y_true"] != observation["target"]:
            raise ValueError(f"Model A target mismatch for {sentence_id}")

        if model_b_record["y_true"] != observation["target"]:
            raise ValueError(f"Model B target mismatch for {sentence_id}")

        comparison_records.append(
            {
                "sentence_id": sentence_id,
                "essay_id": observation["essay_id"],
                "student_id": observation["student_id"],
                "manual_label": observation["manual_label"],
                "y_true": observation["target"],
                "model_a_prediction": model_a_record["y_pred"],
                "model_a_probability": model_a_record[
                    "unreliable_probability"
                ],
                "model_b_prediction": model_b_record["y_pred"],
                "model_b_probability": model_b_record[
                    "unreliable_probability"
                ],
                "model_b_fold": model_b_record["fold"],
                "frozen_prediction": int(
                    frozen_predictions[observation_index]
                ),
                "features": observation["features"],
            }
        )

    if len(comparison_records) != len(observations):
        raise ValueError("Prediction comparison lost observations")

    return comparison_records


def partition_system_intersections(
    comparison_records: Sequence[dict[str, object]],
) -> dict[str, dict[str, list[dict[str, object]]]]:
    unreliable_groups = {}
    usable_groups = {}

    for group_name in UNRELIABLE_GROUP_NAMES:
        unreliable_groups[group_name] = []

    for group_name in USABLE_GROUP_NAMES:
        usable_groups[group_name] = []

    for record in comparison_records:
        target = record["y_true"]
        model_prediction = record["model_b_prediction"]
        frozen_prediction = record["frozen_prediction"]

        if target == 1:
            if model_prediction == 1 and frozen_prediction == 1:
                group_name = "caught_by_both"
            elif model_prediction == 1 and frozen_prediction == 0:
                group_name = "model_b_only"
            elif model_prediction == 0 and frozen_prediction == 1:
                group_name = "frozen_only"
            else:
                group_name = "missed_by_both"

            unreliable_groups[group_name].append(record)
        else:
            if model_prediction == 0 and frozen_prediction == 0:
                group_name = "accepted_by_both"
            elif model_prediction == 1 and frozen_prediction == 0:
                group_name = "model_b_false_positive_only"
            elif model_prediction == 0 and frozen_prediction == 1:
                group_name = "frozen_false_positive_only"
            else:
                group_name = "false_positive_by_both"

            usable_groups[group_name].append(record)

    return {
        "unreliable": unreliable_groups,
        "usable": usable_groups,
    }


def probability_summary(
    records: Sequence[dict[str, object]],
) -> dict[str, float] | None:
    if len(records) == 0:
        return None

    probabilities = []

    for record in records:
        probabilities.append(record["model_b_probability"])

    return {
        "minimum": float(np.min(probabilities)),
        "median": float(np.median(probabilities)),
        "maximum": float(np.max(probabilities)),
    }


def error_margin_category(record: dict[str, object]) -> str:
    target = record["y_true"]
    prediction = record["model_b_prediction"]
    probability = record["model_b_probability"]

    if target == prediction:
        raise ValueError("Margin categories are defined only for errors")

    if 0.40 <= probability <= 0.60:
        return "very_close"

    if 0.25 <= probability < 0.40:
        return "moderate"

    if 0.60 < probability <= 0.75:
        return "moderate"

    return "confident_error"


def active_heuristic_flags(record: dict[str, object]) -> list[str]:
    features = record["features"]
    active_flags = []

    for signal_name in HEURISTIC_SIGNAL_NAMES:
        if features[signal_name] == 1:
            active_flags.append(signal_name)

    return active_flags


def active_signal_counts(
    records: Sequence[dict[str, object]],
) -> Counter:
    counts = Counter()

    for record in records:
        for signal_name in active_heuristic_flags(record):
            counts[signal_name] += 1

    return counts


def _contribution_value(item: tuple[str, float]) -> float:
    return item[1]


def fold_linear_contributions(
    record: dict[str, object],
    feature_names: Sequence[str],
    fold_estimators: Sequence[Pipeline],
    limit: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    fold_index = record["model_b_fold"]

    if fold_index < 1 or fold_index > len(fold_estimators):
        raise ValueError("OOF record has no corresponding fold estimator")

    pipeline = fold_estimators[fold_index - 1]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    feature_values = []

    for feature_name in feature_names:
        feature_values.append(record["features"][feature_name])

    feature_matrix = np.asarray([feature_values], dtype=float)
    scaled_values = scaler.transform(feature_matrix)[0]
    coefficients = classifier.coef_[0]
    contribution_pairs = []

    for feature_name, scaled_value, coefficient in zip(
        feature_names,
        scaled_values,
        coefficients,
        strict=True,
    ):
        contribution = float(scaled_value * coefficient)
        contribution_pairs.append((feature_name, contribution))

    toward_unreliable = []
    toward_usable = []

    for contribution_pair in contribution_pairs:
        if contribution_pair[1] > 0:
            toward_unreliable.append(contribution_pair)
        elif contribution_pair[1] < 0:
            toward_usable.append(contribution_pair)

    toward_unreliable.sort(key=_contribution_value, reverse=True)
    toward_usable.sort(key=_contribution_value)

    return {
        "toward_unreliable": toward_unreliable[:limit],
        "toward_usable": toward_usable[:limit],
    }


def recurring_contribution_counts(
    records: Sequence[dict[str, object]],
    feature_names: Sequence[str],
    fold_estimators: Sequence[Pipeline],
    direction: str,
    limit_per_record: int = 5,
) -> Counter:
    if direction not in ("toward_unreliable", "toward_usable"):
        raise ValueError(f"Unknown contribution direction: {direction}")

    counts = Counter()

    for record in records:
        contributions = fold_linear_contributions(
            record,
            feature_names,
            fold_estimators,
            limit=limit_per_record,
        )

        for feature_name, contribution in contributions[direction]:
            counts[feature_name] += 1

    return counts


def failure_family_counts(
    records: Sequence[dict[str, object]],
    annotations_by_id: dict[str, dict[str, object]],
) -> Counter:
    counts = Counter()

    for record in records:
        annotation = annotations_by_id[record["sentence_id"]]

        for failure_type in annotation["manual_failure_types"]:
            counts[failure_type] += 1

    return counts


def failure_combination_counts(
    records: Sequence[dict[str, object]],
    annotations_by_id: dict[str, dict[str, object]],
) -> Counter:
    counts = Counter()

    for record in records:
        annotation = annotations_by_id[record["sentence_id"]]
        failure_types = tuple(sorted(annotation["manual_failure_types"]))

        if len(failure_types) > 1:
            counts[failure_types] += 1

    return counts
