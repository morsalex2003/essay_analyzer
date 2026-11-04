"""Re-run the Reliability ML baseline on the frozen ML v2 development set.

Follows the evaluation procedure fixed in
data/syntax_reliability/development_set_v1.json: essay-context parses,
student-grouped cross-validation, threshold 0.5, and the frozen v1.2
heuristic as the reference. Models and feature schema are unchanged from
ML v1; only the data and the evaluation procedure differ.

Output is aggregate only and contains no learner text or sentence IDs.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = PROJECT_ROOT / "data" / "syntax_reliability"

sys.path.insert(0, str(PROJECT_ROOT))

from src.cell_pilot_evaluation import wilson_interval  # noqa: E402
from src.reliability_development_set import (  # noqa: E402
    LOCK_RELATIVE_PATH,
    load_batches,
    lock_mismatches,
)
from src.syntax.reliability_ml import (  # noqa: E402
    MODEL_A_FEATURE_NAMES,
    MODEL_B_FEATURE_NAMES,
    build_dummy_classifier,
    calculate_binary_metrics,
    feature_name_leakage,
    observation_targets,
    prepare_essay_context_observations,
    run_grouped_oof,
    select_grouped_splits,
)


EXPECTED_TARGET_COUNTS = {0: 504, 1: 102}
WARNING_CATEGORIES = ("medium", "low")


def load_observations():
    lock = json.loads((DATASET_DIRECTORY / LOCK_RELATIVE_PATH).read_text())
    changed_files = lock_mismatches(lock, DATASET_DIRECTORY)
    if changed_files:
        raise SystemExit(f"Frozen development files changed: {changed_files}")

    observations = []
    batch_ids = []
    for batch in load_batches(DATASET_DIRECTORY):
        batch_observations = prepare_essay_context_observations(
            batch["essays"],
            batch["annotations"],
        )
        observations.extend(batch_observations)
        batch_ids.extend([batch["batch_id"]] * len(batch_observations))

    return observations, np.asarray(batch_ids, dtype=str)


def frozen_warning_with_abstentions(observations):
    """Return v1.2 warning predictions, with -1 where the heuristic abstained."""
    predictions = []
    for observation in observations:
        reliability = observation["frozen_reliability"]
        if reliability == "not_analyzable":
            predictions.append(-1)
        else:
            predictions.append(int(reliability in WARNING_CATEGORIES))
    return np.asarray(predictions, dtype=int)


def summarize(targets, predictions, probabilities=None):
    metrics = calculate_binary_metrics(targets, predictions, probabilities)
    true_positive = metrics["true_positive"]
    predicted_positive = true_positive + metrics["false_positive"]
    actual_positive = true_positive + metrics["false_negative"]
    metrics["precision_ci"] = wilson_interval(
        true_positive, predicted_positive
    )
    metrics["recall_ci"] = wilson_interval(true_positive, actual_positive)
    return metrics


def print_row(name, metrics):
    average_precision = metrics["average_precision"]
    average_precision_text = (
        "  -  " if average_precision is None else f"{average_precision:.3f}"
    )
    print(
        f"  {name:<28} "
        f"P {metrics['precision']:.3f} {metrics['precision_ci']}  "
        f"R {metrics['recall']:.3f} {metrics['recall_ci']}  "
        f"F1 {metrics['f1']:.3f}  F2 {metrics['f2']:.3f}  "
        f"BA {metrics['balanced_accuracy']:.3f}  AP {average_precision_text}  "
        f"TP {metrics['true_positive']} FP {metrics['false_positive']} "
        f"FN {metrics['false_negative']} TN {metrics['true_negative']}"
    )


def print_table(title, systems, targets, mask):
    print(title, f"(n={int(mask.sum())}, unreliable={int(targets[mask].sum())})")
    for name, predictions, probabilities in systems:
        selected_probabilities = (
            None if probabilities is None else probabilities[mask]
        )
        print_row(
            name,
            summarize(targets[mask], predictions[mask], selected_probabilities),
        )
    print()


def oof_arrays(result):
    records = result["oof_records"]
    predictions = np.asarray([record["y_pred"] for record in records])
    probabilities = np.asarray(
        [record["unreliable_probability"] for record in records]
    )
    return predictions, probabilities


def main() -> None:
    observations, batch_ids = load_observations()
    targets = observation_targets(observations)

    if dict(Counter(targets.tolist())) != EXPECTED_TARGET_COUNTS:
        raise SystemExit("Unexpected binary target distribution")

    leaking_features = feature_name_leakage(MODEL_B_FEATURE_NAMES)
    if leaking_features:
        raise SystemExit(f"Feature leakage detected: {leaking_features}")

    student_groups = np.asarray(
        [observation["student_id"] for observation in observations],
        dtype=str,
    )
    split_count, splits = select_grouped_splits(targets, student_groups)

    model_a = oof_arrays(
        run_grouped_oof(observations, MODEL_A_FEATURE_NAMES, splits)
    )
    model_b = oof_arrays(
        run_grouped_oof(observations, MODEL_B_FEATURE_NAMES, splits)
    )
    dummy = oof_arrays(
        run_grouped_oof(
            observations,
            MODEL_A_FEATURE_NAMES,
            splits,
            estimator=build_dummy_classifier(),
        )
    )
    frozen = frozen_warning_with_abstentions(observations)

    print(
        f"Observations: {len(observations)} "
        f"(usable {EXPECTED_TARGET_COUNTS[0]}, "
        f"unreliable {EXPECTED_TARGET_COUNTS[1]}), "
        f"students: {len(set(student_groups))}"
    )
    print(f"Student-grouped folds: {split_count}")
    for fold_index, (_, validation_indexes) in enumerate(splits, start=1):
        print(
            f"  fold {fold_index}: "
            f"{len(set(student_groups[validation_indexes]))} students, "
            f"{len(validation_indexes)} sentences, "
            f"{int(targets[validation_indexes].sum())} unreliable"
        )
    print(f"Heuristic abstentions: {int((frozen < 0).sum())}")
    print()

    everything = np.ones(len(targets), dtype=bool)
    print_table(
        "All analyzable sentences",
        (
            ("Model A (structural)", *model_a),
            ("Model B (structural+signals)", *model_b),
            ("Dummy (most frequent)", *dummy),
        ),
        targets,
        everything,
    )

    comparison_systems = (
        ("Model A (structural)", *model_a),
        ("Model B (structural+signals)", *model_b),
        ("Frozen v1.2 warning", frozen, None),
    )
    compared = frozen >= 0
    print_table(
        "Sentences where v1.2 did not abstain",
        comparison_systems,
        targets,
        compared,
    )

    for batch_id in sorted(set(batch_ids)):
        print_table(
            f"Batch {batch_id}, v1.2 not abstained",
            comparison_systems,
            targets,
            compared & (batch_ids == batch_id),
        )


if __name__ == "__main__":
    main()
