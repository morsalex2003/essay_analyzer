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
    MODEL_C_FEATURE_NAMES,
    build_dummy_classifier,
    calculate_binary_metrics,
    feature_name_leakage,
    observation_targets,
    prepare_essay_context_observations,
    run_grouped_oof,
    select_grouped_splits,
)
from src.syntax.reliability_thresholds import (  # noqa: E402
    THRESHOLD_RULES,
    WARNING_CATEGORIES,
    nested_threshold_predictions,
)


EXPECTED_TARGET_COUNTS = {0: 504, 1: 102}


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
            include_error_pattern_features=True,
        )
        observations.extend(batch_observations)
        batch_ids.extend([batch["batch_id"]] * len(batch_observations))

    return observations, np.asarray(batch_ids, dtype=str)


def frozen_warning_with_abstentions(observations):
    """Return v1.2 warning predictions, -1 where the heuristic abstained."""
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
    unreliable_count = int(targets[mask].sum())
    print(title, f"(n={int(mask.sum())}, unreliable={unreliable_count})")
    for name, predictions, probabilities in systems:
        selected_probabilities = (
            None if probabilities is None else probabilities[mask]
        )
        print_row(
            name,
            summarize(
                targets[mask], predictions[mask], selected_probabilities
            ),
        )
    print()


def print_overlap_with_heuristic(name, predictions, frozen, targets, mask):
    """Count which unreliable sentences each system catches."""
    unreliable = mask & (targets == 1)
    model_warns = predictions == 1
    heuristic_warns = frozen == 1
    both = int(np.sum(unreliable & model_warns & heuristic_warns))
    model_only = int(np.sum(unreliable & model_warns & ~heuristic_warns))
    heuristic_only = int(np.sum(unreliable & ~model_warns & heuristic_warns))
    neither = int(np.sum(unreliable & ~model_warns & ~heuristic_warns))
    print(
        f"  {name:<28} both {both}, model only {model_only}, "
        f"v1.2 only {heuristic_only}, neither {neither}"
    )


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

    leaking_features = feature_name_leakage(MODEL_C_FEATURE_NAMES)
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
    model_c = oof_arrays(
        run_grouped_oof(observations, MODEL_C_FEATURE_NAMES, splits)
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
            ("Model C (B+error patterns)", *model_c),
            ("Dummy (most frequent)", *dummy),
        ),
        targets,
        everything,
    )

    comparison_systems = (
        ("Model A (structural)", *model_a),
        ("Model B (structural+signals)", *model_b),
        ("Model C (B+error patterns)", *model_c),
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

    print("Thresholds chosen inside training folds (rules fixed in advance)")
    threshold_systems = []
    for model_name, feature_names in (
        ("Model A", MODEL_A_FEATURE_NAMES),
        ("Model B", MODEL_B_FEATURE_NAMES),
        ("Model C", MODEL_C_FEATURE_NAMES),
    ):
        for rule in THRESHOLD_RULES:
            predictions, probabilities, thresholds = (
                nested_threshold_predictions(
                    observations, feature_names, splits, rule
                )
            )
            rounded = [round(threshold, 3) for threshold in thresholds]
            print(f"  {model_name} {rule}: fold thresholds {rounded}")
            threshold_systems.append(
                (f"{model_name} {rule}", predictions, probabilities)
            )
    print()

    nested_systems = (
        *threshold_systems,
        ("Frozen v1.2 warning", frozen, None),
    )
    print_table(
        "Nested thresholds, v1.2 not abstained",
        nested_systems,
        targets,
        compared,
    )
    for batch_id in sorted(set(batch_ids)):
        print_table(
            f"Nested thresholds, batch {batch_id}, v1.2 not abstained",
            nested_systems,
            targets,
            compared & (batch_ids == batch_id),
        )

    print("Unreliable sentences caught, compared with v1.2 (not abstained)")
    for name, predictions, _ in threshold_systems:
        print_overlap_with_heuristic(
            name, predictions, frozen, targets, compared
        )


if __name__ == "__main__":
    main()
