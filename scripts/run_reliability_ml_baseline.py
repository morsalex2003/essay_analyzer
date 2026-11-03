import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.syntax.reliability_ml import (
    MODEL_A_FEATURE_NAMES,
    MODEL_B_FEATURE_NAMES,
    build_dummy_classifier,
    calculate_binary_metrics,
    descriptive_coefficients,
    evaluate_fixed_transfer,
    feature_name_leakage,
    frozen_warning_predictions,
    observation_targets,
    prepare_ml_observations,
    run_grouped_oof,
    select_grouped_splits,
)


DATASET_DIRECTORY = PROJECT_ROOT / "data" / "syntax_reliability"
ESSAYS_PATH = DATASET_DIRECTORY / "essays_v1.jsonl"
ANNOTATIONS_PATH = DATASET_DIRECTORY / "annotations_v1.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []

    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            records.append(json.loads(line))

    return records


def _print_metrics(name: str, metrics: dict[str, object]) -> None:
    print(name)
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  F2: {metrics['f2']:.4f}")
    print(f"  balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")

    if metrics["average_precision"] is not None:
        print(f"  PR-AUC/AP: {metrics['average_precision']:.4f}")

    print(
        "  confusion: "
        f"TN={metrics['true_negative']}, "
        f"FP={metrics['false_positive']}, "
        f"FN={metrics['false_negative']}, "
        f"TP={metrics['true_positive']}"
    )
    print(f"  false discard rate: {metrics['false_discard_rate']:.4f}")
    print(f"  miss rate: {metrics['miss_rate']:.4f}")


def _print_folds(name: str, folds: list[dict[str, object]]) -> None:
    print(name)

    for fold in folds:
        print(
            f"  fold {fold['fold']}: "
            f"train essays={fold['train_essays']}, "
            f"validation essays={fold['validation_essays']}, "
            f"train n={fold['train_observations']}, "
            f"validation n={fold['validation_observations']}, "
            f"validation U={fold['validation_usable']}, "
            f"validation UNR={fold['validation_unreliable']}, "
            f"P={fold['precision']:.3f}, "
            f"R={fold['recall']:.3f}, "
            f"F1={fold['f1']:.3f}, "
            f"F2={fold['f2']:.3f}, "
            f"essay overlap={fold['essay_overlap']}"
        )


def _print_coefficients(
    name: str,
    coefficients: dict[str, list[tuple[str, float]]],
) -> None:
    print(name)
    print("  toward UNRELIABLE:")

    for feature_name, coefficient in coefficients["positive"]:
        print(f"    {feature_name}: {coefficient:+.4f}")

    print("  toward USABLE:")

    for feature_name, coefficient in coefficients["negative"]:
        print(f"    {feature_name}: {coefficient:+.4f}")


def main() -> None:
    essays = _load_jsonl(ESSAYS_PATH)
    annotations = _load_jsonl(ANNOTATIONS_PATH)
    observations = prepare_ml_observations(annotations)

    essay_ids = set()
    student_counts = Counter()
    target_counts = Counter()

    for observation in observations:
        essay_ids.add(observation["essay_id"])
        student_counts[observation["student_id"]] += 1
        target_counts[observation["target"]] += 1

    if len(essays) != 28 or len(annotations) != 232:
        raise ValueError("Unexpected canonical dataset size")

    if len(observations) != 221 or len(essay_ids) != 28:
        raise ValueError("Unexpected binary dataset size")

    if target_counts != {0: 178, 1: 43}:
        raise ValueError("Unexpected binary target distribution")

    leaking_features = feature_name_leakage(MODEL_B_FEATURE_NAMES)

    if len(leaking_features) > 0:
        raise ValueError(f"Feature leakage detected: {leaking_features}")

    targets = observation_targets(observations)
    groups = []

    for observation in observations:
        groups.append(observation["essay_id"])

    grouped_essay_ids = np.asarray(groups, dtype=str)
    split_count, splits = select_grouped_splits(targets, grouped_essay_ids)

    model_a_result = run_grouped_oof(
        observations,
        MODEL_A_FEATURE_NAMES,
        splits,
    )
    model_b_result = run_grouped_oof(
        observations,
        MODEL_B_FEATURE_NAMES,
        splits,
    )
    dummy_result = run_grouped_oof(
        observations,
        MODEL_A_FEATURE_NAMES,
        splits,
        estimator=build_dummy_classifier(),
    )

    frozen_predictions = frozen_warning_predictions(observations)

    if frozen_predictions is None:
        frozen_metrics = None
    else:
        frozen_metrics = calculate_binary_metrics(
            targets,
            frozen_predictions,
        )

    student_ids = sorted(student_counts)
    student_results = {}

    if student_ids == ["student_m", "student_v"]:
        directions = (
            ("student_v", "student_m"),
            ("student_m", "student_v"),
        )

        for train_student, test_student in directions:
            train_observations = []
            test_observations = []

            for observation in observations:
                if observation["student_id"] == train_student:
                    train_observations.append(observation)
                elif observation["student_id"] == test_student:
                    test_observations.append(observation)

            direction_name = f"{train_student}->{test_student}"
            student_results[direction_name] = {
                "model_a": evaluate_fixed_transfer(
                    train_observations,
                    test_observations,
                    MODEL_A_FEATURE_NAMES,
                ),
                "model_b": evaluate_fixed_transfer(
                    train_observations,
                    test_observations,
                    MODEL_B_FEATURE_NAMES,
                ),
            }

    model_a_coefficients = descriptive_coefficients(
        observations,
        MODEL_A_FEATURE_NAMES,
    )
    model_b_coefficients = descriptive_coefficients(
        observations,
        MODEL_B_FEATURE_NAMES,
    )

    print("DATA")
    print(f"  observations: {len(observations)}")
    print(f"  essays: {len(essay_ids)}")
    print(f"  students: {dict(student_counts)}")
    print(f"  USABLE: {target_counts[0]}")
    print(f"  UNRELIABLE: {target_counts[1]}")
    print(f"  grouped folds: {split_count}")
    print(f"  feature leakage names: {len(leaking_features)}")
    print()

    _print_metrics("MODEL A — STRUCTURAL", model_a_result["metrics"])
    print()
    _print_metrics("MODEL B — STRUCTURAL + SIGNALS", model_b_result["metrics"])
    print()

    if frozen_metrics is None:
        print("Frozen v1.2 binary baseline mapping is not uniquely defined.")
    else:
        _print_metrics("FROZEN v1.2 WARNING BASELINE", frozen_metrics)

    print()
    _print_metrics("DUMMY MOST-FREQUENT", dummy_result["metrics"])
    print()
    _print_folds("MODEL A PER FOLD", model_a_result["folds"])
    print()
    _print_folds("MODEL B PER FOLD", model_b_result["folds"])
    print()
    print("CROSS-STUDENT DEVELOPMENT STRESS TEST")

    for direction_name, direction_results in student_results.items():
        print(f"  {direction_name}")

        for model_name, result in direction_results.items():
            metrics = result["metrics"]
            print(
                f"    {model_name}: train={result['train_size']}, "
                f"test={result['test_size']}, "
                f"test U={result['test_usable']}, "
                f"test UNR={result['test_unreliable']}, "
                f"P={metrics['precision']:.3f}, "
                f"R={metrics['recall']:.3f}, "
                f"F1={metrics['f1']:.3f}, "
                f"F2={metrics['f2']:.3f}, "
                f"BA={metrics['balanced_accuracy']:.3f}, "
                f"TN={metrics['true_negative']}, "
                f"FP={metrics['false_positive']}, "
                f"FN={metrics['false_negative']}, "
                f"TP={metrics['true_positive']}"
            )

    print()
    _print_coefficients("MODEL A COEFFICIENTS", model_a_coefficients)
    print()
    _print_coefficients("MODEL B COEFFICIENTS", model_b_coefficients)


if __name__ == "__main__":
    main()
