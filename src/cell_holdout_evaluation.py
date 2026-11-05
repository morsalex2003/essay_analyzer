"""Final CELL holdout evaluation, as fixed in holdout_v1_evaluation_plan.json.

The functions here only score predictions that were made without the
holdout labels. They compare each candidate with frozen Syntax Reliability
v1.2 on the same sentences: overall metrics with Wilson intervals, paired
discordance with an exact McNemar test, recall by failure type, metrics by
adjudication status, and the plan's decision rule for v1.3.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from src.cell_pilot_evaluation import wilson_interval
from src.syntax.reliability_ml import (
    calculate_binary_metrics,
    target_from_manual_label,
)


WARNING_CATEGORIES = ("medium", "low")
SIGNIFICANCE_LEVEL = 0.05


def exact_mcnemar_p_value(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar test on the discordant pair counts."""
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0

    smaller = min(only_first, only_second)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2 * tail / 2**discordant)


def warning_prediction(verdict: str) -> int | None:
    """1 for a warning, 0 for high, None when the heuristic abstained."""
    if verdict == "not_analyzable":
        return None
    return int(verdict in WARNING_CATEGORIES)


def analysis_rows(
    records: Sequence[Mapping],
    reference_verdicts: Mapping[str, str],
) -> list[Mapping]:
    """Labelled analyzable records where the reference did not abstain."""
    rows = []
    for record in records:
        if target_from_manual_label(record["manual_label"]) is None:
            continue
        verdict = reference_verdicts[record["sentence_id"]]
        if warning_prediction(verdict) is None:
            continue
        rows.append(record)
    return rows


def system_metrics(targets, predictions) -> dict:
    metrics = calculate_binary_metrics(
        np.asarray(targets, dtype=int), np.asarray(predictions, dtype=int)
    )
    true_positive = metrics["true_positive"]
    metrics["precision_ci"] = wilson_interval(
        true_positive, true_positive + metrics["false_positive"]
    )
    metrics["recall_ci"] = wilson_interval(
        true_positive, true_positive + metrics["false_negative"]
    )
    metrics.pop("average_precision")
    return {
        name: round(value, 3) if isinstance(value, float) else value
        for name, value in metrics.items()
    }


def paired_comparison(targets, candidate, reference) -> dict:
    """Discordance of a candidate and the reference on both classes."""
    targets = np.asarray(targets, dtype=int)
    candidate = np.asarray(candidate, dtype=int)
    reference = np.asarray(reference, dtype=int)
    result = {}

    for name, mask in (
        ("unreliable", targets == 1),
        ("usable", targets == 0),
    ):
        candidate_warns = mask & (candidate == 1)
        reference_warns = mask & (reference == 1)
        candidate_only = int(np.sum(candidate_warns & ~reference_warns))
        reference_only = int(np.sum(reference_warns & ~candidate_warns))
        result[name] = {
            "candidate_only_warns": candidate_only,
            "reference_only_warns": reference_only,
            "exact_mcnemar_p": round(
                exact_mcnemar_p_value(candidate_only, reference_only), 4
            ),
        }
    return result


def recall_by_failure_type(rows, targets, predictions) -> dict:
    counts = {}
    for row, target, prediction in zip(rows, targets, predictions):
        if target != 1:
            continue
        for failure_type in row["manual_failure_types"]:
            entry = counts.setdefault(
                failure_type, {"unreliable": 0, "caught": 0}
            )
            entry["unreliable"] += 1
            entry["caught"] += int(prediction)
    return dict(sorted(counts.items()))


def v13_decision(comparison: Mapping) -> str:
    """Apply the plan's decision rule to the v1.3 against v1.2 comparison."""
    unreliable = comparison["unreliable"]
    usable = comparison["usable"]
    recall_gain = (
        unreliable["candidate_only_warns"] > unreliable["reference_only_warns"]
        and unreliable["exact_mcnemar_p"] < SIGNIFICANCE_LEVEL
    )
    added_false_warnings = (
        usable["candidate_only_warns"] > usable["reference_only_warns"]
        and usable["exact_mcnemar_p"] < SIGNIFICANCE_LEVEL
    )

    if recall_gain and not added_false_warnings:
        return "adopt_v1_3"
    if added_false_warnings or (
        unreliable["candidate_only_warns"] < unreliable["reference_only_warns"]
    ):
        return "negative"
    return "inconclusive"


def evaluate_holdout(
    records: Sequence[Mapping],
    verdicts_by_version: Mapping[str, Mapping[str, str]],
    model_b_predictions: Mapping[str, int],
) -> dict:
    """Score v1.2, v1.3, and Model B on the plan's analysis set."""
    reference = verdicts_by_version["1.2"]
    rows = analysis_rows(records, reference)
    sentence_ids = [row["sentence_id"] for row in rows]
    targets = [target_from_manual_label(row["manual_label"]) for row in rows]
    predictions = {
        "v1.2": [warning_prediction(reference[i]) for i in sentence_ids],
        "v1.3": [
            warning_prediction(verdicts_by_version["1.3"][i])
            for i in sentence_ids
        ],
        "model_b": [model_b_predictions[i] for i in sentence_ids],
    }
    if any(prediction is None for prediction in predictions["v1.3"]):
        raise ValueError("v1.3 abstained where v1.2 did not")

    labelled = [r for r in records if r["manual_label"] is not None]
    analyzable = [
        r
        for r in labelled
        if target_from_manual_label(r["manual_label"]) is not None
    ]
    by_adjudication = {}
    for status in ("adjudicated", "single"):
        indexes = [
            index
            for index, row in enumerate(rows)
            if row["adjudication_status"] == status
        ]
        if not indexes:
            by_adjudication[status] = None
            continue
        by_adjudication[status] = {
            system: system_metrics(
                [targets[i] for i in indexes],
                [values[i] for i in indexes],
            )
            for system, values in predictions.items()
        }

    comparisons = {
        "v1.3_vs_v1.2": paired_comparison(
            targets, predictions["v1.3"], predictions["v1.2"]
        ),
        "model_b_vs_v1.2": paired_comparison(
            targets, predictions["model_b"], predictions["v1.2"]
        ),
    }

    return {
        "counts": {
            "sentences": len(records),
            "not_analyzable_labels": len(labelled) - len(analyzable),
            "reference_abstentions": len(analyzable) - len(rows),
            "analysis_set": len(rows),
            "unreliable": int(sum(targets)),
            "usable": len(targets) - int(sum(targets)),
        },
        "metrics": {
            system: system_metrics(targets, values)
            for system, values in predictions.items()
        },
        "paired_comparisons": comparisons,
        "recall_by_failure_type": {
            system: recall_by_failure_type(rows, targets, values)
            for system, values in predictions.items()
        },
        "metrics_by_adjudication_status": by_adjudication,
        "v1_3_decision": v13_decision(comparisons["v1.3_vs_v1.2"]),
    }
