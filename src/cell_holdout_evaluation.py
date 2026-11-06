"""Final CELL holdout evaluations, as fixed in the holdout evaluation plans.

The functions here only score predictions that were made without the
holdout labels. They compare each candidate with frozen Syntax Reliability
v1.2 on the same sentences: overall metrics with Wilson intervals, paired
discordance with an exact McNemar test, recall by failure type, metrics by
adjudication status, and the plan's decision rule for its primary
candidate (v1.3 in holdout v1, v1.4 against v1.3 in holdout v2).
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
REFERENCE_VERSION = "1.2"
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


def candidate_decision(comparison: Mapping, positive_outcome: str) -> str:
    """Apply a plan's decision rule to its primary paired comparison."""
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
        return positive_outcome
    if added_false_warnings or (
        unreliable["candidate_only_warns"] < unreliable["reference_only_warns"]
    ):
        return "negative"
    return "inconclusive"


def evaluate_holdout(
    records: Sequence[Mapping],
    verdicts_by_version: Mapping[str, Mapping[str, str]],
    model_b_predictions: Mapping[str, int],
    candidates: Sequence[str] = ("1.3",),
    decision_pair: tuple[str, str] = ("1.3", REFERENCE_VERSION),
    positive_outcome: str = "adopt_v1_3",
) -> dict:
    """Score v1.2, the candidate versions, and Model B on the analysis set.

    Every system is compared with v1.2; `decision_pair` (candidate,
    baseline) is the plan's primary comparison and gets the decision rule.
    """
    reference = verdicts_by_version[REFERENCE_VERSION]
    rows = analysis_rows(records, reference)
    sentence_ids = [row["sentence_id"] for row in rows]
    targets = [target_from_manual_label(row["manual_label"]) for row in rows]
    predictions = {
        f"v{REFERENCE_VERSION}": [
            warning_prediction(reference[i]) for i in sentence_ids
        ],
    }
    for version in candidates:
        version_predictions = [
            warning_prediction(verdicts_by_version[version][i])
            for i in sentence_ids
        ]
        if any(prediction is None for prediction in version_predictions):
            raise ValueError(f"v{version} abstained where v1.2 did not")
        predictions[f"v{version}"] = version_predictions
    predictions["model_b"] = [model_b_predictions[i] for i in sentence_ids]

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

    reference_name = f"v{REFERENCE_VERSION}"
    comparisons = {}
    for system in [f"v{version}" for version in candidates] + ["model_b"]:
        comparisons[f"{system}_vs_{reference_name}"] = paired_comparison(
            targets, predictions[system], predictions[reference_name]
        )
    candidate, baseline = (f"v{version}" for version in decision_pair)
    decision_key = f"{candidate}_vs_{baseline}"
    if decision_key not in comparisons:
        comparisons[decision_key] = paired_comparison(
            targets, predictions[candidate], predictions[baseline]
        )
    decision_name = candidate.replace(".", "_") + "_decision"

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
        decision_name: candidate_decision(
            comparisons[decision_key], positive_outcome
        ),
    }
