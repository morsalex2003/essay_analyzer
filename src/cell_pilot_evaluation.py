"""Compare frozen Syntax Reliability v1.2 verdicts with CELL pilot labels.

The heuristic runs on each sentence inside its essay parse, exactly as
``analyze_writing()`` does and as the annotator saw it. The canonical Dataset
v1 baseline parsed sentences in isolation, so the two results are related
but not identical measurements.

The report contains aggregate counts and sentence positions only, never
learner text. All pilot students belong to the development split; this is a
development estimate, not untouched holdout evaluation.
"""

import math

import numpy as np

from src.syntax.clauses import detect_clauses
from src.syntax.reliability import assess_sentence_reliability
from src.syntax.reliability_ml import (
    POSITIVE_CLASS,
    calculate_binary_metrics,
    target_from_manual_label,
)


EVALUATION_VERSION = "1.0"
MANUAL_LABELS = (
    "RELIABLE",
    "MOSTLY_RELIABLE",
    "UNRELIABLE",
    "NOT_ANALYZABLE",
)
HEURISTIC_CATEGORIES = ("high", "medium", "low", "not_analyzable")

# "warning" matches the frozen v1.2 baseline in run_reliability_ml_baseline:
# medium and low both predict UNRELIABLE. "low_only" is the stricter cut.
OPERATING_POINTS = {
    "warning": ("medium", "low"),
    "low_only": ("low",),
}

WILSON_Z_95 = 1.959964


def heuristic_verdicts(records, parser) -> dict[str, str]:
    """Return frozen v1.2 reliability per sentence_id in essay context."""
    clause_summaries = {}
    verdicts = {}

    for record in records:
        essay_id = record["essay_id"]

        if essay_id not in clause_summaries:
            clause_summaries[essay_id] = detect_clauses(
                parser.document(essay_id)
            )

        result = assess_sentence_reliability(
            parser.sentence(record),
            clause_summaries[essay_id],
        )
        verdicts[record["sentence_id"]] = result["reliability"]

    return verdicts


def wilson_interval(successes: int, total: int) -> list[float] | None:
    """Return a 95% Wilson score interval for a proportion."""
    if total == 0:
        return None

    proportion = successes / total
    z_squared = WILSON_Z_95 ** 2
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        / denominator
    )

    return [round(centre - margin, 3), round(centre + margin, 3)]


def _rounded_metrics(metrics):
    rounded = {}

    for name, value in metrics.items():
        if isinstance(value, float):
            rounded[name] = round(value, 3)
        else:
            rounded[name] = value

    return rounded


def _operating_point_result(targets, verdicts, positive_categories):
    predictions = []

    for verdict in verdicts:
        if verdict in positive_categories:
            predictions.append(POSITIVE_CLASS)
        else:
            predictions.append(1 - POSITIVE_CLASS)

    metrics = calculate_binary_metrics(
        np.asarray(targets, dtype=int),
        np.asarray(predictions, dtype=int),
    )
    true_positive = metrics["true_positive"]
    predicted_positive = true_positive + metrics["false_positive"]
    actual_positive = true_positive + metrics["false_negative"]

    result = _rounded_metrics(metrics)
    result["precision_95ci"] = wilson_interval(
        true_positive,
        predicted_positive,
    )
    result["recall_95ci"] = wilson_interval(true_positive, actual_positive)

    return result, predictions


def _recall_by_group(rows, group_names_function):
    """Return UNRELIABLE recall (warning point) for each group name."""
    groups = {}

    for row in rows:
        if row["target"] != POSITIVE_CLASS:
            continue

        for group_name in group_names_function(row):
            counts = groups.setdefault(group_name, {"unreliable": 0, "caught": 0})
            counts["unreliable"] += 1
            if row["warning_prediction"] == POSITIVE_CLASS:
                counts["caught"] += 1

    for counts in groups.values():
        counts["recall"] = round(counts["caught"] / counts["unreliable"], 3)

    return dict(sorted(groups.items()))


def evaluate_pilot(records, verdicts_by_sentence_id) -> dict[str, object]:
    """Build the aggregate report for labelled records.

    ``records`` is the full ledger in template order, including pending
    records; ``verdicts_by_sentence_id`` covers at least the labelled ones.
    """
    labelled_records = []
    for record in records:
        if record["manual_label"] is not None:
            labelled_records.append(record)

    cross_table = {}
    for manual_label in MANUAL_LABELS:
        cross_table[manual_label] = dict.fromkeys(HEURISTIC_CATEGORIES, 0)

    rows = []
    heuristic_abstentions = 0
    student_ids = set()
    essay_ids = set()
    origins = {}
    adjudication_statuses = {}

    for position, record in enumerate(records, start=1):
        manual_label = record["manual_label"]
        if manual_label is None:
            continue

        verdict = verdicts_by_sentence_id[record["sentence_id"]]
        cross_table[manual_label][verdict] += 1
        student_ids.add(record["student_id"])
        essay_ids.add(record["essay_id"])
        origin = record["annotation_origin"]
        origins[origin] = origins.get(origin, 0) + 1
        status = record.get("adjudication_status", "single")
        adjudication_statuses[status] = adjudication_statuses.get(status, 0) + 1

        target = target_from_manual_label(manual_label)
        if target is None:
            continue

        if verdict == "not_analyzable":
            # The heuristic abstained on a sentence the annotator could
            # judge; it has no binary prediction to score.
            heuristic_abstentions += 1
            continue

        rows.append(
            {
                "position": position,
                "target": target,
                "verdict": verdict,
                "failure_types": record["manual_failure_types"],
                "structures": record.get("selection_structures", []),
            }
        )

    targets = [row["target"] for row in rows]
    verdicts = [row["verdict"] for row in rows]
    operating_points = {}
    errors = {"missed_unreliable": [], "false_warning": []}

    if len(rows) > 0:
        for name, positive_categories in OPERATING_POINTS.items():
            result, predictions = _operating_point_result(
                targets,
                verdicts,
                positive_categories,
            )
            operating_points[name] = result

            if name == "warning":
                for row, prediction in zip(rows, predictions):
                    row["warning_prediction"] = prediction
                    if row["target"] == POSITIVE_CLASS and prediction == 0:
                        errors["missed_unreliable"].append(row["position"])
                    if row["target"] != POSITIVE_CLASS and prediction == 1:
                        errors["false_warning"].append(row["position"])

    pending_count = len(records) - len(labelled_records)
    status = "complete"
    if pending_count > 0:
        status = "partial"
    if len(labelled_records) == 0:
        status = "no_labels"

    report = {
        "evaluation_version": EVALUATION_VERSION,
        "status": status,
        "status_note": (
            "Development estimate on CELL pilot students; not untouched "
            "holdout evaluation. Partial results describe only the labelled "
            "sentences and can change as labelling continues."
        ),
        "parse_context": "sentence_in_essay",
        "coverage": {
            "total_sentences": len(records),
            "labelled_sentences": len(labelled_records),
            "pending_sentences": pending_count,
            "labelled_students": len(student_ids),
            "labelled_essays": len(essay_ids),
            "annotation_origin_counts": origins,
            "adjudication_status_counts": adjudication_statuses,
        },
        "label_by_heuristic": cross_table,
        "binary_task": {
            "positive_class": "UNRELIABLE",
            "scored_sentences": len(rows),
            "unreliable_sentences": targets.count(POSITIVE_CLASS),
            "heuristic_abstained_on_analyzable": heuristic_abstentions,
            "operating_points": operating_points,
        },
        "warning_recall_by_failure_type": {},
        "warning_recall_by_selection_structure": {},
        "warning_errors_by_position": errors,
    }

    if len(rows) > 0:
        report["warning_recall_by_failure_type"] = _recall_by_group(
            rows,
            lambda row: row["failure_types"] or ["unspecified"],
        )
        report["warning_recall_by_selection_structure"] = _recall_by_group(
            rows,
            lambda row: row["structures"],
        )

    return report
