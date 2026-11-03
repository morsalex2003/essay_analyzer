import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.pipeline import nlp
from src.syntax.clauses import detect_clauses
from src.syntax.reliability_ml import (
    MODEL_A_FEATURE_NAMES,
    MODEL_B_FEATURE_NAMES,
    frozen_warning_predictions,
    observation_groups,
    observation_targets,
    prepare_ml_observations,
    run_grouped_oof,
    select_grouped_splits,
)
from src.syntax.reliability_ml_errors import (
    active_heuristic_flags,
    active_signal_counts,
    combine_system_predictions,
    error_margin_category,
    failure_combination_counts,
    failure_family_counts,
    fold_linear_contributions,
    partition_system_intersections,
    probability_summary,
    recurring_contribution_counts,
)


ANNOTATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "syntax_reliability"
    / "annotations_v1.jsonl"
)

DEPENDENCY_FEATURE_NAMES = (
    "advcl_count",
    "ccomp_count",
    "xcomp_count",
    "relcl_count",
    "acl_count",
    "conj_count",
    "parataxis_count",
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []

    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            records.append(json.loads(line))

    return records


def _prediction_name(prediction: int) -> str:
    if prediction == 1:
        return "UNRELIABLE"

    return "USABLE"


def _record_sentence_id(record: dict[str, object]) -> str:
    return record["sentence_id"]


def _print_counter(name: str, counts: Counter) -> None:
    print(name)

    if len(counts) == 0:
        print("  none")
        return

    for key, count in counts.most_common():
        print(f"  {key}: {count}")


def _print_probability_summary(
    name: str,
    records: list[dict[str, object]],
) -> None:
    summary = probability_summary(records)

    if summary is None:
        print(f"{name}: none")
        return

    print(
        f"{name}: min={summary['minimum']:.4f}, "
        f"median={summary['median']:.4f}, "
        f"max={summary['maximum']:.4f}"
    )


def _head_names(clause_summary: dict[str, object], key: str) -> str:
    names = []

    for token in clause_summary[key]:
        names.append(f"{token.text}/{token.dep_}")

    if len(names) == 0:
        return "none"

    return ", ".join(names)


def _parse_summary(annotation: dict[str, object]) -> dict[str, str]:
    document = nlp(annotation["original_text"])
    clause_summary = detect_clauses(document)
    root = None

    for token in document:
        if token.dep_ == "ROOT":
            root = token
            break

    if root is None:
        root_description = "none"
    else:
        root_description = f"{root.text}/{root.pos_}"

    return {
        "root": root_description,
        "finite": _head_names(clause_summary, "finite_clause_heads"),
        "dependent": _head_names(
            clause_summary,
            "dependent_clause_heads",
        ),
        "non_finite": _head_names(clause_summary, "non_finite_heads"),
    }


def _format_contributions(
    contribution_pairs: list[tuple[str, float]],
) -> str:
    values = []

    for feature_name, contribution in contribution_pairs:
        values.append(f"{feature_name}={contribution:+.3f}")

    if len(values) == 0:
        return "none"

    return ", ".join(values)


def _gold_reason(annotation: dict[str, object]) -> str:
    reasons = []
    annotation_notes = annotation.get("annotation_notes")
    adjudication_notes = annotation.get("adjudication_notes")

    if annotation_notes:
        reasons.append(annotation_notes)

    if adjudication_notes:
        reasons.append(adjudication_notes)

    if len(reasons) == 0:
        return "No annotation note."

    return " ".join(reasons)


def _print_unreliable_case(
    record: dict[str, object],
    annotation: dict[str, object],
    fold_estimators: list[object],
) -> None:
    features = record["features"]
    parse_summary = _parse_summary(annotation)
    flags = active_heuristic_flags(record)
    contributions = fold_linear_contributions(
        record,
        MODEL_B_FEATURE_NAMES,
        fold_estimators,
    )
    dependency_relations = []

    for feature_name in DEPENDENCY_FEATURE_NAMES:
        if features[feature_name] > 0:
            dependency_relations.append(
                f"{feature_name}={features[feature_name]}"
            )

    print(f"\n### {record['sentence_id']}")
    print(
        "Metadata: "
        f"essay={record['essay_id']}; student={record['student_id']}; "
        f"stage={annotation['source_stage']}; gold=UNRELIABLE; "
        f"failures={annotation['manual_failure_types']}; "
        f"adjudication={annotation['adjudication_status']}"
    )
    print(
        "Prediction: "
        f"Model B p={record['model_b_probability']:.4f}, "
        f"prediction={_prediction_name(record['model_b_prediction'])}; "
        f"Frozen={_prediction_name(record['frozen_prediction'])}; "
        f"Model A correct={record['model_a_prediction'] == 1}; "
        f"fold={record['model_b_fold']}"
    )
    print(
        "Frozen flags: "
        f"{flags if len(flags) > 0 else 'none'}; "
        f"heuristic_warning_count={features['heuristic_warning_count']}"
    )
    print(
        "Parse: "
        f"ROOT={parse_summary['root']}; finite={parse_summary['finite']}; "
        f"dependent={parse_summary['dependent']}; "
        f"non-finite={parse_summary['non_finite']}"
    )
    print(
        "Counts: "
        f"finite={features['finite_clause_count']}; "
        f"dependent={features['dependent_finite_clause_count']}; "
        f"non-finite={features['nonfinite_predicative_count']}; "
        f"morph-finite={features['morph_finite_verb_count']}; "
        f"subjects={features['subject_count']}; "
        f"dependencies={dependency_relations}"
    )
    print(f"Gold reason: {_gold_reason(annotation)}")
    print(
        "Toward UNRELIABLE: "
        f"{_format_contributions(contributions['toward_unreliable'])}"
    )
    print(
        "Toward USABLE: "
        f"{_format_contributions(contributions['toward_usable'])}"
    )


def _print_model_b_false_positive(
    record: dict[str, object],
    annotation: dict[str, object],
    fold_estimators: list[object],
) -> None:
    features = record["features"]
    flags = active_heuristic_flags(record)
    contributions = fold_linear_contributions(
        record,
        MODEL_B_FEATURE_NAMES,
        fold_estimators,
    )

    print(f"\n### {record['sentence_id']}")
    print(
        f"Gold={record['manual_label']}; "
        f"adjudication={annotation['adjudication_status']}; "
        f"p={record['model_b_probability']:.4f}; "
        f"fold={record['model_b_fold']}; "
        f"Model A correct={record['model_a_prediction'] == 0}"
    )
    print(
        f"Signals ({features['heuristic_warning_count']}): "
        f"{flags if len(flags) > 0 else 'none'}"
    )
    print(
        "Toward UNRELIABLE: "
        f"{_format_contributions(contributions['toward_unreliable'])}"
    )
    print(
        "Toward USABLE: "
        f"{_format_contributions(contributions['toward_usable'])}"
    )


def _error_records(
    records: list[dict[str, object]],
    target: int,
    prediction: int,
) -> list[dict[str, object]]:
    selected_records = []

    for record in records:
        if (
            record["y_true"] == target
            and record["model_b_prediction"] == prediction
        ):
            selected_records.append(record)

    return selected_records


def _correct_records(
    records: list[dict[str, object]],
    target: int,
) -> list[dict[str, object]]:
    selected_records = []

    for record in records:
        if (
            record["y_true"] == target
            and record["model_b_prediction"] == target
        ):
            selected_records.append(record)

    return selected_records


def _distribution(
    records: list[dict[str, object]],
    field_name: str,
    annotations_by_id: dict[str, dict[str, object]],
) -> Counter:
    counts = Counter()

    for record in records:
        if field_name in record:
            value = record[field_name]
        else:
            annotation = annotations_by_id[record["sentence_id"]]
            value = annotation[field_name]

        counts[value] += 1

    return counts


def _margin_counts(records: list[dict[str, object]]) -> Counter:
    counts = Counter()

    for record in records:
        counts[error_margin_category(record)] += 1

    return counts


def _signal_presence_counts(records: list[dict[str, object]]) -> Counter:
    counts = Counter()

    for record in records:
        if record["features"]["heuristic_warning_count"] > 0:
            counts["with_signals"] += 1
        else:
            counts["without_signals"] += 1

    return counts


def _print_student_rates(
    records: list[dict[str, object]],
) -> None:
    students = sorted({record["student_id"] for record in records})

    for student_id in students:
        actual_unreliable = 0
        actual_usable = 0
        false_negatives = 0
        false_positives = 0

        for record in records:
            if record["student_id"] != student_id:
                continue

            if record["y_true"] == 1:
                actual_unreliable += 1

                if record["model_b_prediction"] == 0:
                    false_negatives += 1
            else:
                actual_usable += 1

                if record["model_b_prediction"] == 1:
                    false_positives += 1

        print(
            f"  {student_id}: FN={false_negatives}/{actual_unreliable} "
            f"({false_negatives / actual_unreliable:.3f}); "
            f"FP={false_positives}/{actual_usable} "
            f"({false_positives / actual_usable:.3f})"
        )


def _print_intersection_counts(
    title: str,
    groups: dict[str, list[dict[str, object]]],
) -> None:
    print(title)

    for group_name, records in groups.items():
        print(f"  {group_name}: {len(records)}")


def _print_model_comparison(records: list[dict[str, object]]) -> None:
    model_b_errors_fixed_by_a = 0
    model_a_errors_fixed_by_b = 0
    model_a_error_count = 0
    model_b_error_count = 0

    for record in records:
        target = record["y_true"]
        model_a_correct = record["model_a_prediction"] == target
        model_b_correct = record["model_b_prediction"] == target

        if not model_a_correct:
            model_a_error_count += 1

        if not model_b_correct:
            model_b_error_count += 1

        if not model_b_correct and model_a_correct:
            model_b_errors_fixed_by_a += 1

        if not model_a_correct and model_b_correct:
            model_a_errors_fixed_by_b += 1

    print("MODEL A VS MODEL B")
    print(f"  Model A errors: {model_a_error_count}")
    print(f"  Model B errors: {model_b_error_count}")
    print(f"  Model B errors fixed by Model A: {model_b_errors_fixed_by_a}")
    print(f"  Model A errors fixed by Model B: {model_a_errors_fixed_by_b}")


def main() -> None:
    annotations = _load_jsonl(ANNOTATIONS_PATH)
    annotations_by_id = {}

    for annotation in annotations:
        annotations_by_id[annotation["sentence_id"]] = annotation

    observations = prepare_ml_observations(annotations)
    targets = observation_targets(observations)
    groups = observation_groups(observations)
    _, splits = select_grouped_splits(targets, groups)
    model_a_result = run_grouped_oof(
        observations,
        MODEL_A_FEATURE_NAMES,
        splits,
    )
    model_b_result = run_grouped_oof(
        observations,
        MODEL_B_FEATURE_NAMES,
        splits,
        retain_estimators=True,
    )
    frozen_predictions = frozen_warning_predictions(observations)

    if frozen_predictions is None:
        raise ValueError("Frozen binary mapping is unavailable")

    comparison_records = combine_system_predictions(
        observations,
        model_a_result["oof_records"],
        model_b_result["oof_records"],
        frozen_predictions,
    )
    intersections = partition_system_intersections(comparison_records)
    unreliable_groups = intersections["unreliable"]
    usable_groups = intersections["usable"]
    unreliable_partition_total = 0
    usable_partition_total = 0

    for records in unreliable_groups.values():
        unreliable_partition_total += len(records)

    for records in usable_groups.values():
        usable_partition_total += len(records)

    if unreliable_partition_total != 43:
        raise ValueError("UNRELIABLE intersections do not partition 43 rows")

    if usable_partition_total != 178:
        raise ValueError("USABLE intersections do not partition 178 rows")

    model_b_false_negatives = _error_records(comparison_records, 1, 0)
    model_b_false_positives = _error_records(comparison_records, 0, 1)
    model_b_true_positives = _correct_records(comparison_records, 1)
    model_b_true_negatives = _correct_records(comparison_records, 0)
    frozen_false_negatives = []
    frozen_false_positives = []

    for record in comparison_records:
        if record["y_true"] == 1 and record["frozen_prediction"] == 0:
            frozen_false_negatives.append(record)
        elif record["y_true"] == 0 and record["frozen_prediction"] == 1:
            frozen_false_positives.append(record)

    missed_by_either = []

    for group_name in ("model_b_only", "frozen_only", "missed_by_both"):
        missed_by_either.extend(unreliable_groups[group_name])

    fold_estimators = model_b_result["fold_estimators"]

    print("DEVELOPMENT ERROR ANALYSIS — NO TEXT IS PERSISTED")
    print(f"observations={len(comparison_records)}")
    print()
    _print_intersection_counts("UNRELIABLE INTERSECTION", unreliable_groups)
    print()
    _print_intersection_counts("USABLE INTERSECTION", usable_groups)
    print()

    print("FAILURE FAMILIES")
    _print_counter(
        "Model B FN",
        failure_family_counts(
            model_b_false_negatives,
            annotations_by_id,
        ),
    )
    _print_counter(
        "Frozen FN",
        failure_family_counts(
            frozen_false_negatives,
            annotations_by_id,
        ),
    )
    _print_counter(
        "Missed by both",
        failure_family_counts(
            unreliable_groups["missed_by_both"],
            annotations_by_id,
        ),
    )
    _print_counter(
        "Model B FN combinations",
        failure_combination_counts(
            model_b_false_negatives,
            annotations_by_id,
        ),
    )
    _print_counter(
        "Frozen FN combinations",
        failure_combination_counts(
            frozen_false_negatives,
            annotations_by_id,
        ),
    )
    _print_counter(
        "Missed-by-both combinations",
        failure_combination_counts(
            unreliable_groups["missed_by_both"],
            annotations_by_id,
        ),
    )
    print()

    print("PROBABILITIES")
    _print_probability_summary("Model B FN", model_b_false_negatives)
    _print_probability_summary("Model B FP", model_b_false_positives)
    _print_probability_summary("Model B TP", model_b_true_positives)
    _print_probability_summary("Model B TN", model_b_true_negatives)
    _print_counter("Model B FN margins", _margin_counts(model_b_false_negatives))
    _print_counter("Model B FP margins", _margin_counts(model_b_false_positives))
    print()

    print("HEURISTIC DEPENDENCE")
    _print_counter(
        "Model B FN signal presence",
        _signal_presence_counts(model_b_false_negatives),
    )
    _print_counter(
        "Model B FP signal presence",
        _signal_presence_counts(model_b_false_positives),
    )
    _print_counter(
        "Model B FN active signals",
        active_signal_counts(model_b_false_negatives),
    )
    _print_counter(
        "Model B FP active signals",
        active_signal_counts(model_b_false_positives),
    )
    print()

    print("MODEL B FP DISTRIBUTIONS")
    _print_counter(
        "Gold labels",
        _distribution(
            model_b_false_positives,
            "manual_label",
            annotations_by_id,
        ),
    )
    _print_counter(
        "Adjudication",
        _distribution(
            model_b_false_positives,
            "adjudication_status",
            annotations_by_id,
        ),
    )
    _print_counter(
        "Students",
        _distribution(
            model_b_false_positives,
            "student_id",
            annotations_by_id,
        ),
    )
    _print_counter(
        "Source stages",
        _distribution(
            model_b_false_positives,
            "source_stage",
            annotations_by_id,
        ),
    )
    _print_counter(
        "Essays",
        _distribution(
            model_b_false_positives,
            "essay_id",
            annotations_by_id,
        ),
    )
    print()

    print("MODEL B FN DISTRIBUTIONS")
    _print_counter(
        "Adjudication",
        _distribution(
            model_b_false_negatives,
            "adjudication_status",
            annotations_by_id,
        ),
    )
    _print_counter(
        "Students",
        _distribution(
            model_b_false_negatives,
            "student_id",
            annotations_by_id,
        ),
    )
    _print_counter(
        "Source stages",
        _distribution(
            model_b_false_negatives,
            "source_stage",
            annotations_by_id,
        ),
    )
    _print_counter(
        "Essays",
        _distribution(
            model_b_false_negatives,
            "essay_id",
            annotations_by_id,
        ),
    )
    print("Student rates (small samples):")
    _print_student_rates(comparison_records)
    print()

    _print_model_comparison(comparison_records)
    print()

    print("RECURRING TOP-5 CONTRIBUTION FEATURES")
    _print_counter(
        "Model B FN pressure toward USABLE",
        recurring_contribution_counts(
            model_b_false_negatives,
            MODEL_B_FEATURE_NAMES,
            fold_estimators,
            "toward_usable",
        ),
    )
    _print_counter(
        "Model B FP pressure toward UNRELIABLE",
        recurring_contribution_counts(
            model_b_false_positives,
            MODEL_B_FEATURE_NAMES,
            fold_estimators,
            "toward_unreliable",
        ),
    )
    print()

    print("UNRELIABLE MISSED BY AT LEAST ONE SYSTEM")

    for record in sorted(
        missed_by_either,
        key=_record_sentence_id,
    ):
        annotation = annotations_by_id[record["sentence_id"]]
        _print_unreliable_case(record, annotation, fold_estimators)

    print("\nMODEL B FALSE POSITIVES")

    for record in sorted(
        model_b_false_positives,
        key=_record_sentence_id,
    ):
        annotation = annotations_by_id[record["sentence_id"]]
        _print_model_b_false_positive(record, annotation, fold_estimators)

    print("\nFROZEN-ONLY FALSE POSITIVES")

    frozen_only_false_positives = usable_groups[
        "frozen_false_positive_only"
    ]

    if len(frozen_only_false_positives) == 0:
        print("none")
    else:
        for record in frozen_only_false_positives:
            print(record["sentence_id"])

    print("\nPROVISIONAL QC WATCHLIST")
    watchlist = []

    for record in model_b_false_negatives + model_b_false_positives:
        annotation = annotations_by_id[record["sentence_id"]]

        if annotation["adjudication_status"] != "single":
            continue

        if error_margin_category(record) == "confident_error":
            watchlist.append(record["sentence_id"])

    if len(watchlist) == 0:
        print("none")
    else:
        for sentence_id in sorted(watchlist):
            print(f"  {sentence_id}: provisional reference")


if __name__ == "__main__":
    main()
