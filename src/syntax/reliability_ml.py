from collections.abc import Iterable, Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from spacy.language import Language

from src.cell_annotation import SentenceParser
from src.nlp.pipeline import nlp
from src.syntax.clauses import detect_clauses
from src.syntax.reliability import assess_sentence_reliability
from src.syntax.reliability_error_features import (
    ERROR_PATTERN_FEATURE_NAMES,
    extract_error_pattern_features,
)
from src.syntax.reliability_features import (
    ALL_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    extract_reliability_features,
)


POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0
RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.5

MODEL_A_FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES
MODEL_B_FEATURE_NAMES = ALL_FEATURE_NAMES
MODEL_C_FEATURE_NAMES = (*ALL_FEATURE_NAMES, *ERROR_PATTERN_FEATURE_NAMES)

FORBIDDEN_FEATURE_NAME_FRAGMENTS = (
    "label",
    "gold",
    "manual",
    "usable",
    "unreliable",
    "student",
    "essay_id",
    "source_stage",
    "adjudication",
    "reliability_level",
)


class ReliabilityObservation(TypedDict):
    sentence_id: str
    essay_id: str
    student_id: str
    manual_label: str
    target: int
    features: dict[str, int | float]
    frozen_reliability: str


class FoldSummary(TypedDict):
    fold: int
    train_essays: int
    validation_essays: int
    train_observations: int
    validation_observations: int
    validation_usable: int
    validation_unreliable: int
    precision: float
    recall: float
    f1: float
    f2: float
    essay_overlap: int


def target_from_manual_label(manual_label: str) -> int | None:
    if manual_label == "UNRELIABLE":
        return POSITIVE_CLASS

    if manual_label in ("RELIABLE", "MOSTLY_RELIABLE"):
        return NEGATIVE_CLASS

    if manual_label == "NOT_ANALYZABLE":
        return None

    raise ValueError(f"Unknown manual label: {manual_label}")


def prepare_ml_observations(
    annotation_records: Iterable[dict[str, object]],
    nlp_pipeline: Language = nlp,
) -> list[ReliabilityObservation]:
    observations = []

    for record in annotation_records:
        manual_label = record["manual_label"]
        target = target_from_manual_label(manual_label)

        if target is None:
            continue

        text = record["original_text"]
        document = nlp_pipeline(text)
        sentences = list(document.sents)

        if len(sentences) != 1:
            sentence_id = record["sentence_id"]
            raise ValueError(
                f"{sentence_id} must contain exactly one parsed sentence"
            )

        clause_summary = detect_clauses(document)
        features = extract_reliability_features(
            document,
            clause_summary,
            include_heuristic_signals=True,
        )
        frozen_result = assess_sentence_reliability(
            document,
            clause_summary,
        )
        observations.append(
            {
                "sentence_id": record["sentence_id"],
                "essay_id": record["essay_id"],
                "student_id": record["student_id"],
                "manual_label": manual_label,
                "target": target,
                "features": features,
                "frozen_reliability": frozen_result["reliability"],
            }
        )

    return observations


def prepare_essay_context_observations(
    essay_records: Iterable[dict[str, object]],
    annotation_records: Iterable[dict[str, object]],
    nlp_pipeline: Language = nlp,
    include_error_pattern_features: bool = False,
) -> list[ReliabilityObservation]:
    """Build observations from each sentence's span in its parsed essay.

    Labels were judged on these spans, so this is the parse context fixed for
    Reliability ML v2. ``prepare_ml_observations`` parses sentences in
    isolation and stays unchanged for the frozen ML v1 baseline.
    """

    essays_by_id = {essay["essay_id"]: essay for essay in essay_records}
    parser = SentenceParser(nlp_pipeline, essays_by_id)
    clause_summaries = {}
    observations = []

    for record in annotation_records:
        target = target_from_manual_label(record["manual_label"])

        if target is None:
            continue

        essay_id = record["essay_id"]
        if essay_id not in clause_summaries:
            clause_summaries[essay_id] = detect_clauses(
                parser.document(essay_id)
            )

        sentence = parser.sentence(record)
        clause_summary = clause_summaries[essay_id]
        features = extract_reliability_features(
            sentence,
            clause_summary,
            include_heuristic_signals=True,
        )
        if include_error_pattern_features:
            features.update(
                extract_error_pattern_features(sentence, clause_summary)
            )

        observations.append(
            {
                "sentence_id": record["sentence_id"],
                "essay_id": essay_id,
                "student_id": record["student_id"],
                "manual_label": record["manual_label"],
                "target": target,
                "features": features,
                "frozen_reliability": assess_sentence_reliability(
                    sentence,
                    clause_summary,
                )["reliability"],
            }
        )

    return observations


def feature_name_leakage(feature_names: Sequence[str]) -> list[str]:
    leaking_feature_names = []

    for feature_name in feature_names:
        for forbidden_fragment in FORBIDDEN_FEATURE_NAME_FRAGMENTS:
            if forbidden_fragment in feature_name:
                leaking_feature_names.append(feature_name)
                break

    return leaking_feature_names


def build_feature_matrix(
    observations: Sequence[ReliabilityObservation],
    feature_names: Sequence[str],
) -> NDArray[np.float64]:
    feature_rows = []

    for observation in observations:
        feature_row = []

        for feature_name in feature_names:
            feature_row.append(observation["features"][feature_name])

        feature_rows.append(feature_row)

    return np.asarray(feature_rows, dtype=float)


def observation_targets(
    observations: Sequence[ReliabilityObservation],
) -> NDArray[np.int64]:
    targets = []

    for observation in observations:
        targets.append(observation["target"])

    return np.asarray(targets, dtype=int)


def observation_groups(
    observations: Sequence[ReliabilityObservation],
) -> NDArray[np.str_]:
    groups = []

    for observation in observations:
        groups.append(observation["essay_id"])

    return np.asarray(groups, dtype=str)


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=(
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        )
    )


def build_dummy_classifier() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def _split_is_valid(
    targets: NDArray[np.int64],
    groups: NDArray[np.str_],
    splits: Sequence[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> bool:
    validation_assignments = np.zeros(len(targets), dtype=int)

    for train_indexes, validation_indexes in splits:
        train_groups = set(groups[train_indexes])
        validation_groups = set(groups[validation_indexes])

        if not train_groups.isdisjoint(validation_groups):
            return False

        if len(np.unique(targets[validation_indexes])) != 2:
            return False

        if len(np.unique(targets[train_indexes])) != 2:
            return False

        validation_assignments[validation_indexes] += 1

    return bool(np.all(validation_assignments == 1))


def select_grouped_splits(
    targets: NDArray[np.int64],
    groups: NDArray[np.str_],
    candidate_splits: Sequence[int] = (5, 4, 3),
) -> tuple[int, list[tuple[NDArray[np.int64], NDArray[np.int64]]]]:
    placeholder_features = np.zeros((len(targets), 1), dtype=float)

    for split_count in candidate_splits:
        splitter = StratifiedGroupKFold(
            n_splits=split_count,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        splits = list(
            splitter.split(
                placeholder_features,
                targets,
                groups,
            )
        )

        if _split_is_valid(targets, groups, splits):
            return split_count, splits

    raise ValueError(
        "No candidate grouped split assigns both classes to every fold"
    )


def calculate_binary_metrics(
    targets: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64] | None = None,
) -> dict[str, float | int | None]:
    matrix = confusion_matrix(targets, predictions, labels=(0, 1))
    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )
    actual_usable = true_negative + false_positive
    actual_unreliable = false_negative + true_positive

    if probabilities is None:
        average_precision = None
    else:
        average_precision = float(
            average_precision_score(targets, probabilities)
        )

    return {
        "precision": float(
            precision_score(
                targets,
                predictions,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                targets,
                predictions,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                targets,
                predictions,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "f2": float(
            fbeta_score(
                targets,
                predictions,
                beta=2,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "accuracy": float(accuracy_score(targets, predictions)),
        "average_precision": average_precision,
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "false_discard_rate": (
            float(false_positive / actual_usable)
            if actual_usable > 0
            else 0.0
        ),
        "miss_rate": (
            float(false_negative / actual_unreliable)
            if actual_unreliable > 0
            else 0.0
        ),
    }


def _positive_probabilities(
    fitted_estimator,
    feature_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    probabilities = fitted_estimator.predict_proba(feature_matrix)
    classes = fitted_estimator.classes_
    positive_indexes = np.flatnonzero(classes == POSITIVE_CLASS)

    if len(positive_indexes) != 1:
        raise ValueError("Estimator does not expose the positive class")

    return probabilities[:, positive_indexes[0]]


def run_grouped_oof(
    observations: Sequence[ReliabilityObservation],
    feature_names: Sequence[str],
    splits: Sequence[tuple[NDArray[np.int64], NDArray[np.int64]]],
    estimator: Pipeline | DummyClassifier | None = None,
    retain_estimators: bool = False,
) -> dict[str, object]:
    feature_matrix = build_feature_matrix(observations, feature_names)
    targets = observation_targets(observations)
    groups = observation_groups(observations)

    if estimator is None:
        estimator = build_logistic_pipeline()

    oof_predictions = np.full(len(observations), -1, dtype=int)
    oof_probabilities = np.full(len(observations), np.nan, dtype=float)
    fold_assignments = np.full(len(observations), -1, dtype=int)
    fold_summaries = []
    fold_estimators = []

    for fold_index, split in enumerate(splits, start=1):
        train_indexes, validation_indexes = split
        fitted_estimator = clone(estimator)
        fitted_estimator.fit(
            feature_matrix[train_indexes],
            targets[train_indexes],
        )

        if retain_estimators:
            fold_estimators.append(fitted_estimator)

        probabilities = _positive_probabilities(
            fitted_estimator,
            feature_matrix[validation_indexes],
        )
        predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(
            int
        )
        oof_predictions[validation_indexes] = predictions
        oof_probabilities[validation_indexes] = probabilities
        fold_assignments[validation_indexes] = fold_index

        fold_metrics = calculate_binary_metrics(
            targets[validation_indexes],
            predictions,
            probabilities,
        )
        train_groups = set(groups[train_indexes])
        validation_groups = set(groups[validation_indexes])
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_essays": len(train_groups),
                "validation_essays": len(validation_groups),
                "train_observations": len(train_indexes),
                "validation_observations": len(validation_indexes),
                "validation_usable": int(
                    np.sum(targets[validation_indexes] == NEGATIVE_CLASS)
                ),
                "validation_unreliable": int(
                    np.sum(targets[validation_indexes] == POSITIVE_CLASS)
                ),
                "precision": fold_metrics["precision"],
                "recall": fold_metrics["recall"],
                "f1": fold_metrics["f1"],
                "f2": fold_metrics["f2"],
                "essay_overlap": len(
                    train_groups.intersection(validation_groups)
                ),
            }
        )

    if np.any(oof_predictions < 0):
        raise ValueError("Some observations have no OOF prediction")

    if np.any(fold_assignments < 0):
        raise ValueError("Some observations have no fold assignment")

    if not np.all(np.isfinite(oof_probabilities)):
        raise ValueError("OOF probabilities contain non-finite values")

    oof_records = []

    for observation_index, observation in enumerate(observations):
        oof_records.append(
            {
                "sentence_id": observation["sentence_id"],
                "essay_id": observation["essay_id"],
                "student_id": observation["student_id"],
                "manual_label": observation["manual_label"],
                "y_true": int(targets[observation_index]),
                "y_pred": int(oof_predictions[observation_index]),
                "unreliable_probability": float(
                    oof_probabilities[observation_index]
                ),
                "fold": int(fold_assignments[observation_index]),
            }
        )

    result = {
        "metrics": calculate_binary_metrics(
            targets,
            oof_predictions,
            oof_probabilities,
        ),
        "folds": fold_summaries,
        "oof_records": oof_records,
    }

    if retain_estimators:
        result["fold_estimators"] = fold_estimators

    return result


def evaluate_fixed_transfer(
    train_observations: Sequence[ReliabilityObservation],
    test_observations: Sequence[ReliabilityObservation],
    feature_names: Sequence[str],
) -> dict[str, object]:
    train_features = build_feature_matrix(
        train_observations,
        feature_names,
    )
    test_features = build_feature_matrix(test_observations, feature_names)
    train_targets = observation_targets(train_observations)
    test_targets = observation_targets(test_observations)
    pipeline = build_logistic_pipeline()
    pipeline.fit(train_features, train_targets)
    probabilities = _positive_probabilities(pipeline, test_features)
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

    return {
        "train_size": len(train_observations),
        "test_size": len(test_observations),
        "test_usable": int(np.sum(test_targets == NEGATIVE_CLASS)),
        "test_unreliable": int(np.sum(test_targets == POSITIVE_CLASS)),
        "metrics": calculate_binary_metrics(
            test_targets,
            predictions,
            probabilities,
        ),
    }


def descriptive_coefficients(
    observations: Sequence[ReliabilityObservation],
    feature_names: Sequence[str],
    limit: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    feature_matrix = build_feature_matrix(observations, feature_names)
    targets = observation_targets(observations)
    pipeline = build_logistic_pipeline()
    pipeline.fit(feature_matrix, targets)
    classifier = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    coefficient_pairs = []

    for feature_name, coefficient in zip(
        feature_names,
        coefficients,
        strict=True,
    ):
        coefficient_pairs.append((feature_name, float(coefficient)))

    positive = sorted(
        coefficient_pairs,
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    negative = sorted(
        coefficient_pairs,
        key=lambda item: item[1],
    )[:limit]

    return {
        "positive": positive,
        "negative": negative,
    }


def frozen_warning_predictions(
    observations: Sequence[ReliabilityObservation],
) -> NDArray[np.int64] | None:
    predictions = []

    for observation in observations:
        reliability = observation["frozen_reliability"]

        if reliability == "high":
            predictions.append(NEGATIVE_CLASS)
        elif reliability == "medium" or reliability == "low":
            predictions.append(POSITIVE_CLASS)
        elif reliability == "not_analyzable":
            return None
        else:
            raise ValueError(
                f"Unknown frozen reliability category: {reliability}"
            )

    return np.asarray(predictions, dtype=int)
