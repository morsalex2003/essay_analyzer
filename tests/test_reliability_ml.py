import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cell_annotation import SentenceParser
from src.cell_pilot_evaluation import heuristic_verdicts
from src.nlp.pipeline import nlp
from src.syntax.reliability_features import (
    ALL_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
)
from src.syntax.reliability_ml import (
    CLASSIFICATION_THRESHOLD,
    MODEL_A_FEATURE_NAMES,
    MODEL_B_FEATURE_NAMES,
    build_feature_matrix,
    build_logistic_pipeline,
    calculate_binary_metrics,
    feature_name_leakage,
    frozen_warning_predictions,
    observation_groups,
    observation_targets,
    prepare_essay_context_observations,
    prepare_ml_observations,
    run_grouped_oof,
    select_grouped_splits,
    target_from_manual_label,
)
from src.syntax.reliability_ml_errors import (
    combine_system_predictions,
    fold_linear_contributions,
    partition_system_intersections,
)


def _feature_values(seed: int) -> dict[str, int | float]:
    features = {}

    for feature_index, feature_name in enumerate(ALL_FEATURE_NAMES):
        features[feature_name] = seed + feature_index / 100

    return features


def _synthetic_observations():
    observations = []

    for group_index in range(6):
        for target in (0, 1):
            observation_index = group_index * 2 + target
            observations.append(
                {
                    "sentence_id": f"sentence_{observation_index}",
                    "essay_id": f"essay_{group_index}",
                    "student_id": "student_test",
                    "manual_label": (
                        "RELIABLE" if target == 0 else "UNRELIABLE"
                    ),
                    "target": target,
                    "features": _feature_values(observation_index),
                    "frozen_reliability": (
                        "high" if target == 0 else "medium"
                    ),
                }
            )

    return observations


@pytest.mark.parametrize(
    "manual_label, expected_target",
    (
        ("RELIABLE", 0),
        ("MOSTLY_RELIABLE", 0),
        ("UNRELIABLE", 1),
        ("NOT_ANALYZABLE", None),
    ),
)
def test_target_mapping(manual_label, expected_target):
    assert target_from_manual_label(manual_label) == expected_target


def test_unknown_target_label_is_rejected():
    with pytest.raises(ValueError):
        target_from_manual_label("UNKNOWN")


def test_feature_matrix_uses_deterministic_requested_order():
    observations = _synthetic_observations()[:2]
    feature_names = (
        "finite_clause_count",
        "token_count",
        "heuristic_warning_count",
    )
    matrix = build_feature_matrix(observations, feature_names)

    assert matrix.shape == (2, 3)
    assert matrix[0].tolist() == [
        observations[0]["features"][feature_name]
        for feature_name in feature_names
    ]


def test_model_feature_sets_match_frozen_schema():
    assert MODEL_A_FEATURE_NAMES == STRUCTURAL_FEATURE_NAMES
    assert MODEL_B_FEATURE_NAMES == ALL_FEATURE_NAMES
    assert len(MODEL_A_FEATURE_NAMES) == 44
    assert len(MODEL_B_FEATURE_NAMES) == 63


def test_observation_groups_are_metadata_not_features():
    observations = _synthetic_observations()
    groups = observation_groups(observations)

    assert groups[0] == "essay_0"
    assert groups[-1] == "essay_5"
    assert "essay_id" not in MODEL_B_FEATURE_NAMES
    assert "student_id" not in MODEL_B_FEATURE_NAMES


def test_grouped_splits_preserve_essays_and_assign_each_row_once():
    observations = _synthetic_observations()
    targets = observation_targets(observations)
    groups = observation_groups(observations)
    split_count, splits = select_grouped_splits(
        targets,
        groups,
        candidate_splits=(3,),
    )
    validation_assignments = np.zeros(len(observations), dtype=int)

    assert split_count == 3

    for train_indexes, validation_indexes in splits:
        train_groups = set(groups[train_indexes])
        validation_groups = set(groups[validation_indexes])

        assert train_groups.isdisjoint(validation_groups)
        assert set(targets[validation_indexes]) == {0, 1}
        validation_assignments[validation_indexes] += 1

    assert np.all(validation_assignments == 1)


def test_scaler_is_inside_logistic_regression_pipeline():
    pipeline = build_logistic_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(
        pipeline.named_steps["classifier"],
        LogisticRegression,
    )


def test_grouped_oof_returns_one_prediction_per_observation():
    observations = _synthetic_observations()
    targets = observation_targets(observations)
    groups = observation_groups(observations)
    _, splits = select_grouped_splits(
        targets,
        groups,
        candidate_splits=(3,),
    )
    result = run_grouped_oof(
        observations,
        MODEL_A_FEATURE_NAMES,
        splits,
    )
    oof_records = result["oof_records"]

    assert len(oof_records) == len(observations)
    assert {record["sentence_id"] for record in oof_records} == {
        observation["sentence_id"] for observation in observations
    }
    assert all(record["fold"] in (1, 2, 3) for record in oof_records)
    assert all("manual_label" in record for record in oof_records)
    assert all("original_text" not in record for record in oof_records)


def test_grouped_oof_can_retain_fold_specific_estimators():
    observations = _synthetic_observations()
    targets = observation_targets(observations)
    groups = observation_groups(observations)
    _, splits = select_grouped_splits(
        targets,
        groups,
        candidate_splits=(3,),
    )
    result = run_grouped_oof(
        observations,
        MODEL_B_FEATURE_NAMES,
        splits,
        retain_estimators=True,
    )

    assert len(result["fold_estimators"]) == 3
    frozen_predictions = np.zeros(len(observations), dtype=int)
    comparison_records = combine_system_predictions(
        observations,
        result["oof_records"],
        result["oof_records"],
        frozen_predictions,
    )

    for record in comparison_records:
        contributions = fold_linear_contributions(
            record,
            MODEL_B_FEATURE_NAMES,
            result["fold_estimators"],
        )

        assert "toward_unreliable" in contributions
        assert "toward_usable" in contributions


def test_prediction_intersections_partition_both_target_classes():
    observations = _synthetic_observations()[:8]
    model_a_records = []
    model_b_records = []
    frozen_predictions = np.asarray((0, 1, 0, 1, 0, 1, 0, 1))
    model_b_predictions = (0, 0, 1, 1, 0, 0, 1, 1)

    for index, observation in enumerate(observations):
        base_record = {
            "sentence_id": observation["sentence_id"],
            "essay_id": observation["essay_id"],
            "student_id": observation["student_id"],
            "manual_label": observation["manual_label"],
            "y_true": observation["target"],
            "unreliable_probability": 0.5,
            "fold": 1,
        }
        model_a_record = dict(base_record)
        model_a_record["y_pred"] = observation["target"]
        model_a_records.append(model_a_record)
        model_b_record = dict(base_record)
        model_b_record["y_pred"] = model_b_predictions[index]
        model_b_records.append(model_b_record)

    comparison_records = combine_system_predictions(
        observations,
        model_a_records,
        model_b_records,
        frozen_predictions,
    )
    intersections = partition_system_intersections(comparison_records)
    unreliable_total = 0
    usable_total = 0

    for records in intersections["unreliable"].values():
        unreliable_total += len(records)

    for records in intersections["usable"].values():
        usable_total += len(records)

    assert unreliable_total == 4
    assert usable_total == 4
    assert len(comparison_records) == len(observations)
    assert all("original_text" not in record for record in comparison_records)


def test_error_analysis_does_not_change_classification_threshold():
    assert CLASSIFICATION_THRESHOLD == 0.5


def test_metrics_use_unreliable_as_positive_class():
    targets = np.asarray((0, 0, 1, 1), dtype=int)
    predictions = np.asarray((0, 1, 0, 1), dtype=int)
    metrics = calculate_binary_metrics(targets, predictions)

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_discard_rate"] == 0.5
    assert metrics["miss_rate"] == 0.5


def test_feature_names_have_no_label_or_metadata_leakage():
    assert feature_name_leakage(MODEL_A_FEATURE_NAMES) == []
    assert feature_name_leakage(MODEL_B_FEATURE_NAMES) == []
    assert feature_name_leakage(("manual_label",)) == ["manual_label"]


def test_not_analyzable_is_excluded_during_preparation():
    records = (
        {
            "sentence_id": "reliable",
            "essay_id": "essay_1",
            "student_id": "student_1",
            "manual_label": "RELIABLE",
            "original_text": "Birds fly.",
        },
        {
            "sentence_id": "mostly",
            "essay_id": "essay_1",
            "student_id": "student_1",
            "manual_label": "MOSTLY_RELIABLE",
            "original_text": "Cats sleep.",
        },
        {
            "sentence_id": "unreliable",
            "essay_id": "essay_2",
            "student_id": "student_2",
            "manual_label": "UNRELIABLE",
            "original_text": "Dogs run.",
        },
        {
            "sentence_id": "excluded",
            "essay_id": "essay_2",
            "student_id": "student_2",
            "manual_label": "NOT_ANALYZABLE",
            "original_text": "Project overview.",
        },
    )
    observations = prepare_ml_observations(records)

    assert len(observations) == 3
    assert [observation["target"] for observation in observations] == [
        0,
        0,
        1,
    ]
    assert all(
        observation["sentence_id"] != "excluded"
        for observation in observations
    )


def test_frozen_warning_mapping_uses_historical_medium_low_semantics():
    observations = _synthetic_observations()[:3]
    observations[0]["frozen_reliability"] = "high"
    observations[1]["frozen_reliability"] = "medium"
    observations[2]["frozen_reliability"] = "low"

    predictions = frozen_warning_predictions(observations)

    assert predictions.tolist() == [0, 1, 1]


def test_frozen_not_analyzable_stops_binary_mapping():
    observations = _synthetic_observations()[:1]
    observations[0]["frozen_reliability"] = "not_analyzable"

    assert frozen_warning_predictions(observations) is None


ESSAY_TEXT = (
    "The students wrote essays. They was tired because the test long. "
    "Teachers who care help them."
)


def _essay_context_records():
    essay = {
        "essay_id": "essay_1",
        "student_id": "student_1",
        "original_text": ESSAY_TEXT,
    }
    sentences = [sentence.text for sentence in nlp(ESSAY_TEXT).sents]
    labels = ("RELIABLE", "UNRELIABLE", "NOT_ANALYZABLE")
    assert len(sentences) == len(labels)
    annotations = []
    for index, (text, label) in enumerate(zip(sentences, labels), start=1):
        annotations.append(
            {
                "sentence_id": f"essay_1_s{index}",
                "essay_id": "essay_1",
                "student_id": "student_1",
                "sentence_index": index,
                # Ledgers may store the sentence without its trailing space.
                "original_text": text.strip(),
                "manual_label": label,
            }
        )
    return essay, annotations


def test_essay_context_observations_use_essay_spans():
    essay, annotations = _essay_context_records()

    observations = prepare_essay_context_observations([essay], annotations)

    assert [observation["sentence_id"] for observation in observations] == [
        "essay_1_s1",
        "essay_1_s2",
    ]
    assert [observation["target"] for observation in observations] == [0, 1]
    assert set(observations[0]["features"]) == set(ALL_FEATURE_NAMES)

    parser = SentenceParser(nlp, {"essay_1": essay})
    verdicts = heuristic_verdicts(annotations[:2], parser)
    for observation in observations:
        assert observation["frozen_reliability"] == verdicts[
            observation["sentence_id"]
        ]


def test_essay_context_rejects_sentence_not_in_segmentation():
    essay, annotations = _essay_context_records()
    annotations[0]["original_text"] = "A different sentence."

    with pytest.raises(ValueError, match="does not reproduce"):
        prepare_essay_context_observations([essay], annotations)

