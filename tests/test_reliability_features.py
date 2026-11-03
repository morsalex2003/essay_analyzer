import math

import pytest

from src.nlp.pipeline import nlp
from src.syntax import (
    ALL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    HEURISTIC_SIGNAL_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    detect_clauses,
    extract_reliability_features,
)
from src.syntax.reliability import STRONG_SIGNALS, WARNING_SIGNALS
from src.syntax.reliability_features import HEURISTIC_SIGNAL_NAMES


def _extract(
    text: str,
    include_heuristic_signals: bool = False,
) -> dict[str, int | float]:
    doc = nlp(text)
    clause_summary = detect_clauses(doc)
    return extract_reliability_features(
        doc,
        clause_summary,
        include_heuristic_signals=include_heuristic_signals,
    )


def test_feature_schema_is_deterministic():
    assert FEATURE_SCHEMA_VERSION == "1.0"
    assert len(STRUCTURAL_FEATURE_NAMES) == 44
    assert len(HEURISTIC_SIGNAL_FEATURE_NAMES) == 19
    assert len(ALL_FEATURE_NAMES) == 63
    assert ALL_FEATURE_NAMES == (
        *STRUCTURAL_FEATURE_NAMES,
        *HEURISTIC_SIGNAL_FEATURE_NAMES,
    )
    assert len(set(ALL_FEATURE_NAMES)) == len(ALL_FEATURE_NAMES)


def test_heuristic_allowlist_matches_frozen_v1_2_signals():
    frozen_signals = set(STRONG_SIGNALS)
    frozen_signals.update(WARNING_SIGNALS)

    assert set(HEURISTIC_SIGNAL_NAMES) == frozen_signals


@pytest.mark.parametrize(
    "text",
    (
        "Birds fly.",
        "I think that education is important.",
        "I want to study abroad.",
        "I sing and she dances.",
        "That’s why we should respect wildlife.",
    ),
)
def test_every_sentence_has_the_expected_keys_and_numeric_values(text):
    structural_features = _extract(text)
    all_features = _extract(text, include_heuristic_signals=True)

    assert tuple(structural_features) == STRUCTURAL_FEATURE_NAMES
    assert tuple(all_features) == ALL_FEATURE_NAMES

    for value in all_features.values():
        assert type(value) in (int, float)
        assert math.isfinite(value)


def test_empty_input_returns_safe_numeric_features():
    structural_features = _extract("")
    all_features = _extract("", include_heuristic_signals=True)

    assert tuple(structural_features) == STRUCTURAL_FEATURE_NAMES
    assert tuple(all_features) == ALL_FEATURE_NAMES
    assert structural_features["token_count"] == 0
    assert structural_features["finite_clause_count"] == 0
    assert structural_features["dependent_to_finite_ratio"] == 0.0
    assert structural_features["nonfinite_to_finite_ratio"] == 0.0
    assert structural_features["subject_to_finite_ratio"] == 0.0
    assert structural_features["max_dependency_distance"] == 0
    assert structural_features["max_tree_depth"] == 0

    for value in all_features.values():
        assert type(value) in (int, float)
        assert math.isfinite(value)


def test_simple_matrix_clause_features():
    features = _extract("Birds fly.")

    assert features["token_count"] == 3
    assert features["non_punct_token_count"] == 2
    assert features["verb_aux_token_count"] == 1
    assert features["punct_token_count"] == 1
    assert features["root_is_verb"] == 1
    assert features["root_has_subject"] == 1
    assert features["finite_clause_count"] == 1
    assert features["dependent_finite_clause_count"] == 0
    assert features["independent_finite_clause_count"] == 1


def test_dependent_clause_features():
    features = _extract("I think that education is important.")

    assert features["finite_clause_count"] == 2
    assert features["dependent_finite_clause_count"] == 1
    assert features["independent_finite_clause_count"] == 1
    assert features["ccomp_count"] == 1
    assert features["mark_count"] == 1
    assert features["dependent_to_finite_ratio"] == 0.5


def test_non_finite_predicative_features():
    features = _extract("I want to study abroad.")

    assert features["finite_clause_count"] == 1
    assert features["nonfinite_predicative_count"] == 1
    assert features["xcomp_count"] == 1
    assert features["nonfinite_to_finite_ratio"] == 1.0


def test_coordination_features_distinguish_separate_subject():
    shared_subject = _extract("I sing and dance.")
    separate_subject = _extract("I sing and she dances.")

    assert shared_subject["conj_count"] == 1
    assert shared_subject["cc_count"] == 1
    assert shared_subject["finite_clause_count"] == 1
    assert separate_subject["conj_count"] == 1
    assert separate_subject["cc_count"] == 1
    assert separate_subject["finite_clause_count"] == 2


def test_non_verbal_root_features():
    features = _extract("That’s why we should respect wildlife.")

    assert features["root_is_noun"] == 1
    assert features["root_is_verb"] == 0
    assert features["root_is_aux"] == 0
    assert features["root_is_adj"] == 0
    assert features["root_is_other"] == 0


def test_known_reliability_conflict_is_an_individual_signal():
    text = (
        "For instance, in few regions of our planet we cannot to grow "
        "plants anymore, so we need to find new sorts that can live there."
    )
    features = _extract(text, include_heuristic_signals=True)

    assert features["nonfinite_head_finite_support_conflict"] == 1
    assert features["heuristic_warning_count"] >= 1


def test_heuristic_features_exclude_final_reliability_verdict():
    features = _extract(
        "That’s why we should respect wildlife.",
        include_heuristic_signals=True,
    )

    forbidden_names = (
        "reliability",
        "reliability_level",
        "high",
        "medium",
        "low",
        "not_analyzable",
        "abstention",
    )

    for forbidden_name in forbidden_names:
        assert forbidden_name not in features


def test_doc_level_clause_summary_is_filtered_for_one_sentence_span():
    doc = nlp("Birds fly. I think that education is important.")
    clause_summary = detect_clauses(doc)
    second_sentence = list(doc.sents)[1]
    features = extract_reliability_features(
        second_sentence,
        clause_summary,
    )

    assert features["finite_clause_count"] == 2
    assert features["dependent_finite_clause_count"] == 1
    assert features["token_count"] == 7


def test_feature_names_pass_label_leakage_audit():
    forbidden_fragments = (
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

    for feature_name in ALL_FEATURE_NAMES:
        for forbidden_fragment in forbidden_fragments:
            assert forbidden_fragment not in feature_name
