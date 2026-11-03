import hashlib
import json
from pathlib import Path

from src.cell import CellManifestRecord, parse_cell_filename
from src.cell_pilot import (
    PilotSentence,
    assign_course_targets,
    select_pilot_sentences,
    select_pilot_students,
    sentence_categories,
)


def make_record(student_id: str, period: str, course: str, index: int):
    filename = f"{period}_{course}_t01_a01_q01_{student_id}.txt"
    return CellManifestRecord(
        archive_path=f"{period}/folder-{index}/{filename}",
        filename=filename,
        sha256=hashlib.sha256(filename.encode()).hexdigest(),
        parsed_filename=parse_cell_filename(filename),
        duplicate_metadata={},
    )


def base_features(**updates):
    features = {
        "conj_count": 0,
        "cc_count": 0,
        "dependent_finite_clause_count": 0,
        "relcl_count": 0,
        "ccomp_count": 0,
        "xcomp_count": 0,
        "nonfinite_predicative_count": 0,
        "modal_count": 0,
        "aux_count": 0,
        "cop_count": 0,
        "comma_count": 0,
        "semicolon_count": 0,
        "colon_count": 0,
        "parataxis_count": 0,
        "finite_clause_count": 1,
        "non_punct_token_count": 10,
    }
    features.update(updates)
    return features


def test_student_selection_is_development_only_and_deterministic():
    records = []
    roles = {}
    for index in range(12):
        student_id = f"s{index:05d}"
        roles[student_id] = "untouched_holdout" if index == 0 else "development"
        records.append(
            make_record(
                student_id,
                "2018_01" if index % 2 else "2019_01",
                "engle102f" if index % 3 else "edue220f",
                index,
            )
        )

    first = select_pilot_students(records, roles, student_count=5)
    second = select_pilot_students(list(reversed(records)), roles, student_count=5)

    assert first == second
    assert "s00000" not in first
    assert len(first) == 5


def test_sentence_categories_include_structures_and_simple_controls():
    assert sentence_categories(base_features()) == ("simple_control",)
    categories = sentence_categories(
        base_features(
            conj_count=1,
            dependent_finite_clause_count=1,
            relcl_count=1,
            xcomp_count=1,
            nonfinite_predicative_count=1,
            modal_count=1,
            comma_count=1,
            finite_clause_count=2,
        )
    )
    assert set(categories) == {
        "coordination",
        "finite_subordination",
        "relative_clause",
        "ccomp_xcomp",
        "nonfinite_predicative",
        "modal_auxiliary",
        "punctuation_linked",
    }


def test_course_targets_maximize_distinct_metadata_coverage():
    records = [
        make_record("s00001", "2018_01", "course1", 1),
        make_record("s00001", "2018_01", "course2", 2),
        make_record("s00002", "2019_01", "course1", 3),
        make_record("s00003", "2019_01", "course3", 4),
    ]

    targets = assign_course_targets(
        ["s00001", "s00002", "s00003"],
        records,
    )

    assert set(targets.values()) == {"course1", "course2", "course3"}
    assert len(targets) == 3


def test_sentence_selection_retains_controls_and_diverse_structures():
    record = make_record("s12345", "2018_01", "engle102f", 1)
    feature_rows = [
        base_features(),
        base_features(),
        base_features(),
        base_features(conj_count=1, finite_clause_count=2),
        base_features(relcl_count=1, dependent_finite_clause_count=1),
        base_features(xcomp_count=1, nonfinite_predicative_count=1),
        base_features(modal_count=1, comma_count=1, finite_clause_count=2),
        base_features(semicolon_count=1, finite_clause_count=2),
    ]
    candidates = [
        PilotSentence(
            essay_record=record,
            sentence_index=index,
            text=f"Sentence {index}.",
            categories=sentence_categories(features),
            structural_features=features,
        )
        for index, features in enumerate(feature_rows, start=1)
    ]
    selected = select_pilot_sentences(candidates, sentence_count=8)
    selected_categories = {
        category for sentence in selected for category in sentence.categories
    }

    assert len(selected) == 8
    assert sum("simple_control" in item.categories for item in selected) >= 3
    assert "coordination" in selected_categories
    assert "relative_clause" in selected_categories
    assert "ccomp_xcomp" in selected_categories


def test_frozen_pilot_report_is_public_safe_and_in_target_range():
    report_path = (
        Path(__file__).parents[1]
        / "data/syntax_reliability/external/cell/pilot_v1_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["data_role"] == "development"
    assert report["holdout_opened"] is False
    assert 30 <= report["counts"]["students"] <= 40
    assert 300 <= report["counts"]["sentences"] <= 500
    assert report["structural_coverage"]["simple_control"] >= 100
    assert all(
        report["structural_coverage"][category] > 0
        for category in (
            "coordination",
            "finite_subordination",
            "relative_clause",
            "ccomp_xcomp",
            "nonfinite_predicative",
            "modal_auxiliary",
            "punctuation_linked",
        )
    )
    privacy_flags = {
        key: value
        for key, value in report["privacy"].items()
        if key != "raw_git_status"
    }
    assert all(not value for value in privacy_flags.values())
