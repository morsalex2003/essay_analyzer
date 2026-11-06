"""Deterministic, label-free selection helpers for CELL Reliability Pilot v1."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from src.cell import CellManifestRecord


CELL_PILOT_SEED = 20260824
CELL_PILOT_STUDENT_COUNT = 35
CELL_PILOT_SENTENCES_PER_STUDENT = 12
CELL_PILOT_CONTROL_TARGET_PER_STUDENT = 3

STRUCTURAL_CATEGORIES = (
    "coordination",
    "finite_subordination",
    "relative_clause",
    "ccomp_xcomp",
    "nonfinite_predicative",
    "modal_auxiliary",
    "punctuation_linked",
    "simple_control",
)


@dataclass(frozen=True)
class PilotSentence:
    """A parsed sentence candidate with text-free selection features."""

    essay_record: CellManifestRecord
    sentence_index: int
    text: str
    categories: tuple[str, ...]
    structural_features: Mapping[str, int | float]


def stable_rank(value: str, seed: int = CELL_PILOT_SEED) -> str:
    """Return a stable rank that does not depend on learner text."""

    material = f"{seed}:{value}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def essay_count_bin(essay_count: int) -> str:
    if essay_count == 1:
        return "1"
    if essay_count <= 3:
        return "2-3"
    if essay_count <= 7:
        return "4-7"
    return "8+"


def select_pilot_students(
    records: Sequence[CellManifestRecord],
    student_roles: Mapping[str, str],
    *,
    student_count: int = CELL_PILOT_STUDENT_COUNT,
    seed: int = CELL_PILOT_SEED,
    role: str = "development",
    excluded_students: Collection[str] = (),
) -> list[str]:
    """Select students of one split role using structural metadata only.

    `excluded_students` removes students that an earlier batch already
    used, such as a holdout batch that has been opened.
    """

    excluded = set(excluded_students)
    role_records = [
        record
        for record in records
        if student_roles.get(record.student_id) == role
        and record.student_id not in excluded
    ]
    essay_counts = Counter(record.student_id for record in role_records)
    student_features: dict[str, set[str]] = {}
    for record in role_records:
        student_features.setdefault(record.student_id, set()).update(
            (
                f"period:{record.period}",
                f"course:{record.course_id}",
                f"essay_bin:{essay_count_bin(essay_counts[record.student_id])}",
            )
        )

    if student_count > len(student_features):
        raise ValueError(f"student_count exceeds available {role} students")

    feature_frequency = Counter(
        feature
        for features in student_features.values()
        for feature in features
    )
    selected: list[str] = []
    covered: set[str] = set()
    remaining = set(student_features)
    while len(selected) < student_count:
        student_id = min(
            remaining,
            key=lambda candidate: (
                -sum(
                    1 / feature_frequency[feature]
                    for feature in student_features[candidate] - covered
                ),
                stable_rank(candidate, seed),
                candidate,
            ),
        )
        selected.append(student_id)
        covered.update(student_features[student_id])
        remaining.remove(student_id)

    return selected


def assign_course_targets(
    selected_students: Sequence[str],
    records: Sequence[CellManifestRecord],
    *,
    seed: int = CELL_PILOT_SEED,
) -> dict[str, str]:
    """Match distinct courses to selected students when coverage permits."""

    selected = set(selected_students)
    students_by_course: dict[str, set[str]] = {}
    for record in records:
        if record.student_id in selected:
            students_by_course.setdefault(record.course_id, set()).add(
                record.student_id
            )

    course_for_student: dict[str, str] = {}

    def match(course_id: str, visited: set[str]) -> bool:
        candidates = sorted(
            students_by_course[course_id],
            key=lambda student_id: (
                stable_rank(f"course:{course_id}:{student_id}", seed),
                student_id,
            ),
        )
        for student_id in candidates:
            if student_id in visited:
                continue
            visited.add(student_id)
            previous_course = course_for_student.get(student_id)
            if previous_course is None or match(previous_course, visited):
                course_for_student[student_id] = course_id
                return True
        return False

    ordered_courses = sorted(
        students_by_course,
        key=lambda course_id: (
            len(students_by_course[course_id]),
            stable_rank(f"course-order:{course_id}", seed),
            course_id,
        ),
    )
    for course_id in ordered_courses:
        match(course_id, set())

    return course_for_student


def sentence_categories(
    features: Mapping[str, int | float],
) -> tuple[str, ...]:
    """Map parser-derived structural features to pilot coverage categories."""

    categories = []
    if features["conj_count"] > 0 or features["cc_count"] > 0:
        categories.append("coordination")
    if features["dependent_finite_clause_count"] > 0:
        categories.append("finite_subordination")
    if features["relcl_count"] > 0:
        categories.append("relative_clause")
    if features["ccomp_count"] > 0 or features["xcomp_count"] > 0:
        categories.append("ccomp_xcomp")
    if features["nonfinite_predicative_count"] > 0:
        categories.append("nonfinite_predicative")
    if (
        features["modal_count"] > 0
        or features["aux_count"] > 0
        or features["cop_count"] > 0
    ):
        categories.append("modal_auxiliary")
    if (
        features["comma_count"] > 0
        or features["semicolon_count"] > 0
        or features["colon_count"] > 0
        or features["parataxis_count"] > 0
    ):
        categories.append("punctuation_linked")

    if (
        features["finite_clause_count"] == 1
        and features["dependent_finite_clause_count"] == 0
        and features["nonfinite_predicative_count"] == 0
        and features["conj_count"] == 0
        and features["parataxis_count"] == 0
        and 5 <= features["non_punct_token_count"] <= 30
    ):
        categories.append("simple_control")
    return tuple(categories)


def select_pilot_essay(
    candidates_by_essay: Mapping[str, Sequence[PilotSentence]],
    *,
    seed: int = CELL_PILOT_SEED,
) -> str:
    """Choose one complete essay with broad structures and usable controls."""

    if not candidates_by_essay:
        raise ValueError("A selected student has no parsed essay candidates")

    def essay_key(essay_id: str) -> tuple[int, int, int, str]:
        sentences = candidates_by_essay[essay_id]
        coverage = {
            category
            for sentence in sentences
            for category in sentence.categories
        }
        control_count = sum(
            "simple_control" in sentence.categories for sentence in sentences
        )
        return (
            -min(control_count, CELL_PILOT_CONTROL_TARGET_PER_STUDENT),
            -len(coverage),
            -min(len(sentences), CELL_PILOT_SENTENCES_PER_STUDENT),
            stable_rank(essay_id, seed),
        )

    return min(candidates_by_essay, key=essay_key)


def select_pilot_sentences(
    candidates: Sequence[PilotSentence],
    *,
    sentence_count: int = CELL_PILOT_SENTENCES_PER_STUDENT,
    control_target: int = CELL_PILOT_CONTROL_TARGET_PER_STUDENT,
    seed: int = CELL_PILOT_SEED,
) -> list[PilotSentence]:
    """Select diverse structural cases plus explicit simple controls."""

    selected: list[PilotSentence] = []
    selected_keys: set[tuple[str, int]] = set()

    def candidate_key(sentence: PilotSentence) -> tuple[str, int]:
        return sentence.essay_record.record_id, sentence.sentence_index

    def add(sentence: PilotSentence) -> None:
        key = candidate_key(sentence)
        if key not in selected_keys and len(selected) < sentence_count:
            selected.append(sentence)
            selected_keys.add(key)

    ranked = sorted(
        candidates,
        key=lambda sentence: (
            stable_rank(
                f"{sentence.essay_record.archive_path}:{sentence.sentence_index}",
                seed,
            ),
            sentence.sentence_index,
        ),
    )
    controls = [
        sentence
        for sentence in ranked
        if "simple_control" in sentence.categories
    ]
    for sentence in controls[:control_target]:
        add(sentence)

    for category in STRUCTURAL_CATEGORIES:
        if category == "simple_control":
            continue
        for sentence in ranked:
            if category in sentence.categories:
                add(sentence)
                break

    category_counts = Counter(
        category for sentence in selected for category in sentence.categories
    )
    while len(selected) < sentence_count:
        remaining = [
            sentence
            for sentence in ranked
            if candidate_key(sentence) not in selected_keys
        ]
        if not remaining:
            break
        next_sentence = min(
            remaining,
            key=lambda sentence: (
                sum(category_counts[category] for category in sentence.categories),
                stable_rank(
                    f"fill:{sentence.essay_record.archive_path}:"
                    f"{sentence.sentence_index}",
                    seed,
                ),
            ),
        )
        add(next_sentence)
        category_counts.update(next_sentence.categories)

    return sorted(selected, key=lambda sentence: sentence.sentence_index)
