"""Frozen development data for Syntax Reliability ML v2.

The development set combines labelled batches whose students are all on the
development side of the data protocol. Freezing it records, before any ML v2
work starts, exactly which students, essays, and labels it contains and how
ML v2 will be evaluated on it.

Two artifacts are produced:

- a public-safe manifest with aggregate counts and per-student status records
  (no learner text, sentence text, per-sentence labels, hashes, or paths);
- a private local lock with the SHA-256 of every frozen source file, so any
  later change to essays or labels is detected.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from spacy.language import Language

from src.alignment_experiments import sha256_file
from src.cell import load_student_roles
from src.cell_annotation import load_jsonl


DEVELOPMENT_SET_ID = "reliability_development_v1"
MANIFEST_VERSION = "1.0"
FROZEN_AT = "2026-11-04"
MANIFEST_RELATIVE_PATH = "development_set_v1.json"
LOCK_RELATIVE_PATH = "local/development_set_v1_lock.json"

# Paths are relative to data/syntax_reliability.
BATCH_DEFINITIONS = (
    {
        "batch_id": "syntax_reliability_dataset_v1",
        "essays_path": "essays_v1.jsonl",
        "annotations_path": "annotations_v1.jsonl",
        "role_assigned_at": "2026-08-17",
        "task_types": ["unrecorded"],
        "duplicate_audit_status": "not_recorded",
    },
    {
        "batch_id": "cell_reliability_pilot_v1",
        "essays_path": "external/cell/local/pilot_v1/essays.jsonl",
        "annotations_path": (
            "external/cell/local/pilot_v1/annotations_v1.jsonl"
        ),
        "role_assigned_at": "2026-08-24",
        "task_types": ["unrecorded"],
        "duplicate_audit_status": "complete",
    },
)
HOLDOUT_SPLIT_PATH = "external/cell/split_v1.json"

EXCLUSIONS = (
    {
        "name": "corpus_a_reference_v1",
        "reason": (
            "auxiliary development evidence with a different annotation "
            "history; excluded from ML v1 and kept out for comparability"
        ),
    },
    {
        "name": "cell_untouched_holdout",
        "reason": "reserved for the final ML v2 evaluation",
    },
    {
        "name": "cell_unannotated_development",
        "reason": "development students without labels",
    },
)

KNOWN_LIMITATIONS = (
    "Labels are model-assisted reference annotations, not blind human gold.",
    "222 model RELIABLE labels in the CELL pilot are unreviewed; a seeded "
    "sample of 40 changed 3 to MOSTLY_RELIABLE, which keeps the binary "
    "target.",
    "Dataset v1 has two students; student_v contributes 21 of 28 essays, so "
    "one student-grouped fold is dominated by one author.",
    "Dataset v1 was labelled under annotation guideline 1.0; the 1.2 "
    "clarifications from the CELL pilot review were not re-applied to it.",
    "Labels were judged against Clause Detection v2.1 and sentence "
    "segmentation v1; they do not transfer to v2.2 or segmentation v2 "
    "without re-judging.",
    "The CELL pilot was used for the v1.2 error analysis and to design "
    "Clause Detection v2.2 and sentence segmentation v2, so it is not "
    "independent of those components.",
)

# Labelled sentences were cut from complete essays by the frozen pipeline.
LABEL_SENTENCE_SEGMENTATION_VERSION = "1"

MANUAL_LABELS = (
    "RELIABLE",
    "MOSTLY_RELIABLE",
    "UNRELIABLE",
    "NOT_ANALYZABLE",
)
SHARED_PROVENANCE_FIELDS = (
    "spacy_version",
    "spacy_model_name",
    "spacy_model_version",
    "clause_detector_version",
)

ML_V2_EVALUATION_PROTOCOL = {
    "target": {
        "positive": ["UNRELIABLE"],
        "negative": ["RELIABLE", "MOSTLY_RELIABLE"],
        "excluded": ["NOT_ANALYZABLE"],
    },
    "parse_context": (
        "essay: each labelled sentence is taken from the frozen pipeline's "
        "parse of its complete essay (sentence segmentation v1), because "
        "labels were judged on those sentence spans"
    ),
    "cross_validation": {
        "splitter": "StratifiedGroupKFold",
        "group": "student_id",
        "fold_candidates": [5, 4, 3],
        "fold_rule": "first candidate that puts both classes in every fold",
        "shuffle": True,
        "random_state": 42,
    },
    "threshold": (
        "0.5 unless chosen inside training folds only; never tuned on "
        "out-of-fold predictions"
    ),
    "metrics": [
        "precision",
        "recall",
        "f1",
        "f2",
        "balanced_accuracy",
        "average_precision",
    ],
    "confidence_intervals": "Wilson 95% for precision and recall",
    "baseline": (
        "frozen Syntax Reliability v1.2 warning (medium or low) on the same "
        "essay-context parses"
    ),
    "breakdowns": ["batch_id"],
    "forbidden_features": [
        "student_id",
        "essay_id",
        "batch_id",
        "source_stage",
        "data_role",
        "annotation fields",
    ],
    "final_evaluation": (
        "untouched CELL holdout, only after model, features, threshold, and "
        "this procedure are frozen"
    ),
}


def _count_labels(annotations: Iterable[dict]) -> dict[str, int]:
    counts = Counter(record["manual_label"] for record in annotations)
    return {label: counts.get(label, 0) for label in MANUAL_LABELS}


def _binary_counts(label_counts: dict[str, int]) -> dict[str, int]:
    return {
        "usable": label_counts["RELIABLE"] + label_counts["MOSTLY_RELIABLE"],
        "unreliable": label_counts["UNRELIABLE"],
        "excluded_not_analyzable": label_counts["NOT_ANALYZABLE"],
    }


def _single_value(records: Iterable[dict], field: str, batch_id: str):
    values = {record[field] for record in records}
    if len(values) != 1:
        raise ValueError(
            f"{batch_id} mixes {field} values: {sorted(map(str, values))}"
        )
    return values.pop()


def check_batch_integrity(
    batch: dict,
    holdout_students: set[str],
) -> None:
    """Raise ValueError if a batch cannot be part of the development set."""

    batch_id = batch["batch_id"]
    essays = batch["essays"]
    annotations = batch["annotations"]
    essay_ids = [essay["essay_id"] for essay in essays]
    sentence_ids = [record["sentence_id"] for record in annotations]

    if len(set(essay_ids)) != len(essay_ids):
        raise ValueError(f"{batch_id} has duplicate essay IDs")

    if len(set(sentence_ids)) != len(sentence_ids):
        raise ValueError(f"{batch_id} has duplicate sentence IDs")

    essay_students = {
        essay["essay_id"]: essay["student_id"] for essay in essays
    }
    for record in annotations:
        sentence_id = record["sentence_id"]
        if record["manual_label"] not in MANUAL_LABELS:
            raise ValueError(f"{sentence_id} has no final manual label")
        if essay_students.get(record["essay_id"]) != record["student_id"]:
            raise ValueError(f"{sentence_id} does not match a frozen essay")

    for essay in essays:
        if essay["data_status"] != "development":
            raise ValueError(f"{essay['essay_id']} is not development data")

    leaked = set(essay_students.values()) & holdout_students
    if leaked:
        raise ValueError(f"{batch_id} contains holdout students")

    for field in SHARED_PROVENANCE_FIELDS:
        _single_value([*essays, *annotations], field, batch_id)


def essay_context_mismatches(
    essays: Sequence[dict],
    annotations: Sequence[dict],
    pipeline: Language,
) -> list[str]:
    """Return sentence IDs not reproduced by segmenting their essay.

    Sentence indexes are 1-based. Surrounding whitespace is ignored because
    some ledgers stored sentence text without the trailing space.
    """

    essay_texts = {
        essay["essay_id"]: essay["original_text"] for essay in essays
    }
    sentences_by_essay = {}
    mismatches = []

    for record in annotations:
        essay_id = record["essay_id"]
        if essay_id not in sentences_by_essay:
            document = pipeline(essay_texts[essay_id])
            sentences_by_essay[essay_id] = [
                sentence.text.strip() for sentence in document.sents
            ]

        sentences = sentences_by_essay[essay_id]
        position = record["sentence_index"] - 1
        if not (
            0 <= position < len(sentences)
            and sentences[position] == record["original_text"].strip()
        ):
            mismatches.append(record["sentence_id"])

    return mismatches


def summarize_batch(batch: dict) -> dict:
    """Return public-safe aggregate counts and provenance for one batch."""

    essays = batch["essays"]
    annotations = batch["annotations"]
    label_counts = _count_labels(annotations)
    all_records = [*essays, *annotations]
    provenance = {
        field: _single_value(all_records, field, batch["batch_id"])
        for field in SHARED_PROVENANCE_FIELDS
    }
    provenance["reliability_versions"] = sorted(
        {record["reliability_version"] for record in all_records}
    )
    provenance["sentence_segmentation_version"] = (
        LABEL_SENTENCE_SEGMENTATION_VERSION
    )

    return {
        "batch_id": batch["batch_id"],
        "source_stages": sorted({essay["source_stage"] for essay in essays}),
        "counts": {
            "students": len({essay["student_id"] for essay in essays}),
            "essays": len(essays),
            "sentences": len(annotations),
        },
        "label_counts": label_counts,
        "binary_counts": _binary_counts(label_counts),
        "annotation": {
            "guideline_versions": sorted(
                {
                    record["annotation_guideline_version"]
                    for record in annotations
                }
            ),
            "origins": dict(
                sorted(
                    Counter(
                        record["annotation_origin"] for record in annotations
                    ).items()
                )
            ),
            "annotator_ids": sorted(
                {record["annotator_id"] for record in annotations}
            ),
            "adjudicated": sum(
                record.get("adjudication_status") == "adjudicated"
                for record in annotations
            ),
        },
        "provenance": provenance,
    }


def student_records(batch: dict) -> list[dict]:
    """Return one protocol status record per student in a batch."""

    essays_per_student = Counter()
    stages_per_student = {}
    for essay in batch["essays"]:
        student_id = essay["student_id"]
        essays_per_student[student_id] += 1
        stages_per_student.setdefault(student_id, set()).add(
            essay["source_stage"]
        )

    records = []

    for student_id in sorted(essays_per_student):
        records.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "batch_id": batch["batch_id"],
                "student_id": student_id,
                "data_role": "development",
                "role_assigned_at": batch["role_assigned_at"],
                # A student can span stages, so this is a list rather than
                # the protocol's single source_stage.
                "source_stages": sorted(stages_per_student[student_id]),
                "essay_count": essays_per_student[student_id],
                "task_types": batch["task_types"],
                "collection_status": "complete",
                "annotation_status": "frozen",
                "freeze_status": "annotation_frozen",
                "raw_git_status": "ignored_untracked",
                "duplicate_audit_status": batch["duplicate_audit_status"],
                "holdout_opened_at": None,
                "holdout_open_reason": None,
            }
        )

    return records


def _totals(batch_summaries: Sequence[dict]) -> dict:
    counts = Counter()
    label_counts = Counter()

    for summary in batch_summaries:
        counts.update(summary["counts"])
        label_counts.update(summary["label_counts"])

    labels = {label: label_counts[label] for label in MANUAL_LABELS}
    return {
        "counts": dict(counts),
        "label_counts": labels,
        "binary_counts": _binary_counts(labels),
    }


def build_development_manifest(
    batches: Sequence[dict],
    holdout_students: set[str],
    exclusions: Sequence[dict],
    known_limitations: Sequence[str],
) -> dict:
    """Build the public-safe development set manifest."""

    seen_students = {}
    for batch in batches:
        check_batch_integrity(batch, holdout_students)
        for essay in batch["essays"]:
            owner = seen_students.setdefault(
                essay["student_id"], batch["batch_id"]
            )
            if owner != batch["batch_id"]:
                raise ValueError(
                    f"{essay['student_id']} appears in two batches"
                )

    batch_summaries = [summarize_batch(batch) for batch in batches]
    students = []
    for batch in batches:
        students.extend(student_records(batch))

    return {
        "manifest_version": MANIFEST_VERSION,
        "development_set_id": DEVELOPMENT_SET_ID,
        "frozen_at": FROZEN_AT,
        "data_role": "development",
        "holdout_opened": False,
        "batches": batch_summaries,
        "totals": _totals(batch_summaries),
        "students": students,
        "exclusions": list(exclusions),
        "ml_v2_evaluation_protocol": ML_V2_EVALUATION_PROTOCOL,
        "known_limitations": list(known_limitations),
        "privacy": {
            "learner_text_in_manifest": False,
            "sentence_labels_in_manifest": False,
            "hashes_in_manifest": False,
            "private_paths_in_manifest": False,
            "raw_git_status": "ignored_untracked",
        },
    }


def load_batches(dataset_directory: Path) -> list[dict]:
    """Load every frozen batch from its local private files."""

    batches = []
    for definition in BATCH_DEFINITIONS:
        batch = dict(definition)
        batch["essays"] = load_jsonl(
            dataset_directory / definition["essays_path"]
        )
        batch["annotations"] = load_jsonl(
            dataset_directory / definition["annotations_path"]
        )
        batches.append(batch)
    return batches


def load_holdout_students(dataset_directory: Path) -> set[str]:
    roles = load_student_roles(dataset_directory / HOLDOUT_SPLIT_PATH)
    return {
        student_id
        for student_id, role in roles.items()
        if role == "untouched_holdout"
    }


def build_lock(dataset_directory: Path) -> dict:
    """Return the private lock: SHA-256 and line count of each source file."""

    files = {}
    for definition in BATCH_DEFINITIONS:
        for key in ("essays_path", "annotations_path"):
            relative_path = definition[key]
            path = dataset_directory / relative_path
            files[relative_path] = {
                "sha256": sha256_file(path),
                "records": len(load_jsonl(path)),
            }

    return {
        "development_set_id": DEVELOPMENT_SET_ID,
        "frozen_at": FROZEN_AT,
        "files": files,
    }


def lock_mismatches(lock: dict, dataset_directory: Path) -> list[str]:
    """Return the source files whose content differs from the lock."""

    current = build_lock(dataset_directory)["files"]
    return sorted(
        relative_path
        for relative_path in set(lock["files"]) | set(current)
        if lock["files"].get(relative_path) != current.get(relative_path)
    )
