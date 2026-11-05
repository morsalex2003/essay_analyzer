"""Prepare a private CELL reliability annotation batch.

Defaults reproduce CELL Reliability Pilot v1 from development students.
With --role untouched_holdout it prepares the holdout batch with the same
procedure; only counts and structural coverage are printed, never text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import en_core_web_sm
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import (  # noqa: E402
    CellCorpus,
    extract_cell_learner_text,
    load_cell_manifest,
    load_student_roles,
)
from src.cell_pilot import (  # noqa: E402
    CELL_PILOT_SEED,
    CELL_PILOT_SENTENCES_PER_STUDENT,
    CELL_PILOT_STUDENT_COUNT,
    STRUCTURAL_CATEGORIES,
    PilotSentence,
    assign_course_targets,
    select_pilot_essay,
    select_pilot_sentences,
    select_pilot_students,
    sentence_categories,
)
from src.nlp.pipeline import nlp  # noqa: E402
from src.syntax.clauses import detect_clauses  # noqa: E402
from src.syntax.reliability_features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    extract_reliability_features,
)


SOURCE_STAGE = "cell_reliability_pilot_v1"
SPLIT_ROLES = ("development", "untouched_holdout")
# Pilot v1 selection is frozen on Clause Detection v2.1.
CLAUSE_DETECTOR_VERSION = "2.1"
RELIABILITY_GUIDELINE_VERSION = "1.2"


def parse_arguments() -> argparse.Namespace:
    base = PROJECT_ROOT / "data/syntax_reliability/external/cell"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=base / "local/CELL_manifest.csv",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=base / "local/corpus",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=base / "split_v1.json",
    )
    parser.add_argument(
        "--local-output",
        type=Path,
        default=base / "local/pilot_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=base / "pilot_v1_report.json",
    )
    parser.add_argument(
        "--students",
        type=int,
        default=CELL_PILOT_STUDENT_COUNT,
    )
    parser.add_argument(
        "--sentences-per-student",
        type=int,
        default=CELL_PILOT_SENTENCES_PER_STUDENT,
    )
    parser.add_argument("--role", choices=SPLIT_ROLES, default="development")
    parser.add_argument("--seed", type=int, default=CELL_PILOT_SEED)
    parser.add_argument("--source-stage", default=SOURCE_STAGE)
    return parser.parse_args()


def jsonl_write(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def provenance() -> dict[str, str]:
    repository_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    return {
        "repository_commit": repository_commit,
        "spacy_version": spacy.__version__,
        "spacy_model_name": "en_core_web_sm",
        "spacy_model_version": en_core_web_sm.__version__,
        "clause_detector_version": CLAUSE_DETECTOR_VERSION,
        "reliability_version": RELIABILITY_GUIDELINE_VERSION,
        "structural_feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def main() -> None:
    arguments = parse_arguments()
    records = load_cell_manifest(arguments.manifest)
    roles = load_student_roles(arguments.split)
    selected_students = select_pilot_students(
        records,
        roles,
        student_count=arguments.students,
        seed=arguments.seed,
        role=arguments.role,
    )
    selected_student_set = set(selected_students)
    selected_records = [
        record
        for record in records
        if record.student_id in selected_student_set
    ]
    if any(
        roles[record.student_id] != arguments.role
        for record in selected_records
    ):
        raise RuntimeError("Selection crossed the development/holdout split")

    parsed_by_student: dict[
        str, dict[str, list[PilotSentence]]
    ] = defaultdict(dict)
    text_by_record_id: dict[str, str] = {}
    with CellCorpus(arguments.corpus) as corpus:
        for record in selected_records:
            raw_bytes = corpus.read_bytes(record)
            learner_text = extract_cell_learner_text(
                raw_bytes.decode("utf-8")
            )
            text_by_record_id[record.record_id] = learner_text
            document = nlp(learner_text)
            clause_summary = detect_clauses(
                document,
                detector_version=CLAUSE_DETECTOR_VERSION,
            )
            candidates = []
            for sentence_index, sentence in enumerate(document.sents, start=1):
                features = extract_reliability_features(
                    sentence,
                    clause_summary,
                    include_heuristic_signals=False,
                )
                candidates.append(
                    PilotSentence(
                        essay_record=record,
                        sentence_index=sentence_index,
                        text=sentence.text,
                        categories=sentence_categories(features),
                        structural_features=features,
                    )
                )
            parsed_by_student[record.student_id][record.record_id] = candidates

    selected_sentences = []
    selected_essay_records = []
    course_targets = assign_course_targets(
        selected_students,
        selected_records,
        seed=arguments.seed,
    )
    for student_id in selected_students:
        candidates_by_essay = parsed_by_student[student_id]
        course_target = course_targets.get(student_id)
        if course_target is not None:
            candidates_by_essay = {
                essay_id: candidates
                for essay_id, candidates in candidates_by_essay.items()
                if candidates[0].essay_record.course_id == course_target
            }
        essay_id = select_pilot_essay(
            candidates_by_essay,
            seed=arguments.seed,
        )
        essay_candidates = candidates_by_essay[essay_id]
        selected_essay_records.append(essay_candidates[0].essay_record)
        selected_sentences.extend(
            select_pilot_sentences(
                essay_candidates,
                sentence_count=arguments.sentences_per_student,
                seed=arguments.seed,
            )
        )

    provenance_fields = provenance()
    essay_rows = []
    for record in selected_essay_records:
        learner_text = text_by_record_id[record.record_id]
        essay_rows.append(
            {
                "schema_version": "1.0",
                "essay_id": record.record_id,
                "student_id": record.student_id,
                "task_id": record.parsed_filename.task_id,
                "prompt_id": record.parsed_filename.question_id,
                "response_family_id": None,
                "source_stage": arguments.source_stage,
                "data_status": arguments.role,
                "original_text": learner_text,
                "original_text_sha256": hashlib.sha256(
                    learner_text.encode("utf-8")
                ).hexdigest(),
                **provenance_fields,
            }
        )

    annotation_rows = []
    for sentence in selected_sentences:
        sentence_hash = hashlib.sha256(
            sentence.text.encode("utf-8")
        ).hexdigest()
        annotation_rows.append(
            {
                "schema_version": "1.0",
                "sentence_id": (
                    f"{sentence.essay_record.record_id}:"
                    f"s{sentence.sentence_index:04d}"
                ),
                "essay_id": sentence.essay_record.record_id,
                "student_id": sentence.essay_record.student_id,
                "source_stage": arguments.source_stage,
                "sentence_index": sentence.sentence_index,
                "original_text": sentence.text,
                "original_text_sha256": sentence_hash,
                "manual_label": None,
                "usable_binary_label": None,
                "abstention_status": None,
                "manual_failure_types": [],
                "annotation_notes": None,
                "annotation_origin": "model_assisted_manual_review",
                "annotator_id": None,
                "annotation_guideline_version": (
                    RELIABILITY_GUIDELINE_VERSION
                ),
                "adjudication_status": "single",
                "selection_structures": list(sentence.categories),
                **provenance_fields,
            }
        )

    schema = {
        "batch_id": arguments.source_stage,
        "status": "annotation_template",
        "allowed_manual_labels": [
            "RELIABLE",
            "MOSTLY_RELIABLE",
            "UNRELIABLE",
            "NOT_ANALYZABLE",
        ],
        "pending_value": None,
        "binary_mapping": {
            "RELIABLE": True,
            "MOSTLY_RELIABLE": True,
            "UNRELIABLE": False,
            "NOT_ANALYZABLE": None,
        },
        "source_schema": "data/syntax_reliability/schema_v1.json",
        "learner_text_policy": "local ignored untracked only",
        "provenance": provenance_fields,
    }

    jsonl_write(arguments.local_output / "essays.jsonl", essay_rows)
    jsonl_write(
        arguments.local_output / "annotations_template.jsonl",
        annotation_rows,
    )
    arguments.local_output.mkdir(parents=True, exist_ok=True)
    (arguments.local_output / "annotation_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    category_counts = Counter(
        category
        for sentence in selected_sentences
        for category in sentence.categories
    )
    period_counts = Counter(
        record.period for record in selected_essay_records
    )
    course_counts = Counter(
        record.course_id for record in selected_essay_records
    )
    report = {
        "pilot_version": arguments.source_stage,
        "seed": arguments.seed,
        "data_role": arguments.role,
        "holdout_opened": False,
        "selection_inputs": [
            "student_id",
            "period",
            "course_id",
            "essays_per_student_bin",
            "parser_derived_structural_features",
        ],
        "selection_excluded_inputs": [
            "reliability_labels",
            "parser_error_review",
            "model_predictions",
            "quality",
            "word_count",
            "duplicate_flags",
        ],
        "counts": {
            "students": len(selected_students),
            "essays": len(selected_essay_records),
            "sentences": len(selected_sentences),
        },
        "structural_coverage": {
            category: category_counts[category]
            for category in STRUCTURAL_CATEGORIES
        },
        "essay_metadata_coverage": {
            "periods": dict(sorted(period_counts.items())),
            "courses": dict(sorted(course_counts.items())),
        },
        "annotation_status": "not_started",
        "privacy": {
            "learner_text_in_report": False,
            "source_hashes_in_report": False,
            "private_paths_in_report": False,
            "student_ids_in_report": False,
            "raw_git_status": "ignored_untracked",
        },
        "provenance": provenance_fields,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], sort_keys=True))
    print(json.dumps(report["structural_coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
