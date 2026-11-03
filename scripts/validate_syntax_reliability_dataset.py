import hashlib
import json
from collections import Counter
from pathlib import Path


DATASET_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "data" / "syntax_reliability"
)
ESSAYS_PATH = DATASET_DIRECTORY / "essays_v1.jsonl"
ANNOTATIONS_PATH = DATASET_DIRECTORY / "annotations_v1.jsonl"

EXPECTED_ANNOTATION_ORIGIN = "model_assisted_manual_review"
EXPECTED_ANNOTATOR_ID = "codex_review_v1"
EXPECTED_ADJUDICATOR_ID = "human_owner_v1"
EXPECTED_ADJUDICATION_DATE = "2026-08-17"
EXPECTED_STAGE_CONFIG = {
    "v11_dev7": {
        "essay_count": 7,
        "sentence_count": 62,
        "adjudicated_count": 23,
        "data_status": "development",
        "label_counts": {
            "RELIABLE": 43,
            "MOSTLY_RELIABLE": 2,
            "UNRELIABLE": 13,
            "NOT_ANALYZABLE": 4,
        },
        "provenance": {
            "schema_version": "1.0",
            "repository_commit": (
                "6f86cf593ecc1b19b00023ebef6c9e60e550c3a3"
            ),
            "spacy_version": "3.8.13",
            "spacy_model_name": "en_core_web_sm",
            "spacy_model_version": "3.8.0",
            "clause_detector_version": "2.1",
            "reliability_version": "1.1",
        },
    },
    "v12_holdout12": {
        "essay_count": 12,
        "sentence_count": 99,
        "adjudicated_count": 8,
        "data_status": "development",
        "label_counts": {
            "RELIABLE": 78,
            "MOSTLY_RELIABLE": 5,
            "UNRELIABLE": 12,
            "NOT_ANALYZABLE": 4,
        },
        "provenance": {
            "schema_version": "1.0",
            "repository_commit": (
                "d50bdc81bad5a9e6fc2f19b9a5bd48a790579fc9"
            ),
            "spacy_version": "3.8.13",
            "spacy_model_name": "en_core_web_sm",
            "spacy_model_version": "3.8.0",
            "clause_detector_version": "2.1",
            "reliability_version": "1.2",
        },
    },
    "v11_holdout9": {
        "essay_count": 9,
        "sentence_count": 71,
        "adjudicated_count": 20,
        "data_status": "development",
        "label_counts": {
            "RELIABLE": 43,
            "MOSTLY_RELIABLE": 7,
            "UNRELIABLE": 18,
            "NOT_ANALYZABLE": 3,
        },
        "provenance": {
            "schema_version": "1.0",
            "repository_commit": (
                "064658cbfc502fae2481bdd9b5da9377425af32f"
            ),
            "spacy_version": "3.8.13",
            "spacy_model_name": "en_core_web_sm",
            "spacy_model_version": "3.8.0",
            "clause_detector_version": "2.1",
            "reliability_version": "1.1",
        },
    },
}
EXPECTED_ESSAY_COUNT = 28
EXPECTED_SENTENCE_COUNT = 232
EXPECTED_ADJUDICATED_COUNT = 51
EXPECTED_LABEL_COUNTS = {
    "RELIABLE": 164,
    "MOSTLY_RELIABLE": 14,
    "UNRELIABLE": 43,
    "NOT_ANALYZABLE": 11,
}

VALID_MANUAL_LABELS = set(EXPECTED_LABEL_COUNTS)
VALID_ABSTENTION_STATUSES = {
    "none",
    "source_fragment",
    "parser_abstention",
    "other",
}
VALID_FAILURE_TYPES = {
    "finite_inventory",
    "dependent_inventory",
    "matrix_hierarchy",
    "nonfinite_inventory",
    "major_dependency_structure",
    "sentence_segmentation",
    "source_fragment",
    "parser_abstention",
    "other",
}
VALID_ADJUDICATION_STATUSES = {
    "single",
    "agreed",
    "adjudicated",
}
VALID_DATA_STATUSES = {
    "development",
    "reserve",
    "sealed_test",
}

ESSAY_REQUIRED_FIELDS = {
    "schema_version",
    "essay_id",
    "student_id",
    "task_id",
    "prompt_id",
    "response_family_id",
    "source_stage",
    "data_status",
    "original_text",
    "original_text_sha256",
    "repository_commit",
    "spacy_version",
    "spacy_model_name",
    "spacy_model_version",
    "clause_detector_version",
    "reliability_version",
}

ANNOTATION_REQUIRED_FIELDS = {
    "schema_version",
    "sentence_id",
    "essay_id",
    "student_id",
    "source_stage",
    "sentence_index",
    "original_text",
    "original_text_sha256",
    "manual_label",
    "usable_binary_label",
    "abstention_status",
    "manual_failure_types",
    "annotation_notes",
    "annotation_origin",
    "annotator_id",
    "annotation_guideline_version",
    "adjudication_status",
    "repository_commit",
    "spacy_version",
    "spacy_model_name",
    "spacy_model_version",
    "clause_detector_version",
    "reliability_version",
}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []

    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid JSON: {error}"
                ) from error

            records.append(record)

    return records


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_fields(
    record: dict[str, object],
    required_fields: set[str],
    record_name: str,
    errors: list[str],
) -> None:
    missing_fields = sorted(required_fields - set(record))

    if missing_fields:
        errors.append(
            f"{record_name}: missing required fields: {missing_fields}"
        )


def validate_hash(
    record: dict[str, object],
    record_name: str,
    errors: list[str],
) -> None:
    original_text = record.get("original_text")
    stored_hash = record.get("original_text_sha256")

    if not isinstance(original_text, str) or original_text == "":
        errors.append(f"{record_name}: original_text must be non-empty")
        return

    expected_hash = text_sha256(original_text)

    if stored_hash != expected_hash:
        errors.append(f"{record_name}: original_text_sha256 does not match")


def validate_provenance(
    record: dict[str, object],
    record_name: str,
    errors: list[str],
) -> None:
    source_stage = record.get("source_stage")
    stage_config = EXPECTED_STAGE_CONFIG.get(source_stage)

    if stage_config is None:
        return

    expected_provenance = stage_config["provenance"]
    for field_name, expected_value in expected_provenance.items():
        if record.get(field_name) != expected_value:
            errors.append(
                f"{record_name}: {field_name} must be {expected_value}"
            )


def expected_binary_label(manual_label: str) -> bool | None:
    if manual_label == "RELIABLE" or manual_label == "MOSTLY_RELIABLE":
        return True

    if manual_label == "UNRELIABLE":
        return False

    return None


def validate_essays(
    essays: list[dict[str, object]],
    errors: list[str],
) -> tuple[dict[str, dict[str, object]], int]:
    essays_by_id = {}
    essay_hash_counts = Counter()
    source_stage_counts = Counter()

    for record_number, essay in enumerate(essays, start=1):
        record_name = f"essay record {record_number}"
        require_fields(essay, ESSAY_REQUIRED_FIELDS, record_name, errors)
        essay_id = essay.get("essay_id")

        if not isinstance(essay_id, str) or essay_id == "":
            errors.append(f"{record_name}: essay_id must be non-empty")
            continue

        if essay_id in essays_by_id:
            errors.append(f"duplicate essay_id: {essay_id}")
        else:
            essays_by_id[essay_id] = essay

        source_stage = essay.get("source_stage")
        stage_config = EXPECTED_STAGE_CONFIG.get(source_stage)
        if stage_config is None:
            errors.append(f"{essay_id}: unexpected source_stage")
        else:
            source_stage_counts[source_stage] += 1
            if essay.get("data_status") != stage_config["data_status"]:
                errors.append(f"{essay_id}: unexpected data_status")

        if essay.get("data_status") not in VALID_DATA_STATUSES:
            errors.append(f"{essay_id}: invalid data_status")

        validate_hash(essay, essay_id, errors)
        validate_provenance(essay, essay_id, errors)
        essay_hash = essay.get("original_text_sha256")
        essay_hash_counts[essay_hash] += 1

    if len(essays) != EXPECTED_ESSAY_COUNT:
        errors.append(
            f"expected {EXPECTED_ESSAY_COUNT} essays, found {len(essays)}"
        )

    for source_stage, stage_config in EXPECTED_STAGE_CONFIG.items():
        expected_count = stage_config["essay_count"]
        if source_stage_counts[source_stage] != expected_count:
            errors.append(
                f"source stage {source_stage} must contain "
                f"{expected_count} essays"
            )

    duplicate_hash_count = 0
    for hash_count in essay_hash_counts.values():
        if hash_count > 1:
            duplicate_hash_count += hash_count - 1

    return essays_by_id, duplicate_hash_count


def validate_annotations(
    annotations: list[dict[str, object]],
    essays_by_id: dict[str, dict[str, object]],
    errors: list[str],
) -> tuple[Counter, int]:
    sentence_ids = set()
    sentence_hash_counts = Counter()
    label_counts = Counter()
    source_stage_counts = Counter()
    stage_label_counts = {}
    stage_adjudicated_counts = Counter()
    adjudicated_count = 0

    for source_stage in EXPECTED_STAGE_CONFIG:
        stage_label_counts[source_stage] = Counter()

    for record_number, annotation in enumerate(annotations, start=1):
        record_name = f"annotation record {record_number}"
        require_fields(
            annotation,
            ANNOTATION_REQUIRED_FIELDS,
            record_name,
            errors,
        )
        sentence_id = annotation.get("sentence_id")

        if not isinstance(sentence_id, str) or sentence_id == "":
            errors.append(f"{record_name}: sentence_id must be non-empty")
            continue

        if sentence_id in sentence_ids:
            errors.append(f"duplicate sentence_id: {sentence_id}")
        sentence_ids.add(sentence_id)

        essay_id = annotation.get("essay_id")
        if essay_id not in essays_by_id:
            errors.append(f"{sentence_id}: unknown essay_id: {essay_id}")
        else:
            essay = essays_by_id[essay_id]
            if annotation.get("student_id") != essay.get("student_id"):
                errors.append(f"{sentence_id}: student_id differs from essay")
            if annotation.get("source_stage") != essay.get("source_stage"):
                errors.append(f"{sentence_id}: source_stage differs from essay")

        source_stage = annotation.get("source_stage")
        if source_stage not in EXPECTED_STAGE_CONFIG:
            errors.append(f"{sentence_id}: unexpected source_stage")
        else:
            source_stage_counts[source_stage] += 1

        manual_label = annotation.get("manual_label")
        if manual_label not in VALID_MANUAL_LABELS:
            errors.append(f"{sentence_id}: invalid manual_label")
        else:
            label_counts[manual_label] += 1
            if source_stage in stage_label_counts:
                stage_label_counts[source_stage][manual_label] += 1
            expected_label = expected_binary_label(manual_label)
            if annotation.get("usable_binary_label") != expected_label:
                errors.append(
                    f"{sentence_id}: usable_binary_label does not match "
                    f"{manual_label}"
                )

        abstention_status = annotation.get("abstention_status")
        if abstention_status not in VALID_ABSTENTION_STATUSES:
            errors.append(f"{sentence_id}: invalid abstention_status")
        elif manual_label == "NOT_ANALYZABLE":
            if abstention_status == "none":
                errors.append(
                    f"{sentence_id}: NOT_ANALYZABLE needs abstention reason"
                )
        elif abstention_status != "none":
            errors.append(
                f"{sentence_id}: analyzable label must use abstention_status none"
            )

        failure_types = annotation.get("manual_failure_types")
        if not isinstance(failure_types, list):
            errors.append(f"{sentence_id}: manual_failure_types must be a list")
        else:
            invalid_failure_types = set(failure_types) - VALID_FAILURE_TYPES
            if invalid_failure_types:
                errors.append(
                    f"{sentence_id}: invalid failure types: "
                    f"{sorted(invalid_failure_types)}"
                )

            if abstention_status == "source_fragment":
                if manual_label != "NOT_ANALYZABLE":
                    errors.append(
                        f"{sentence_id}: source_fragment must be "
                        "NOT_ANALYZABLE"
                    )
                if "source_fragment" not in failure_types:
                    errors.append(
                        f"{sentence_id}: source_fragment abstention needs "
                        "source_fragment failure type"
                    )

        adjudication_status = annotation.get("adjudication_status")
        if adjudication_status not in VALID_ADJUDICATION_STATUSES:
            errors.append(f"{sentence_id}: invalid adjudication_status")
        elif adjudication_status == "adjudicated":
            adjudicated_count += 1
            if source_stage in EXPECTED_STAGE_CONFIG:
                stage_adjudicated_counts[source_stage] += 1
            if annotation.get("adjudicator_id") != EXPECTED_ADJUDICATOR_ID:
                errors.append(f"{sentence_id}: invalid adjudicator_id")
            if (
                annotation.get("adjudication_date")
                != EXPECTED_ADJUDICATION_DATE
            ):
                errors.append(f"{sentence_id}: invalid adjudication_date")

        pre_adjudication_label = annotation.get("pre_adjudication_label")
        if pre_adjudication_label is not None:
            if pre_adjudication_label not in VALID_MANUAL_LABELS:
                errors.append(
                    f"{sentence_id}: invalid pre_adjudication_label"
                )
            if adjudication_status != "adjudicated":
                errors.append(
                    f"{sentence_id}: pre_adjudication_label requires "
                    "adjudicated status"
                )

        adjudication_notes = annotation.get("adjudication_notes")
        if adjudication_notes is not None:
            if (
                not isinstance(adjudication_notes, str)
                or not adjudication_notes
            ):
                errors.append(f"{sentence_id}: invalid adjudication_notes")
            if adjudication_status != "adjudicated":
                errors.append(
                    f"{sentence_id}: adjudication_notes require "
                    "adjudicated status"
                )

        if annotation.get("annotation_origin") != EXPECTED_ANNOTATION_ORIGIN:
            errors.append(f"{sentence_id}: invalid annotation_origin")

        if annotation.get("annotator_id") != EXPECTED_ANNOTATOR_ID:
            errors.append(f"{sentence_id}: invalid annotator_id")

        sentence_index = annotation.get("sentence_index")
        if not isinstance(sentence_index, int) or sentence_index < 1:
            errors.append(f"{sentence_id}: invalid sentence_index")

        validate_hash(annotation, sentence_id, errors)
        validate_provenance(annotation, sentence_id, errors)
        sentence_hash = annotation.get("original_text_sha256")
        sentence_hash_counts[sentence_hash] += 1

    if len(annotations) != EXPECTED_SENTENCE_COUNT:
        errors.append(
            f"expected {EXPECTED_SENTENCE_COUNT} sentences, "
            f"found {len(annotations)}"
        )

    if dict(label_counts) != EXPECTED_LABEL_COUNTS:
        errors.append(
            f"label count mismatch: expected {EXPECTED_LABEL_COUNTS}, "
            f"found {dict(label_counts)}"
        )

    for source_stage, stage_config in EXPECTED_STAGE_CONFIG.items():
        expected_sentence_count = stage_config["sentence_count"]
        if source_stage_counts[source_stage] != expected_sentence_count:
            errors.append(
                f"source stage {source_stage} must contain "
                f"{expected_sentence_count} sentences"
            )

        expected_stage_labels = stage_config["label_counts"]
        found_stage_labels = dict(stage_label_counts[source_stage])
        if found_stage_labels != expected_stage_labels:
            errors.append(
                f"source stage {source_stage} label count mismatch: "
                f"expected {expected_stage_labels}, "
                f"found {found_stage_labels}"
            )

        expected_adjudicated = stage_config["adjudicated_count"]
        if stage_adjudicated_counts[source_stage] != expected_adjudicated:
            errors.append(
                f"source stage {source_stage} must contain "
                f"{expected_adjudicated} adjudicated records"
            )

    if adjudicated_count != EXPECTED_ADJUDICATED_COUNT:
        errors.append(
            f"expected {EXPECTED_ADJUDICATED_COUNT} adjudicated records, "
            f"found {adjudicated_count}"
        )

    duplicate_hash_count = 0
    for hash_count in sentence_hash_counts.values():
        if hash_count > 1:
            duplicate_hash_count += hash_count - 1

    return label_counts, duplicate_hash_count


def main() -> None:
    essays = load_jsonl(ESSAYS_PATH)
    annotations = load_jsonl(ANNOTATIONS_PATH)
    errors = []
    essays_by_id, duplicate_essay_hash_count = validate_essays(
        essays,
        errors,
    )
    label_counts, duplicate_sentence_hash_count = validate_annotations(
        annotations,
        essays_by_id,
        errors,
    )

    duplicate_hash_count = (
        duplicate_essay_hash_count + duplicate_sentence_hash_count
    )

    if duplicate_essay_hash_count > 0:
        errors.append(
            f"found {duplicate_essay_hash_count} duplicate exact essay hashes"
        )

    if duplicate_sentence_hash_count > 0:
        errors.append(
            f"found {duplicate_sentence_hash_count} duplicate exact sentence "
            "hashes"
        )

    if errors:
        print("Syntax Reliability Dataset v1 validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    usable_count = (
        label_counts["RELIABLE"] + label_counts["MOSTLY_RELIABLE"]
    )

    print("Syntax Reliability Dataset v1 validation passed.")
    print(f"Essays: {len(essays)}")
    print(f"Sentences: {len(annotations)}")
    print(f"RELIABLE: {label_counts['RELIABLE']}")
    print(f"MOSTLY_RELIABLE: {label_counts['MOSTLY_RELIABLE']}")
    print(f"UNRELIABLE: {label_counts['UNRELIABLE']}")
    print(f"NOT_ANALYZABLE: {label_counts['NOT_ANALYZABLE']}")
    print(f"Binary USABLE: {usable_count}")
    print(f"Binary UNRELIABLE: {label_counts['UNRELIABLE']}")
    print(
        "Binary analyzable: "
        f"{usable_count + label_counts['UNRELIABLE']}"
    )
    print(f"Human adjudicated: {EXPECTED_ADJUDICATED_COUNT}")
    print(f"Duplicate exact hashes: {duplicate_hash_count}")
    for source_stage, stage_config in EXPECTED_STAGE_CONFIG.items():
        stage_labels = stage_config["label_counts"]
        print(
            f"{source_stage}: {stage_config['sentence_count']} sentences; "
            f"R={stage_labels['RELIABLE']}, "
            f"MR={stage_labels['MOSTLY_RELIABLE']}, "
            f"U={stage_labels['UNRELIABLE']}, "
            f"NA={stage_labels['NOT_ANALYZABLE']}"
        )


if __name__ == "__main__":
    main()
