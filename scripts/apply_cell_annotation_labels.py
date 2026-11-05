"""Apply a filled label sheet to a private CELL reliability batch.

Reads `labels.tsv` from a blind annotation packet and writes the labels to
the batch ledger `annotations_v1.jsonl` with the same validation as the
annotation tool. Already labelled records are never overwritten. Only
counts are printed, so applying labels does not reveal them.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell_annotation import (  # noqa: E402
    MODEL_ASSISTED_ANNOTATION_ORIGIN,
    create_annotated_record,
    load_annotation_enums,
    load_jsonl,
    merge_progress,
    write_jsonl_atomically,
)


SCHEMA_PATH = PROJECT_ROOT / "data/syntax_reliability/schema_v1.json"


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def apply_label_rows(
    records: list[dict],
    rows: list[dict[str, str]],
    annotator_id: str,
    binary_mapping: dict,
    enums: dict,
) -> int:
    """Label pending records in place; return how many were labelled."""
    applied = 0
    for row in rows:
        position = int(row["position"])
        index = position - 1
        if not 0 <= index < len(records):
            raise ValueError(f"position {position} is outside the batch")

        record = records[index]
        if record["sentence_id"] != row["sentence_id"]:
            raise ValueError(f"position {position}: sentence_id mismatch")
        if record["manual_label"] is not None:
            raise ValueError(f"position {position} is already labelled")

        failure_types = []
        if row["failure_types"].strip() not in ("", "-"):
            failure_types = [
                name.strip() for name in row["failure_types"].split(",")
            ]

        records[index] = create_annotated_record(
            record,
            row["label"].strip(),
            failure_types,
            row["abstention_status"].strip(),
            row.get("note") or None,
            annotator_id,
            binary_mapping,
            enums,
            annotation_origin=MODEL_ASSISTED_ANNOTATION_ORIGIN,
        )
        applied += 1
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--labels", type=Path)
    arguments = parser.parse_args()

    batch_dir = arguments.batch_dir
    labels_path = arguments.labels or batch_dir / "packet" / "labels.tsv"
    template = load_jsonl(batch_dir / "annotations_template.jsonl")
    ledger_path = batch_dir / "annotations_v1.jsonl"
    if ledger_path.exists():
        records = merge_progress(template, load_jsonl(ledger_path))
    else:
        records = [dict(record) for record in template]

    enums = load_annotation_enums(json.loads(SCHEMA_PATH.read_text()))
    binary_mapping = json.loads(
        (batch_dir / "annotation_schema.json").read_text()
    )["binary_mapping"]

    applied = apply_label_rows(
        records,
        read_label_rows(labels_path),
        arguments.annotator_id,
        binary_mapping,
        enums,
    )
    write_jsonl_atomically(ledger_path, records)
    labelled = sum(record["manual_label"] is not None for record in records)
    print(f"Applied {applied}; labelled {labelled} of {len(records)}")


if __name__ == "__main__":
    main()
