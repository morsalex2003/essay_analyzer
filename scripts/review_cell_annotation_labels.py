"""Owner review of model labels in a private CELL reliability batch.

`--sheet` writes `packet/review.tsv`: every label other than RELIABLE plus a
seeded random sample of RELIABLE labels, with empty decision columns.
`--apply` reads the filled sheet and marks each reviewed record as
adjudicated, keeping a changed label's original in `pre_adjudication_label`.
Only counts are printed.

Decisions: `confirm` keeps the label; `change` replaces it with
`new_label`, `new_failure_types`, and `new_abstention_status`.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell_annotation import (  # noqa: E402
    adjudicate_record,
    create_annotated_record,
    load_annotation_enums,
    load_jsonl,
    merge_progress,
    write_jsonl_atomically,
)


SCHEMA_PATH = PROJECT_ROOT / "data/syntax_reliability/schema_v1.json"
REVIEW_COLUMNS = (
    "position",
    "sentence_id",
    "label",
    "failure_types",
    "abstention_status",
    "note",
    "decision",
    "new_label",
    "new_failure_types",
    "new_abstention_status",
    "review_note",
)


def review_positions(
    records: list[dict],
    reliable_sample_size: int,
    seed: int,
) -> list[int]:
    """Return 1-based positions: all non-RELIABLE and a RELIABLE sample."""
    if any(record["manual_label"] is None for record in records):
        raise ValueError("Every record must be labelled before review")

    reliable = []
    others = []
    for position, record in enumerate(records, start=1):
        if record["manual_label"] == "RELIABLE":
            reliable.append(position)
        else:
            others.append(position)

    sample_size = min(reliable_sample_size, len(reliable))
    sample = random.Random(seed).sample(reliable, sample_size)
    return sorted(others + sample)


def review_rows(records: list[dict], positions: list[int]) -> list[dict]:
    rows = []
    for position in positions:
        record = records[position - 1]
        rows.append(
            {
                "position": position,
                "sentence_id": record["sentence_id"],
                "label": record["manual_label"],
                "failure_types": ",".join(record["manual_failure_types"])
                or "-",
                "abstention_status": record["abstention_status"],
                "note": record["annotation_notes"] or "",
                "decision": "",
                "new_label": "",
                "new_failure_types": "",
                "new_abstention_status": "",
                "review_note": "",
            }
        )
    return rows


def _names(value: str) -> list[str]:
    if value.strip() in ("", "-"):
        return []
    return [name.strip() for name in value.split(",")]


def apply_review_rows(
    records: list[dict],
    rows: list[dict[str, str]],
    adjudicator_id: str,
    adjudication_date: str,
    binary_mapping: dict,
    enums: dict,
) -> dict[str, int]:
    """Adjudicate reviewed records in place; return decision counts."""
    counts = {"confirm": 0, "change": 0}
    for row in rows:
        index = int(row["position"]) - 1
        record = records[index]
        if record["sentence_id"] != row["sentence_id"]:
            raise ValueError(f"position {row['position']}: id mismatch")

        decision = row["decision"].strip()
        revised = None
        if decision == "change":
            revised = create_annotated_record(
                record,
                row["new_label"].strip(),
                _names(row["new_failure_types"]),
                row["new_abstention_status"].strip() or "none",
                record["annotation_notes"],
                record["annotator_id"],
                binary_mapping,
                enums,
                annotation_origin=record["annotation_origin"],
            )
        elif decision != "confirm":
            raise ValueError(
                f"position {row['position']}: decision must be confirm or "
                "change"
            )

        records[index] = adjudicate_record(
            record,
            adjudicator_id,
            adjudication_date,
            revised_record=revised,
            adjudication_notes=row.get("review_note") or None,
        )
        counts[decision] += 1
    return counts


def load_records(batch_dir: Path) -> list[dict]:
    template = load_jsonl(batch_dir / "annotations_template.jsonl")
    return merge_progress(
        template, load_jsonl(batch_dir / "annotations_v1.jsonl")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sheet", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--reliable-sample", type=int, default=40)
    parser.add_argument("--adjudicator-id", default="human_owner_v1")
    parser.add_argument("--date", help="Review date, YYYY-MM-DD")
    arguments = parser.parse_args()

    batch_dir = arguments.batch_dir
    sheet_path = batch_dir / "packet" / "review.tsv"
    records = load_records(batch_dir)

    if arguments.sheet:
        positions = review_positions(
            records, arguments.reliable_sample, arguments.seed
        )
        with sheet_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=REVIEW_COLUMNS, delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(review_rows(records, positions))
        print(f"Review sheet: {len(positions)} sentences")
        return

    if not arguments.date:
        raise SystemExit("--apply needs --date")

    with sheet_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    enums = load_annotation_enums(json.loads(SCHEMA_PATH.read_text()))
    binary_mapping = json.loads(
        (batch_dir / "annotation_schema.json").read_text()
    )["binary_mapping"]
    counts = apply_review_rows(
        records,
        rows,
        arguments.adjudicator_id,
        arguments.date,
        binary_mapping,
        enums,
    )
    write_jsonl_atomically(batch_dir / "annotations_v1.jsonl", records)
    print(f"Reviewed {sum(counts.values())}: {counts}")


if __name__ == "__main__":
    main()
