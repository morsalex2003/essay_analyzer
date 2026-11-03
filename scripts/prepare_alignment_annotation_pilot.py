import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.alignment_pilot import (  # noqa: E402
    create_independent_reviewer_worklists,
)


DATASET_DIRECTORY = REPOSITORY_ROOT / "data" / "alignment_evaluation"
TASKS_PATH = DATASET_DIRECTORY / "tasks_v1.jsonl"
RESPONSES_PATH = DATASET_DIRECTORY / "responses_v1.jsonl"
LOCAL_DIRECTORY = DATASET_DIRECTORY / "local"


def load_jsonl(path):
    records = []

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip() == "":
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    path.name
                    + " contains invalid JSON on line "
                    + str(line_number)
                ) from error

            records.append(record)

    return records


def write_jsonl(path, records):
    with path.open("x", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
            )
            output_file.write("\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare private independent alignment annotation worklists."
    )
    parser.add_argument(
        "--reviewer-id",
        action="append",
        required=True,
        help="Pseudonymous reviewer ID; provide at least twice.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    worklists = create_independent_reviewer_worklists(
        load_jsonl(TASKS_PATH),
        load_jsonl(RESPONSES_PATH),
        arguments.reviewer_id,
    )
    LOCAL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for reviewer_id, records in worklists.items():
        output_path = (
            LOCAL_DIRECTORY
            / ("reviewer_" + reviewer_id + "_worklist_v1.jsonl")
        )
        write_jsonl(output_path, records)
        print(str(output_path))


if __name__ == "__main__":
    main()
