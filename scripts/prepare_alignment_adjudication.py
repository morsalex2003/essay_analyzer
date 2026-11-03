import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.alignment_pilot import create_annotation_review_batch  # noqa: E402


LOCAL_DIRECTORY = (
    REPOSITORY_ROOT / "data" / "alignment_evaluation" / "local"
)
OUTPUT_PATH = LOCAL_DIRECTORY / "annotation_review_batch_v1.json"


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare private completed alignment worklists and prepare "
            "a human adjudication queue."
        )
    )
    parser.add_argument(
        "--worklist",
        action="append",
        required=True,
        type=Path,
        help="Completed reviewer JSONL worklist; provide at least twice.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    completed_worklists = []

    for worklist_path in arguments.worklist:
        completed_worklists.append(load_jsonl(worklist_path))

    review_batch = create_annotation_review_batch(completed_worklists)
    LOCAL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("x", encoding="utf-8") as output_file:
        json.dump(
            review_batch,
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
