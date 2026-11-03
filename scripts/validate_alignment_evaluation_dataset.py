import json
from pathlib import Path

from src.alignment_dataset import validate_alignment_dataset


DATASET_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "data" / "alignment_evaluation"
)
TASKS_PATH = DATASET_DIRECTORY / "tasks_v1.jsonl"
RESPONSES_PATH = DATASET_DIRECTORY / "responses_v1.jsonl"
ASSESSMENTS_PATH = DATASET_DIRECTORY / "assessments_v1.jsonl"


def load_jsonl(path):
    records = []

    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
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


def main():
    missing_paths = []

    for path in (TASKS_PATH, RESPONSES_PATH, ASSESSMENTS_PATH):
        if not path.exists():
            missing_paths.append(path)

    if len(missing_paths) > 0:
        missing_names = []
        for path in missing_paths:
            missing_names.append(path.name)

        raise FileNotFoundError(
            "Private local dataset files are missing: "
            + ", ".join(missing_names)
        )

    summary = validate_alignment_dataset(
        load_jsonl(TASKS_PATH),
        load_jsonl(RESPONSES_PATH),
        load_jsonl(ASSESSMENTS_PATH),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
