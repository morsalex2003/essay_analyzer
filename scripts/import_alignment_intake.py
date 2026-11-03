import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.alignment_intake import import_alignment_intake  # noqa: E402
from src.analysis import ANALYSIS_SCHEMA_VERSION  # noqa: E402


ALIGNMENT_DIRECTORY = REPOSITORY_ROOT / "data" / "alignment_evaluation"
DEFAULT_INTAKE_DIRECTORY = ALIGNMENT_DIRECTORY / "local" / "intake"
TASKS_OUTPUT_PATH = ALIGNMENT_DIRECTORY / "tasks_v1.jsonl"
RESPONSES_OUTPUT_PATH = ALIGNMENT_DIRECTORY / "responses_v1.jsonl"


def load_json(path):
    with path.open(encoding="utf-8") as input_file:
        try:
            return json.load(input_file)
        except json.JSONDecodeError as error:
            raise ValueError(path.name + " contains invalid JSON.") from error


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


def require_git_ignored(path):
    """Refuse to write private data to a path that Git would track."""
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(
            "Refusing to write private data: path is not ignored by Git: "
            + str(path.relative_to(REPOSITORY_ROOT))
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Convert private task-alignment intake files into canonical "
            "development JSONL records. Prints aggregate counts only."
        )
    )
    parser.add_argument(
        "--intake-dir",
        type=Path,
        default=DEFAULT_INTAKE_DIRECTORY,
        help="Directory containing tasks.json and responses.jsonl.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    tasks_path = arguments.intake_dir / "tasks.json"
    responses_path = arguments.intake_dir / "responses.jsonl"

    for output_path in (TASKS_OUTPUT_PATH, RESPONSES_OUTPUT_PATH):
        if output_path.exists():
            raise SystemExit(
                "Refusing to overwrite existing file: "
                + str(output_path.relative_to(REPOSITORY_ROOT))
            )
        require_git_ignored(output_path)

    task_records, response_records, summary = import_alignment_intake(
        load_json(tasks_path),
        load_jsonl(responses_path),
        ANALYSIS_SCHEMA_VERSION,
    )

    write_jsonl(TASKS_OUTPUT_PATH, task_records)
    write_jsonl(RESPONSES_OUTPUT_PATH, response_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
