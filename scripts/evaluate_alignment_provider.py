import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.alignment_experiments import (  # noqa: E402
    evaluate_frozen_alignment_experiment,
    sha256_file,
)
from src.alignment_batch import copy_run_records_from_batch  # noqa: E402


LOCAL_DIRECTORY = (
    REPOSITORY_ROOT / "data" / "alignment_evaluation" / "local"
)


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


def load_json(path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one alignment provider against private human references."
        )
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--assessments", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--provider-configuration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-role",
        choices=("development", "sealed_evaluation"),
        required=True,
    )
    parser.add_argument(
        "--authorize-sealed-evaluation",
        action="store_true",
        help="Explicitly authorize opening the frozen sealed evaluation.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    source_file_sha256 = {
        "tasks": sha256_file(arguments.tasks),
        "responses": sha256_file(arguments.responses),
        "assessments": sha256_file(arguments.assessments),
    }
    manifest = load_json(arguments.manifest)
    provider_configuration = load_json(arguments.provider_configuration)
    batch_result = load_json(arguments.batch)

    if manifest.get("data_role") != arguments.data_role:
        raise ValueError("CLI data role does not match frozen manifest.")

    result = evaluate_frozen_alignment_experiment(
        manifest=manifest,
        task_records=load_jsonl(arguments.tasks),
        response_records=load_jsonl(arguments.responses),
        assessment_records=load_jsonl(arguments.assessments),
        source_file_sha256=source_file_sha256,
        provider_configuration=provider_configuration,
        run_records=copy_run_records_from_batch(
            batch_result,
            manifest,
            provider_configuration,
        ),
        sealed_evaluation_authorized=(
            arguments.authorize_sealed_evaluation
        ),
    )
    LOCAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = LOCAL_DIRECTORY / (
        "provider_evaluation_" + result["experiment_id"] + "_v1.json"
    )

    with output_path.open("x", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(str(output_path))


if __name__ == "__main__":
    main()
