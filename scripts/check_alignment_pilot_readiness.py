import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.alignment_readiness import (  # noqa: E402
    assess_development_pilot_readiness,
    create_readiness_declarations,
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Check private development-pilot operational readiness."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--confirm-direct-identifiers-removed",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-consent-scope-reviewed",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-two-independent-reviewers",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-separate-adjudicator",
        action="store_true",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    declarations = create_readiness_declarations(
        direct_identifiers_removed=(
            arguments.confirm_direct_identifiers_removed
        ),
        consent_scope_reviewed=(
            arguments.confirm_consent_scope_reviewed
        ),
        two_independent_reviewers_confirmed=(
            arguments.confirm_two_independent_reviewers
        ),
        separate_adjudicator_confirmed=(
            arguments.confirm_separate_adjudicator
        ),
    )
    result = assess_development_pilot_readiness(
        task_records=load_jsonl(arguments.tasks),
        response_records=load_jsonl(arguments.responses),
        declarations=declarations,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
