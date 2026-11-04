"""Freeze the Syntax Reliability development set for ML v2.

Writes the public-safe manifest (tracked) and the private file lock (local,
ignored). Both files are created once; an existing file is never replaced.
"""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = PROJECT_ROOT / "data" / "syntax_reliability"

sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.pipeline import nlp  # noqa: E402
from src.reliability_development_set import (  # noqa: E402
    EXCLUSIONS,
    KNOWN_LIMITATIONS,
    LOCK_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    build_development_manifest,
    build_lock,
    essay_context_mismatches,
    load_batches,
    load_holdout_students,
)


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def main() -> None:
    batches = load_batches(DATASET_DIRECTORY)

    for batch in batches:
        mismatches = essay_context_mismatches(
            batch["essays"],
            batch["annotations"],
            nlp,
        )
        if mismatches:
            raise ValueError(
                f"{batch['batch_id']}: {len(mismatches)} sentences are not "
                "reproduced by segmenting their essay"
            )

    manifest = build_development_manifest(
        batches,
        load_holdout_students(DATASET_DIRECTORY),
        EXCLUSIONS,
        KNOWN_LIMITATIONS,
    )
    lock = build_lock(DATASET_DIRECTORY)

    write_new_json(DATASET_DIRECTORY / MANIFEST_RELATIVE_PATH, manifest)
    write_new_json(DATASET_DIRECTORY / LOCK_RELATIVE_PATH, lock)

    totals = manifest["totals"]
    print("Frozen", manifest["development_set_id"])
    print("Counts:", totals["counts"])
    print("Binary:", totals["binary_counts"])


if __name__ == "__main__":
    main()
