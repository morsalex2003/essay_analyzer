"""Check that the frozen Syntax Reliability development set is unchanged."""

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
    essay_context_mismatches,
    load_batches,
    load_holdout_students,
    lock_mismatches,
)


def main() -> None:
    manifest = json.loads(
        (DATASET_DIRECTORY / MANIFEST_RELATIVE_PATH).read_text()
    )
    lock = json.loads((DATASET_DIRECTORY / LOCK_RELATIVE_PATH).read_text())

    changed_files = lock_mismatches(lock, DATASET_DIRECTORY)
    if changed_files:
        raise SystemExit(f"Frozen source files changed: {changed_files}")

    batches = load_batches(DATASET_DIRECTORY)
    rebuilt = build_development_manifest(
        batches,
        load_holdout_students(DATASET_DIRECTORY),
        EXCLUSIONS,
        KNOWN_LIMITATIONS,
    )
    if rebuilt != manifest:
        raise SystemExit("Manifest does not match the frozen sources")

    for batch in batches:
        if essay_context_mismatches(
            batch["essays"], batch["annotations"], nlp
        ):
            raise SystemExit(
                f"{batch['batch_id']} no longer matches essay segmentation"
            )

    manifest_text = json.dumps(manifest)
    for batch in batches:
        for record in [*batch["essays"], *batch["annotations"]]:
            for field in ("essay_id", "sentence_id", "original_text_sha256"):
                if field in record and record[field] in manifest_text:
                    raise SystemExit(f"Manifest exposes a private {field}")

    totals = manifest["totals"]
    print("Development set", manifest["development_set_id"], "is intact")
    print("Counts:", totals["counts"])
    print("Binary:", totals["binary_counts"])


if __name__ == "__main__":
    main()
