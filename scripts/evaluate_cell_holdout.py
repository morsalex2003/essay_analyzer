"""Lock, open, and evaluate one CELL holdout batch exactly once.

`--batch v1`, `v2`, or `v3` picks the batch; each follows its own
holdout_<batch>_evaluation_plan.json. v3 scores Syntax Reliability v1.5 and
needs requirements-transformer.txt. Steps, in order:

1. --lock-labels: after labelling and owner review are complete, store the
   hash of the holdout ledger. Prints counts only.
2. --record-opening --reason TEXT: add the opening date and reason to
   `holdout_openings` in split_v1.json.
3. --evaluate: check the label lock, the development set lock, and the
   opening record, then score v1.2, the batch's candidate versions, and
   Model B once and write the aggregate report. An existing report is never
   replaced.
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment_experiments import sha256_file  # noqa: E402
from src.cell_annotation import (  # noqa: E402
    SentenceParser,
    load_jsonl,
    merge_progress,
)
from src.cell_holdout_evaluation import evaluate_holdout  # noqa: E402
from src.cell_pilot_evaluation import heuristic_verdicts  # noqa: E402
from src.nlp.pipeline import nlp  # noqa: E402
from src.reliability_development_set import (  # noqa: E402
    LOCK_RELATIVE_PATH,
    load_batches,
    lock_mismatches,
)
from src.syntax.reliability_ml import (  # noqa: E402
    MODEL_B_FEATURE_NAMES,
    POSITIVE_CLASS,
    build_feature_matrix,
    build_logistic_pipeline,
    observation_targets,
    prepare_essay_context_observations,
    run_grouped_oof,
    select_grouped_splits,
)
from src.syntax.reliability_thresholds import (  # noqa: E402
    heuristic_warning_precision,
    select_precision_floor_threshold,
)


DATASET_DIRECTORY = PROJECT_ROOT / "data/syntax_reliability"
CELL_DIRECTORY = DATASET_DIRECTORY / "external/cell"
SPLIT_PATH = CELL_DIRECTORY / "split_v1.json"

# Systems and decision rule of each batch, as fixed in its plan.
HOLDOUT_BATCHES = {
    "v1": {
        "batch_id": "cell_reliability_holdout_v1",
        "candidates": ("1.3",),
        "decision_pair": ("1.3", "1.2"),
        "positive_outcome": "adopt_v1_3",
    },
    "v2": {
        "batch_id": "cell_reliability_holdout_v2",
        "candidates": ("1.3", "1.4"),
        "decision_pair": ("1.4", "1.3"),
        "positive_outcome": "confirm_v1_4",
    },
    # v1.5 needs the optional transformer model (requirements-transformer.txt).
    "v3": {
        "batch_id": "cell_reliability_holdout_v3",
        "candidates": ("1.3", "1.4", "1.5"),
        "decision_pair": ("1.5", "1.4"),
        "positive_outcome": "confirm_v1_5",
    },
}


def batch_paths(batch: str) -> dict[str, Path]:
    directory = CELL_DIRECTORY / f"local/holdout_{batch}"
    return {
        "directory": directory,
        "ledger": directory / "annotations_v1.jsonl",
        "review": directory / "packet/review.tsv",
        "label_lock": directory / "labels_lock.json",
        "plan": CELL_DIRECTORY / f"holdout_{batch}_evaluation_plan.json",
        "report": CELL_DIRECTORY / f"holdout_{batch}_evaluation.json",
    }


def load_holdout_records(paths: dict[str, Path]) -> list[dict]:
    template = load_jsonl(paths["directory"] / "annotations_template.jsonl")
    return merge_progress(template, load_jsonl(paths["ledger"]))


def opening_record(split: dict, batch_id: str) -> dict | None:
    for opening in split.get("holdout_openings", []):
        if opening["batch_id"] == batch_id:
            return opening
    return None


def check_review_complete(
    records: list[dict], review_path: Path
) -> int:
    if any(record["manual_label"] is None for record in records):
        raise SystemExit("Not every holdout sentence is labelled")

    with review_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    reviewed = [int(row["position"]) for row in rows]
    for position in reviewed:
        if records[position - 1]["adjudication_status"] != "adjudicated":
            raise SystemExit(f"Review is incomplete at position {position}")
    return len(reviewed)


def lock_labels(batch: str) -> None:
    paths = batch_paths(batch)
    records = load_holdout_records(paths)
    reviewed = check_review_complete(records, paths["review"])
    lock = {
        "batch_id": HOLDOUT_BATCHES[batch]["batch_id"],
        "locked_at": date.today().isoformat(),
        "ledger_sha256": sha256_file(paths["ledger"]),
        "records": len(records),
        "reviewed": reviewed,
    }
    with paths["label_lock"].open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2)
        stream.write("\n")
    print(f"Locked {len(records)} labels ({reviewed} reviewed)")


def record_opening(batch: str, reason: str) -> None:
    if not batch_paths(batch)["label_lock"].exists():
        raise SystemExit("Lock the holdout labels before opening")

    batch_id = HOLDOUT_BATCHES[batch]["batch_id"]
    split = json.loads(SPLIT_PATH.read_text())
    if opening_record(split, batch_id) is not None:
        raise SystemExit("The holdout opening is already recorded")

    opening = {
        "batch_id": batch_id,
        "opened_at": date.today().isoformat(),
        "reason": reason,
    }
    split.setdefault("holdout_openings", []).append(opening)
    if split.get("holdout_opened_at") is None:
        # The first opening of the holdout as a whole.
        split["holdout_opened_at"] = opening["opened_at"]
        split["holdout_open_reason"] = reason
    SPLIT_PATH.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Recorded holdout opening:", opening["opened_at"])


def model_b_predictions(holdout_essays, holdout_records):
    """Refit Model B on development data; threshold by heuristic_precision."""
    development = []
    for batch in load_batches(DATASET_DIRECTORY):
        development.extend(
            prepare_essay_context_observations(
                batch["essays"], batch["annotations"]
            )
        )
    targets = observation_targets(development)
    groups = np.asarray(
        [observation["student_id"] for observation in development]
    )
    _, splits = select_grouped_splits(targets, groups)
    out_of_fold = run_grouped_oof(development, MODEL_B_FEATURE_NAMES, splits)
    probabilities = np.asarray(
        [r["unreliable_probability"] for r in out_of_fold["oof_records"]]
    )
    threshold = select_precision_floor_threshold(
        targets, probabilities, heuristic_warning_precision(development)
    )

    model = build_logistic_pipeline()
    model.fit(
        build_feature_matrix(development, MODEL_B_FEATURE_NAMES), targets
    )
    holdout = prepare_essay_context_observations(
        holdout_essays, holdout_records
    )
    positive_column = list(model.classes_).index(POSITIVE_CLASS)
    holdout_probabilities = model.predict_proba(
        build_feature_matrix(holdout, MODEL_B_FEATURE_NAMES)
    )[:, positive_column]

    predictions = {
        observation["sentence_id"]: int(probability >= threshold)
        for observation, probability in zip(holdout, holdout_probabilities)
    }
    return predictions, threshold


def evaluate(batch: str) -> None:
    paths = batch_paths(batch)
    settings = HOLDOUT_BATCHES[batch]
    if paths["report"].exists():
        raise SystemExit("The holdout was already evaluated once")

    lock = json.loads(paths["label_lock"].read_text())
    if sha256_file(paths["ledger"]) != lock["ledger_sha256"]:
        raise SystemExit("Holdout labels changed after locking")

    development_lock = json.loads(
        (DATASET_DIRECTORY / LOCK_RELATIVE_PATH).read_text()
    )
    if lock_mismatches(development_lock, DATASET_DIRECTORY):
        raise SystemExit("The development set changed")

    split = json.loads(SPLIT_PATH.read_text())
    opening = opening_record(split, settings["batch_id"])
    if opening is None:
        raise SystemExit("Record the holdout opening first")

    records = load_holdout_records(paths)
    essays = load_jsonl(paths["directory"] / "essays.jsonl")
    parser = SentenceParser(
        nlp, {essay["essay_id"]: essay for essay in essays}
    )
    verdicts = {
        version: heuristic_verdicts(records, parser, version)
        for version in ("1.2", *settings["candidates"])
    }
    model_b, threshold = model_b_predictions(essays, records)

    report = {
        "plan": paths["plan"].relative_to(PROJECT_ROOT).as_posix(),
        "evaluated_at": date.today().isoformat(),
        "holdout_opened_at": opening["opened_at"],
        "label_lock_date": lock["locked_at"],
        "model_b_threshold": round(float(threshold), 4),
        **evaluate_holdout(
            records,
            verdicts,
            model_b,
            candidates=settings["candidates"],
            decision_pair=settings["decision_pair"],
            positive_outcome=settings["positive_outcome"],
        ),
    }
    with paths["report"].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", choices=sorted(HOLDOUT_BATCHES), required=True
    )
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--lock-labels", action="store_true")
    step.add_argument("--record-opening", action="store_true")
    step.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reason")
    arguments = parser.parse_args()

    if arguments.lock_labels:
        lock_labels(arguments.batch)
    elif arguments.record_opening:
        if not arguments.reason:
            raise SystemExit("--record-opening needs --reason")
        record_opening(arguments.batch, arguments.reason)
    else:
        evaluate(arguments.batch)


if __name__ == "__main__":
    main()
