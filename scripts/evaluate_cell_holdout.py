"""Lock, open, and evaluate CELL holdout v1 exactly once.

Steps, in order, as fixed in holdout_v1_evaluation_plan.json:

1. --lock-labels: after labelling and owner review are complete, store the
   hash of the holdout ledger. Prints counts only.
2. --record-opening --reason TEXT: write the opening date and reason into
   split_v1.json.
3. --evaluate: check the label lock, the development set lock, and the
   opening record, then score v1.2, v1.3, and Model B once and write the
   aggregate report. An existing report is never replaced.
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
BATCH_DIRECTORY = CELL_DIRECTORY / "local/holdout_v1"
LEDGER_PATH = BATCH_DIRECTORY / "annotations_v1.jsonl"
REVIEW_PATH = BATCH_DIRECTORY / "packet/review.tsv"
LABEL_LOCK_PATH = BATCH_DIRECTORY / "labels_lock.json"
SPLIT_PATH = CELL_DIRECTORY / "split_v1.json"
PLAN_PATH = CELL_DIRECTORY / "holdout_v1_evaluation_plan.json"
REPORT_PATH = CELL_DIRECTORY / "holdout_v1_evaluation.json"


def load_holdout_records() -> list[dict]:
    template = load_jsonl(BATCH_DIRECTORY / "annotations_template.jsonl")
    return merge_progress(template, load_jsonl(LEDGER_PATH))


def check_review_complete(records: list[dict]) -> int:
    if any(record["manual_label"] is None for record in records):
        raise SystemExit("Not every holdout sentence is labelled")

    with REVIEW_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    reviewed = [int(row["position"]) for row in rows]
    for position in reviewed:
        if records[position - 1]["adjudication_status"] != "adjudicated":
            raise SystemExit(f"Review is incomplete at position {position}")
    return len(reviewed)


def lock_labels() -> None:
    records = load_holdout_records()
    reviewed = check_review_complete(records)
    lock = {
        "batch_id": "cell_reliability_holdout_v1",
        "locked_at": date.today().isoformat(),
        "ledger_sha256": sha256_file(LEDGER_PATH),
        "records": len(records),
        "reviewed": reviewed,
    }
    with LABEL_LOCK_PATH.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2)
        stream.write("\n")
    print(f"Locked {len(records)} labels ({reviewed} reviewed)")


def record_opening(reason: str) -> None:
    if not LABEL_LOCK_PATH.exists():
        raise SystemExit("Lock the holdout labels before opening")

    split = json.loads(SPLIT_PATH.read_text())
    if split["holdout_opened_at"] is not None:
        raise SystemExit("The holdout opening is already recorded")

    split["holdout_opened_at"] = date.today().isoformat()
    split["holdout_open_reason"] = reason
    SPLIT_PATH.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Recorded holdout opening:", split["holdout_opened_at"])


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


def evaluate() -> None:
    if REPORT_PATH.exists():
        raise SystemExit("The holdout was already evaluated once")

    lock = json.loads(LABEL_LOCK_PATH.read_text())
    if sha256_file(LEDGER_PATH) != lock["ledger_sha256"]:
        raise SystemExit("Holdout labels changed after locking")

    development_lock = json.loads(
        (DATASET_DIRECTORY / LOCK_RELATIVE_PATH).read_text()
    )
    if lock_mismatches(development_lock, DATASET_DIRECTORY):
        raise SystemExit("The development set changed")

    split = json.loads(SPLIT_PATH.read_text())
    if split["holdout_opened_at"] is None:
        raise SystemExit("Record the holdout opening first")

    records = load_holdout_records()
    essays = load_jsonl(BATCH_DIRECTORY / "essays.jsonl")
    parser = SentenceParser(
        nlp, {essay["essay_id"]: essay for essay in essays}
    )
    verdicts = {
        version: heuristic_verdicts(records, parser, version)
        for version in ("1.2", "1.3")
    }
    model_b, threshold = model_b_predictions(essays, records)

    report = {
        "plan": PLAN_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "evaluated_at": date.today().isoformat(),
        "holdout_opened_at": split["holdout_opened_at"],
        "label_lock_date": lock["locked_at"],
        "model_b_threshold": round(float(threshold), 4),
        **evaluate_holdout(records, verdicts, model_b),
    }
    with REPORT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--lock-labels", action="store_true")
    step.add_argument("--record-opening", action="store_true")
    step.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reason")
    arguments = parser.parse_args()

    if arguments.lock_labels:
        lock_labels()
    elif arguments.record_opening:
        if not arguments.reason:
            raise SystemExit("--record-opening needs --reason")
        record_opening(arguments.reason)
    else:
        evaluate()


if __name__ == "__main__":
    main()
