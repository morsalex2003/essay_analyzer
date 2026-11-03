"""Score frozen Syntax Reliability v1.2 against CELL pilot labels.

Prints an aggregate JSON report: no learner text. Error lists contain
1-based ledger positions that can be reopened with
``annotate_cell_reliability_pilot.py --start-at N``.
"""

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.cell_annotation import (  # noqa: E402
    SentenceParser,
    load_jsonl,
    merge_progress,
)
from src.cell_pilot_evaluation import (  # noqa: E402
    evaluate_pilot,
    heuristic_verdicts,
)


CELL_DIRECTORY = REPOSITORY_ROOT / "data/syntax_reliability/external/cell"
PILOT_DIRECTORY = CELL_DIRECTORY / "local/pilot_v1"
TEMPLATE_PATH = PILOT_DIRECTORY / "annotations_template.jsonl"
PROGRESS_PATH = PILOT_DIRECTORY / "annotations_v1.jsonl"
ESSAYS_PATH = PILOT_DIRECTORY / "essays.jsonl"
SPLIT_PATH = CELL_DIRECTORY / "split_v1.json"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Score frozen v1.2 against CELL pilot labels."
    )
    parser.add_argument(
        "--origin",
        choices=("human_manual_review", "model_assisted_manual_review"),
        help=(
            "Score only labels with this annotation_origin; other labels "
            "are treated as pending."
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if not PROGRESS_PATH.exists():
        raise SystemExit(
            "No labels yet: run scripts/annotate_cell_reliability_pilot.py "
            "first."
        )

    records = merge_progress(
        load_jsonl(TEMPLATE_PATH),
        load_jsonl(PROGRESS_PATH),
    )

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    development_students = set(split["students"]["development"])
    for record in records:
        if record["student_id"] not in development_students:
            raise SystemExit(
                "Refusing to evaluate: a pilot record is outside the "
                "development split."
            )

    if arguments.origin is not None:
        filtered_records = []
        for record in records:
            if (
                record["manual_label"] is not None
                and record["annotation_origin"] != arguments.origin
            ):
                record = dict(record)
                record["manual_label"] = None
            filtered_records.append(record)
        records = filtered_records

    labelled_records = []
    for record in records:
        if record["manual_label"] is not None:
            labelled_records.append(record)

    essays_by_id = {}
    for essay in load_jsonl(ESSAYS_PATH):
        essays_by_id[essay["essay_id"]] = essay

    from src.nlp.pipeline import nlp

    verdicts = heuristic_verdicts(
        labelled_records,
        SentenceParser(nlp, essays_by_id),
    )
    report = evaluate_pilot(records, verdicts)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
