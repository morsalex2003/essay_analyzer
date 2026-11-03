"""Label CELL Reliability Pilot v1 sentences in the terminal.

Labels are saved after every sentence to the ignored local
``annotations_v1.jsonl``; the frozen template is never changed. Run the
script again to continue from the first unlabelled sentence.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.cell_annotation import (  # noqa: E402
    HUMAN_ANNOTATION_ORIGIN,
    MODEL_ASSISTED_ANNOTATION_ORIGIN,
    SentenceParser,
    label_counts,
    load_annotation_enums,
    load_jsonl,
    merge_progress,
    run_annotation_session,
    write_jsonl_atomically,
)


PILOT_DIRECTORY = (
    REPOSITORY_ROOT / "data/syntax_reliability/external/cell/local/pilot_v1"
)
TEMPLATE_PATH = PILOT_DIRECTORY / "annotations_template.jsonl"
PROGRESS_PATH = PILOT_DIRECTORY / "annotations_v1.jsonl"
ESSAYS_PATH = PILOT_DIRECTORY / "essays.jsonl"
PILOT_SCHEMA_PATH = PILOT_DIRECTORY / "annotation_schema.json"
DATASET_SCHEMA_PATH = (
    REPOSITORY_ROOT / "data/syntax_reliability/schema_v1.json"
)


def require_git_ignored(path):
    """Refuse to write learner text to a path that Git would track."""
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
            "Label CELL Reliability Pilot v1 sentences. Heuristic and ML "
            "predictions are not shown."
        )
    )
    parser.add_argument(
        "--annotator-id",
        required=True,
        help=(
            "Pseudonymous annotator ID stored with each label (never a "
            "real name). Labels are recorded as human_manual_review "
            "unless --model-assisted is given."
        ),
    )
    parser.add_argument(
        "--model-assisted",
        action="store_true",
        help=(
            "Record this session's labels as model_assisted_manual_review: "
            "a model suggested each label and the annotator reviewed it. "
            "Use a separate --annotator-id for such sessions."
        ),
    )
    parser.add_argument(
        "--start-at",
        type=int,
        help=(
            "Sentence number (1-based) to start from, including labelled "
            "sentences; use it to review earlier labels."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print label counts and exit.",
    )
    return parser.parse_args()


def check_parser_provenance(nlp, pilot_schema):
    import spacy

    provenance = pilot_schema["provenance"]
    current = {
        "spacy_version": spacy.__version__,
        "spacy_model_name": nlp.meta["lang"] + "_" + nlp.meta["name"],
        "spacy_model_version": nlp.meta["version"],
    }

    for field_name, current_value in current.items():
        if provenance[field_name] != current_value:
            raise SystemExit(
                "Parser differs from the frozen pilot provenance: "
                + field_name
                + " is "
                + current_value
                + ", expected "
                + provenance[field_name]
            )


def main():
    arguments = parse_arguments()
    template_records = load_jsonl(TEMPLATE_PATH)

    if PROGRESS_PATH.exists():
        records = merge_progress(template_records, load_jsonl(PROGRESS_PATH))
    else:
        records = [dict(record) for record in template_records]

    if arguments.summary:
        print(json.dumps(label_counts(records), indent=2, sort_keys=True))
        return

    require_git_ignored(PROGRESS_PATH)

    start_position = None
    if arguments.start_at is not None:
        if not 1 <= arguments.start_at <= len(records):
            raise SystemExit(
                "--start-at must be between 1 and " + str(len(records))
            )
        start_position = arguments.start_at - 1

    pilot_schema = json.loads(PILOT_SCHEMA_PATH.read_text(encoding="utf-8"))
    enums = load_annotation_enums(
        json.loads(DATASET_SCHEMA_PATH.read_text(encoding="utf-8"))
    )

    print("Loading spaCy...")
    from src.nlp.pipeline import nlp

    check_parser_provenance(nlp, pilot_schema)
    essays_by_id = {}
    for essay in load_jsonl(ESSAYS_PATH):
        essays_by_id[essay["essay_id"]] = essay

    annotation_origin = HUMAN_ANNOTATION_ORIGIN
    if arguments.model_assisted:
        annotation_origin = MODEL_ASSISTED_ANNOTATION_ORIGIN
    print("Labels in this session are recorded as " + annotation_origin + ".")

    saved_count = run_annotation_session(
        records,
        SentenceParser(nlp, essays_by_id),
        arguments.annotator_id,
        pilot_schema["binary_mapping"],
        enums,
        lambda current_records: write_jsonl_atomically(
            PROGRESS_PATH,
            current_records,
        ),
        start_position=start_position,
        annotation_origin=annotation_origin,
    )

    print("")
    print("Saved " + str(saved_count) + " label(s) this session.")
    print(json.dumps(label_counts(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Every confirmed label is already saved.")
