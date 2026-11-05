"""Build a blind annotation packet for a private CELL reliability batch.

The packet holds only what a labeller needs: each sentence with its
neighbours, the token parse, the clause heads, the annotation guideline,
and an empty label sheet. It contains no Syntax Reliability verdicts or
flags and no ML output, so a separate labelling session can work from the
packet alone. The packet stays in the ignored local batch directory.
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell_annotation import (  # noqa: E402
    SentenceParser,
    format_sentence_view,
    load_jsonl,
)
from src.nlp.pipeline import nlp  # noqa: E402


DATASET_README = PROJECT_ROOT / "data/syntax_reliability/README.md"
SCHEMA_PATH = PROJECT_ROOT / "data/syntax_reliability/schema_v1.json"
GUIDELINE_START = "## Reference labels"
GUIDELINE_END = "## Data integrity"
LABEL_SHEET_COLUMNS = (
    "position",
    "sentence_id",
    "label",
    "failure_types",
    "abstention_status",
    "note",
)


def guideline_text(readme_text: str) -> str:
    start = readme_text.index(GUIDELINE_START)
    end = readme_text.index(GUIDELINE_END)
    return readme_text[start:end].strip() + "\n"


def instructions_text(record_count: int, enums: dict) -> str:
    failure_types = ", ".join(enums["manual_failure_type"])
    abstentions = ", ".join(enums["abstention_status"])
    columns = "\t".join(LABEL_SHEET_COLUMNS)
    return f"""# Annotation task

Label all {record_count} sentences in `sentences.txt` following
`GUIDELINE.md`. Work only from the files in this folder. Do not open other
project files: the labels must not depend on any parser-reliability rule or
model, and none is shown here.

Each sentence is shown with its neighbours, its spaCy tokens (part of
speech, dependency, head), and the clause heads found by Clause Detection.
Judge whether the parse-derived clause inventory and structure are usable,
not whether the English is correct.

Fill `labels.tsv`, one tab-separated row per sentence, keeping the header:

    {columns}

- `position` and `sentence_id`: copy from the sentence header.
- `label`: RELIABLE, MOSTLY_RELIABLE, UNRELIABLE, or NOT_ANALYZABLE.
- `failure_types`: comma-separated names, or `-` for none. Allowed:
  {failure_types}. RELIABLE uses `-`.
- `abstention_status`: `none` for analyzable labels; for NOT_ANALYZABLE one
  of: {abstentions}.
- `note`: a short reason for any label other than RELIABLE (no tabs), or
  empty.
"""


def sentence_sheet(records: list[dict], parser: SentenceParser) -> str:
    blocks = []
    for position, record in enumerate(records, start=1):
        sentence = parser.sentence(record)
        sentences = parser.sentences(record["essay_id"])
        view = format_sentence_view(record, sentence, sentences)
        blocks.append(
            f"=== {position} / {len(records)}   id {record['sentence_id']}\n"
            f"{view}\n"
        )
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    arguments = parser.parse_args()

    batch_dir = arguments.batch_dir
    packet_dir = batch_dir / "packet"
    essays = load_jsonl(batch_dir / "essays.jsonl")
    records = load_jsonl(batch_dir / "annotations_template.jsonl")
    enums = json.loads(SCHEMA_PATH.read_text())["enums"]

    sentence_parser = SentenceParser(
        nlp, {essay["essay_id"]: essay for essay in essays}
    )
    sentences = sentence_sheet(records, sentence_parser)
    guideline = guideline_text(DATASET_README.read_text())
    instructions = instructions_text(len(records), enums)

    packet_dir.mkdir(exist_ok=False)
    (packet_dir / "sentences.txt").write_text(sentences, encoding="utf-8")
    (packet_dir / "GUIDELINE.md").write_text(guideline, encoding="utf-8")
    (packet_dir / "INSTRUCTIONS.md").write_text(
        instructions, encoding="utf-8"
    )
    (packet_dir / "labels.tsv").write_text(
        "\t".join(LABEL_SHEET_COLUMNS) + "\n", encoding="utf-8"
    )
    print(f"Packet written: {len(records)} sentences")


if __name__ == "__main__":
    main()
