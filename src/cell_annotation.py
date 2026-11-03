"""Annotation session for the private CELL Reliability Pilot v1 batch.

The frozen ``annotations_template.jsonl`` is never modified. Labels are
written to a separate progress ledger with the same records in the same
order, so an interrupted session can continue where it stopped.

The annotator sees the exact sentence, its neighbouring sentences, and the
spaCy parse. Syntax Reliability heuristic verdicts and ML predictions are
deliberately not shown, so that they cannot steer the reference labels.
"""

import json
import os
from pathlib import Path

from src.syntax.clauses import detect_clauses


# Fields that the annotator may change. Every other template field is frozen
# provenance and must stay identical in the progress ledger.
ANNOTATION_FIELDS = (
    "annotation_origin",
    "manual_label",
    "usable_binary_label",
    "abstention_status",
    "manual_failure_types",
    "annotation_notes",
    "annotator_id",
)

# human_manual_review: the annotator chooses each label directly.
# model_assisted_manual_review: a model suggests the label and the annotator
# accepts or changes it. The session uses one origin for all its labels.
HUMAN_ANNOTATION_ORIGIN = "human_manual_review"
MODEL_ASSISTED_ANNOTATION_ORIGIN = "model_assisted_manual_review"

LABEL_KEYS = {
    "r": "RELIABLE",
    "m": "MOSTLY_RELIABLE",
    "u": "UNRELIABLE",
    "n": "NOT_ANALYZABLE",
}

# These abstention reasons have a failure type with the same name. The
# canonical Dataset v1 ledger always records both together.
ABSTENTION_FAILURE_TYPES = ("source_fragment", "parser_abstention")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []

    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip() != "":
                records.append(json.loads(line))

    return records


def write_jsonl_atomically(path: Path, records) -> None:
    """Replace the file in one step so a crash never leaves half a ledger."""
    temporary_path = path.with_name(path.name + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
            )
            output_file.write("\n")

    os.replace(temporary_path, path)


def load_annotation_enums(schema: dict[str, object]) -> dict[str, list]:
    """Return the allowed annotation values from ``schema_v1.json``."""
    enums = schema["enums"]

    return {
        "manual_label": list(enums["manual_label"]),
        "abstention_status": list(enums["abstention_status"]),
        "manual_failure_type": list(enums["manual_failure_type"]),
        "annotation_origin": list(enums["annotation_origin"]),
    }


def merge_progress(template_records, progress_records):
    """Return template records with labels from an existing progress ledger.

    The progress ledger must describe exactly the same frozen sentences.
    Only ``ANNOTATION_FIELDS`` may differ from the template.
    """
    if len(progress_records) != len(template_records):
        raise ValueError(
            "Progress ledger has a different number of records than the "
            "template."
        )

    merged_records = []

    for template_record, progress_record in zip(
        template_records,
        progress_records,
    ):
        sentence_id = template_record["sentence_id"]

        if set(progress_record) != set(template_record):
            raise ValueError(sentence_id + ": progress fields differ.")

        for field_name, template_value in template_record.items():
            if field_name in ANNOTATION_FIELDS:
                continue
            if progress_record[field_name] != template_value:
                raise ValueError(
                    sentence_id + ": frozen field changed: " + field_name
                )

        merged_records.append(dict(progress_record))

    return merged_records


def create_annotated_record(
    template_record: dict[str, object],
    manual_label: str,
    failure_types: list[str],
    abstention_status: str,
    annotation_notes,
    annotator_id: str,
    binary_mapping: dict[str, object],
    enums: dict[str, list],
    annotation_origin: str = HUMAN_ANNOTATION_ORIGIN,
) -> dict[str, object]:
    """Return a copy of one record with a validated label.

    The rules match ``validate_syntax_reliability_dataset.py`` so that the
    pilot ledger can later join the canonical dataset without rework.
    """
    if annotation_origin not in enums["annotation_origin"]:
        raise ValueError(
            "Dataset schema does not allow " + annotation_origin + "."
        )

    if manual_label not in enums["manual_label"]:
        raise ValueError("Unsupported manual_label.")

    if abstention_status not in enums["abstention_status"]:
        raise ValueError("Unsupported abstention_status.")

    for failure_type in failure_types:
        if failure_type not in enums["manual_failure_type"]:
            raise ValueError("Unsupported failure type: " + failure_type)

    if len(set(failure_types)) != len(failure_types):
        raise ValueError("Failure types must not repeat.")

    if manual_label == "NOT_ANALYZABLE":
        if abstention_status == "none":
            raise ValueError("NOT_ANALYZABLE needs an abstention reason.")
    elif abstention_status != "none":
        raise ValueError("Analyzable labels must use abstention_status none.")

    if (
        abstention_status in ABSTENTION_FAILURE_TYPES
        and abstention_status not in failure_types
    ):
        failure_types = failure_types + [abstention_status]

    if not isinstance(annotator_id, str) or annotator_id.strip() == "":
        raise ValueError("annotator_id must be a non-empty string.")

    if annotation_notes is not None:
        annotation_notes = annotation_notes.strip()
        if annotation_notes == "":
            annotation_notes = None

    record = dict(template_record)
    record["annotation_origin"] = annotation_origin
    record["manual_label"] = manual_label
    record["usable_binary_label"] = binary_mapping[manual_label]
    record["abstention_status"] = abstention_status
    record["manual_failure_types"] = list(failure_types)
    record["annotation_notes"] = annotation_notes
    record["annotator_id"] = annotator_id.strip()

    return record


def is_labelled(record) -> bool:
    return record["manual_label"] is not None


def label_counts(records) -> dict[str, int]:
    counts = {"pending": 0}

    for record in records:
        label = record["manual_label"]
        if label is None:
            counts["pending"] += 1
        else:
            counts[label] = counts.get(label, 0) + 1

    return counts


def first_pending_position(records) -> int | None:
    for position, record in enumerate(records):
        if not is_labelled(record):
            return position

    return None


class SentenceParser:
    """Parse each essay once and return frozen sentence spans."""

    def __init__(self, nlp, essays_by_id):
        self._nlp = nlp
        self._essays_by_id = essays_by_id
        self._documents_by_essay = {}
        self._sentences_by_essay = {}

    def document(self, essay_id: str):
        if essay_id not in self._documents_by_essay:
            self._documents_by_essay[essay_id] = self._nlp(
                self._essays_by_id[essay_id]["original_text"]
            )

        return self._documents_by_essay[essay_id]

    def sentences(self, essay_id: str):
        if essay_id not in self._sentences_by_essay:
            self._sentences_by_essay[essay_id] = list(
                self.document(essay_id).sents
            )

        return self._sentences_by_essay[essay_id]

    def sentence(self, record):
        sentences = self.sentences(record["essay_id"])
        sentence = sentences[record["sentence_index"] - 1]

        if sentence.text != record["original_text"]:
            raise ValueError(
                record["sentence_id"]
                + ": current parser segmentation does not reproduce the "
                "frozen sentence."
            )

        return sentence


def _head_list(heads, sentence) -> str:
    """Show clause heads as 'word (token number)' to match the token table."""
    parts = []

    for head in heads:
        parts.append(head.text + " (" + str(head.i - sentence.start + 1) + ")")

    if len(parts) == 0:
        return "-"

    return ", ".join(parts)


def format_sentence_view(record, sentence, sentences) -> str:
    """Return the text shown for one sentence: context, tokens, clauses."""
    sentence_index = record["sentence_index"]
    lines = []

    if sentence_index > 1:
        lines.append("  before: " + sentences[sentence_index - 2].text)

    lines.append(">> " + sentence.text)

    if sentence_index < len(sentences):
        lines.append("  after:  " + sentences[sentence_index].text)

    lines.append("")
    lines.append("  #  token            POS    dep         head")

    for token in sentence:
        if token.is_space:
            continue

        token_number = token.i - sentence.start + 1
        head_number = token.head.i - sentence.start + 1
        head_text = "ROOT"
        if token.head.i != token.i:
            head_text = str(head_number) + " " + token.head.text

        lines.append(
            "  "
            + str(token_number).rjust(2)
            + " "
            + token.text[:16].ljust(16)
            + " "
            + token.pos_.ljust(6)
            + " "
            + token.dep_.ljust(11)
            + " "
            + head_text
        )

    clauses = detect_clauses(sentence)
    lines.append("")
    for label, field_name in (
        ("finite clause heads:   ", "finite_clause_heads"),
        ("dependent clause heads:", "dependent_clause_heads"),
        ("non-finite heads:      ", "non_finite_heads"),
        ("unclassified heads:    ", "unknown_heads"),
    ):
        if field_name == "unknown_heads" and len(clauses[field_name]) == 0:
            continue
        lines.append(
            "  " + label + " " + _head_list(clauses[field_name], sentence)
        )

    return "\n".join(lines)


def _choose_numbers(prompt, options, input_function, allow_empty):
    """Ask for comma-separated option numbers; return the chosen values."""
    while True:
        answer = input_function(prompt).strip()

        if answer == "" and allow_empty:
            return []

        chosen = []
        valid = answer != ""

        for part in answer.split(","):
            part = part.strip()
            if not part.isdigit() or not 1 <= int(part) <= len(options):
                valid = False
                break
            chosen.append(options[int(part) - 1])

        if valid:
            return chosen


def _numbered_options(options) -> str:
    parts = []

    for number, option in enumerate(options, start=1):
        parts.append(str(number) + "=" + option)

    return "  " + "  ".join(parts)


def ask_annotation(
    record,
    annotator_id,
    binary_mapping,
    enums,
    input_function,
    print_function,
    annotation_origin=HUMAN_ANNOTATION_ORIGIN,
):
    """Ask for one label. Return a record, or a navigation command."""
    while True:
        answer = input_function(
            "Label [r]eliable [m]ostly [u]nreliable [n]ot analyzable, "
            "[s]kip, [b]ack, [q]uit: "
        ).strip().lower()

        if answer in ("s", "b", "q"):
            return answer

        if answer in LABEL_KEYS:
            break

    manual_label = LABEL_KEYS[answer]
    abstention_status = "none"
    failure_types = []

    if manual_label == "NOT_ANALYZABLE":
        reasons = []
        for status in enums["abstention_status"]:
            if status != "none":
                reasons.append(status)
        print_function(_numbered_options(reasons))
        abstention_status = _choose_numbers(
            "Abstention reason (one number): ",
            reasons,
            input_function,
            allow_empty=False,
        )[0]

    if manual_label in ("MOSTLY_RELIABLE", "UNRELIABLE"):
        print_function(_numbered_options(enums["manual_failure_type"]))
        failure_types = _choose_numbers(
            "Failure types (numbers separated by commas, Enter for none): ",
            enums["manual_failure_type"],
            input_function,
            allow_empty=True,
        )

    notes = input_function("Note (Enter for none): ")

    return create_annotated_record(
        record,
        manual_label,
        failure_types,
        abstention_status,
        notes,
        annotator_id,
        binary_mapping,
        enums,
        annotation_origin,
    )


def run_annotation_session(
    records,
    parser,
    annotator_id,
    binary_mapping,
    enums,
    save_function,
    input_function=input,
    print_function=print,
    start_position=None,
    annotation_origin=HUMAN_ANNOTATION_ORIGIN,
):
    """Walk through pending sentences and save after every label.

    ``records`` is updated in place. Returns the number of labels saved.
    """
    position = start_position
    if position is None:
        position = first_pending_position(records)

    saved_count = 0
    history = []

    while position is not None and position < len(records):
        record = records[position]
        sentence = parser.sentence(record)
        sentences = parser.sentences(record["essay_id"])
        labelled_count = len(records) - label_counts(records)["pending"]

        print_function("")
        print_function(
            "Sentence "
            + str(position + 1)
            + "/"
            + str(len(records))
            + "  (labelled "
            + str(labelled_count)
            + ")  "
            + record["sentence_id"]
        )
        if is_labelled(record):
            print_function(
                "Current label: "
                + record["manual_label"]
                + " "
                + str(record["manual_failure_types"])
            )
        print_function(format_sentence_view(record, sentence, sentences))

        result = ask_annotation(
            record,
            annotator_id,
            binary_mapping,
            enums,
            input_function,
            print_function,
            annotation_origin,
        )

        if result == "q":
            break

        if result == "b":
            if len(history) > 0:
                position = history.pop()
            continue

        if result != "s":
            records[position] = result
            save_function(records)
            saved_count += 1

        history.append(position)
        position += 1

        if start_position is None:
            remaining = first_pending_position(records[position:])
            if remaining is None:
                position = None
            else:
                position += remaining

    return saved_count
