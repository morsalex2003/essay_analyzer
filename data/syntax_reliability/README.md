# Syntax Reliability Dataset v1

## Purpose

This dataset records provisional model-assisted reference annotations about
whether parser-derived syntax and clause metrics are usable. It does not
evaluate the quality of a student's English and does not contain a writing
score.

Dataset v1 currently contains only the 12 responses from the frozen
Syntax Reliability v1.2 holdout (`source_stage = "v12_holdout12"`). It
contains canonical source text, provisional labels, and parser provenance.
Structural ML features will be generated separately in a later stage.

Corpus D contains provisional model-assisted reference annotations. Eight
high-priority records have additionally been reviewed and adjudicated by the
human dataset owner. The dataset as a whole must not be described as human
gold.

## Reference labels

- `RELIABLE`: the main parser-derived syntax and clause structure is usable.
- `MOSTLY_RELIABLE`: local POS or attachment errors exist, but the key
  syntax and clause metrics remain usable.
- `UNRELIABLE`: clause inventories, clause hierarchy, non-finite units,
  sentence segmentation, or another major structure is materially distorted.
- `NOT_ANALYZABLE`: the sentence is an abstention case, such as a source
  fragment or a parse state for which metrics should not be calculated.

The first binary task is defined as follows:

- `USABLE = RELIABLE + MOSTLY_RELIABLE`
- `UNRELIABLE = UNRELIABLE`
- `NOT_ANALYZABLE` is excluded from binary classification.

`manual_label` is the historical field name. The current labels are
provisional model-assisted reference annotations, not independent human gold.
Independent human adjudication of the complete dataset has not been completed.
Records with `adjudication_status = "adjudicated"` form the human-adjudicated
subset. `usable_binary_label` remains a provisional ML target.

Syntax Reliability heuristic predictions are not reference labels and are not
stored in this annotation ledger.

`annotation_origin` separates two ways a label was produced:

- `model_assisted_manual_review`: a model proposed the label and a person
  reviewed it. All Dataset v1 labels use this origin with
  `annotator_id = "codex_review_v1"`.
- `human_manual_review`: a person chose the label directly while viewing the
  parser output, without heuristic verdicts or ML predictions. CELL pilot
  labels made with `scripts/annotate_cell_reliability_pilot.py` use this
  origin with a pseudonymous `annotator_id` by default.

CELL pilot sessions run with `--model-assisted` record
`model_assisted_manual_review`: a model suggested each label and the person
accepted or changed it. Such sessions use their own `annotator_id`, so the
two kinds of labels stay separable in the ledger and in the evaluation.

Both origins are single-annotator labels unless `adjudication_status` says
otherwise; neither is independent multi-annotator gold by itself.

## Annotation guidelines

- `original_text` is the exact written learner text. Annotation must not
  silently reconstruct or correct spelling, grammar, or punctuation.
- If correcting a parser analysis would change the finite clause inventory,
  dependent clause inventory, non-finite predicative inventory, or matrix
  hierarchy, label the analyzable sentence `UNRELIABLE`.
- If a parser issue is local and the core syntax metrics remain usable, label
  the sentence `MOSTLY_RELIABLE`.
- A source fragment without a comparable matrix structure is
  `NOT_ANALYZABLE`, with `abstention_status = "source_fragment"`.

### Clarification: verbs without agreement

Learner texts often contain a predicate verb in the base form where standard
English needs agreement or tense marking (for example, a synthetic
"She always walk to school."). A verb counts as a **finite predication**
in the reference analysis when all of these hold:

- it has its own overt subject;
- it stands in predicate position: the matrix predicate, a conjoined
  predicate, or the predicate of a dependent clause;
- it has no infinitive marker `to`, and no modal or auxiliary already
  carries finiteness for it.

The missing agreement is the learner's grammar choice; the text is not
corrected, but the clause still counts as finite. If the parser or Clause
Detection treats such a verb as non-finite, the finite and non-finite
inventories both change, so the sentence is `UNRELIABLE` with
`finite_inventory` and `nonfinite_inventory`.

This clarification does not change two adjudicated Dataset v1 decisions:

- a base-form verb without its own subject that acts as a clausal subject
  (synthetic "Write clearly is important.") stays non-finite;
- a `to`-infinitive with no finite matrix predicate is an infinitival source
  fragment and stays `NOT_ANALYZABLE`.

The clarification was added on 2026-11-03, before the first CELL pilot label.
It resolves a case that guideline 1.2 left open and agrees with all existing
adjudicated labels, so `annotation_guideline_version` stays `1.2`.

### Clarifications from the CELL pilot review

These rules were applied consistently during the CELL Reliability Pilot v1
review on 2026-11-04. They interpret guideline 1.2 for recurring cases and
change no earlier label. Examples are synthetic.

- **Segmentation.** If the sentence segmenter moves a sentence boundary (for
  example, it splits a citation at "et." or glues a heading or list marker to
  the next sentence), the sentence is `UNRELIABLE` with
  `sentence_segmentation`, even when the clauses inside the span are parsed
  correctly, because sentence length, sentence count, and clauses per
  sentence are distorted. A full stop the learner omitted is not a segmenter
  error; such a sentence is labelled by what the parser did with its clauses.
  A paragraph or line break that the segmenter ignored is a segmenter error.
- **Missing verbs.** When the learner omits a verb but a modal or auxiliary
  with its own subject is present ("The prices will very high."), the clause
  is still a finite predication; if the parser loses it, the sentence is
  `UNRELIABLE` with `finite_inventory`. When no finite element is present at
  all ("Many students learning online."), the sentence is a fragment:
  `NOT_ANALYZABLE` with `source_fragment`.
- **Spurious non-finite units.** A noun tagged as a verb ("the risks of
  cyberbullying", "to curriculum demands") that adds a non-finite predicative
  unit makes the sentence `UNRELIABLE` with `nonfinite_inventory`, even when
  only one such unit is added.
- **Reduced participles.** A reduced relative or participial phrase ("the
  group taught online") counted as a finite dependent clause makes the
  sentence `UNRELIABLE` with `finite_inventory`, `dependent_inventory`, and
  `nonfinite_inventory`.

## Data integrity

- `student_id` is pseudonymized. Real student names must not be stored here.
- Original text is preserved without spelling, grammar, or punctuation fixes.
- All sentences from one essay must remain in the same ML split.
- Random sentence splitting is prohibited.
- Future revisions of one response must share a `response_family_id`.
- Serialized spaCy objects are not the canonical source. Exact text and frozen
  parser provenance allow the parse to be reproduced.
- Raw learner essays and sentence annotations are intentionally not
  version-controlled in the public repository. Dataset schema and validation
  code may be public; learner-written text remains private and local.

No train, development, or test split is defined in Dataset v1.

## Files

- `essays_v1.jsonl`: private local file with one complete learner response per
  line; intentionally ignored by Git.
- `annotations_v1.jsonl`: private local file with one frozen spaCy sentence
  and its provisional reference annotation per line; intentionally ignored by
  Git.
- `schema_v1.json`: fields, enum values, nullability, and label semantics.
- `../../scripts/validate_syntax_reliability_dataset.py`: integrity checks and
  frozen Corpus D totals.
