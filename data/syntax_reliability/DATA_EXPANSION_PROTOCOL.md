# Syntax Reliability Data Expansion v1

## Purpose

This protocol governs collection of new learner responses after Syntax
Reliability ML Baseline v1. New data should increase independent author and
essay diversity, extend parser-failure coverage, and provide structurally
similar usable controls. Increasing sentence count alone is not the goal.

The frozen canonical B+C+D dataset remains unchanged. Corpus A is
development-only and must not be used as final evaluation data.

## Collection unit

The collection unit is a complete learner response, not an isolated sentence.
Every response must retain its essay boundary and, where available, task,
prompt, response date, and revision provenance. Sentences may be created only
from a preserved complete response.

Real names and identity mappings must remain outside the public repository.
New learners receive pseudonymous IDs such as `student_n01`; existing IDs
`student_v` and `student_m` are not renamed.

## Data roles

Each new student receives exactly one role before model or label inspection.
All essays and revisions from that student inherit the same role.

### Development expansion

`data_role = "development"`

Development responses may be parsed, annotated, human-adjudicated, used for
error analysis, and used during Reliability ML v2 development and training.
Once inspected for development, they are not untouched evaluation data.

### Untouched student holdout

`data_role = "untouched_holdout"`

The holdout must contain at least one new student who is absent from the
current development dataset. Before the ML v2 model, features, threshold, and
evaluation procedure are frozen, holdout labels and predictions must not be
used for feature design, model comparison, threshold tuning, hyperparameter
selection, training, or error analysis.

Technical intake may preserve exact responses, essay boundaries, hashes,
pseudonymous identity, and provenance. Gold annotation may be prepared by an
independent workflow, but its results remain sealed from model development.

A holdout student cannot be moved into development because model performance
is poor. If holdout results influence a design decision, record the opening
event and retire that pool from final-holdout status.

## Student, essay, and revision grouping

- Never split one student's sentences or essays across development and
  untouched holdout.
- Never split one essay across ML partitions.
- Give revisions of one response a shared `response_family_id` and distinct
  `revision_index` values.
- Keep every revision family in one role and one ML group.
- Do not use `student_id`, `essay_id`, `data_role`, or provenance fields as
  predictive features.

## Private response metadata

The private intake ledger for each complete response should preserve:

- `student_id`;
- `essay_id`;
- `data_role`;
- `source_stage` and `batch_id`;
- `task_type`;
- `prompt_id`, when known;
- `response_date`, when known;
- `response_family_id`;
- `revision_index`;
- exact-text hash and parser provenance;
- sentence indexes derived from the complete response.

The private learner-to-pseudonym mapping, learner text, response hashes, and
private storage paths must not be placed in tracked public metadata.

## Public-safe batch manifest design

A future tracked manifest may contain one aggregate record per student within
each batch using the following fields. It must not contain learner text,
sentence text, real names, direct identifiers, private paths, or model
predictions.

| Field | Meaning |
|---|---|
| `manifest_version` | Public manifest schema version |
| `batch_id` | Pseudonymous batch identifier |
| `student_id` | Pseudonymous student identifier |
| `data_role` | `development` or `untouched_holdout` |
| `role_assigned_at` | Date when the role was fixed |
| `source_stage` | Dataset intake stage |
| `essay_count` | Number of complete responses |
| `task_types` | Non-identifying task categories |
| `collection_status` | `planned`, `received`, or `complete` |
| `annotation_status` | `not_started`, `in_progress`, `frozen`, or `sealed` |
| `freeze_status` | `open`, `source_frozen`, `annotation_frozen`, or `retired` |
| `raw_git_status` | Must report `ignored_untracked` for private raw data |
| `duplicate_audit_status` | Status of exact, normalized, and revision audit |
| `holdout_opened_at` | Null until a holdout is opened |
| `holdout_open_reason` | Null until a holdout is opened |

Role assignment and holdout-opening events should be append-only audit facts.
Changing an existing fact requires an explicit correction record rather than
silently overwriting history.

## Structural coverage targets

Collection must use complete responses and must not cherry-pick only parser
failures. After development annotation, track coverage of both
`UNRELIABLE` failures and `USABLE` controls for:

- subordinate clauses: `because`, `although`, `while`, `if`, `when`, `as`,
  relative clauses, and that-complements;
- shared-subject predicates, coordinated matrix clauses, coordinated
  subordinate clauses, and `and`/`but`/`or` coordination;
- infinitives, coordinated infinitives, participial modifiers, gerund
  clauses, and xcomp/ccomp-like learner structures;
- modal and auxiliary support such as `can`, `could`, `will`, `would`,
  `should`, and `might`;
- colon, semicolon, comma-linked structures, and discourse markers.

Especially retain correctly parsed negative controls with legitimate
infinitival chains, coordination, modal plus non-finite structures, relative
clauses, ccomp/xcomp structures, introductory non-finite phrases, and several
valid clause heads. Suspicious-looking structural counts alone are not a
reason to exclude a usable response.

Existing failure-family coverage remains:

- `finite_inventory`;
- `dependent_inventory`;
- `matrix_hierarchy`;
- `nonfinite_inventory`;
- `major_dependency_structure`;
- `sentence_segmentation`;
- `source_fragment`;
- `parser_abstention`;
- `other`.

Do not create a new annotation taxonomy during intake.

## Development annotation workflow

1. Preserve the complete exact response and provenance privately.
2. Audit sentence segmentation.
3. Run the frozen parser and clause detector.
4. Create provisional model-assisted reference annotations without using
   Model B predictions as gold.
5. Human-adjudicate high-priority cases.
6. Human-review a sample of `RELIABLE` controls.
7. Complete duplicate and revision audits.
8. Freeze the expansion batch.
9. Permit ML v2 training only after the freeze.

Model disagreement may prioritize later human review. It must not decide
which complete responses enter the collection.

## Untouched holdout workflow

1. Assign the student's holdout role before content, labels, or predictions
   are inspected for model development.
2. Preserve complete exact responses and provenance privately.
3. Create hashes and freeze source data.
4. Prepare annotations through an independent or operationally sealed path.
5. Freeze the model, features, threshold, and evaluation procedure.
6. Record the holdout-opening timestamp and reason.
7. Run one planned final evaluation pass.
8. Keep the opened holdout out of subsequent claims of untouched evaluation.

If complete operational separation of annotations is impossible, developers
must still avoid all holdout-derived design decisions before the evaluation
freeze.

## Duplicate and leakage protection

- Audit exact hashes, normalized duplicates, and near-duplicate revisions.
- Do not delete revisions automatically; group them to prevent leakage.
- Preserve prompt and revision metadata where available.
- Do not use sentence-level random splitting.
- Do not use current-model errors as the primary sampling mechanism.
- Keep raw learner responses ignored and untracked in the public repository.
- Do not expose real-name mappings, private paths, raw text, labels, or hashes
  through a public batch manifest.
- Treat Corpus A as auxiliary development evidence, never as a new final
  holdout.

## Reserve-data assignment

Do not read or parse reserve learner text before assigning its role. First,
using private identity metadata only:

1. establish which responses belong to the same student and revision family;
2. verify that the student is absent from the other pool;
3. assign the whole student to development or untouched holdout;
4. record the assignment and freeze it;
5. only then begin the workflow allowed for that role.

If student identity or revision relationships cannot be resolved safely, keep
the batch unassigned. Do not infer identity from writing content. Prefer
several new development students over many additional responses from one
existing student, and reserve at least one entirely new student for holdout.

## Intake checklist

- [ ] Full responses preserved
- [ ] Pseudonymous `student_id` assigned
- [ ] Development/holdout role assigned before ML inspection
- [ ] All responses from one student use one role
- [ ] Essay boundaries preserved
- [ ] Prompt and revision metadata preserved where available
- [ ] Exact hashes generated in the private ledger
- [ ] Exact, normalized, and near-duplicate audit completed
- [ ] Revision families grouped together
- [ ] Raw data ignored and untracked
- [ ] No sentence-level random split
- [ ] No model prediction used as a gold label
- [ ] No performance-driven response selection
- [ ] Holdout remains untouched, if applicable
- [ ] Holdout-opening event recorded, if applicable

## Progress and freeze criteria

Do not use a magic sentence-count threshold. Track:

- number of new students by role;
- complete essays per student;
- analyzable sentences;
- `UNRELIABLE` observations;
- existing failure-family coverage;
- structurally similar `USABLE` controls;
- duplicate and revision-family status;
- annotation and human-adjudication status.

Reliability ML v2 development may begin only after the selected development
expansion is frozen. Final holdout evaluation may begin only after the ML v2
model, feature set, threshold, and evaluation procedure are frozen.

## CELL integration v1

CELL intake is fixed at 2,857 canonical essays from 491 student IDs. The
`All/` tree is a mirror and is excluded. Source identity is the pair
`archive_path + sha256`; basename alone is unsafe because two different texts
share one basename. Student IDs are parsed as `s\d+` and must not be assumed
to have a fixed width.

`data/syntax_reliability/external/cell/split_v1.json` assigns 98 complete
students to untouched holdout and the remaining 393 to development using seed
`20260824`. Assignment uses `student_id`, period/course coverage, and
essays-per-student bins; labels, parser errors, text length, short/long flags,
and duplicate flags are excluded from the split decision. All essays,
duplicates, and possible revisions from one student inherit that student's
role.

The corpus-wide technical intake audit preceded role assignment. Its text
length and duplicate outputs were not used for assignment, and no Reliability
labels, predictions, parser-error analysis, or model results were inspected.
Accordingly, untouched status here means untouched by Reliability ML
development; it does not mean the files were absent from the technical audit.

CELL text for NLP is the HTML-unescaped content of `Intro`, `Body`, and
`Conclusion`, extracted case-insensitively by tag name and joined in that
canonical order. Learner language is not corrected. `Title`, `Ref`, and `App`
are excluded. Missing, duplicate, nested, mismatched, or unclosed structural
tags are intake errors rather than silently repaired input.

Exact and normalized duplicate metadata is retained without deleting source
rows. Near-duplicate/revision candidates are also retained and are never
removed automatically. Raw CELL files, source hashes, and the detailed intake
manifest remain local, ignored, and untracked.
