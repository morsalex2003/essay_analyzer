# Project History

This document records important architectural decisions,
design choices, and reasons behind them.

It should be updated when major changes affect the project structure,
analysis methodology, or future development direction.


---

# 2026-08-10

## Project Goal

### Decision

The project is being developed as an NLP-based English essay analysis system.

The long-term goal is to analyze IELTS and TOEFL writing using:

- linguistic features;
- statistical NLP;
- machine learning;
- transformer-based models;
- automated feedback generation.


### Reason

A simple rule-based checker is not sufficient for high-quality writing evaluation.

The project should combine interpretable linguistic metrics with future ML models.


### Consequences

New features should contribute to the overall NLP analysis pipeline.

Avoid building isolated tools without integration into the main architecture.


---

# 2026-08-10

## Lexical Analysis Module Separation

### Decision

Lexical analysis was separated into an independent module:
src/lexical/
├── preprocessing.py
├── diversity.py
├── frequency.py
├── sophistication.py
├── cefr.py
└── efllex.py


### Reason

Lexical analysis became too large and contained multiple independent components:

- lexical diversity;
- frequency analysis;
- lexical sophistication;
- CEFR vocabulary analysis;
- EFLLex analysis.

Keeping all logic in one file made the project harder to maintain and extend.

### Consequences

All future lexical-related features should be implemented inside:
src/lexical/

Do not add new lexical analysis functions to unrelated modules.

---

# 2026-08-10

## Shared NLP Pipeline Extraction

### Decision

Created `src/nlp/` as a common infrastructure layer.

### Reason

Lexical and syntax modules should not depend on each other.

### Consequences

Future NLP modules should reuse `src/nlp/` instead of creating their own
pipelines.

---

# 2026-08-10

## Lexical Normalization Boundary

### Decision

The shared `src/nlp/pipeline.py` module owns only the common spaCy pipeline.
Frequency-aware normalization and lexical token preprocessing belong to
`src/lexical/preprocessing.py`.

### Reason

Syntax analysis should not depend indirectly on `wordfreq` or other
lexical-specific behavior when it imports the shared spaCy pipeline.

### Consequences

Lexical modules use `src/lexical/preprocessing.py`, which depends on the shared
`src/nlp/pipeline.py`. Syntax and other non-lexical modules import the shared
pipeline directly and do not depend on lexical analysis.

---

# 2026-08-10

## Syntax Responsibility Migration

### Decision

Syntax responsibility moved from `statistics.py` to `src/syntax/`.

### Reason

Separating linguistic domains makes the project architecture clearer and the
syntax analysis easier to maintain.

### Consequences

`src/statistics.py` contains only general character and word-count utilities.
Sentence, POS, clause, and syntactic complexity functions belong to
`src/syntax/` and reuse the shared `src/nlp/pipeline.py`.

---

# 2026-08-16

## Clause Detection Methodology v2

### Decision

Created a single clause detector in `src/syntax/clauses.py`.

Clause-related metrics now distinguish:

- finite clauses;
- dependent finite clauses;
- non-finite predicative units.

`total_clauses` remains as a backward-compatible alias for the number of
finite clauses. Coordinated predicates are not automatically counted as
separate clauses; a coordinated finite predicate needs evidence of separate
predication, primarily its own subject.

Validation on learner English led to three conservative detector refinements:

- finite `parataxis` is recognized when it has independent predicate evidence;
- an explicit infinitive marker takes precedence over conflicting finite
  morphology in complement structures;
- verbal non-finite `pcomp` structures are counted as non-finite predicative
  units when they have no finite auxiliary or copular support.

### Reason

The previous `total_clauses` implementation counted `VERB` tokens. That proxy
mixed finite clauses with infinitives and participles, while also treating
shared-subject coordinated predicates as separate clauses.

### Consequences

All clause-related metrics in `src/syntax/complexity.py` use the shared clause
detector. Non-finite predicative units no longer increase `total_clauses` or
`clause_density`. Sentence-based `subordination_ratio` includes only sentences
with at least one finite dependent clause. `complex_sentence_percentage` uses
the same finite-only numerator and equals `subordination_ratio * 100`.

---

# 2026-08-16

## Syntax Reliability Layer

### Decision

Created an experimental syntax reliability layer in
`src/syntax/reliability.py`.

### Purpose

Flag potentially unreliable syntax interpretation without correcting grammar
or dependency parses. The layer uses an existing spaCy parse and the existing
clause detector results; it does not create another NLP pipeline or change
clause detection.

### Consequences

Syntax Reliability v1.1 replaces length-based fragment handling with
structural analyzability checks. It adds conservative checks for conflicts
between a non-finite root and finite dependency evidence, and for relative
roots whose dependency hierarchy suggests a finite matrix clause. The
unmarked comma-link warning now requires leftward hierarchy evidence.

The seven Academic Discussion responses used while developing v1.1 are
development regression data, not an untouched holdout corpus.

Syntax Reliability v1.2 adds conservative cross-consistency warnings for
coordination topology, non-root finite/non-finite evidence conflicts,
unrepresented finite predication, and a non-verbal ROOT with finite auxiliary
support. No new strong signals were added, and sentence and essay thresholds
remain unchanged. Mixed ADJ/VERB predicate coordination such as H5.S4 remains
a documented limitation because a broader rule also flags valid structures.

The nine responses previously used as the untouched v1.1 holdout are now v1.2
development evidence. A new untouched holdout corpus is required before v1.2
generalization can be evaluated.

Reliability categories are engineering heuristics, not probabilities,
proficiency scores, or grammar judgements. The provisional signals and
thresholds require validation on a new independent learner-English corpus
before the layer can be considered stable or used by scoring and feedback
systems.

---

# 2026-08-17

## Structural Reliability Features v1

### Decision

Created a deterministic structural feature schema for a future binary syntax
reliability classifier. The extractor consumes an existing spaCy sentence
parse and Clause Detection v2.1 summary without running another NLP pipeline.

Schema v1 provides two feature sets:

- structural-only parse, morphology, clause, dependency, and tree features;
- the same structural features plus individual frozen Syntax Reliability
  v1.2 evidence signals.

### Boundaries

The schema contains no lexical identity, lexical metrics, labels, student or
essay identifiers, source-stage metadata, final heuristic verdict, or
abstention decision. Corpus A remains excluded from the canonical dry run and
is development-only.

### Consequences

The first development baseline now compares structural features against
structural-plus-signal features, using essay IDs only for grouped data
splitting and never as predictive inputs.

## Structural Reliability ML Baseline v1

### Decision

Created a reproducible binary development baseline for distinguishing
`UNRELIABLE` syntax parses from `USABLE` parses. The experiment uses only the
canonical B+C+D data: 221 analyzable sentence observations from 28 essays,
with 178 `USABLE` and 43 `UNRELIABLE` targets. Corpus A and 11
`NOT_ANALYZABLE` records are excluded.

Both models use an essay-grouped, five-fold `StratifiedGroupKFold` evaluation
and a fold-local `StandardScaler` followed by balanced Logistic Regression.
Model A uses the 44 structural features. Model B uses those features plus the
19 frozen Syntax Reliability v1.2 evidence signals. The classification
threshold remains 0.5; no feature selection, threshold tuning, or
hyperparameter search was performed.

### Development OOF results

- Model A: precision 0.509, recall 0.674, F1 0.580, F2 0.633, balanced
  accuracy 0.759, average precision 0.605, TN 150, FP 28, FN 14, TP 29.
- Model B: precision 0.667, recall 0.744, F1 0.703, F2 0.727, balanced
  accuracy 0.827, average precision 0.809, TN 162, FP 16, FN 11, TP 32.
- Frozen v1.2 warning baseline: precision 0.941, recall 0.744, F1 0.831,
  F2 0.777, balanced accuracy 0.867, TN 176, FP 2, FN 11, TP 32.

### Boundaries

These are development estimates, not independent validation or final
generalization results. The dataset contains only 221 analyzable sentences,
28 essays, and two currently represented pseudonymized students, and it
participated in methodology development. The baseline does not claim TOEFL or
IELTS performance. Descriptive full-data coefficients are not causal and can
be unstable when features are correlated. More untouched data is required
before model or threshold selection.

## Reliability ML Baseline v1 Error Analysis

The Model B and frozen v1.2 true-positive sets overlap on 31 observations;
each system uniquely catches one additional `UNRELIABLE` observation, and
both miss 10. Matrix-hierarchy failures are the most frequent family among
the shared misses.

Model B produces 14 false positives not produced by frozen v1.2, while both
systems share two false positives. Fourteen of Model B's 16 false positives
have no active heuristic signal, so their main mechanism is structural-feature
pressure rather than direct warning-signal activation. Many false positives
and false negatives are far from the unchanged 0.5 threshold. The next
methodological step is more untouched, structurally targeted data rather than
threshold tuning, a stronger model, or an immediate feature-schema revision.

## Syntax Reliability Data Expansion v1

Future collection is divided into development expansion and an untouched
student-level holdout. Complete learner responses are assigned by student to
one role before model inspection; essays, sentences, and revisions from that
student cannot cross the boundary. Collection prioritizes additional authors,
parser-failure coverage, and structurally similar usable controls rather than
performance-driven sampling of suspicious sentences.

Corpus A remains auxiliary development evidence and cannot serve as final
evaluation data. Reliability ML v2 development may begin only after the
development expansion is frozen. The untouched holdout may be evaluated only
after its source and annotations are sealed and the ML v2 model, features,
threshold, and evaluation procedure are frozen.

## CELL Data Integration v1

Added a deterministic intake boundary for the CELL corpus without changing
Reliability ML features, model code, or thresholds. The canonical inventory
contains 2,857 essays from 491 student IDs; the duplicate `All/` mirror is
excluded, and source records use collision-safe `archive_path + sha256`
identity rather than basename.

The frozen split uses seed `20260824` and assigns whole students to either
development or untouched holdout. Assignment uses only student identity,
period/course coverage, and essays-per-student bins. Learner text is extracted
by structural tag name as `Intro + Body + Conclusion`, then HTML-unescaped
without language correction;
`Title`, `Ref`, and `App` are excluded from NLP input.

Raw learner files and the hash-bearing intake manifest remain ignored and
untracked. Exact, normalized, and near-duplicate/revision metadata is retained
without automatic source-row deletion.

---

# 2026-08-27

## CELL Reliability Pilot v1

Prepared a deterministic annotation pilot using only CELL development
students. Student selection uses structural filename metadata; essay and
sentence selection uses parser-derived structural coverage without
Reliability labels, reviewed errors, heuristic verdicts, ML predictions,
quality, length, or duplicate flags. The batch intentionally includes simple
negative controls alongside coordination, finite subordination, relative
clauses, ccomp/xcomp, non-finite predicative units, modal/auxiliary support,
and punctuation-linked structures.

Complete selected essays and the R/MR/U/NA annotation template remain in the
ignored local CELL directory. The tracked pilot report contains aggregate
composition and frozen parser provenance only. The untouched holdout remains
unopened, and Reliability ML v1 was neither retrained nor tuned.

The frozen batch contains 35 development students, 35 complete essays, and
420 annotation sentences. It covers all eight periods and 23 courses. Its
overlapping structural counts are: coordination 136, finite subordination
110, relative clauses 60, ccomp/xcomp 93, non-finite predicative units 137,
modal/auxiliary support 187, punctuation-linked structures 185, and 130 simple
controls.

---

# 2026-10-29

## Writing Analysis Application Boundary v1

### Decision

Created a versioned `analyze_writing()` application interface that combines
the existing lexical, syntax, and syntax-reliability results in one
JSON-compatible structure. The interface accepts prepared vocabulary lookups
instead of loading datasets for every request and reuses one spaCy `Doc`
through the analysis pipeline.

### Reason

The existing modules were reusable, but `main.py` was the only place where
their results were assembled. A stable application boundary is required
before adding a web resource, Telegram client, revision workflow, or model
orchestration.

### Consequences

TTR and MATTR retain their existing algorithms, but their different token
policies are now explicit in the result schema. Repetition uses observed
surface forms, while frequency-dependent normalization remains limited to
metrics that require word-frequency lookup. Sentence-level reliability records
include text offsets for future evidence-based feedback. Lexical evidence links
repeated surface forms and unknown frequency-list words to their exact text
offsets and sentence numbers. EFLLex stays outside the default result and is
available only through an explicit research-metrics option because its license
is noncommercial. `main.py` now consumes the shared application result instead
of independently rerunning all analysis modules.

## Evidence-Based Reporting Boundary v1

### Decision

Added separate deterministic student and teacher report builders on top of the
versioned analysis result. The student report limits itself to concise,
evidence-backed review prompts and keeps parser reliability warnings separate
from writing feedback. The teacher report preserves the complete diagnostic
metrics, evidence offsets, parser flags, and token-policy notes.

### Boundaries

The reports do not assign TOEFL, IELTS, proficiency, or writing-quality scores.
An unknown frequency-list word is not labelled as a spelling error, repetition
is not automatically labelled as poor writing, and syntax reliability remains
a warning about measurement quality rather than a grammar judgement. No model
or external service is used to generate the reports.

## Revision Comparison Boundary v1

### Decision

Added a deterministic comparison for two versioned writing-analysis results.
It reports metric deltas and tracks which unknown frequency-list words and
repeated surface forms are no longer present, remain present, or are newly
present, preserving evidence offsets for both revisions.

### Boundaries

Metric direction is intentionally not labelled as improvement or decline.
Syntax reliability changes describe parser confidence rather than writing
quality, and optional CEFR percentage changes remain descriptive. Revision
comparison does not use an external model or modify either analysis result.

## Writing Task Context Boundary v1

### Decision

Added a versioned, validated task-context structure for prompts, response modes,
instructions, source materials, and explicit requirements. Task context remains
separate from learner response text and is copied into the application result
without entering lexical or syntax measurements. Revision comparison requires
both analyses to refer to the same canonical task context.

### Boundaries

The current system stores task context but does not infer prompt fulfillment,
source use, argument quality, or an exam score. Student and teacher reports
explicitly mark task alignment as `not_evaluated`. The response modes are
generic writing modes rather than claims about a particular current exam
format.

## Evidence-Grounded Task Alignment Contract v1

### Decision

Task requirements now have stable IDs, and a separate alignment-assessment
contract records requirement-level statuses, exact response offsets, and
explicit assessor provenance. Assessments are bound to hashes of the canonical
task context and exact learner response. Reports can display the supplied
assessment but do not calculate one themselves.

### Boundaries

The application does not infer semantic task fulfillment, aggregate statuses
into a score, or treat model output as hidden ground truth. `met` and
`partially_met` require response evidence; `not_met` requires an explanatory
note; `not_assessed` is an explicit abstention. Human, model, and model-assisted
assessors remain visibly distinguishable.

## Alignment Review Orchestration v1

### Decision

Added a provider-neutral coordination layer for comparing two or more valid
task-alignment assessments. It preserves each assessor's status, evidence, and
note for every requirement and records either provisional agreement or explicit
disagreement. Teacher reports may include the unresolved review.

### Boundaries

The orchestration layer does not call external models, vote, weight assessors,
or convert model agreement into ground truth. Any disagreement remains
unresolved until a separately identified human adjudicator supplies a complete
evidence-grounded assessment. Adjudication creates a new result and does not
mutate the original review.

## Alignment Provider Interface v1

### Decision

Added a provider-neutral synchronous adapter for future local or remote
alignment models. It sends only canonical task context and learner response,
requires structured completion or explicit abstention, validates all returned
evidence through the alignment contract, and records provider/model provenance,
hashes, elapsed time, and a bounded diagnostic reason code.

### Boundaries

No concrete external API, credentials, retry policy, or paid service is
configured. Local and remote providers require distinct explicit data-handling
policies; remote processing must be authorized by the caller before learner
text is sent. The synchronous adapter passes a timeout to the provider and
fails closed on `TimeoutError` or a late return, but the concrete provider is
responsible for interrupting its own blocking request. Invalid model output
never becomes an assessment. Provider run records do not retain a standalone
full-response field or raw exception messages, but completed assessments retain
exact learner evidence quotations and remain private. Only completed compatible
runs enter orchestration; abstentions and failures remain visible in provenance
but cannot vote.

## Task Alignment Evaluation Infrastructure v1

### Decision

Added a public-safe schema and validator for a future requirement-level task
alignment evaluation dataset. Canonical task, learner-response, and reference
assessment JSONL files are local-only. The validator checks exact text hashes,
canonical task and alignment contracts, one human reference per response,
pseudonymous participant declarations, consent/use status, and task/response
identity.

### Boundaries

No learner records, prompts, source material, examples derived from real work,
or evaluation split assignments were added. Development and sealed-evaluation
roles are defined, and participants and response families cannot cross them.
Provider predictions and orchestration agreement are not reference labels.

## Task Alignment Annotation Pilot Protocol v1

### Decision

Added a public annotation protocol and a local-only worklist generator for the
development portion of the task-alignment dataset. At least two pseudonymous
reviewers receive independent copies containing only canonical task context,
exact learner response text, and blank requirement-level annotation fields.

### Boundaries

Worklists exclude provider predictions, NLP metrics, other reviewers' labels,
and suggested decisions. Every requirement-level disagreement requires human
adjudication; voting is not used. The pilot workflow rejects sealed-evaluation
records, stores private worklists only under an ignored local directory, and
does not add learner data to the repository. Development-pilot results cannot
be reported as untouched or production evaluation results.

Completed worklists are validated through the canonical evidence-grounded
alignment contract before comparison. Reviewers must cover identical task and
response records, and unexpected prediction fields are rejected. Status
agreement remains provisional; disagreements produce a private human
adjudication queue preserving all source reviews. Final adjudication is a new,
complete human assessment rather than a vote or mutation of reviewer records.

## Task Alignment Evaluation Protocol v1

### Decision

Added a provider-neutral offline evaluator for comparing one provider's run
records with canonical human task-alignment references. It reports run-status
coverage, conditional requirement-status agreement, strict response agreement,
a four-status confusion matrix, per-status diagnostics, and privacy-safe
mismatch identifiers. Evidence-span equality and overlap are retained only as
descriptive diagnostics.

### Boundaries

Development and sealed-evaluation inputs must remain role-pure and are never
combined. Sealed evaluation requires explicit authorization. Conditional
agreement excludes non-completed runs but must always be interpreted alongside
coverage; undefined denominators remain `None`. Evidence overlap is not treated
as semantic correctness. Aggregate output contains no learner text and does
not represent an essay, proficiency, or exam score.

---

# 2026-10-30

## Task Alignment Experiment Freeze v1

### Decision

Added a deterministic experiment manifest that binds role-pure dataset files,
canonical dataset content, protocol versions, provider specification, adapter
version, prompt-template hash, and a closed generation configuration. The
frozen evaluator validates all current inputs against this manifest and binds
the result to its experiment ID and manifest SHA-256.

### Boundaries

The manifest contains counts and hashes but no prompt, source material, learner
text, evidence quotation, or participant ID. Provider configuration cannot
carry arbitrary credential fields. SHA-256 detects configuration drift but is
not an authenticity signature. Development and sealed manifests remain
separate; sealed creation and evaluation both require explicit authorization.
Local artifacts are stored only under the ignored alignment directory and are
never overwritten automatically.

## Task Alignment Strict JSON Adapter v1

### Decision

Added a versioned, hashed task-alignment prompt template and a provider-neutral
JSON adapter. The prompt separates fixed instructions from untrusted task and
learner data, prohibits scoring and correction, and requests requirement-level
statuses with exact evidence offsets. The adapter accepts only plain JSON with
a closed output shape and binds every run to its adapter and prompt provenance.

### Boundaries

No external model SDK, endpoint, credential, or paid API was added. A concrete
completion callable must be injected and remains responsible for timeout and
authorized data handling. Markdown, commentary, unknown fields, and malformed
JSON are rejected rather than repaired. Raw completion text is not retained;
canonical evidence quotations remain in private provider-run assessments.
Schema validity does not establish semantic correctness, so human-reference
evaluation remains required before learner-facing use.

## Task Alignment Frozen Batch Runner v1

### Decision

Added an in-memory batch runner that validates the frozen experiment before the
first provider call and records one outcome for every response. Human references
are used only for manifest validation and never enter prompt packages. Batch
artifacts retain experiment, manifest, provider, adapter, and run-status
provenance and can be passed directly to the frozen evaluator.

### Boundaries

One response failure does not stop later records, and every completed,
abstained, timed-out, failed, or invalid run remains visible. Batch summaries
are validated against their run records; detached or tampered summaries are
rejected. The project still provides no concrete external client. Completed
runs contain learner evidence quotations, so batches remain private and must be
saved only under the ignored local directory using create-only persistence.

## Task Alignment Data-First Development Pilot v1

### Decision

Deferred concrete model selection until a human-annotated development pilot is
complete. Added an aggregate readiness gate for canonical development tasks and
responses. It checks task requirements, role purity, task representation,
response-mode coverage, participant coverage, consent/right categories, and
explicit privacy and reviewer declarations without returning learner text.

### Boundaries

The initial operational targets are 12 responses, 3 tasks, 6 pseudonymous
participants, 2 response modes, 2 independent reviewers, and a separately
recorded adjudication role. These are manageable annotation-pilot targets, not
a power analysis or production-performance threshold. Smaller valid data are
reported as below plan; missing requirements or unconfirmed privacy, consent,
reviewer, and adjudication safeguards block annotation. Existing corpora cannot
be repurposed without confirming their rights and consent scope.

---

# 2026-11-03

## Task Alignment Intake Importer v1

### Decision

Added `src/alignment_intake.py` and `scripts/import_alignment_intake.py` to
convert hand-prepared intake files (`local/intake/tasks.json` and
`local/intake/responses.jsonl`) into the canonical private
`tasks_v1.jsonl` and `responses_v1.jsonl` records. The importer reuses
`create_task_context()` and `validate_alignment_source_records()` instead of
duplicating validation.

### Boundaries

All imported records are `development`; the importer cannot create sealed
evaluation data. Learner text is copied exactly and hashed without trimming or
correction. Closed intake field sets reject unexpected columns, duplicate
requirement descriptions are rejected instead of silently dropped, and
source-based tasks require source material. The script refuses to overwrite
existing canonical files or to write to a path not ignored by Git, and prints
aggregate counts only. Readiness declarations remain explicit CLI
confirmations in the existing readiness audit.

## Short-Text Measurement Notes v1

### Decision

`calculate_mattr()` returns plain TTR when a text has fewer content words than
one window, but results and reports still labelled the value MATTR. Analysis
results now record `lexical.diversity.mattr_method` (`moving_average`,
`ttr_fallback_below_window`, or `no_tokens`). Reports add a
`text_length_measurement` note when a text has fewer than 150 words or when
MATTR fell back to TTR; revision comparisons show the MATTR method for both
revisions. The CLI demo prints the method and the short-text note.

### Boundaries

The 150-word threshold is the existing README guidance, not a validated
cut-off. The note is a measurement warning, not a judgement about the writing,
and no metric value or score changes. The field is additive, so
`ANALYSIS_SCHEMA_VERSION` stays 1.0.

## Command-Line Interface v1

### Decision

`main.py` used to analyze only `essays/sample.txt` and print raw metrics, so
student and teacher reports and revision comparison were reachable only from
Python. It now accepts an essay path, `--report metrics|student|teacher`,
`--compare REVISED_ESSAY`, `--json`, and `--research`. Text formatting moved
into `src/text_reports.py`, which only formats results built elsewhere, so the
text and JSON views describe the same data. `main.py` only parses arguments,
runs the analysis, and prints.

### Boundaries

EFLLex is now opt-in on the command line (`--research`), matching its
research-only license and the existing `include_research_metrics` option; the
default metric listing is otherwise unchanged, and with `--research` it is
identical to the previous demo output. Missing, empty, and non-UTF-8 files fail
with a short message. `--compare` cannot be combined with `--report`.

## CELL Reliability Pilot Annotation Tool v1

### Decision

The 420 prepared CELL pilot sentences could only be labelled by editing long
JSONL records by hand. Added `src/cell_annotation.py` and
`scripts/annotate_cell_reliability_pilot.py`, a terminal session that shows
each sentence with its neighbours, the frozen spaCy parse, and detected clause
heads, and records R/MR/U/NA labels, failure types, abstention reasons, and
notes. Allowed values are read from `schema_v1.json`, and record rules match
the canonical dataset validator.

### Boundaries

Heuristic verdicts and ML predictions are not shown to the annotator. Labels go
to a separate ignored `annotations_v1.jsonl` written atomically after every
sentence; the frozen template stays unchanged, and frozen fields cannot drift
because the ledger is checked against the template on every load. The tool
refuses to run with a spaCy version or model that differs from the pilot
provenance. `annotation_origin` keeps the template value
`model_assisted_manual_review`; the schema has no separate value for direct
human annotation yet. The holdout remains unopened.

## Human Annotation Origin v1

### Decision

Added `human_manual_review` to the `annotation_origin` enum in
`data/syntax_reliability/schema_v1.json`. It marks labels that a person chose
directly while viewing parser output, as opposed to
`model_assisted_manual_review`, where a model proposed the label and a person
reviewed it. The CELL pilot annotation tool now records
`human_manual_review` on every label it saves and refuses to run against a
schema without that value; the pilot validator requires it on labelled
records. This supersedes the open point in Annotation Tool v1.

### Boundaries

The decision was made before any CELL pilot label existed, so no record needs
relabelling. Dataset v1 is unchanged: its 232 labels stay
`model_assisted_manual_review` / `codex_review_v1`, and its validator still
requires exactly that. The enum change is additive, so record
`schema_version` stays 1.0. Pending template records keep the template origin
until they are labelled. A single human annotator is still not
multi-annotator gold.

## CELL Reliability Pilot Evaluation v1

### Decision

Added `src/cell_pilot_evaluation.py` and
`scripts/evaluate_cell_reliability_pilot.py` to score the frozen Syntax
Reliability v1.2 heuristic against human CELL pilot labels. The heuristic runs
on each sentence inside its essay parse with the essay-level clause summary,
which is what `analyze_writing()` reports and what the annotator sees. The
report gives a four-label by four-verdict table, binary UNRELIABLE metrics at
the `warning` operating point (medium or low, as in the Dataset v1 baseline)
and a stricter `low_only` point, 95% Wilson intervals for precision and
recall, warning recall by failure type and selection structure, and ledger
positions of errors for review in the annotation tool.

### Boundaries

Dataset v1 parsed sentences in isolation, so pilot and Dataset v1 numbers are
related but not identical measurements. Analyzable sentences on which the
heuristic abstains are counted separately rather than scored. Partial
labelling is reported as `partial` with coverage counts. The script refuses
records outside the development split; the holdout remains unopened. No
threshold, rule, or model was changed, and the report contains no learner
text.

## Annotation Guideline Clarification: Verbs Without Agreement

### Decision

A base-form predicate verb with its own overt subject, in predicate position,
without `to` and without a modal or auxiliary carrying finiteness, counts as a
finite predication in reference labels, even though the learner omitted
agreement or tense marking. When the parser or Clause Detection treats such a
verb as non-finite, the sentence is `UNRELIABLE` with `finite_inventory` and
`nonfinite_inventory`. The rule is written in
`data/syntax_reliability/README.md`.

### Reason

The case arose on the first CELL pilot sentence and will recur often in learner
text. Guideline 1.2 did not cover it, and inconsistent handling would distort
the pilot evaluation of the heuristic.

### Boundaries

The rule was checked against Dataset v1 before adoption. Two adjudicated
decisions stay unchanged because they fall outside its conditions: a base-form
clausal subject without its own subject remains non-finite, and a
`to`-infinitive without a finite matrix predicate remains an infinitival
source fragment. No CELL pilot label existed yet, so nothing needs
relabelling, and `annotation_guideline_version` stays 1.2. Public examples are
synthetic; no learner sentence is quoted.

## CELL Pilot Model-Assisted Labelling Option

### Decision

The dataset owner chose to label the CELL pilot with model suggestions. The
annotation tool gained `--model-assisted`, which records every label of that
session as `model_assisted_manual_review` instead of `human_manual_review`;
such sessions use a separate `annotator_id`. The pilot validator accepts both
origins, and `evaluate_cell_reliability_pilot.py --origin ...` scores one
kind of label at a time.

### Boundaries

The origin is fixed per session, so one label never mixes the two processes.
Model-assisted pilot labels have the same status as Dataset v1 labels:
provisional reference annotations, not independent human gold, and the
evaluation report must state their origin counts. No label existed when the
option was added.

# 2026-11-04

## CELL Reliability Pilot v1 Labelling and Review

### Decision

All 420 pilot sentences were labelled as `model_assisted_manual_review`:
sentences 1-20 in an owner session with model suggestions
(`owner_model_assisted_v1`), sentences 21-420 by the model
(`claude_review_v1`). The dataset owner then reviewed every `UNRELIABLE`,
`NOT_ANALYZABLE`, and `MOSTLY_RELIABLE` model label (138) and a seeded random
sample of 40 of 262 model `RELIABLE` labels (seed 20261105). All 138 were
confirmed; in the sample, 37 were confirmed and 3 were changed from
`RELIABLE` to `MOSTLY_RELIABLE` for missed local errors. Reviewed records are
`adjudicated` with `adjudicator_id = human_owner_v1`; the ledger and the
annotation tool now support the optional schema adjudication fields, and a
relabel clears an earlier review. Recurring cases were written down as
guideline clarifications in `data/syntax_reliability/README.md`.

### Results

Label totals: 270 `RELIABLE`, 56 `MOSTLY_RELIABLE`, 59 `UNRELIABLE`, 35
`NOT_ANALYZABLE`. Frozen Syntax Reliability v1.2, scored in essay context on
379 analyzable sentences (53 `UNRELIABLE`; 6 analyzable sentences where the
heuristic abstained are excluded):

- `warning` (medium or low): precision 0.786 (95% CI 0.605-0.898), recall
  0.415 (0.293-0.549), F1 0.543, balanced accuracy 0.698, TP 22, FP 6, FN 31,
  TN 320.
- `low_only`: precision 1.000 (0.646-1.000), recall 0.132 (0.065-0.248).
- Warning recall by failure type: matrix hierarchy 0.737, dependent inventory
  0.448, finite inventory 0.417, non-finite inventory 0.360, major dependency
  structure 0.333, sentence segmentation 0.250.

On Dataset v1 the same operating point had precision 0.941 and recall 0.744.
Several misses come from Clause Detection rather than the spaCy parse (for
example, finite "should have prepared" or "children progress" classified as
non-finite), which is a separate follow-up.

### Boundaries

These are development estimates: the pilot uses development students only,
and the holdout remains unopened. Labels are model-proposed; the owner's
review was not blind, since it saw the proposed label and rationale. 222
model `RELIABLE` labels remain unreviewed; the sample suggests some should be
`MOSTLY_RELIABLE`, which does not change the binary target. Dataset v1 parsed
sentences in isolation, so its numbers are not directly comparable. No rule,
threshold, or model was changed.

## Clause Detection v2.2

### Decision

The CELL pilot review showed Clause Detection v2.1 classifying some finite
predicates as non-finite even when the spaCy parse was usable. v2.2 counts a
verb as finite when (A) a modal is attached to the next auxiliary instead of
the main verb ("should" -> "have" in "should have prepared"), (B) a VBN-tagged
verb has an active subject and no auxiliary (a past tense mis-tagged as a
participle, "than Piaget suspected"), or (C) a base-form verb has an active
subject, no "to", and no auxiliary, unless it follows a bare-infinitive
governor such as "make", "let", "help", or "see". Rule C implements the
guideline clarification on verbs without agreement. `detect_clauses()` takes
`detector_version` (default "2.2") and records it in the summary.

### Boundaries

Syntax Reliability v1.2 and Reliability ML feature schema v1 are frozen on
v2.1: given a newer summary, they recompute a v2.1 summary from the same parse
(`frozen_clause_summary()`). Without this, rule C would silence the v1.2
`no_finite_clause` and `nonfinite_root_with_subject` signals, which fire on
the old misclassification. After the change the Reliability ML v1 baseline,
the frozen v1.2 warning baseline, and the CELL pilot evaluation reproduce
their recorded numbers exactly. The CELL pilot tool shows the detector version
stamped on each record (v2.1), and pilot preparation and ML error analysis
stay on v2.1.

On the CELL pilot, v2.2 changes the clause inventory of exactly four
sentences, all `UNRELIABLE` under v2.1; on Dataset v1, rules A and B change
nothing and rule C changes four `UNRELIABLE` sentences, two of them corrected
and two already-broken parses whose error changes form. No `RELIABLE` or
`MOSTLY_RELIABLE` sentence changes. Reference labels stay tied to v2.1 through
`clause_detector_version`; a v2.2 evaluation needs labels judged against v2.2
output. The rules were designed on development data and are not independent
validation.

## CELL Pilot v1 Error Analysis of Syntax Reliability v1.2

### Decision

Grouped the frozen v1.2 heuristic's errors on the reviewed CELL pilot in
`data/syntax_reliability/external/cell/pilot_v1_error_analysis.md`
(aggregate counts and synthetic examples only). Comma-spliced clauses are its
strongest area. The 31 misses split into nouns parsed as verbs (8),
segmentation errors (6), clause attachment errors without a comma cue (6),
noun/verb part-of-speech swaps (4), reduced participles counted as finite (3),
Clause Detection v2.1 misclassifications now fixed in v2.2 (2), and modals
with a missing verb outside the root (2). Five of six false warnings come from
`nominal_predicate_coordination` on genuine but local attachment errors.

### Boundaries

The heuristic was not changed. The listed candidate signals are directions
for a future version, with counts that are upper bounds from the same
development data; they must be evaluated on data they were not designed on,
ultimately the untouched holdout.

## Sentence Segmentation v2

### Decision

The CELL pilot review found segmenter errors in 8 `UNRELIABLE` sentences:
headings and list markers glued to the next sentence and citations split at
"et." or "para.". Added the `learner_sentence_boundaries` spaCy component
(`src/nlp/segmentation.py`), which runs before the parser and presets only
boundaries it can decide from layout and punctuation: no boundary after a
citation abbreviation followed by a lowercase word or digit; a boundary after
a blank line when the next word is not lowercase; and a boundary after a
single line break only when the previous line has at most 8 tokens (or is a
numbered list marker) and the next line starts with a capital letter. A
single line break alone is not a boundary, because learners hard-wrap long
sentences. `analyze_writing()` uses this pipeline and records
`processing_versions` (sentence segmentation "2", clause detection "2.2").

### Boundaries

The shared `nlp` pipeline keeps the original en_core_web_sm segmentation,
because Syntax Reliability datasets, the CELL pilot, and Reliability ML v1
must reproduce their recorded sentences; the pilot validator, the pilot
evaluation, and the ML baseline reproduce their recorded results exactly.
On the 35 pilot essays the component corrects 7 of the 8 reviewed
segmentation errors; the remaining one follows a long unpunctuated line. All
49 added boundaries were inspected and fall before headings, numbered items,
or labels such as "Revision:"; five wrong boundaries were removed. The
Dataset v1 essays are segmented exactly as before. The rules were designed on
pilot development essays.

## Reliability Development Set v1

### Decision

Froze the development data for Reliability ML v2 as
`reliability_development_v1`: Dataset v1 (2 students, 28 essays, 232
sentences) and the labelled CELL Reliability Pilot v1 (35 students, 35
essays, 420 sentences). Totals: 37 students, 63 essays, 652 sentences; 504
usable, 102 `UNRELIABLE`, and 46 `NOT_ANALYZABLE` excluded from the binary
target. Both batches share spaCy 3.8.13 with en_core_web_sm 3.8.0, Clause
Detection v2.1, and sentence segmentation v1; every labelled sentence is
reproduced by segmenting its complete essay with the frozen pipeline.

The tracked manifest `data/syntax_reliability/development_set_v1.json`
follows the public-safe batch manifest design of the data expansion protocol:
aggregate counts, one status record per student, exclusions, and known
limitations, with no text, sentence labels, hashes, or paths. A private
ignored lock stores the SHA-256 and record count of each source file. The
validator rebuilds the manifest from the local files and fails if a file, a
count, a student, or the essay-context alignment changes, or if a holdout
student appears.

### ML v2 evaluation procedure, fixed before development

- Target: `UNRELIABLE` positive; `RELIABLE` and `MOSTLY_RELIABLE` negative;
  `NOT_ANALYZABLE` excluded.
- Parse context: each sentence comes from the frozen pipeline's parse of its
  complete essay, because labels were judged on those spans. Reliability ML
  v1 parsed Dataset v1 sentences in isolation, so its numbers are not
  directly comparable.
- Cross-validation grouped by student (`StratifiedGroupKFold`, 5 folds, or
  4 or 3 if a fold would lack a class; shuffle, seed 42), stricter than the
  essay grouping of ML v1.
- Threshold 0.5 unless chosen inside training folds only.
- Metrics: precision, recall, F1, F2, balanced accuracy, and average
  precision, with Wilson 95% intervals for precision and recall, reported
  overall and by batch, against frozen v1.2 `warning` on the same parses.
- The untouched CELL holdout is evaluated once, after the model, features,
  threshold, and this procedure are frozen.

### Boundaries

Excluded: Corpus A (different annotation history, not used by ML v1), the
CELL holdout, and unlabelled CELL development students. Known limitations
recorded in the manifest: labels are model-assisted rather than blind human
gold; 222 model `RELIABLE` pilot labels are unreviewed; Dataset v1 has two
students, one with 21 of 28 essays; Dataset v1 used annotation guideline 1.0;
labels are tied to Clause Detection v2.1 and segmentation v1; and the pilot
already shaped Clause Detection v2.2 and segmentation v2. No model was
trained and the holdout remains unopened.

