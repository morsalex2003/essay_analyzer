# Essay Analyzer

An interpretable NLP engine for analyzing English learner writing, developed
as a learning and research project around exam-style writing (TOEFL®/IELTS-like
tasks).

The project deliberately does **not** predict an exam score. It reports
descriptive lexical and syntactic measurements, links them to exact evidence in
the text, and estimates how far parser-derived syntax metrics can be trusted
for a given learner text.

TOEFL is a registered trademark of ETS. This project is not endorsed or
approved by ETS. IELTS is a registered trademark of its respective owners.

## Design principles

- **Interpretable metrics over opaque scores.** Every metric has a documented
  token policy, and student-facing feedback cites exact text offsets.
- **Descriptive is not evaluative.** An unknown word is not labelled as a
  spelling error, repetition is not automatically poor writing, and a syntax
  reliability warning is a statement about measurement quality, not a grammar
  judgement.
- **Learner English is the default input.** Non-standard forms, typos, and
  parser failures are expected and handled explicitly rather than corrected.
- **Separated domains.** Lexical and syntax analysis depend only on a shared
  NLP layer, never on each other.
- **Honest evaluation.** Development results are not reported as independent
  validation; holdout data is assigned by student before model inspection.
- **Private learner data stays local.** Raw learner texts and annotations are
  ignored by Git and never published.

## Architecture

```text
                    src/analysis.py  ── analyze_writing()  (versioned JSON result)
                   /        |        \
       src/lexical/     src/syntax/     src/tasks.py, src/alignment.py
   (diversity, frequency,  (clauses, complexity,   (task context, evidence-grounded
    sophistication, CEFR,   reliability, features,   requirement-level alignment)
    EFLLex)                 ML baseline)
             \               /
              src/nlp/pipeline.py   (single shared spaCy pipeline)

  Clients: main.py (CLI), src/reporting.py (student/teacher reports),
           src/revisions.py (revision comparison)
```

One spaCy `Doc` is created per text and reused by every component.

## What it measures

**Lexical**
- TTR and moving-average TTR (MATTR, window 50) with explicit token policies.
- Frequency profile with `wordfreq` Zipf values; out-of-vocabulary tokens are
  reported separately and never counted as sophisticated vocabulary.
- CEFR vocabulary profile (CEFR-J + Octanove C1/C2): token-level A1–C2 and
  Unknown distribution.
- EFLLex learner-association profile (research-only, opt-in; see licensing).

**Syntax**
- Sentence segmentation v2 for essays: headings and list markers on their own
  line are separated from the next sentence, and citation abbreviations such
  as "et al." or "para." do not end a sentence. Hard-wrapped lines stay
  together.
- Sentence length statistics, dependency distance, and tree depth.
- Clause Detection v2.2: finite clauses, dependent finite clauses, and
  non-finite predicative units are counted separately. Coordinated predicates
  sharing a subject are not counted as separate clauses. Frozen components
  (Syntax Reliability v1.2, Reliability ML v1, CELL pilot v1) keep v2.1.
- Clause density, dependent clause ratio, and sentence-level subordination
  ratio based on finite dependent clauses only.

**Syntax reliability**
- Rule-based consistency checks (v1.4) that flag sentences whose parse likely
  misrepresents the clause inventory or hierarchy. Categories (HIGH, MEDIUM,
  LOW) are engineering heuristics, not probabilities. v1.3 adds three
  warnings from the CELL pilot error analysis: a bare gerund after a
  preposition, a finite-classified reduced clause without a subject, and a
  modal attached to a non-verbal head. v1.4 adds a warning when a sentence
  span disagrees with a boundary that segmentation v2 decides (a heading
  glued to the next sentence, a citation split after "et."); it never fires
  on segmentation v2 spans, so it matters only for frozen segmentation v1
  parses. Datasets, the CELL pilot, feature schema 1.0, and Reliability ML
  keep frozen v1.2.
- Optional transformer check (v1.5): the essay is parsed a second time with
  `en_core_web_trf`, and a sentence is flagged when a clausal dependent that
  `en_core_web_sm` parses as a verb or auxiliary is nominal (noun, proper
  noun, adjective, or number) in the transformer parse (usually a noun
  misread as a verb). The transformer
  parse is evidence only; every metric still uses `en_core_web_sm`. The
  check is off by default because the model is a large extra install.
- A structural feature schema (44 features + 19 heuristic signals) and a
  logistic-regression development baseline. See
  [Methodology notes](#methodology-notes).

**Application layer**
- Student and teacher reports with evidence offsets and no score.
- Revision comparison: metric deltas plus resolved, persistent, and new
  repeated or unknown words, without labelling changes as improvement.
- Task context and an evidence-grounded task-alignment contract
  (`met`, `partially_met`, `not_met`, `not_assessed`). Statuses are never
  aggregated into a score.
- A provider-neutral interface for future alignment models. No concrete model
  or API is configured.

## Installation

Requires Python 3.14.

```text
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
git config core.hooksPath scripts/hooks
```

The last line enables the pre-commit hook that refuses commits of private
learner data (see [Data and privacy](#data-and-privacy)).

The optional transformer reliability check needs a further, large install
(torch and `en_core_web_trf`):

```text
.venv/bin/python -m pip install -r requirements-transformer.txt
```

## Usage

Analyze an essay from the command line (the default file is
`essays/sample.txt`):

```text
.venv/bin/python main.py                                  # full metric listing
.venv/bin/python main.py my_essay.txt --report student    # student report
.venv/bin/python main.py my_essay.txt --report teacher    # teacher report
.venv/bin/python main.py draft.txt --compare revised.txt  # revision comparison
.venv/bin/python main.py my_essay.txt --report student --json
```

`--json` prints the same result as JSON. EFLLex metrics are added only with
`--research`, because the EFLLex license is research-only.
`--transformer-check` (or `analyze_writing(text,
reference_parse_check=True)`) switches Syntax Reliability to v1.5; it needs
`requirements-transformer.txt`.

Use the engine from code:

```python
from src.analysis import analyze_writing
from src.cefr import load_cefr_vocabulary
from src.reporting import build_student_report, build_teacher_report

text = "Some people think zoos are cruel. I disagree because ..."
analysis = analyze_writing(text, cefr_vocabulary=load_cefr_vocabulary())

analysis["lexical"]["diversity"]["mattr"]
analysis["syntax"]["subordination_ratio"]
analysis["syntax_reliability"]

student_report = build_student_report(analysis)
teacher_report = build_teacher_report(analysis)
```

Compare two versions of the same response:

```python
from src.revisions import compare_revisions

changes = compare_revisions(analyze_writing(draft), analyze_writing(revision))
```

## Tests

```text
.venv/bin/python -m pytest -q
```

## Methodology notes

**Clause detection.** An earlier version counted VERB tokens as a clause
proxy, which mixed infinitives and participles with finite clauses and
over-counted shared-subject coordination. Clause Detection v2 uses subject
evidence, finiteness, and dependency topology, with conservative refinements
for learner English (finite parataxis, infinitive-marker precedence, non-finite
`pcomp`). v2.2 also counts a verb as finite when a modal is attached to the
next auxiliary ("should" -> "have"), when a past tense with its own subject is
tagged as a participle, and when a base-form verb has its own subject without
"to" or an auxiliary (learner agreement errors), except bare infinitives after
verbs such as "make" or "let".

**Syntax reliability ML baseline (development estimates only).** Canonical
development data: 221 analyzable sentences from 28 essays (178 usable, 43
unreliable), essay-grouped 5-fold `StratifiedGroupKFold`, balanced logistic
regression, fixed threshold 0.5, no feature selection or tuning.

| System | Precision | Recall | F1 | Balanced accuracy |
|---|---|---|---|---|
| Model A: 44 structural features | 0.509 | 0.674 | 0.580 | 0.759 |
| Model B: structural + 19 heuristic signals | 0.667 | 0.744 | 0.703 | 0.827 |
| Frozen rule-based v1.2 | 0.941 | 0.744 | 0.831 | 0.867 |

These numbers come from data that also informed method development and from
only two pseudonymized students. They are not evidence of generalization. A
student-level split of the CELL corpus (seed 20260824; 393 development and 98
untouched holdout students) was frozen for future work; the holdout has not
been opened.

**Development set for Reliability ML v2.** Dataset v1 and the labelled CELL
pilot are frozen together as `reliability_development_v1`: 37 students, 63
essays, 652 sentences (504 usable, 102 unreliable, 46 not analyzable). The
public manifest `data/syntax_reliability/development_set_v1.json` holds
aggregate counts, per-student status, known limitations, and the ML v2
evaluation procedure fixed in advance (student-grouped cross-validation on
essay-context parses, compared with frozen v1.2). A private local lock stores
the hash of every source file; check it with
`python scripts/validate_reliability_development_set.py`.

Re-running the unchanged ML v1 models under this procedure
(`python scripts/run_reliability_ml_v2_baseline.py`; 600 sentences where
v1.2 did not abstain, 96 unreliable, 5 student-grouped folds):

| System | Precision (95% CI) | Recall (95% CI) | F1 | Average precision |
|---|---|---|---|---|
| Model A: structural | 0.421 (0.348-0.497) | 0.719 (0.622-0.799) | 0.531 | 0.529 |
| Model B: structural + signals | 0.516 (0.429-0.601) | 0.677 (0.578-0.762) | 0.586 | 0.716 |
| Frozen rule-based v1.2 | 0.869 (0.762-0.932) | 0.552 (0.453-0.648) | 0.675 | - |

At the fixed 0.5 threshold neither model is a better warning than v1.2: they
find more unreliable sentences but raise 8-12 times as many false warnings.
With thresholds chosen inside training folds only, Model B matches v1.2
instead of improving on it: at v1.2's training precision it reaches
precision 0.864 and recall 0.531 (v1.2: 0.869 and 0.552), and 48 of its 51
caught errors are ones v1.2 also catches. 40 of 96 unreliable sentences are
caught by neither, so the current structural features carry no evidence
about them.

Adding nine error-pattern features from the pilot error analysis (feature
schema 2.0, `src/syntax/reliability_error_features.py`) made the model worse
(Model C average precision 0.577 against 0.716): the patterns are rare, and
after standardisation a single occurrence dominates the logistic model.

The three most precise of these patterns became warning signals in
Syntax Reliability v1.3. On the same 600 sentences v1.3 reaches precision
0.882 and recall 0.625 (v1.2: 0.869 and 0.552) with no new false warnings;
all seven extra catches are in the CELL pilot the patterns were taken from,
and Dataset v1 is unchanged. This is an optimistic development estimate
until the holdout is evaluated.

**Holdout evaluation (2026-11-06).** The plan fixed in
`holdout_v1_evaluation_plan.json` was run once on 25 untouched CELL holdout
students: 295 sentences labelled blind by a separate session and reviewed by
the owner; 277 analyzable sentences where v1.2 gives a verdict, 54 of them
unreliable.

| System | Precision (95% CI) | Recall (95% CI) | FP |
|---|---|---|---|
| Frozen v1.2 | 0.812 (0.570-0.934) | 0.241 (0.146-0.369) | 3 |
| v1.3 | 0.824 (0.590-0.938) | 0.259 (0.161-0.389) | 3 |
| Model B (threshold 0.9238 from development) | 0.850 (0.640-0.948) | 0.315 (0.207-0.447) | 3 |

v1.3 adds one catch and no false warning (McNemar p = 1.0), so the
pre-registered result is inconclusive; Model B adds five catches and loses
one (p = 0.22). Warnings remain precise, but most unreliable holdout
sentences are not flagged: 1 of 13 segmentation errors and 1 of 16
non-finite inventory errors are caught by v1.2.

**Syntax Reliability v1.4 (after the holdout).** On development data, four
of the five missed segmentation errors were headings glued to the next
sentence and one a citation split after "et.", exactly the boundaries
segmentation v2 corrects. v1.4 warns when a span disagrees with them. On the
same 600 development sentences it reaches precision 0.890 and recall 0.677
(v1.3: 0.882 and 0.625): five more catches, no new false warnings. The
pattern was found on these sentences, so this is an in-sample estimate; an
out-of-sample check needs the 73 still untouched holdout students.

**Holdout evaluation v2 (2026-11-06).** The plan fixed in
`holdout_v2_evaluation_plan.json` was run once on 36 of those 73 students:
432 sentences labelled blind by a new sealed session, 207 labels reviewed by
the owner; 362 analyzable sentences where v1.2 gives a verdict, 66 of them
unreliable.

| System | Precision (95% CI) | Recall (95% CI) | FP |
|---|---|---|---|
| Frozen v1.2 | 0.826 (0.629-0.930) | 0.288 (0.193-0.406) | 4 |
| v1.3 | 0.852 (0.675-0.941) | 0.348 (0.245-0.469) | 4 |
| v1.4 | 0.865 (0.720-0.941) | 0.485 (0.368-0.603) | 5 |
| Model B (threshold 0.9238) | 0.724 (0.543-0.853) | 0.318 (0.218-0.438) | 8 |

v1.4 catches nine unreliable sentences that v1.3 misses and loses none
(exact McNemar p = 0.004), with one extra false warning (p = 1.0): the
pre-registered result is **confirm_v1_4**, and v1.4 stays the default.
Segmentation errors caught rise from 3 to 12 of 22. Against v1.2, v1.3 adds
four catches (p = 0.125, again not significant on its own). Non-finite
inventory errors remain the largest gap (5 of 23 caught). 37 holdout
students are still untouched.

**Syntax Reliability v1.5 (optional, after holdout v2).** A second parser of
the same family (`en_core_web_md`) mostly repeats the mistakes of
`en_core_web_sm` and was rejected. The transformer model `en_core_web_trf`
disagrees on 12 of the 14 non-finite errors v1.4 misses on development
data. v1.5 warns when a clausal dependent that sm parses as a verb or
auxiliary is nominal (noun, proper noun, adjective, or number) in the
transformer parse. On the same 600 development sentences it
reaches precision 0.880 and recall 0.760 (v1.4: 0.890 and 0.677): eight more
catches, two more false warnings. The rule was chosen from four variants on
these sentences, so this is optimistic; `holdout_v3_evaluation_plan.json`
fixes its test on the last 37 untouched holdout students.

**Holdout evaluation v3 (2026-11-07).** The plan was run once on those 37
students: 444 sentences labelled blind by a new sealed session, 175 labels
reviewed by the owner (all confirmed); 410 analyzable sentences where v1.2
gives a verdict, 75 of them unreliable.

| System | Precision (95% CI) | Recall (95% CI) | FP |
|---|---|---|---|
| Frozen v1.2 | 0.821 (0.644-0.921) | 0.307 (0.214-0.418) | 5 |
| v1.3 | 0.812 (0.647-0.911) | 0.347 (0.249-0.459) | 6 |
| v1.4 | 0.824 (0.665-0.917) | 0.373 (0.273-0.486) | 6 |
| v1.5 (transformer check) | 0.821 (0.673-0.910) | 0.427 (0.321-0.539) | 7 |
| Model B (threshold 0.9238) | 0.684 (0.525-0.809) | 0.347 (0.249-0.459) | 12 |

v1.5 catches four unreliable sentences that v1.4 misses and loses none
(exact McNemar p = 0.125), with one extra false warning (p = 1.0): the
pre-registered result is **inconclusive**. The transformer check stays
optional as a development candidate, and v1.4 stays the default.
Non-finite inventory errors caught rise only from 10 to 12 of 30. Against
v1.2, v1.5 adds nine catches (p = 0.004) and v1.4 five (p = 0.0625). No
untouched CELL holdout students remain.

**Reference annotations.** Existing sentence-level reliability labels are
provisional model-assisted reference annotations, not independent human gold.

Detailed decisions and their reasons are recorded in
[PROJECT_HISTORY.md](PROJECT_HISTORY.md).

## Limitations

- No exam score or proficiency level is produced, by design.
- spaCy `en_core_web_sm` is trained on mostly native-speaker text; learner
  errors can break parses, which is why the reliability layer exists.
- Reliability rules and ML were evaluated on independent data (three
  pre-registered CELL holdout batches, 25, 36, and 37 students, blind
  labels):
  every system finds far fewer parser failures there than on development
  data (v1.4 holdout v2 recall 0.49, development 0.68; v1.2 0.29 and 0.55).
  Non-finite inventory errors are still mostly missed, also with the
  optional transformer check (holdout v3: 12 of 30). The CELL holdout is
  now fully used; further reliability changes need new independent data.
- Lexical and syntactic metrics are unstable on very short texts (for example,
  under about 150 words). Student and teacher reports add a
  `text_length_measurement` note below this threshold, and
  `lexical.diversity.mattr_method` shows when MATTR fell back to plain TTR
  because a text had fewer content words than one 50-word window.
- The task-alignment pilot is designed (protocols, validators, blind reviewer
  worklists) but has not been run with human reviewers.

## Project structure

```text
main.py                  command-line interface
src/analysis.py          application entry point: analyze_writing()
src/nlp/                 shared spaCy pipeline
src/lexical/             lexical metrics
src/syntax/              clauses, complexity, reliability, ML baseline
src/reporting.py         student and teacher reports
src/text_reports.py      plain-text views of results and reports
src/revisions.py         revision comparison
src/tasks.py             task context
src/alignment*.py        task-alignment contract, datasets, evaluation
src/orchestration.py     multi-assessor comparison (no voting)
src/providers.py         provider-neutral model interface
src/cell*.py             CELL corpus intake and annotation pilot
scripts/                 dataset validation, baselines, pilot preparation
data/                    vocabulary datasets and public protocols/schemas
tests/                   pytest suite
```

## Data and privacy

Raw learner essays, annotations, CELL corpus files, and task-alignment intake
data are excluded by `.gitignore`. Only protocols, schemas, aggregate reports,
and text-free split assignments are tracked.

Two further guards cover mistakes that `.gitignore` cannot stop:

- `scripts/hooks/pre-commit` runs `scripts/check_private_data.py` on every
  commit. It refuses staged files in private locations, new data files
  under `data/` that are not listed as public, and added lines in `data/`
  files that give a value to a learner-text field such as `original_text`.
- `.claude/settings.json` stops Claude Code from reading the raw CELL
  corpus, force-adding ignored files, skipping the hook, or changing
  `core.hooksPath`. `.claude/agents/reviewer.md` defines a read-only
  reviewer subagent for a check before committing.

## CEFR vocabulary data attribution

The project includes the CEFR-J Wordlist Version 1.5, compiled by Yukio Tono
at Tokyo University of Foreign Studies and provided through
[Open Language Profiles](https://github.com/openlanguageprofiles/olp-en-cefrj).
The source repository permits research and commercial use at no charge when
the dataset is cited properly. Copyright belongs to Tono Laboratory at TUFS.

The supplemental Octanove Vocabulary Profile C1/C2 Version 1.0 was created by
Octanove Labs and is distributed under the
[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

See [data/README.md](data/README.md) and the upstream repository for the full
attribution, terms, disclaimers, and references.

## EFLLex

The project also includes
[EFLLex](https://cental.uclouvain.be/cefrlex/efllex/download/), created by
Luise Dürlich and Thomas François. EFLLex describes how English lemmas are
distributed across learner-oriented pedagogical materials at levels A1-C1.
It is licensed under CC BY-NC-SA 4.0 and therefore cannot be used commercially.

EFLLex does not replace CEFR-J. CEFR-J associates lexical entries with CEFR
bands, while EFLLex provides level-specific normalized frequency distributions
from pedagogical materials. See [data/efllex/README.md](data/efllex/README.md)
for attribution and dataset details.
