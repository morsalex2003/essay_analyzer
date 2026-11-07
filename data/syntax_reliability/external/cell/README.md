# CELL integration v1

`split_v1.json` is the tracked, text-free assignment of all 491 CELL student
IDs to development or untouched holdout. It was created from the canonical
2,857-row intake manifest with seed `20260824` before any Reliability ML error
inspection. Initial seeded hash ranking is followed only by deterministic
student-level swaps for period/course coverage, preserving the
essays-per-student bin when possible.

The holdout is untouched with respect to Reliability labels, predictions,
error analysis, and model development. The corpus-wide technical intake audit
preceded role assignment, but its word counts and duplicate flags were not
used by the split algorithm. `holdout_opened_at` remains null.

The raw corpus, learner texts, source hashes, and intake manifest remain local
and untracked. Put them under this directory's ignored `local/` folder when
running the importer, for example:

```text
local/CELL_manifest.csv
local/CELL_near_duplicate_candidates.tsv
local/corpus/
```

The importer also accepts a local CELL ZIP instead of `local/corpus/`.

Regenerate the split from metadata only:

```text
python scripts/create_cell_split.py \
  data/syntax_reliability/external/cell/local/CELL_manifest.csv \
  data/syntax_reliability/external/cell/split_v1.json
```

The importer in `src/cell.py` excludes `All/`, validates each
`archive_path + sha256` identity, preserves duplicate/revision metadata, and
extracts only `Intro + Body + Conclusion` by tag name. It HTML-unescapes those
sections, accepting the upper- and lowercase tag variants found in CELL, but
does not correct learner language. `Title`, `Ref`, and `App` are never included
in the NLP text.

Prepare CELL Reliability Pilot v1 from development students only:

```text
python scripts/prepare_cell_reliability_pilot.py
```

The private, ignored `local/pilot_v1/` directory receives complete selected
essays, an annotation template compatible with the R/MR/U/NA guideline, and
parser provenance. The tracked `pilot_v1_report.json` contains aggregate
composition only: no learner text, hashes, private paths, selected student
IDs, labels, heuristic verdicts, or model predictions.

Validate the private batch, frozen split membership, hashes, and public-report
privacy with `python scripts/validate_cell_reliability_pilot.py`.

Label the pilot sentences in the terminal:

```text
python scripts/annotate_cell_reliability_pilot.py --annotator-id owner_review_v1
```

Each screen shows the exact sentence with its neighbours, the frozen spaCy
token table (POS, dependency, head), and the detected finite, dependent, and
non-finite clause heads. Syntax Reliability heuristic verdicts and ML
predictions are not shown. Keys: `r` RELIABLE, `m` MOSTLY_RELIABLE,
`u` UNRELIABLE (then failure types), `n` NOT_ANALYZABLE (then the abstention
reason), `s` skip, `b` back, `q` quit. Allowed values come from
`data/syntax_reliability/schema_v1.json`.

Every confirmed label is saved immediately to the ignored
`local/pilot_v1/annotations_v1.jsonl`; `annotations_template.jsonl` is never
changed. Running the script again continues at the first unlabelled sentence,
`--start-at N` reviews from sentence N, and `--summary` prints label counts.
The script refuses to run if the installed spaCy version or model differs from
the frozen pilot provenance. The validator checks the labelled ledger as well
when it exists.

Score the frozen Syntax Reliability v1.2 heuristic against the labels so far:

```text
python scripts/evaluate_cell_reliability_pilot.py
```

The heuristic runs on each sentence inside its essay parse, as in
`analyze_writing()` and as the annotator saw it. The JSON report contains a
label-by-verdict table, binary UNRELIABLE metrics at two operating points
(`warning` = medium or low, matching the Dataset v1 baseline; `low_only`),
95% Wilson intervals for precision and recall, recall by failure type and by
selection structure, and the ledger positions of misses and false warnings.
It contains no learner text. `status` is `partial` until all 420 sentences are
labelled. The pilot uses development students only, so its results are
development estimates, not holdout evaluation.

Labelling with model suggestions is recorded honestly as a separate origin:

```text
python scripts/annotate_cell_reliability_pilot.py \
    --annotator-id owner_model_assisted_v1 --model-assisted
```

Every label from such a session is `model_assisted_manual_review`. The
evaluation report counts labels by origin, and
`evaluate_cell_reliability_pilot.py --origin human_manual_review` (or
`model_assisted_manual_review`) scores one kind only.

Records reviewed by the dataset owner carry `adjudication_status =
"adjudicated"`, `adjudicator_id`, `adjudication_date`, and, when the label was
changed, `pre_adjudication_label` and `adjudication_notes`. Relabelling a
reviewed record in the annotation tool resets it to `single`.

## Holdout evaluation v1

`holdout_v1_evaluation_plan.json` fixes the final holdout evaluation before
any holdout sentence is selected, labelled, or predicted: 25 holdout
students and 12 sentences each, selected with the pilot procedure (seed
20261105); frozen v1.2 as reference, v1.3 as the primary candidate, and
Reliability ML Model B as a secondary candidate; metrics, paired McNemar
comparisons, and the decision rule for v1.3.

The batch is prepared with
`scripts/prepare_cell_reliability_pilot.py --role untouched_holdout --seed
20261105 --students 25 --source-stage cell_reliability_holdout_v1
--local-output .../local/holdout_v1 --report .../holdout_v1_report.json`.
`scripts/build_cell_annotation_packet.py` writes a blind packet (sentences
with context, token parse, clause heads, the guideline, and an empty label
sheet; no heuristic or ML output) for a separate labelling session, and
`scripts/apply_cell_annotation_labels.py` writes the filled sheet to the
ledger, printing counts only.

The holdout batch has 25 students, 25 essays, and 295 sentences (some
essays have fewer than 12 sentences); `holdout_v1_report.json` holds its
aggregate composition. `scripts/review_cell_annotation_labels.py` builds the
owner review sheet and records the decisions. `scripts/evaluate_cell_holdout.py`
then runs the plan's last steps in order: `--lock-labels` (ledger hash),
`--record-opening --reason ...` (opening date and reason in
`split_v1.json`), and `--evaluate`, which checks both locks and the opening
record, scores v1.2, v1.3, and Model B once, and refuses to overwrite
`holdout_v1_evaluation.json`.

## Holdout evaluation v2

The 25 holdout v1 students were opened on 2026-11-06. Every opening is
listed in `holdout_openings` in `split_v1.json`; the older
`holdout_opened_at` and `holdout_open_reason` fields keep the first opening
of the holdout as a whole.

`holdout_v2_evaluation_plan.json` fixes the second evaluation before any
selection: 36 of the 73 remaining holdout students (12 sentences each, seed
20261106), labelled blind by a new sealed session and reviewed by the owner.
The primary question is whether Syntax Reliability v1.4 (the segmentation
boundary warning) raises recall over v1.3; v1.3, v1.4, and Model B are also
compared with v1.2. The other 37 holdout students stay untouched.

The batch is prepared with
`scripts/prepare_cell_reliability_pilot.py --role untouched_holdout --seed
20261106 --students 36 --source-stage cell_reliability_holdout_v2
--exclude-students-from .../local/holdout_v1/essays.jsonl --local-output
.../local/holdout_v2 --report .../holdout_v2_report.json`, then packet,
labels, and review follow the v1 steps. `scripts/evaluate_cell_holdout.py`
takes `--batch v1` or `--batch v2`; each batch has its own label lock,
opening record, and report.

The batch was evaluated on 2026-11-06 (`holdout_v2_evaluation.json`):
v1.4 caught nine unreliable sentences that v1.3 missed and lost none
(exact McNemar p = 0.004), so the pre-registered decision is
`confirm_v1_4`.

## Holdout evaluation v3

`holdout_v3_evaluation_plan.json` fixes the third evaluation before any
selection: all 37 remaining holdout students (12 sentences each, seed
20261107), labelled blind by a new sealed session and reviewed by the
owner. The primary question is whether the optional Syntax Reliability v1.5
(the transformer reference-parse warning) raises recall over v1.4. After
this batch no CELL holdout student is untouched.

The batch is prepared like v2, excluding both earlier batches
(`--exclude-students-from` once for `local/holdout_v1/essays.jsonl` and once
for `local/holdout_v2/essays.jsonl`). The blind packet shows only the
`en_core_web_sm` parse. `evaluate_cell_holdout.py --batch v3` needs
`requirements-transformer.txt`.

The batch was evaluated on 2026-11-07 (`holdout_v3_evaluation.json`):
v1.5 caught four unreliable sentences that v1.4 missed and lost none
(exact McNemar p = 0.125), so the pre-registered decision is
`inconclusive`. All 98 holdout students have now been opened.
