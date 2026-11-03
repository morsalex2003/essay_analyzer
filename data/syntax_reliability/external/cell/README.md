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
