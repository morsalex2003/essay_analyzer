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
