# CELL Reliability Pilot v1: Error Analysis of Syntax Reliability v1.2

This report groups the errors of the frozen Syntax Reliability v1.2 heuristic
on the reviewed CELL Reliability Pilot v1 labels (2026-11-04). It contains
aggregate counts only. All examples are synthetic and only illustrate a
pattern; no learner sentence, sentence position, or identifier is included.

The analysis uses development students only. Any heuristic change designed
from it must be evaluated on data it was not designed on, ultimately the
untouched holdout. Nothing in the heuristic, its thresholds, or its signals
was changed.

## Scope

Heuristic verdicts are computed in essay context with Clause Detection v2.1,
the version the heuristic and the labels are frozen on. The binary task treats
`UNRELIABLE` as positive and `RELIABLE` plus `MOSTLY_RELIABLE` as negative, at
the `warning` operating point (medium or low).

| Outcome | Count |
|---|---|
| `UNRELIABLE` caught (true positives) | 22 |
| `UNRELIABLE` missed with a `high` verdict (false negatives) | 31 |
| `UNRELIABLE` where the heuristic abstained (`not_analyzable`) | 6 |
| Usable sentences warned about (false positives) | 6 |

## What the heuristic already catches

Comma-spliced independent clauses parsed as nested complements are the
heuristic's strongest area: 11 of the 22 true positives carry
`unmarked_comma_clause_link`. Missing finite clauses at the root
(`no_finite_clause`, `nonfinite_root_with_subject`) account for five more.

## Missed errors by mechanism

Each missed sentence is assigned to its primary mechanism.

| Mechanism | Misses | Synthetic illustration | Detectable from the parse? |
|---|---|---|---|
| Noun or compound parsed as a verb, adding a spurious non-finite unit | 8 | "the risks **of cyberbullying**", "to **curriculum** goals" | Partly: a bare gerund `pcomp` without dependents, or an infinitive whose lemma is attested only as a noun, needs a lexical resource |
| Segmenter boundary errors | 6 | a heading glued to the next sentence across a line break; a citation cut at "et." | Yes: internal line break after an unpunctuated line, sentence ending in a known abbreviation, sentence starting with a citation remnant |
| Clause attachment errors without a comma cue | 6 | a missing full stop ("we left early **They** stayed"), clauses joined by ";", an adverbial clause attached as a conjunct, an omitted "that" before a modal clause | Partly: a capitalised subject pronoun mid-sentence; a `ccomp` linked across ";"; a modal-supported `acomp` with its own subject |
| Noun/verb part-of-speech swap that moves a finite clause | 4 | "that pupils **access** new levels" (verb as NOUN `ccomp`); a misspelled noun parsed as the finite verb | Partly: a NOUN or ADJ head of `ccomp`/`relcl` with its own `nsubj` is structural (the existing `non_predicative_clausal_dependency` signal excludes `ccomp`); a noun parsed as a finite verb needs lexical evidence |
| Reduced participle counted as a finite dependent clause | 3 | "the group **taught** online" tagged as past tense | Yes: a finite-classified `acl` with no own subject and no relative word |
| Finite clause misclassified by Clause Detection v2.1 | 2 | "than the authors **suspected**" tagged VBN | Fixed in Clause Detection v2.2 for metrics; the frozen heuristic keeps v2.1 |
| Modal with a missing verb outside the root | 2 | "prices **will** very high" | Yes: a finite modal attached to a non-verbal, non-root head; the existing `nonverbal_root_finite_auxiliary_conflict` only checks the root |
| **Total** | **31** | | |

## Abstentions on reviewed errors

In all six abstentions, the parser lost the only finite verb (a verb tagged as
a noun at the root, or a left-dislocated topic taking the root), so the
heuristic found no comparable structure. If downstream metrics are suppressed
for `not_analyzable` sentences, these behave like caught errors in practice;
they are reported separately rather than scored.

## False warnings

Five of the six false warnings come from `nominal_predicate_coordination`
firing on a noun phrase attached as a conjunct of a verb. In four of them the
reviewers saw a real but local attachment error (`MOSTLY_RELIABLE`), so the
signal points at genuine misparses whose effect on clause metrics is small.
The sixth is `unmarked_comma_clause_link` on a correctly parsed quotation.

## Candidate directions for a future heuristic version

Ordered by expected yield and by how little they depend on lexical resources:

1. Segmentation anomaly signal (up to 6 misses), purely structural.
2. Dependent-clause head with noun or adjective POS and its own subject
   (up to 2 of the 4 part-of-speech swaps).
3. Reduced participle classified as finite (up to 3).
4. Finite modal on a non-verbal head outside the root (up to 2).
5. Bare gerund or infinitive heads attested only as nouns (up to 8), which
   needs a part-of-speech lexicon and carries the highest false-positive risk.

These counts are upper bounds from the same data the patterns were found in,
not expected recall gains.
