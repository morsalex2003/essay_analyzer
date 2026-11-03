# Task Alignment Evaluation Protocol v1

## Evaluation target

The evaluation unit is one explicit task requirement in one complete learner
response. The provider predicts one of `met`, `partially_met`, `not_met`, or
`not_assessed`. The human reference follows the same canonical alignment
contract.

This protocol evaluates task-requirement assessment. It does not evaluate
writing quality, language proficiency, an exam score, or the learner.

## Dataset roles

Development and sealed-evaluation records must be evaluated separately. A
single evaluation input must contain exactly one data role. Development
results may guide implementation and therefore are not untouched performance
estimates.

Sealed evaluation may be opened only after the provider specification, prompt,
output contract, evaluation code, and decision procedure are frozen. Running
sealed evaluation requires an explicit authorization flag. Participant and
response-family separation remains mandatory.

## Coverage before agreement

Every eligible response must have one provider run record, including
abstention, timeout, execution failure, or invalid output. Report all run
statuses and completion coverage before interpreting agreement.

Status agreement is conditional on valid completed runs. It must never be
reported without coverage, because excluding abstentions and failures can make
an unreliable provider look stronger than it is. If there are no completed
runs, conditional rates are undefined rather than zero.

## Status evaluation

Report:

- the number of reference and evaluated requirements;
- exact requirement-status agreement;
- strict all-requirements response agreement;
- the complete reference-by-prediction confusion matrix;
- per-status conditional precision and recall;
- pseudonymous response and requirement IDs for mismatches.

Do not collapse the four statuses into a score. `not_assessed` is an explicit
decision and remains separate from runtime abstention or failure.

## Evidence comparison

For status-matched `met` and `partially_met` requirements, report exact span
agreement and any character overlap as descriptive diagnostics. These measures
are boundary-sensitive and do not establish semantic correctness. A different
valid quotation can support the same decision, while an overlapping quotation
can still be irrelevant. Human error analysis remains necessary.

## Reporting discipline

Aggregate reports must not contain raw prompts, source material, response
text, or evidence quotations. Provider identity and version must be preserved.
Small development pilots support debugging only; they do not justify product,
population, or exam-performance claims.
