# Task Alignment Development Pilot Plan v1

## Decision

The project follows a data-first sequence. A concrete model provider will not
be selected or connected until the development annotation pilot is complete
and its disagreements and annotation burden are understood.

The pilot tests whether task requirements, evidence rules, reviewer workflow,
and adjudication are usable. It does not estimate production model accuracy.

## Operational pilot targets

The first annotation pilot aims for at least:

- 12 complete responses;
- 3 canonical writing tasks;
- 6 pseudonymous participants;
- 2 generic response modes;
- 2 independent initial reviewers;
- 1 explicit adjudication role.

These are workload and coverage targets, not a statistical power analysis.
A smaller valid collection may still be used to debug the protocol but must be
reported as `below_plan`.

The adjudication role is a separate documented pass. A third independent human
is preferable; if that is not feasible, the dataset owner must record who
adjudicated and must not describe the result as multi-party independent gold.

## Intake requirements

Before annotation:

1. Freeze each task's prompt, response mode, source material, and explicit
   requirement IDs.
2. Confirm rights to use every prompt and source.
3. Confirm that consent or the applicable use agreement covers this internal
   task-alignment development purpose.
4. Replace participant identity with stable pseudonymous IDs and remove direct
   identifiers from text and metadata.
5. Keep exact, uncorrected learner responses and their SHA-256 values.
6. Keep every participant and response family in the development role only.
7. Confirm two human reviewers before generating blind worklists.

Existing learner corpora must not be reused merely because they are available.
Their consent and content rights must cover this new annotation purpose.

## Composition

Use more than one task and response mode. Include naturally occurring complete,
partial, omitted, and genuinely unassessable requirements. Do not select only
known model failures, unusually strong writing, or unusually weak writing.

Source-based tasks must include the exact source material used by the learner.
Every task must contain at least one explicit requirement. A task with no pilot
responses may remain in intake data, but it is a readiness warning and should
not be counted as represented.

## Workflow

1. Prepare private canonical task and response JSONL files.
2. Run the aggregate readiness audit and resolve all blockers.
3. Generate two blind worklists without model or NLP outputs.
4. Freeze both completed reviewer worklists.
5. Compare statuses and send every disagreement to explicit human
   adjudication.
6. Validate canonical reference assessments.
7. Perform annotation error analysis and revise the protocol only on
   development data.
8. Select a model/provider and freeze its configuration only after the pilot.

No provider run, heuristic prediction, or syntax/lexical metric may influence
the initial human labels.

## Readiness interpretation

`meets_development_pilot_plan` means that the operational intake targets and
manual declarations are satisfied. It does not mean that labels are correct or
that the dataset supports generalization claims.

`below_plan` means the records are structurally valid but have limited pilot
coverage. `blocked` means annotation should not start until privacy, consent,
reviewer, adjudication, or task-definition problems are resolved.
