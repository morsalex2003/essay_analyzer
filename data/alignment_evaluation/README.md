# Task Alignment Evaluation Dataset v1

## Purpose

This dataset is intended to evaluate requirement-level task-alignment
assessments produced by a single provider or by the alignment orchestration
layer. It does not contain an essay score and does not evaluate general writing
quality.

Reference assessments must identify an explicit human assessor. A
model-assisted record is permitted only when a human adjudicator supplied the
final evidence-grounded assessment. Model agreement is not a reference label.
Reviewer identifiers must also be pseudonymous.

## Privacy and repository policy

The repository is public. Raw prompts, source materials, learner responses,
and evidence quotations are intentionally local-only and must not be committed.
The following canonical files are ignored by Git:

- `tasks_v1.jsonl`
- `responses_v1.jsonl`
- `assessments_v1.jsonl`

Only this documentation, the public schema, validation code, and aggregate
statistics that contain no learner text may be version-controlled. Do not add
real names, email addresses, institution IDs, or other direct identifiers.
`participant_id` must be pseudonymous, and the record must explicitly confirm
that fact.

## Dataset roles

- `development`: may be inspected while developing providers and prompts.
- `sealed_evaluation`: must remain unopened until the provider set, prompts,
  orchestration policy, and evaluation procedure have been frozen.

All responses and revisions from one participant must remain in one role.
Every `response_family_id` must also remain in one role. Random sentence-level
splitting is prohibited.

## Canonical records

`tasks_v1.jsonl` contains one canonical task context per line, including its
requirements and any source material. `content_rights_status` must use one of
the controlled permission/restriction values in the schema.

`responses_v1.jsonl` contains exact uncorrected learner text, its SHA-256,
pseudonymous participant identity, consent/use status, and dataset role.

`assessments_v1.jsonl` contains exactly one reference assessment per response.
The embedded assessment follows `src/alignment.py`: `met` and `partially_met`
require exact response evidence, `not_met` requires a note, and
`not_assessed` is an explicit abstention. Requirement statuses are not combined
into a score.

## Evaluation discipline

Provider outputs, heuristic predictions, and orchestration agreement must not
be used to silently change reference assessments. Any later adjudication should
preserve provenance. Development data cannot later be described as untouched
evaluation data.

No real learner records are included in the public repository.

## Development annotation pilot

The annotation procedure is defined in
`ANNOTATION_PROTOCOL_V1.md`, and the data-first collection sequence is defined
in `DEVELOPMENT_PILOT_PLAN_V1.md`. Development responses are reviewed by at least
two pseudonymous reviewers using separate worklists. Worklists intentionally
exclude provider predictions, lexical and syntax metrics, other reviewers'
decisions, and suggested labels.

Hand-prepared intake files can be converted into the canonical private JSONL
files. Place `tasks.json` (a JSON list) and `responses.jsonl` under the ignored
`local/intake/` directory and run:

```text
python scripts/import_alignment_intake.py
```

The importer assigns every record to `development`, computes SHA-256 values
from the exact uncorrected `original_text`, and validates the result with the
canonical dataset validator. Unexpected intake fields are rejected so that
accidental identifier columns are not carried forward. It refuses to overwrite
existing canonical files or to write to a path that Git would track, and it
prints aggregate counts only.

`intake_template/` contains a tracked, complete example of both intake files:
3 tasks (independent, source-based, and discussion modes), 12 responses, and
6 pseudonymous participants. All prompts, sources, and responses are synthetic
texts written for this template, not learner data. The template meets the
development pilot plan and is checked by
`tests/test_alignment_intake.py`, so it can be copied into `local/intake/` as
a starting point and then replaced record by record with real data. It must
never be imported as pilot data itself.

Before generating worklists, run the private aggregate readiness audit:

```text
python scripts/check_alignment_pilot_readiness.py \
    --tasks data/alignment_evaluation/tasks_v1.jsonl \
    --responses data/alignment_evaluation/responses_v1.jsonl \
    --confirm-direct-identifiers-removed \
    --confirm-consent-scope-reviewed \
    --confirm-two-independent-reviewers \
    --confirm-separate-adjudicator
```

The confirmations are deliberate human declarations, not automated privacy or
legal conclusions. The report contains only aggregate counts, readiness
checks, blockers, and warnings.

After the private task and response JSONL files have been prepared locally,
independent worklists can be generated with:

```text
python scripts/prepare_alignment_annotation_pilot.py \
    --reviewer-id reviewer_01 \
    --reviewer-id reviewer_02
```

The generated worklists are stored under `local/`, which is ignored by Git.
They contain private prompt and learner text and must not be copied into public
tests, documentation, logs, or commits. The development pilot is protocol
development evidence, not an estimate of production performance. Sealed
evaluation records are rejected by the worklist generator.

After both reviewers mark every item `completed`, compare the private
worklists and prepare the disagreement queue with:

```text
python scripts/prepare_alignment_adjudication.py \
    --worklist data/alignment_evaluation/local/reviewer_01_worklist_v1.jsonl \
    --worklist data/alignment_evaluation/local/reviewer_02_worklist_v1.jsonl
```

The resulting `annotation_review_batch_v1.json` also remains under `local/`.
The comparison records provisional status agreement but never votes or creates
a reference label automatically. Every status disagreement requires an
explicit full human adjudication through the canonical alignment contract.
Both preparation scripts refuse to overwrite an existing local artifact.

## Provider evaluation

The frozen evaluation definitions are documented in
`EVALUATION_PROTOCOL_V1.md`. Evaluation inputs must contain exactly one data
role and exactly one recorded provider outcome for every response, including
abstentions, timeouts, failures, and invalid outputs. Development and sealed
evaluation are never combined. Dataset and provider configuration freezing is
defined in `EXPERIMENT_FREEZE_PROTOCOL_V1.md`.

Before any provider runs, create a strict provider configuration with
`src.alignment_experiments.create_provider_configuration()`. It contains the
canonical provider specification, adapter identity and version, prompt-template
SHA-256, and a closed set of generation parameters. It must not contain a
prompt, credential, or learner text.

The fixed prompt and strict JSON contract are documented in
`PROVIDER_PROMPT_PROTOCOL_V1.md`. The built-in
`src.alignment_adapter.run_json_alignment_adapter()` accepts an injected local
or authorized remote completion callable. No external SDK, endpoint, account,
or credential is configured by the project. Raw prompt packages must not be
logged. Completed provider runs contain canonical evidence quotations and must
remain private even though they do not store a standalone response-text field.

Batch execution is defined in `BATCH_RUN_PROTOCOL_V1.md` and implemented by
`src.alignment_batch.run_frozen_alignment_batch()`. It validates the manifest
before the first call, excludes human references and NLP metrics from prompts,
and retains one run outcome for every response. A real local or remote client
must be injected separately and must save the returned batch only under the
ignored `local/` directory.

Freeze the experiment locally:

```text
python scripts/freeze_alignment_experiment.py \
    --experiment-id alignment_dev_001 \
    --tasks PRIVATE_ROLE_TASKS.jsonl \
    --responses PRIVATE_ROLE_RESPONSES.jsonl \
    --assessments PRIVATE_ROLE_ASSESSMENTS.jsonl \
    --provider-configuration PRIVATE_PROVIDER_CONFIGURATION.json \
    --data-role development
```

The local evaluator accepts explicit role-specific JSONL paths:

```text
python scripts/evaluate_alignment_provider.py \
    --tasks PRIVATE_ROLE_TASKS.jsonl \
    --responses PRIVATE_ROLE_RESPONSES.jsonl \
    --assessments PRIVATE_ROLE_ASSESSMENTS.jsonl \
    --batch PRIVATE_PROVIDER_BATCH.json \
    --manifest data/alignment_evaluation/local/experiment_manifest_alignment_dev_001_v1.json \
    --provider-configuration PRIVATE_PROVIDER_CONFIGURATION.json \
    --data-role development
```

Sealed evaluation additionally requires `--authorize-sealed-evaluation` and
should only be run after the provider and procedure are frozen. The aggregate
result is written under ignored `local/`, excludes learner text, and refuses
to overwrite an existing result. Conditional status agreement must always be
read together with provider completion coverage. Evaluation stops if current
files, protocol versions, provider configuration, or prompt hash differ from
the frozen manifest.
