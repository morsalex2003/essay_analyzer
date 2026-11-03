# Task Alignment Batch Run Protocol v1

## Scope

One batch applies one frozen provider configuration to every response in one
role-pure experiment. The batch runner validates the complete experiment
manifest before the first completion call.

Human reference assessments are required to verify the frozen dataset, but
they are never passed to the provider. Each prompt package contains only the
canonical task context, exact response, fixed output contract, and frozen
generation parameters. Lexical and syntax metrics are also excluded.

## Complete run accounting

Every response produces exactly one run record. The retained status is one of:

- `completed`;
- `abstained`;
- `timed_out`;
- `failed`;
- `invalid_output`.

A failure or invalid output for one response does not remove later responses
from the batch. The batch summary must equal the statuses in its run records.
Duplicate or missing response IDs are rejected during evaluation.

## Provenance and evaluation

The batch records experiment ID, manifest SHA-256, data role, complete provider
configuration, and adapter provenance. The evaluator accepts the validated
batch artifact rather than a detached list of runs. This preserves the link
between provider outputs and the frozen configuration.

## Privacy and persistence

The in-memory runner accepts an injected completion callable; no external model
client is configured. A concrete client may save the returned batch only under
the ignored local alignment directory and must use create-only semantics.

Completed assessments contain exact learner evidence quotations, so the batch
is private even though run records do not include a standalone full-response
field. Batch artifacts, raw completions, and prompt packages must never be
committed or logged. Aggregate evaluation output remains text-free.
