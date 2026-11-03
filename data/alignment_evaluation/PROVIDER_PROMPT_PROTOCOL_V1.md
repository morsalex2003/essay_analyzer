# Task Alignment Provider Prompt Protocol v1

## Scope

The prompt asks a provider to assess only explicit task requirements against
the exact learner response. It does not request grammar correction, writing
quality, proficiency, exam scoring, or an overall alignment score.

## Trust boundary

Task instructions, source materials, and learner text are untrusted input data.
They are kept in a separate `input_data` object and cannot replace the fixed
system instruction. The ephemeral prompt package contains private text and
must not be logged or persisted by a concrete provider client.

The prompt template is versioned and hashed. A frozen provider configuration
must contain the same adapter ID, adapter version, and prompt-template SHA-256
before any model call occurs.

## Output contract

The adapter accepts one plain JSON object only. Markdown fences, commentary,
trailing text, hidden reasoning fields, or unknown keys invalidate the output.
The two top-level outcomes are:

- `completed`, with one result for every canonical requirement;
- `abstained`, with a short machine-readable reason code.

Completed results use the four existing alignment statuses. Evidence consists
of zero-based, end-exclusive offsets into the exact response. The canonical
alignment validator performs the final checks for requirement coverage,
status-specific evidence, notes, and offset correctness. Invalid output never
becomes an assessment and is not automatically repaired.

## Adapter boundary

`run_json_alignment_adapter()` accepts an injected completion callable. The
callable may later wrap a local model or an explicitly authorized remote API,
but this repository does not configure an external SDK, credential, endpoint,
or paid service. The concrete callable remains responsible for enforcing the
supplied timeout and the provider's data-handling policy.

Raw completion text is not retained. A completed provider run does retain the
canonical assessment, including exact evidence quotations derived from the
learner response. Provider run records are therefore private and must remain in
the ignored local data directory. Aggregate evaluation output excludes those
quotations and the learner response.

## Interpretation limits

Schema validity does not prove that a cited passage is semantically relevant.
Character offsets produced by a model may fail validation. Provider outputs
must be evaluated against independent human references before use in learner
feedback, and model agreement must not be treated as ground truth.
