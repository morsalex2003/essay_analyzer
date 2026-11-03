# Task Alignment Experiment Freeze Protocol v1

## Purpose

The freeze manifest binds an evaluation result to one exact dataset snapshot,
provider specification, adapter version, prompt-template hash, generation
configuration, and evaluation policy. It prevents accidental post-hoc changes
from being reported as the original experiment.

## Required freeze inputs

Freeze the following before provider runs begin:

- role-pure task, response, and reference-assessment JSONL files;
- one canonical provider specification;
- adapter identifier and version;
- SHA-256 of the exact prompt template;
- temperature, top-p, output-token limit, seed, and reasoning configuration;
- current annotation, provider-contract, and evaluation schema versions.

The provider configuration has a closed schema and cannot contain arbitrary
fields such as credentials. The prompt template itself remains local; only its
hash enters the manifest.

## Dataset integrity

The manifest stores SHA-256 values for the exact source-file bytes and for
canonical record content. Exact file hashes detect any byte-level change.
Canonical hashes are stable when JSONL record order changes but content does
not. Counts and reference-status aggregates are included without participant
IDs or learner text.

## Evaluation binding

The frozen evaluator verifies the manifest against all current inputs before
calculating metrics. Provider runs must match the frozen provider
specification. The evaluation result records the experiment ID and manifest
hash so results from different configurations cannot be silently combined.

## Privacy and interpretation

The manifest and generated evaluation remain under the ignored local data
directory. Neither artifact contains prompts, source materials, learner text,
evidence quotations, or participant IDs.

SHA-256 detects drift but is not a digital signature and does not establish
authorship or authenticity. Development manifests describe development
experiments only. Creating or evaluating a sealed manifest requires explicit
authorization after the complete procedure has been frozen.
