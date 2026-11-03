# Task Alignment Annotation Protocol v1

## Scope

Annotators judge only whether the exact learner response addresses each
explicit task requirement. They do not score language proficiency, grammar,
style, lexical sophistication, or overall essay quality.

The unit of annotation is one requirement for one complete response. Each
requirement receives one of four statuses:

- `met`: the response clearly addresses the requirement;
- `partially_met`: relevant content is present but incomplete or only partly
  addresses the requirement;
- `not_met`: the required content is absent or contradicted;
- `not_assessed`: the annotator cannot make a defensible judgement from the
  available task and response.

## Evidence rules

`met` and `partially_met` require one or more exact spans from the learner
response. Select the smallest span that still supports the judgement. Do not
rewrite spelling or grammar and do not quote a corrected version.

`not_met` requires a short note explaining what required content is missing.
Absence cannot be represented by a fabricated text span.

`not_assessed` must contain no response evidence and should include a brief
reason when the cause is not obvious.

## Independent review

Two reviewers annotate the same development responses independently. Reviewer
worklists must not contain:

- another reviewer's labels or notes;
- provider predictions or model probabilities;
- orchestration agreement;
- lexical, syntax, CEFR, or reliability metrics;
- a suggested final label.

Reviewers must not discuss cases until both worklists are frozen. Agreement is
measured per requirement. Every disagreement, including `not_assessed` versus
another status, enters human adjudication. Majority vote is not used.

Before comparison, every item must be marked `completed`. Completed worklists
are validated against the same canonical task and evidence contracts used by
the application. Missing requirements, invalid offsets, changed prompts or
responses, unexpected model fields, mixed reviewer identities, and incomplete
items invalidate the worklist.

## Adjudication

The adjudicator sees the original task, exact response, and both completed
reviews. The final record must be a complete evidence-grounded human assessment
with preserved provenance. Adjudication does not silently overwrite the two
independent source reviews.

Status agreement remains provisional agreement. The comparison workflow does
not choose between different evidence spans, manufacture a final reference, or
promote agreement to ground truth. A disagreement queue contains the complete
task, exact response, both source assessments, and a blank full-response
assessment for the human adjudicator. The adjudicator must assess every
requirement, not only the disputed requirement, so the resulting reference is
canonical and internally complete.

## Pilot composition

The first pilot is methodological development, not an estimate of production
accuracy. It should contain complete responses from multiple pseudonymous
participants and more than one generic response mode. Include straightforward
positive cases, partial fulfillment, clear omissions, and legitimate
`not_assessed` cases. Do not select only examples already known to confuse a
model.

No sealed-evaluation participant, task, response, or revision may enter the
development pilot. After the protocol and provider configuration are frozen, a
later sealed evaluation must use different participants and response families.

## Privacy

Worklists contain raw prompts, source material, and learner responses. They are
private operational artifacts under `data/alignment_evaluation/local/` and
must never be committed to the public repository. Reviewer IDs must be
pseudonymous.
