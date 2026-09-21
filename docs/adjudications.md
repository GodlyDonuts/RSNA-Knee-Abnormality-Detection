# Preserved report corrections

Published reader and reviewed JSONL files remain immutable. A later correction is
an additional, independently adjudicated report annotation. It never replaces an
official gold value or its evaluation split.

`rsna.labels` always reads `adjudications/registry.json` beside the supplied
`reviewed/` folder. There is no flag to skip this gate. Registry and evidence paths
are repository-relative; run from the repository root or supply `--artifact-root`.
An absent registry fails even when no changes are intended. An explicit empty
registry is valid only when its `cases/` and `qa/` directories contain no records.

The registry has exactly these fields:

```json
{
  "schema_version": 1,
  "source_reports_sha256": "<canonical blinded report fingerprint>",
  "rubric_sha256": "<frozen rubric checksum>",
  "cases": [{"path": "initiative/reviews-v2/adjudications/cases/example.json", "sha256": "<checksum>"}],
  "semantic_qa": [{"path": "initiative/reviews-v2/adjudications/qa/example.json", "sha256": "<checksum>"}]
}
```

Every JSON file in each directory must appear exactly once in the matching list.
Case IDs and study/target corrections must be unique. Overlapping corrections
require a new adjudicated resolution; application order cannot decide a conflict.

## Correction evidence

A version-1 case binds `source_reports_sha256`, `rubric_sha256`, a unique `id`,
and exact path/checksum references for `proposal`, `independent_assessment` and
`decision`. `supporting_evidence` maps additional checksum fields from the accepted
decision to their referenced artifacts. Every decision checksum field must have
a verified binding; evidence paths cannot escape the supplied artifact root.

The loader supports two existing preserved decision formats:

- `accepted_masks`: the original five batch-000 corrections. The case includes
  one `base_review` and one `source_batch` reference. Proposal `proposed_masks`,
  the independent assessment's `records`, and the decision's `accepted_masks`
  must agree on each accepted target and resulting annotation. The exact before
  state and evidence are checked against the checksum-bound original JSONL.
- `accepted_changes`: decisions covering one or more batches. Every case change
  includes its own `base_review` and `source_batch` references. Proposal
  `proposed_errata`, independent assessment `records`, and accepted changes bind
  each target's source and review hashes. The accepted decision must contain exact
  `before_state`, `before_evidence`, `after_state`, and `after_evidence`, matching
  the case's `before` and `after` objects. Each object has `state` and `evidence`.

Each case must cover its entire accepted decision. The proposer and adjudicator
must be distinct, match their respective evidence artifacts, and have separate
session identities when available. Assessment session identity is required;
legacy proposal artifacts lacking a session use their canonical agent identity
for that comparison. Original decision status must explicitly accept the changes;
pending proposals cannot authorize changes. Before/after evidence must be exact
contiguous source text, except `not_mentioned`, which requires empty evidence.

The adapter does not evaluate medical meaning. The bound proposal and separate
source assessment supply that reasoning. Do not fabricate approval metadata or
normalize the original decision files to satisfy the loader. Preserve those
files byte for byte and record integration status in a separate artifact.

## Semantic QA remains a release gate

Each QA record has version 1, a unique `id`, the same source/rubric bindings,
`scope: "all_release_reviews"`, an explicit `pending` or `resolved` status, a
nonempty reason, and a nonempty list of checksum-bound `evidence` references.
Pending records always block release, including when all correction cases are
valid. `inspect_adjudications` validates and reports pending work without creating
effective annotations; `apply_adjudications` requires it to be resolved.

A resolved issue additionally references an immutable resolution artifact with
`status: "resolved"`, the matching `issue_id`, source/rubric hashes, actual
`agent_id` and `session_id`, and completion `notes`. Its `review_artifacts` maps
every reviewed JSONL path to its exact checksum. Its `correction_cases` maps every
registered case path to its checksum. A new or changed reviewed batch or correction
case invalidates that scope binding and requires further assessment.

The present registry retains cartilage consistency, a focused bone-terminology
check, and qualified-negation/duplicate consistency as separate pending issues.
The latter binds the all4,407-report inspection and root's seven unaccepted
proposals across six studies. A distinct source assessor is still required;
these proposals are not registered correction cases and do not alter labels.
The early focused audits and retained decisions do not certify all future report
batches. Preserve their dispositions as evidence while completing that broader
scope.

## Release outputs

`report_annotations_original.parquet` preserves the validated review table.
`report_annotations.parquet`, `studies.parquet` and `study_labels.csv` include
effective states/evidence plus per-target `__original_state`,
`__original_evidence` and `__correction`. The latter contains a JSON trace of the
accepted decision, identities, exact before/after annotations and artifact hashes;
uncorrected targets have an empty trace. `adjudication_trace.json` and
`label_audit.json` preserve the registry hash and complete correction/QA lineage.

Observed corrected weak targets use `report_adjudicated` provenance. Uncertain or
unmentioned weak targets retain zero masks. Official gold stays authoritative,
with its original value, mask 1, `official_gold` provenance and evaluation split,
even when its report-derived annotation has a correction trace. The audit reports
original reviewer changes and later adjudicated state changes separately.
