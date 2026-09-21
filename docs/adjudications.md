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

The registry now contains **205 accepted target corrections**. The second focused
wording batch adds 87 accepted changes: 74 unsupported observed negatives and
three insufficiently specified positives become uncertain, six already-masked
states become unmentioned, and four quotations gain necessary context without
changing state. Root read all 50 complete sources and froze 600 judgments before
the other readers' case findings. Its initial states matched 73 of the 75 primary
proposals; all thirteen root reconsiderations across the batch remain explicit.
A distinct agent accepted twelve of thirteen additional root proposals after a
separate source reread. One cartilage grading disagreement remains pending,
including after a balanced literature follow-up; it is excluded from registration.

The [integration proof](../initiative/audits/qualified-negation-focused-001-integration-v1/complete205-integration-validation.json)
replays all 205 corrections and preserves the earlier 118 and 297 protected files.
All 87 new annotations match the final root judgments and exact evidence; 599 of
600 root states agree with the resulting inspection, with the remaining disputed
cell explicitly identified. The
[current consistency check](../initiative/audits/qualified-negation-focused-001-integration-v1/current205-consistency-validation.json)
finds no differing target states in 54 exact normalized-report groups/204 studies.
This does not resolve report-similarity leakage or any global semantic QA gate.

At the previous checkpoint, the registry contained **118 accepted target corrections** and retains cartilage,
bone terminology and qualified-negation/duplicate consistency as three pending
global issues. Those two cases added 84 corrections from the first focused
wording batch: 78 unsupported negatives and one insufficiently graded positive
become uncertain, two already-masked fracture states become not_mentioned, and
three negative states retain their state with more complete source evidence.

Both focused readers froze all 600 judgments for 50 complete reports before
their case comparison. Root then read all 48 attention reports and froze 576
judgments before opening their case-specific findings, with historical exposure
disclosed. Root accepted the first reader's 74 proposals; 73 proposed states
matched the root freeze and one required explicit reconsideration. Root's ten
additional proposals received an actual distinct assessor's acceptance, including
six explicit reconsiderations of that assessor's earlier states. Post-comparison
agreement is not initial blinded agreement. Every reading, quote, disagreement
and reconsideration remains preserved.

The [complete integration proof](../initiative/audits/qualified-negation-focused-000-integration-v1/complete118-integration-validation.json)
replays all 118 corrections, preserves the earlier 34 and 298 protected artifacts,
and reconciles all 576 final root states with inspection copies. The older
ten-report root snapshot remains unchanged; the completed assessment is separately
archived in `qualified-negation-focused-000-root-complete-v1/`. The previous
seven-correction duplicate case remains preserved in
`duplicate-consistency-adjudication-v1/`.

No effective label release was performed. The other wording batches and cartilage
candidate inventories are pending work, not additional accepted corrections or
evidence that a complete global gate has been resolved.

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
