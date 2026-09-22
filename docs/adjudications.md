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

The registry contains **491 active accepted target corrections**. Wording003
adds 64 second-reader changes and two first-reader changes, while preserving all 425
prior whole objects, four selected overlaps and three earlier retirements. All 540
root source judgments preceded the current comparison; five later state changes
remain explicit. All 59 original first proposals and 68 second proposals remain
preserved, including two rejected second proposals and one rejected first state.

The [491 integration receipt](../initiative/audits/qualified-negation-focused-003-integration-v1/complete491-integration-validation.json)
and [independent technical replay](../initiative/audits/qualified003-integration-independent-verification-v1/independent-integration-validation.json)
verify 66 exact additions, 539 final root states/quotations and zero conflicts among
54 identical-report groups / 204 reports. The one extra root proposal for 026:03
Lateral OA remains a semantic disagreement after a distinct assessor's rejection;
it is unregistered and its current annotation is unchanged. It remains work for
QA resolution, not evidence of agreement. All three global gates remain pending.

The preceding 425 checkpoint added 63 changes from wording002, preserving the
previous 362 objects. Its [integration receipt](../initiative/audits/qualified-negation-focused-002-integration-v1/complete425-integration-validation.json)
and [verified private publication](../initiative/audits/qualified002-verified425-publication-v1/root-publication-receipt.json)
are preserved. The original wording002 and wording003 assessments must not be
rewritten to match accepted final decisions.

The earlier registry contained **362 active accepted target corrections**. Cartilage005
adds four after six complete root source rereads / 18 frozen OA judgments. Two
negatives become uncertain, and one positive and one negative gain exact evidence
that supports the retained state. All 18 root states are unchanged; one reason
clarifies that a measured tibial lesion is not explicitly sided. The medial femoral
finding independently supports that report's retained positive.

The [362 integration proof](../initiative/audits/cartilage-observed-005-integration-v1/complete362-integration-validation.json)
reconciles all 18 final states and four accepted quotes. It preserves all 358 prior
correction objects, original proposal objects, source/report files, three previous
retirements and the existing v14 source freeze. All three first-reader proposals
are supported through the four exact second-reader proposals. All 30 selected
cells replayed from historical205 to358 without overlap or drift. The 54 identical-
report groups / 204 reports still have no conflicting states. No global QA gate closes.

The preceding registry contained **358 active accepted target corrections**. Cartilage004
adds 25 after 39 complete root source rereads / 117 frozen OA judgments.
Seventeen positives and five negatives become uncertain; three quotations gain
anatomical or report-local depth context. Nine actual root state reconsiderations
remain explicit. Three other second-reader proposals were rejected, retaining
current annotations. All 333 previous correction objects remain unchanged.

The [358 integration proof](../initiative/audits/cartilage-observed-004-integration-v1/complete358-integration-validation.json)
reconciles all 117 final states and 25 accepted quotes, preserves the protected
sources and earlier retirements, and finds no conflicts across 54 identical-report
groups / 204 reports. All 21 first-reader proposed states are supported through
accepted second proposals. All 28 original second proposals, including the three
rejected proposals, remain unchanged. No global QA gate closes.

The bound [terminology check](../initiative/audits/cartilage-observed-004-root-review-v1/terminology-source-check.json)
distinguishes a specifically named ICRS classification from an unidentified grade
number. It records primary-source support and retrieval limitations. Direct
high-grade partial-thickness and low-grade thinning decisions are explicitly
identified as report-level morphology inferences, not added numeric measurements.

The preceding registry contained **333 active accepted target corrections**. Cartilage003
adds 47 independently accepted second-reader proposals after 38 complete root
source rereads / 114 frozen OA judgments. Thirty positives and twelve negatives
become uncertain; five quotations gain explicit anatomy, normal-remaining-cartilage
scope, qualitative extent or a report-internal depth definition. Four root state
reconsiderations are preserved, and all 286 earlier corrections remain exact.

The [333 integration proof](../initiative/audits/cartilage-observed-003-integration-v1/complete333-integration-validation.json)
reconciles all 114 root final states and 47 new quotes and finds no conflict in the
54 identical-report groups / 204 reports. All 39 first-reader proposed states are
supported via the accepted second proposals. Three first/second state disagreements
are resolved in favor of uncertainty after source review. No global QA gate closes.

The preceding registry contained **286 active accepted target corrections**. Cartilage002
adds 32 after 40 complete root attention-source rereads and 120 frozen OA judgments.
Root independently accepts 31 exact second-reader proposals; a distinct assessor
accepts the additional root proposal. Twelve positive and nineteen negative targets
become uncertain, and one positive quotation expands to include its extent evidence.
All 254 prior corrections and the earlier three retirements remain unchanged.

The [286 integration proof](../initiative/audits/cartilage-observed-002-integration-v1/complete286-integration-validation.json)
reconciles all 120 final root states and all 32 exact new quotations, preserves all
prior corrections and protected source artifacts, and verifies zero conflicts in
54 identical-report groups/204 studies. Seven root state reconsiderations remain
explicit. The additional assessor's fresh uncertainty and historical positive are
both preserved with its actual independent acceptance. The actual proposer's separate
canonical source-context correction changes no clinical content and preserves the
original normalized proposal and manifest.

The preceding milestone contained **254 active accepted target corrections**. Its
cartilage001 milestone adds 25 after root's 34 complete attention-source reads and
102 frozen OA judgments. Root actually accepts 24 proposals from the distinct
second reader; a separate source assessor accepts root's additional fraying
proposal. The25 changes mask nine previously positive and twelve previously
negative targets, and expand four positive quotations to include compartment
identity. All229 prior changes and the earlier three retirements remain exact.

The [254 integration proof](../initiative/audits/cartilage-observed-001-integration-v1/complete254-integration-validation.json)
reconciles every final root state and exact new quotation and verifies zero
conflicts across54 normalized identical-report groups/204 studies. Four root
source-state reconsiderations and the additional assessor's reconsiderations
remain preserved; agreement after reconsideration is not initial unanimity.
The frozen rubric has not changed. In particular, root's earlier two-axis size
requirement was withdrawn explicitly, rather than made into a new rule.

The preceding milestone contained **229 active accepted target corrections**. The latest
cartilage000 milestone adds27 corrections and independently retires three earlier
ones. Root read26 complete attention sources and froze78 OA judgments before reader
case findings. Its20 primary assessments yield18 new changes and two exact matches
to already accepted corrections. Four root source-state reconsiderations remain
separate. A distinct assessor reread seven complete sources/21 OA judgments, then
accepted nine additional proposals, with eight proposal-related state reconsiderations
and one further reconsideration supporting an already accepted primary entry.
This is actual post-comparison agreement, not initial unanimity.

The [integration proof](../initiative/audits/cartilage-observed-000-integration-v1/complete229-integration-validation.json)
reconciles all78 final root states and the exact27 new quotations with inspection.
The [current duplicate check](../initiative/audits/cartilage-observed-000-integration-v1/current229-consistency-validation.json)
finds zero conflicting states in54 normalized identical-report groups/204 studies.
The earlier010:40 grading disagreement is now resolved through a distinct cartilage
proposal; the earlier contrary judgment and balanced reference review are preserved.

The [retirement decision](../initiative/audits/cartilage-legacy-three-retirement-integration-v1/accepted-retirement-decision.json)
withdraws the entire earlier `cartilage-morphology-004-088` case after three complete
source rereads and a distinct assessor's explicit acceptance. The three PF annotations
return exactly to their original uncertain states/evidence. The original case wrapper
is preserved byte for byte in the audit and removed from active `cases/` and its
registry list. All prior proposal, assessment and decision files remain unchanged.
The still-pending cartilage QA record binds the retirement decision and original
wrapper. No duplicate or no-op correction is added; final export must verify all
three masks remain zero. The [execution proof](../initiative/audits/cartilage-legacy-three-retirement-integration-v1/retirement-integration-validation.json)
validates exact preservation of every other220 change object.

At the preceding205 checkpoint, the second focused wording batch contributed87
accepted changes:74 unsupported negatives and three insufficiently specified positives
became uncertain, six already-masked states became unmentioned, and four quotations
improved. Those artifacts remain historical evidence, including their then-pending
grading disagreement. The three global semantic gates remain pending.

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
