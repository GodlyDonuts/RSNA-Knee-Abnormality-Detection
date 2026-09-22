# Conservative report supervision and release verification

The data release preserves clinical report annotations and makes a separate decision
about whether each target may contribute to training loss. A held positive remains a
positive report annotation; its training mask is zero. Original reader opinions,
quotes, accepted corrections and unresolved disagreements remain available.

The fixed candidate was independently checked across all 4,407 reports and 52,884
report-target cells. It contains 4,412 overlapping hold records covering 3,629 definite
observations, with 23,844 definite observations retained before gold, technical and
split filtering. The final export reproduces this membership. See the [readiness report](data-readiness-report.md)
for post-filtering counts and the completed runtime acceptance.

## Evidence and exclusions

Every cell has a ledger entry, including uncertain, unmentioned and excluded cases.
A retained definite observation requires genuine independently authored source
readings, current-state support and an explicit disposition for every relevant
obligation. A contrary judgment is preserved even when a later reasoned resolution
supports the retained state. Quote-only resolutions cannot erase state dissent.

The narrow negative reuse cohort applies only to unchanged reviewed negatives with
two genuine negative source readings and no concrete contrary or specific evidence
objection. A lexical retrieval flag alone is neither a diagnosed error nor a new
clinical reading. Separate unresolved obligations still mask a cohort member.

Text-intrinsic holds propagate to exact normalized report copies for the same target.
Record-specific issues stay local. The known near-report pair remains a leakage
edge with explicit per-target dispositions; similarity does not automatically
transfer diagnoses or count as an additional source reading.

Clinical corrections remain in the adjudication registry. The supervision registry
binds a separate proposal, distinct assessment and root decision to exact context,
holds and ledger hashes. Changing any reviewed source, correction, required policy
or bound data-processing code invalidates the release until its updated context is
explicitly accepted. Model architecture and experiment configuration are outside
these data-policy code pins.

## Evaluation and splits

Official gold is evaluation-only. Its values do not select holds or repair the weak
split. Gold-related image/report groups are quarantined transitively. Uncertain,
unmentioned, excluded, technically ineligible and quarantined report supervision
has zero loss mask.

The initial deterministic group hash split left no MCL-positive observation in weak
validation, even before conservative holds. The reviewed repair acts after fixed
masks: it moves the smallest safe whole leakage group, with deterministic hash
tie-breaking, only when a target/class is absent in a development split. The donor
must retain every class it already had, protected groups cannot move, and only split
values may change. An impossible repair fails. The final export required one
group/study move, yielding 3,904 training, 441 weak-validation, 58 gold and 4 quarantined
studies. The persisted-table and runtime checks passed.

Both classes being present is only a minimum viability check. Several targets have
one positive in weak validation. Their AUC is correspondingly unstable; presence
must not be presented as a precise generalization estimate. Earlier historical
experiments also used the small gold set, so it cannot be described as globally
untouched by the entire project history.

## Runtime boundary

`rsna.labels` verifies the complete policy, original review lineage, corrected image
inventory, series sampling and gold isolation. It builds a new output directory
atomically and writes `verified_release.json` only after a fresh full-table replay.
A failure does not publish a partial output. Existing release directories are not
overwritten.

`rsna.train` requires the verified full release and exact series/source binding
before selecting training rows. A diagnostic study limit applies only after full
validation; it is not a policy bypass. Checkpoints bind the release, ordered selected
study identities, settings, optimizer and random state. Resume rejects incompatible
inputs or incomplete smoke-test epochs.

Labeled inference verifies the full release before selecting an evaluation split.
Public-test inference needs a checkpoint's valid release identity but no training
reports or gold files. Its new test inputs have separate provenance. Conflicting
duplicated checkpoint identities are rejected.

The final diagnostic acceptance must demonstrate actual optimizer updates, exact
CPU checkpoint equality between uninterrupted and resumed training, zero gradients
at masked training targets, prediction coverage for all 58 gold studies, and the
released three-study submission template. This establishes pipeline operability;
it is not competitive training or evidence of clinical diagnostic accuracy.

## Current status

Final context approval, all three QA resolutions, persisted export and actual
training/resume/inference acceptance have passed. The preread path bug was fixed
using actual preserved provenance, with independent review and regression tests.
The release is locally ready; verified Git/HF publication is recorded separately
in the [readiness report](data-readiness-report.md).
