# Report-to-image correspondence exclusions

`rsna.labels` requires `initiative/audits/correspondence-release-registry.json`.
The registry must bind every preserved `correspondence-release-dispositions-v*.json`
policy in that directory, with exact hashes and no duplicate studies. An empty
registry is accepted only when there are no policy files. There is no command-line
switch to omit these checks.

Each exclusion binds a study, its exact original blinded report batch and text,
and a verified technical proof identifying the same case and study. Older native
proofs without a top-level study ID use a separate immutable identity wrapper.
The wrapper binds the original proof, the exact source handoff already bound by
that proof, and the source report. The original observations remain unchanged.

After target adjudications and transitive duplicate grouping, the builder sets
all twelve report loss masks to zero for the excluded non-gold studies. It keeps
the report, target states, evidence, values, split and leakage group unchanged.
The `report_correspondence_masked` provenance and report correspondence columns
explain the exclusion. Official gold labels and evaluation masks remain intact;
report-versus-gold agreement omits reports whose correspondence is excluded.

The release exports `correspondence_trace.json`, includes that trace in
`label_audit.json`, and records the registry hash among release inputs. A missing,
changed or incomplete registry, wrong source identity, unverified proof or partial
mask fails before the command writes a new release.

The current policies enforce 18 whole-study report-supervision exclusions:
052:20, 075:14, 048:12, 086:13, 017:07, 018:19, 038:37, 067:37,
052:01, 052:17, 001:39, 007:16, 062:21, 048:09, 048:38, 053:47, 064:46 and 069:10. They cover a shoulder report,
unresolved two-knee acquisitions, explicitly bilateral report aggregates and
unresolved initial-versus-recall examination linkage and an unresolved explicit
report/header side conflict.
Literal side tags alone never cause an exclusion.

The v3 policy adds nine previously verified cases using proof binding version 2.
Each immutable wrapper links the exact original report/index/hash, literal source
quotes, original root exclusion decision and a distinct independent assessment.
Cache cases also bind their original source handoff. The validator rejects
wrong selectors, altered decisions, noncanonical assessor identities and a bare
verified status substituted for evidence. Original v1/v2 policies remain unchanged.
See `initiative/audits/correspondence-extension-v3/` for the independent review,
negative tests and validation against all 4,407 blinded source reports.

Runtime verification checks the original root decision, independent assessment
and joint handoff. It does not rerun the historical image review or recursively
validate earlier source-first reading order; the preserved root verification
artifacts remain the evidence for that work. No official gold columns were used
to develop or test the extension; mask tests use fabricated targets and splits.

The 048:12/086:13 evidence retains its source-origin limit: 83 files came from
fresh Kaggle downloads; fresh official whole-byte confirmation remains deferred
for 576 corrected pinned mirror files. Original files and reports are preserved.
The 25 completed mixed-tag investigations do not justify blanket exclusions or
named-side corrections.

The v4 policy adds 048:09, 048:38 and 053:47 using the exact exclusion action
in each preserved root cache-review record. It binds both full-source freezes,
both visual assessments, the joint handoff and completed root numerical review.
Distinct assessor identities and recorded source/visual/handoff chronology are
validated. These checks authenticate recorded evidence; they do not prove past
access order or rerun the image interpretation. Case048:31 has an explicit retain
action and is not added. All 4,407 blinded source reports pass live validation;
390 full-suite tests pass. See `initiative/audits/correspondence-extension-v4/`.
The original staged markers and initial failed integration tests are retained
with the later root promotion and successful live-validation receipts.

Other correspondence and semantic flags remain individually reviewable; the
18 enforced exclusions do not establish every remaining case's identity.

The v5 policy adds only 064:46. Its report joins initial findings to a later MR
recall, while the four bound image series cannot be assigned to either examination
from the available metadata. Both assessors accepted whole-study non-gold report
supervision exclusion; all images and original target states remain unchanged.
Proof binding version 3 validates the actual temporal evidence formats, source
selectors, literal quotations, distinct identities, explicit accepted actions,
recorded chronology, twelve displayed cache checks and the exact prior registry.
The earlier root visual record contains a mistaken source hash pointing to a PNG;
its original bytes remain intact beside an explicit, required correction. Neither
that correction nor chronology validation proves historical access order or reruns
the MRI review. The independent technical review of all 71 cached slices in
064:14's four compared series supports a preference for series 11/12 when selecting
among those repeated roles; no production series-selection or normalization change
is included. See `initiative/audits/independent-quality-temporal-v1/` and
`initiative/audits/repeated-temporal-001021064-v1/`.

The v6 policy adds only 069:10. Its single RIGHT report conflicts with literal L
tags in five series; the sixth has no side tag. Root and the distinct assessor
separately froze source readings and policy decisions, then accepted exclusion.
Proof binding version 4 verifies exact full-source selectors and quotations,
source/header membership and literal fields, separately bound decisions, actual
identities, accepted twelve-mask actions and recorded chronology. These are
source and metadata checks; no new pixel interpretation or named-side correction
is asserted. All prior five policies and seventeen exclusions remain unchanged.
See `initiative/audits/correspondence-source-header-v1/`.

The consolidated 28-case follow-up records three accepted retain decisions:
021:44, 042:14 and 042:41. Their complete source wording and metadata do not
justify a whole-study exclusion; their clinical states remain subject to semantic
QA. After excluding 069:10, the remaining 24 reports already have no observed
target supervision. The consolidated closure binds their exact original sources
and effective annotation states. Their anatomical or temporal identity remains
unresolved. Before final release, verify all twelve masks are still zero for
every one of these 24 cases after applying all final adjudications; stop release
if any becomes observed without a separately accepted correspondence decision.

The later [sampling preference](series-preferences.md) applies the reviewed
064:14 acquisition preferences and a bounded 084:27 preference while preserving
all original images. It does not change these report-supervision dispositions.

These controls do not complete global semantic QA or produce final manifests.
Final leakage checks and actual final-data training, resume and inference
validation remain release requirements.
