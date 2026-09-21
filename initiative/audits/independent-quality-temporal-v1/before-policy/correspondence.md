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

The current policies enforce 16 whole-study report-supervision exclusions:
052:20, 075:14, 048:12, 086:13, 017:07, 018:19, 038:37, 067:37,
052:01, 052:17, 001:39, 007:16, 062:21, 048:09, 048:38 and 053:47. They cover a shoulder report,
unresolved two-knee acquisitions and explicitly bilateral report aggregates.
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
16 enforced exclusions do not close the remaining queue.

These controls do not certify completion of report review, semantic QA or the
remaining technical correspondence queue. Final manifests and actual final-data
training, resume and inference validation remain release requirements.
