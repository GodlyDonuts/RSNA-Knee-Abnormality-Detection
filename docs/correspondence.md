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

The current policies cover the shoulder report at 052:20, the unsided report
with two knee protocols at 075:14, and the integrated bilateral aggregates at
048:12 and 086:13. The latter two retain explicit source-origin limits: 83 files
came from fresh Kaggle downloads; fresh official whole-byte confirmation remains
deferred for 576 corrected pinned mirror files. Original files and reports are
preserved. The 25 completed mixed-tag investigations favor coherent single
protocols; conflicting tags remain recorded and do not automatically cause an
exclusion or named-side correction.

The independently reviewed 052:01/17 bilateral report aggregates require withholding report supervision because their image-to-report-block correspondence remains unresolved. Their 245 source memberships, 12 complete cache volumes and 36 sampled views are preserved in `initiative/audits/correspondence-052001-017-cache-v1/`; adding their final registry policies remains pending.

The newly preserved001:39/007:16 bilateral source/cache investigations and062:21 dual Greek report investigation likewise require withholding supervision while report-block correspondence remains unresolved. Their final registry policies are pending; the four enforced whole-study exclusions above are still the complete current policy set. See `technical-tier4-bilateral-001007-cache-v1/` and `correspondence-055058060062063-v1/`. Other cases in the latter audit retain individual adjudication needs, including explicit performed-side clarification at063:35; no blanket exclusion was inferred from the group.

These controls do not certify completion of report review, semantic QA or the
remaining technical correspondence queue. Final manifests and actual final-data
training, resume and inference validation remain release requirements.
