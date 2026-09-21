# Neutral source packets for 28 correspondence cases

All 28 full reports are preserved exactly, including whitespace and source markers, in five disjoint packets of **6, 6, 6, 5 and 5** cases. Each case includes its immutable blinded batch hash, zero-based index, study UID, report hash, exact UTF-8 text file and neutral DICOM/header evidence references. No previous assessor opinion or target state appears in those packets. The preparer had prior technical/source exposure, disclosed in the index; packet construction is not claimed to be an independently blinded assessment.

The source packets were frozen **before** opening current clinical annotation status. A separate derived inventory applies only the existing annotation-state mask rule. **24 cases already mask all 12 targets; four have at least one observed target.** No per-target states or new labels are emitted, and this does not assess effective final export eligibility or gold roles.

Start independent full-source reconciliation with **021:44, 042:14, 042:41 and 069:10**, using `source-packet-priority-01.json`. This subset copies the earlier frozen source/metadata records exactly. Keep `annotation-mask-inventory.json` and `reconciliation-priority-plan.json` separate from the assessors until they freeze their source-first observations. The prioritization reasons are source scope and existing mask status, not a conclusion that annotations are wrong or a decision to exclude studies.

The other 24 cases remain in the five numbered packets. Their existing masks should be preserved while source/examination correspondence and any need for explicit export controls are reconciled. Existing masks do not establish source-image linkage.

Independent binding review verified all28 exact identities and all five packets. For27cases, inventory-to-endpoint SOP/series/path/SHA/size bindings passed. Case013:36 is bound to its preserved survey manifest; a full inventory replay is not claimed. The independent evidence is in `../correspondence28-neutral-binding-review-v1/`, including its complete immutable file list.

`source-packet-freeze.json` binds the original35source-construction files. `validation.json` confirms they remain unchanged and verifies packet membership, report bytes, neutral references, current annotation index/file hashes and the exact four-case subset. `final-file-list.json` binds every other output in this directory.

No exclusion was accepted, no labels inferred, and no runtime policy, gold, original source, cache, shared documentation or publication changed. Training readiness remains false.
