# Research memo 1: evidence assessment and next decisions

The owner's supplied memo correctly redirects attention from incremental DINO changes toward supervision coverage and a strong fully trained compact MRI model. It is a research proposal, not proof that its combined recipe will reach 0.95. Its file-citation tokens are not independently accessible sources. The linked public author disclosure reports combined changes rather than controlled causal effects.

## Verified coverage opportunity

UID-only comparison of the pinned public supervision table against the accepted release finds 4,407 distinct IDs, no duplicates and full coverage. The release contains 3,904 train, 441 weak-validation, 58 gold and four quarantined studies. UID coverage does not establish image availability, correct label semantics or admission to a new fit.

Existing leakage-group folds cover all 3,904 training rows. Preserving fold-0 groups excludes all 1,301 fold-0 rows, leaving 2,603 fitting rows: 356 more than the current 2,247. The extra 145 previously all-masked fold-0 rows must remain excluded from fitting even though the current observed endpoint contains only 1,156 rows. Additional original MRI cache availability must be checked before dispatch. No masks, folds, labels or live recipes changed during this audit.

A final recipe-frozen train-only fit could use 3,904 rows. A separately declared final-fit scope could potentially use 4,345 non-gold/non-quarantine rows including weak validation; this is not authority to contaminate the current experiment's held-out endpoint. Gold and quarantine remain excluded.

## Important correction: current supervision is already dense

The memo's old counts of three fracture, five MCL and sixteen effusion positives are not current N3 training supervision. The current fitting-only public overlay observes all 12 targets on all 2,247 studies. Thresholded soft-label counts above 0.5 are fracture 157, MCL 368 and effusion 1,389. These are source probabilities, not clinical adjudications. Sparse old held-out labels remain an evaluation limitation. Do not diagnose current training as receiving almost no positive gradient from obsolete counts.

Fitting-only soft positive-mass weights are recorded as candidate descriptive quantities, not an accepted new loss. Positive weighting can emphasize noisy positives as well as genuine findings; compare a bounded, explicitly specified loss against its control.

## Decisions

1. Preserve the healthy original fixed-16-epoch N3 fit unchanged. It fully trains 73.1 million parameters; it is not another frozen-backbone adapter experiment.
2. Complete raw-DICOM-to-CSV T4 qualification. The actual two-T4 precision/encoder benchmark has already passed, superseding the memo's older status; it does not prove full hidden runtime.
3. Prepare expanded fitting coverage under existing leakage-group boundaries, checking original MRI availability and the full source-label schema before fitting. Distinguish population expansion from a weighted-loss experiment rather than claiming a bundled result isolates either.
4. Keep the preregistered MC128 bag-inference comparison. Do not select per-target transforms using public leaderboard feedback.
5. Preserve late checkpoints for a declared weight-average comparison. CoAtNet has BatchNorm buffers: a naive average of all state entries, especially integer counters, is not valid acceptance. Specify buffer handling and any fitting-only recalibration before evaluation.
6. Freeze the final recipe before reclaiming weak-validation rows for an explicitly declared final fit. Keep held-out and Kaggle scores separate; no winning-performance claim is justified yet.

Current own best solo Kaggle result is 0.872; the earlier owner-submitted 0.936/0.942 systems are different public ensembles. No current CoAtNet clinical/Kaggle result exists. More data, more parameters and a successful engineering benchmark are opportunities, not measured accuracy gains.

See the two aggregate JSON audits here for the reproducible counts. No protected label values or MRI records are included.

## Actual additional-input metadata result

The pinned corrected original inventory contains all356 additional fitting studies:1,849 accepted series/56,825 slices/37,933,598,808 DICOM bytes, no missing study or series. This is metadata completeness, not a new acquisition or decoded cache. No MRI bytes moved, no GPU used and no protected label values read. Actual extra cache construction and resource admission remain required before expanded fitting.

## Subsequent fitting-only quality audit: expansion is conditional

Public probabilities for all356 additional fitting rows are finite and in[0,1], but original eligibility metadata excludes11 from report-supervised fitting:10 unresolved bilateral correspondence and1 confirmed two-knee aggregate report. Preserve these exclusions. Thus the immediate expanded fitting candidate is **2,592 =2,247+345**, not automatic2,603 admission. This supersedes the unconstrained metadata capacity above. Across all training rows17 are correspondence-excluded, giving3,887 report-eligible train cases. Weak validation has440 eligible/one excluded; an explicitly scoped final fit could potentially reach4,327 eligible cases, not blindly4,345.

A further59/356 source-label rows have identical values across all12 targets:37 at0.25,18 at0.025 and4 at0.015, compared with14/2,247 current fitting rows all at0.025. Uniform values are not proof of bad labels (normal examinations can have consistently low probabilities), but the37 at0.25 warrant checking source fallback/uncertainty semantics. None are silently deleted or promoted. No all-zero or all-0.5 extra rows exist. No clinical truth or accuracy improvement established; the admitted release/live N3 remain unchanged.

## New primary documentation and unresolved marker semantics

The detailed follow-up by the [high-scoring CoAtNet author](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792), inspected directly in authenticated Kaggle, calls exact0.25 in the public v4 table a not-addressed marker and uses their own report labels for those cells. This is actionable evidence about a successful alternative supervision procedure, not proof of our table's exact missingness map.

Our [existing upstream source-card audit](../../../docs/research-watch/additional/rsna-steven-v6-provenance-addendum-2026-10-14.md) already establishes byte identity to Steven version6 v4. It records original extraction0.5 as not-addressed, recommends2*abs(p-0.5) confidence weighting, and notes undisclosed blend coefficients/branches. A blended0.25 can therefore reflect a combination of parent evidence rather than a universal missing flag. Do not silently mask either numerical value or claim that an accuracy defect has been proven. Issue92 now focuses on this specific conflict without another target download. Current N3 remains full-weight soft-BCE; a changed loss is an explicit experiment, not a retroactive repair.

Exact released CoAtNet inference/weight contracts are being independently audited in issue93; the public author says the released weights include OAI training and Gold58 selection. Read-only source inspection does not admit those weights to our competition path.
