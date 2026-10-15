Latest addendum: October 15, 12:20 UTC. Read rsna_reports/research_update_20261015.md first: E8 scored .863, prior .872 remains best; running snapshots retained as history. Compact proposal N3 remains untested. Slice-spacing audit added without a new-fit recommendation.

# October 15 update

Read rsna_reports/research_update_20261015.md first for the latest evidence and status. The compact proposal now includes normalization correction Revision N2 and training-contract clarifications Revision N3. Older snapshots remain historical.

# RSNA research dossier

14 October 2026. Start with rsna-research-ledger-20261014/final-week-decision.md: it supersedes earlier candidate rankings in the broader strategy inventory. The two recommended options are a directly deployable compact challenger and one conditional teacher blend. These are proposed comparisons, not completed training results.

The historical ledger and expansion distinguish completed results, engineering acceptance, interrupted runs and proposals. The coverage index classifies all 1,090 audit-root entries but does not claim every receipt or clinical record was read. Its reviewed outcomes cover 22 scored and 12 failed submissions, not 34 training runs or an exhaustive account total.

Historical pin: 8300cfd20c2acaeef9d684a81dbc7fe977282a62. Newer implementation and preservation receipts are separately pinned in the relevant documents. Best verified own public score remains .872.

The source-provenance correction is integrated: the active Shio table matches Steven Kaggle v6, with report-label parents, CC0 declarations and Gold58-guided source choices. New focal-window pooling is already implemented at e4f288d6, with engineering tests but no accuracy result.

This is the private research deliverable. It contains source-linked findings and proposals, no clinical case rows, images, weights or private implementation code. GitHub publication remains pending confirmation.

## Concrete compact comparison

The new [native-cache CoAtNet specification](rsna-compact-native-coatnet-challenger-2026-10-14.md) makes the first comparison concrete: existing native inputs, twelve series-stratified training triplets, all valid within-series triplets at inference, current accepted targets and a fixed epoch-16 endpoint. It is an untested own recipe, not a reproduction or predicted .95 score. The [independent stress test](rsna-compact-recipe-stress-test-2026-10-14.md) records sampling/pooling and BatchNorm risks without adding a sweep.

## Reading order

1. [final-week-decision](rsna-research-ledger-20261014/final-week-decision.md)
2. [experiment-ledger-reconciled](rsna-research-ledger-20261014/experiment-ledger-reconciled.md)
3. [historical-run-expansion](rsna-research-ledger-20261014/historical-run-expansion.md)
4. [experiment-family-coverage](rsna-research-ledger-20261014/experiment-family-coverage.md)
5. [rsna-single-model-evidence-2026-10-14](rsna-single-model-evidence-2026-10-14.md)
6. [rsna_solo_strategy_adversarial_2026-10-14](rsna_solo_strategy_adversarial_2026-10-14.md)
7. [rsna_window_relation_proposal_20261014](rsna_window_relation_proposal_20261014.md)
8. [observation-strategy](rsna-research-ledger-20261014/observation-strategy.md)
9. [multitask-conflict-strategy](rsna-research-ledger-20261014/multitask-conflict-strategy.md)
10. [rsna_extent_severity_critique_2026-10-14](rsna_extent_severity_critique_2026-10-14.md)
11. [rsna_menimv_kneedino_mechanism_review_2026-10-14](rsna_menimv_kneedino_mechanism_review_2026-10-14.md)
12. [privileged_report_distillation_review](../additional/privileged_report_distillation_review.md)
13. [label-provenance](rsna-research-ledger-20261014/label-provenance.md)
14. [research-update-1900](rsna-research-ledger-20261014/research-update-1900.md)
15. [rsna_teacher_strategy_red_team_20261014](rsna_reports/rsna_teacher_strategy_red_team_20261014.md)

## Deeper reasoning update

The [score-gap challenge](rsna-score-gap-selection-challenge-2026-10-14.md) qualifies weak-development-only selection and audits the matched .783-to-.854 supervision comparison. The [supervision limits](rsna_reports/rsna_supervision_ranking_limits_20261014.md) distinguish finite-model denoising from unidentified clinical correction. The [full-bag gradient review](rsna_reports/rsna_full_bag_gradient_review_20261014.md) identifies relevant ICML prior art but rejects a drop-in CoAtNet change because of normalization coupling and cost. No new model has been trained.

## Additional public evidence

The [external OOF-repair audit](rsna_reports/rsna_external_oof_repair_audit_20261014.md) checks a reported failed pseudo-label intervention against its code and finds a different state-gated recipe plus selection/ancestry confounds; it does not invalidate the bounded proposed blend. The [efficiency option](rsna_reports/rsna_efficiency_option_20261014.md) verifies official dual-track eligibility and the runtime tradeoff without changing the primary accuracy objective or adding a fit.

## Anatomical input audit

The [current-pinned anatomical contract audit](rsna-anatomical-input-contract-audit-2026-10-14.md) found no demonstrated ordering, physical-aspect resize or train/inference transform bug. Acquisition-relative positions are not universal medial/lateral coordinates, but adding canonicalization remains a new hypothesis. Seven synthetic checks and their limits are documented; no real clinical inputs were inspected.

## Latest completed receipts and targeted reviews

The [21:19 update](rsna-research-ledger-20261014/research-update-2119.md) supersedes earlier pending-fit and notebook-acceptance statuses: all ten all-development epochs are complete/preserved and E8 passed public-example checks, with no new score. The [recovery diagnosis](rsna_reports/rsna_recovery_returncode_diagnosis_20261014.md) explains why completion JSON is not OS exit evidence; exact cause remains unresolved. The [late-checkpoint soup review](rsna-research-ledger-20261014/lora-soup-design-review.md) establishes that the prior average was already gauge-correct and separates the new rank32 compression hypothesis from a bug fix. New pin: 1f05aa1df34bd92212ce92633b99366f98804e3c.

## Hierarchical knee source gate

The [hierarchical source audit](rsna_reports/rsna_hierarchical_knee_source_gate_20261014.md) keeps the severity branch deferred: the coarse-to-fine paper and released head topology disagree, radiographic KL is not a three-compartment MRI reference, and KJD-25K has endpoint, independence and release-admission gaps. These sources do not justify another final-week fit.

## User-proposed refinements

The [crop audit](rsna_reports/rsna_crop_pipeline_audit_20261014.md) separates artificial padding from acquired background and distinguishes cached framing from pre-downsample detail preservation. The [temperature-sampling review](rsna-research-ledger-20261014/temperature-label-averaging.md) evaluates repeated LLM extraction against the existing blend, with correlated-error and validation caveats. Both are unexecuted proposals. An [additional public blog screening](rsna_reports/rsna_public_blog_screen_20261014.md) excludes a five-fold .903 claim from single-model evidence.
