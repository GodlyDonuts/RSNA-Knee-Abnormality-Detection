# Predeclared fixed-final B/C/C0 comparison

Decision recorded October 20, 2026, before any full predictive model or score exists.
This interprets the owner's [proposal](../predictive-evidence-proposal-20261020-v1/reference/RESEARCH_PLAN.md)
and the accepted [numerical recipe](engineering-decision.json). It creates no new
cohort, label access, job, submission or evaluation authorization.

## Endpoints and principal contrasts

Use **epoch 6 / 2,916 updates** from each completed B, C and C0 arm, with the
original 3,887 fitting studies, target masks, shared initialization, source,
schedule and matched serving policy. Do not select earlier epochs by fitting
metrics or leaderboard outcomes. B has 73,092,652 parameters; C/C0 have 73,271,124.
Image/update budgets match; B/C are different heads, not exactly equal compute.

The two principal score contrasts are:

1. **C minus B:** diagnosis-conditioned predictive evidence versus the strong
   direct classifier with diagnosis-independent reconstruction and observed
   descriptors.
2. **C minus C0:** contribution of the density objective, using identical C/C0
   architecture, initialization and operations, with density weight .1 versus zero.

C0 minus B is secondary. Record all three actual fixed-final Kaggle outcomes,
including failures and unavailable scores. Compare the own-adaptation **0.949**
and external R6 ensemble **0.958** as practical historical references, not matched
causal controls. Preserve the qualified 0.958 baseline and final-run time reserve.

## Minimal result interpretation

- Verify actual completion/export identity and trained T4 qualification first.
  Training loss, clipping, gradient norms and throughput describe optimization
  and deployment; they are not accuracy endpoints. B/C density losses describe
  different distributions and cannot alone establish superior conditional fit.
- Public three-study outputs support exact roster/CSV and numerical-parity checks
  only. Do not infer useful AUC uncertainty, clinical error diversity or rare-target
  performance from those rows.
- Report Kaggle's actual displayed precision. Aggregate scores alone provide no
  paired patient-level confidence interval, per-target AUCs or significance test.
  A small rounded difference is provisional competition evidence; a rounded tie
  does not prove equal underlying scores.
- If already available permitted paired predictions cover a larger identical
  roster, finite-value, saturation, distribution and ranking-disagreement summaries
  are descriptive. Low correlation without paired correctness is not demonstrated
  ensemble value. Do not create a new evaluation sweep for this checklist.
- C exceeding both controls supports this particular density-trained adaptation
  provisionally. C and C0 similarly exceeding B does not establish a density-loss
  benefit. B winning is a negative result for this implementation, not proof that
  predictive learning can never help. No result establishes censoring's causal
  contribution, calibrated clinical likelihoods or lesion localization without
  the proposal's separate mechanism controls.
- Do not redefine this primary comparison after seeing scores. Subsequent epoch,
  loss-weight or ensemble experiments must be identified as new exploratory work,
  with their own stated rationale and qualified serving policy. Disagreement or a
  better standalone score alone does not establish incremental R6 value.

## Exposure and held-out limitations

All 3,887 eligible fitting studies are trained on, including the old development
folds. They are not held out for these models. The exact inherited CoAtNet weights
also [overlap the internal weak holdout, and Gold58 influenced parent selection](../../audits/released0949-baseline-20261016-v1/README.md).
Removing parent logits or training fresh heads does not erase that exposure.
These comparisons are not independent whole-model OOF or a clean clinical test.

The [original evaluation contract](../../../docs/training-strategy.md) reserves
the 441-study weak-validation split for one predeclared confirmation after
freezing the shortlist, inputs, schedule and ensemble rule. Later
[correspondence admission](../../audits/research-paper1-assessment-20261016-v1/README.md)
identifies 440 eligible studies and one excluded. It measures weak report
agreement, has sparse rare targets and cannot restore independence from this
parent. Gold58 has historical calibration exposure and permits only a frozen-model
failure diagnostic, without subsequent selection or calibration. Neither is
opened by this decision.

For the remaining deadline, finish and score the fixed comparison without a new
protected-label detour. The public leaderboard is a noisy development signal;
selecting a practical candidate from it is not independent confirmation or proof
of private-set robustness. No first-place, 0.97 or original accuracy gain is claimed.
