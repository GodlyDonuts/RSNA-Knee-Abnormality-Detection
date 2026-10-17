# Research brief

**October17,2026,01:46UTC.** [Current state](current-state.md) contains actual
scores/jobs; [execution handoff](codex-continuation.md) binds operations. Earlier
[research snapshots](research-brief.history-20261017-0146.md) preserve the full
history and negative results; their dated states are not current facts.

## The actual position

Our independently MRI-trained N3 CoAtNet scored **0.894**, up from0.821/0.869/0.890
at epochs4/8/12. The licensed author-trained CoAtNet scored **0.949** when we
reproduced its unchanged inference. We did not reproduce that training from
scratch. Our sequence-context addition and unchanged-weight full384 field both
scored **0.948**; neither demonstrates an improvement over0.949.

The current experiments address two concrete questions: does careful full-encoder
adaptation improve this strong MRI-trained initialization, and can better sampled
observations provide more useful updates per unit compute? Whole94 adaptation
is actually training on bothT4s. The additional own-model2,592study fit is actually
training on an H100. These have no final/new competition result yet.

## What already works

The strong reference uses one73.1MCoAtNet checkpoint with finding-specific MIL
attention,96selected MRI slices in5sequence/anatomical slots,94three-channel
windows and anatomical mirrorTTA. Its executable prepares384pixels and uses a
320central field. The author reports4,349RSNA studies, blended labels,2,399OAI
knees for3targets and averaged training snapshots. This is not evidence that a
large multi-architecture ensemble is required, or that one isolated ingredient
explains its score. [Exact executable/training contracts](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md).

Our N3 trains every backbone/head parameter from general ImageNet pretraining;
it is not the historical frozen-DINO small-head model. It already uses384inputs,
multiple series,12training observations, dense public soft fitting supervision
and16epochs. Its missing0.055 versus the released parent combines training-data,
initialization, supervision, preprocessing, optimization and inference differences.
It does not isolate a cause or justify merely increasing rank/model size.

## Negative and uncertain evidence

- Physical source-neighbor channels regressed the fixed weak-development point
  estimate:0.92092535→0.91519773, delta-.00572762, conditional95%
  [-.02522802,+.01337630]. Preserve the original selected-slot recipe.
- Positive BCE weighting gave0.92391334, delta+.00298799, interval
  [-.00999470,+.01656811]. No established primary gain or weighted Kaggle score.
- Fixed E12/E14/E16 parameter averaging gave raw-serving+.00031564,
  interval[-.00214034,+.00293081]. Retain unchangedE16 as own-model default.
- Dropping Selection_Tier changed zero selected-series choices on all1,156 outer
  studies. This rules out that field alone, not complete rendering/metadata parity.
- The sequence-context head changes strict study-pair ordering, so it is more
  than calibration; actual0.948 still does not beat0.949. Mechanism is not gain.
- Full384 field produced actual0.948. Keeping more pixels at inference alone
  did not demonstrate improvement; matched training-resolution evidence differs.
- A window-materialization profile was a counterfactual: original N3 already
  selects before materialization. Its saving is not a new N3 throughput result.
- Earlier uniform/stratified DINO sampling differed by+.0009107 with a primary
  interval across zero. Do not confuse that older sampler with the new mechanism.

The N3 weak holdout has just2MCL/fracture positives and3Baker positives. Its macro
0.920925 is weak-label agreement, not clinical accuracy or predicted Kaggle AUC.
Whole-component uncertainty has undefined draws; report them rather than hiding
sparse-target instability. The licensed parent was developed using this cohort/
Gold58, so it and derivatives have no independent internalOOF on that cohort.
Protected clinical values remain unused by us; no blend is tuned on them.

## Registered full-model adaptation

Current whole94 fit uses3,887 eligible fitting studies, all73,136,096 trainable
parameters, effective8,3epochs/1458updates, encoder/headLR3e-6/3e-5, AdamW.02,
warmup10%/cosinefinal1%,clip1, FP16/scaler1024/FP32loss and25%source/75%detached
parent anchoring. BN running buffers stay frozen; affine parameters train.
Exact full/stop/resume model/optimizer/RNG/scaler/prediction recovery passed on
bothT4s before production. Entire immutable epochs are independently verified
and privately preserved before precision/publicMRI acceptance or competition use.

The exposure-preserving low-rate adaptation is a test, not a promised gain.
A wrong teacher can anchor errors, source labels can be noisy, and3passes may
help or harm. Actual checkpoint submissions are the relevant available external
comparison; training loss, green tests and more parameters are not that evidence.

## New importance-guided observation mechanism

This combines [finding-specific attention MIL](https://proceedings.mlr.press/v80/ilse18a.html),
[importance sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html) and
[attention sampling](https://proceedings.mlr.press/v97/katharopoulos19a.html).
It is an adaptation of those ideas, not a claim of a new invention.

A frozen teacher's existing full-study features define a fixed94-window proposal:
25%uniformcoverage plus75%normalized influence, combining both anatomical views
and normalizing per target. On all3,887 fitting studies it reduced full-readout
approximation MSE45.6%, every target improving, without disease labels/newCNN.
This is better teacher fidelity per sampled bag, not better disease classification.

Matched uniform16/influence16 roles use common random variates/study order/view,
IID sampling with replacement, exact duplicates and centered inverse-proposal
attention correction. Both learn the full encoder/head with the same complete
initial model/RNG, original targets/schedule and fixed3epochs/1458updates. Full94
inference stays unchanged. The ratio is biased for finite bags, influence can
become stale during learning, and a bad teacher can amplify mistakes. No accuracy
or94/16 throughput gain is inferred from the mathematical sampling ratio.

Actual two-role MRI GPU recovery139130771 completed: complete full/interrupted/
resumed state and full94 prediction bytes match, both roles start identically
and learn different tensors. Actual full-input CPU bridge and private fixed
proposal Dataset are verified. Storage is admitted; the finite full-comparison
owner waits for enough actual GPU budget to fund all sessions without taking
funds from original training. [Protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md),
[original recovery](../initiative/audits/licensed-importance-encoder-20261017-v1/README.md),
[full readiness](../initiative/audits/licensed-importance-full-comparison-20261017-v1/README.md).

## Decision discipline

Judge by actual external outcomes and matched held-out evidence where independent
labels exist. Preserve negative results, failures and provenance. Keep healthy
training running; match initialization/input/compute when making causal claims.
Do not treat teacher fidelity, software recovery, public3study execution or a
source-task Stanford result as clinical/RSNA/Kaggle improvement. The completed
regional spatial/global-repeat contrast remains useful external evidence to
collect. Do not claim attention identifies verified anatomy or upscaling creates
missing lesion information without an actual matched test.
