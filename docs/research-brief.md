# Research brief

**October 17, 2026, 02:05 UTC.** Read [current state](current-state.md) for actual
scores/jobs and the [research-agent handoff](research-agent-handoff.md) for evidence,
open questions and source paths. [Earlier snapshots](research-brief.history-20261017-0146.md)
remain historical evidence, not simultaneous current facts.

## The measured position

Our independently MRI-trained N3 CoAtNet scored **0.894**, up from 0.821 / 0.869 /
0.890 at epochs 4 / 8 / 12. The licensed author-trained CoAtNet scored **0.949**
when we reproduced its unchanged inference. We did not reproduce that training
from scratch. Our sequence-context head and unchanged-weight full-384 field
both scored **0.948**; neither demonstrates an improvement over 0.949.

The active questions are whether full-encoder adaptation improves a strong
MRI-trained initialization, and whether better observations per update improve
learning under a fixed compute budget. Whole-94 adaptation is actually training
on both T4s. Expanded own-model fitting is actually training on an H100. Matched
uniform-16 / influence-16 production is admitted but waiting for GPU budget;
it has not been pushed. None has a final new competition result yet.

## What the strong reference actually does

One 73.1M-parameter CoAtNet checkpoint uses finding-specific MIL, 96 selected MRI
slices in five sequence/anatomical slots, 94 three-channel windows and anatomical
mirror TTA. Preparation is 384 pixels; executed inference uses a central 320-pixel
field. The author reports 4,349 RSNA studies, blended labels, 2,399 OAI knees for
three tasks and averaged snapshots. That recipe mixes data, supervision,
initialization, optimization and serving changes; no isolated cause is established.
[Exact public executable/training contracts](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md).

Our N3 already trains every backbone/head parameter from ImageNet pretraining.
It uses 384-pixel multi-series inputs, 12 training observations, dense public soft
supervision and 16 epochs. It is not the old frozen-DINO small-head model.
The 0.055 public gap cannot be assigned to LoRA rank, image blur or epoch count
from these confounded comparisons. More parameters alone are not a hypothesis.

## What has not improved the primary result

| Mechanism | Actual finding | Decision |
|---|---|---|
| Source-neighbor channels | Weak-development delta -0.00572762; interval crosses zero | Keep original selected-slot recipe |
| Positive BCE weights | Delta +0.00298799; interval crosses zero | No primary promotion |
| Late parameter averaging | Raw-serving delta +0.00031564; interval crosses zero | Keep unchanged own epoch 16 |
| Sequence-context head | Actual Kaggle 0.948 versus parent 0.949 | No demonstrated public gain |
| Full 384 field at inference | Actual Kaggle 0.948 versus original 320 field 0.949 | More pixels alone did not help this test |
| Removing Selection_Tier | Zero selection changes on all 1,156 outer studies | Does not explain the gap; rendering parity remains separate |
| Earlier DINO stratified sampling | +0.0009107 with interval crossing zero | No established gain; distinct from current influence mechanism |

Exact metrics, reference-serving distinctions and 2,836 defined / 1,164 undefined
bootstrap draws are in [current state](current-state.md). Two MCL/fracture
positives and three Baker's positives make the weak holdout particularly unstable.
Internal AUC is weak-label agreement, not clinical accuracy or predicted Kaggle AUC.
The licensed parent's development exposed this cohort/Gold58; derivatives have
no independent internal OOF on it. We have not tuned using protected clinical values.

## Full-model adaptation now running

Whole-94 training uses 3,887 fitting studies, all 73,136,096 trainable parameters,
effective batch 8, three epochs / 1,458 updates, encoder/head LR 3e-6 / 3e-5,
AdamW 0.02, 10% warmup, cosine to 1%, clipping at 1 and FP16/scaler with FP32 loss.
The loss combines 25% source supervision with 75% detached-parent anchoring.
BN running buffers remain fixed; affine parameters learn. Actual two-T4 recovery
matched complete model/optimizer/RNG/scaler/prediction state before production.

This tests whether low-rate, exposure-preserving adaptation helps. A teacher can
anchor mistakes; weak labels can drive harmful updates. Three passes may improve
or regress the model. Only actual scored checkpoints will establish the available
external result. Epoch completion, lower loss and green tests are not that result.

Expanded N3 adds 345 fitting studies using the original recipe. It changes both
fitting population and update count, so a better result would not isolate a pure
data-volume effect. Its old 1,156-study outer roster remains disjoint.

## Importance-guided observations

The mechanism combines [finding-specific attention MIL](https://proceedings.mlr.press/v80/ilse18a.html),
[importance sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html) and
[attention sampling](https://proceedings.mlr.press/v97/katharopoulos19a.html).
It adapts established ideas; no invention or accuracy gain is claimed.

Existing frozen-teacher features define a fixed 94-window proposal: 25% uniform
coverage plus 75% normalized influence, combining both anatomical views and
normalizing per target. Across all 3,887 fitting studies, it reduced full-readout
approximation MSE by 45.6%, every target improving, without disease labels or
new CNN inference. That is better teacher fidelity per sampled bag.

The matched uniform-16 and influence-16 roles share random variates, study order,
views, initial complete model/RNG, targets and optimizer. They sample IID with
replacement, retain duplicates and use centered inverse-proposal attention
correction. Both train the full encoder/head for three epochs / 1,458 updates;
inference retains full-94 evaluation. The finite-bag ratio is biased, teacher
influence can become stale and teacher errors can be amplified. A 94/16 sampling
ratio is not measured throughput or disease improvement.

Actual both-role MRI GPU recovery, original-input CPU comparison, proposal
Dataset and checkpoint storage acceptance are complete. Production is not yet
GPU-dispatched because its finite owner preserves funding for all unfinished
original sessions. The new serving compiler has synthetic checks only and still
requires an actual completed/preserved sampled epoch plus both-T4 and public-MRI
acceptance. [Protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md),
[recovery](../initiative/audits/licensed-importance-encoder-20261017-v1/README.md),
[full readiness](../initiative/audits/licensed-importance-full-comparison-20261017-v1/README.md),
[serving preparation](../initiative/audits/licensed-importance-serving-preparation-20261017-v1/README.md).

## Where independent research can help most

The [research-agent handoff](research-agent-handoff.md) gives concrete questions:
source/teacher error correction without gold leakage; train/serve observation
coverage; finding-specific localization and extent; a ranking objective that
changes study ordering; and a faithful comparison of source-data/initialization.
Each proposal should identify what measured failure it addresses, a falsifiable
mechanism, the closest already-run control and the smallest informative experiment.

Collect the pending regional external result and actual trained-parent checkpoints
before calling another head successful. Preserve negative results. Match inputs,
initialization and compute for causal claims. Teacher fidelity, software recovery,
three public MRI cases and Stanford source-task scores do not establish a new
RSNA competition result. Public leaderboard selection itself can overfit; a win
or private-score improvement is not promised.
