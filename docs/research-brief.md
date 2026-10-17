# Research brief

**October 17, 2026, 07:41 UTC.** Read [current state](current-state.md) for actual
scores/jobs and the [research-agent handoff](research-agent-handoff.md) for evidence,
open questions and source paths. [Earlier snapshots](research-brief.history-20261017-0146.md)
remain historical evidence, not simultaneous current facts.

## The measured position

Our independently MRI-trained N3 CoAtNet scored **0.894**, up from 0.821 / 0.869 /
0.890 at epochs 4 / 8 / 12. The licensed author-trained CoAtNet scored **0.949**
when we reproduced its unchanged inference. We did not reproduce that training
from scratch. Our sequence-context head and unchanged-weight full-384 field
both scored **0.948**; the regional spatial head also scored **0.948**.
None demonstrates an improvement over 0.949.

The active questions are whether full-encoder adaptation improves a strong
MRI-trained initialization, and whether better observations per update improve
learning under a fixed compute budget. Whole-94 adaptation completed two epochs/972updates on both T4s;
its exact final continuation is API RUNNING. Expanded own-model fitting is actually training on an H100. Matched
uniform16E1 **139137040v1 actually completed486 updates** and all5private objects
were fully verified. **InfluenceE1 139154664v1 also completed486 and all5objects
are private verified; uniformE2 139167076v1 completed972/all5private objects; influenceE2 139181903v1 runs**. Whole-parentE1 passed
bothT4 qualification/public acceptance and was submitted once**57294905**;
actual hidden terminal is **COMPLETE/0.949**, matching the parent without a gain. E2 now completed972/all5private objects verified; original E3
139172537v1 is RUNNING. Expanded N3 has epochs1–15 privately fully verified,
at least4860/5184completed updates, under its original healthy H100 job. The learned whole94E1 now actually scored0.949, matching the external parent;
there is no demonstrated new gain or fixed-final result yet.
[Actual progression, entry and preservation](../initiative/audits/research-progress-20261017-0448-v1/README.md).

The fresh [official Public leaderboard observation](../initiative/audits/github-current-state-refresh-20261017-v2/actual-public-leaderboard.json)
shows first place **0.969** versus our **0.949**. A publicly shared 0.956 blend is not the
leaderboard ceiling; no leading method/private result is inferred.

## New complete public reference to reproduce

The newly pinned Helios138222340v1 actually displays0.956 and combines the
existing CoAtNet with four R7 ConvNeXt-small folds. Its manifest describes384/352
physical inputs,176training windows, state/extent supervision, ordered pairs,
EMA and state/fluid stackers. These differ jointly from the single S6 all-data
checkpoint; they do not establish a resolution-only, supervision-only or fold-only
gain. All8 released R7 objects are privately size/SHA/safe-header verified once.
The extra224 variant also displays0.956, so its title is not evidence of an
additional score improvement. Reproduce the exact complete reference before
claiming a new0.956 baseline; no R7 GPU/entry has been dispatched.
[Actual source/manifest/weight proof and hypotheses](../initiative/audits/r7-public-reference-admission-20261017-v1/README.md).

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
| Regional spatial head | Actual Kaggle 0.948 versus parent 0.949 | No demonstrated public gain |
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
Dataset and checkpoint storage acceptance are complete. UniformE1 is now complete/private verified and influenceE1 runs after the original
funding check. A capacity-aware observer
protects the original main continuation and shares the other actual Kaggle GPU
lane with qualified-model inference; no running scientific recipe was changed. The sampled serving compiler has synthetic checks only. UniformE1 now satisfies
the completed/preserved checkpoint prerequisite; actual learned sampled-model
bothT4/public-MRI serving acceptance remains absent. [Protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md),
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

The regional result is now0.948; it is not an improvement. Collect the actual
trained-parent and matched-sampler competition outcomes before promoting them. Preserve negative results. Match inputs,
initialization and compute for causal claims. Teacher fidelity, software recovery,
three public MRI cases and Stanford source-task scores do not establish a new
RSNA competition result. Public leaderboard selection itself can overfit; a win
or private-score improvement is not promised.

## View-consistency screen: intermediate, no promotion

A new label-free screen tested whether separate-view teacher targets disagree
substantially with the probability consensus used at inference. Across all3,887
fitting cases, mean gaps were1.29–2.71percentage points and strict-pair ordering
reversals2.75–7.41%. The largest fraction with >10-point probability gap was4.37%.
Neither predeclared weak nor strong triage criterion applied: **intermediate**.
These are teacher pairs without clinical labels, not AUC errors. Current training
is unchanged and another GPU arm is not funded by this screen.
[Actual evidence and limits](../initiative/audits/teacher-view-consistency-screen-20261017-v1/README.md).

## New deadline-program assessment

The [owner's new analysis](../initiative/research/owner-deadline-program-20261017-v1/README.md)
points to a distinct strong public parent and structured observation, rather
than another generic backbone search. We actually pinned S6 notebook138285808v2
and its small checkpoint/code manifests, then acquired the original checkpoint
privately once with exact SHA and safe tensor-header verification. Its reported
0.950 remains external. Its exact293-member source/anatomy and unchanged strict
CPU stack now pass; private CPU-only Kaggle MRI gate139151928v1 completed all3 public studies/1586
windows. CPU942028 fully verified original compressed inputs and their decoded
bytes on Newton. CPU942136 additionally scored all3/1586windows with the unchanged classifier/pooling
and device/report-only replacements,355.95s/zero fallback. This is a CPUFP32
reference. Bounded H100 probe942047 remains queued. Original T4 qualification
**139165932v1 actually COMPLETED156.6s**, bothT4s/threeMRI/finite13-column CSV/
zero S6 fallback. The exact source was unchanged; a stale author embedded build ID
required an observer-only repair, not a GPU rerun. Original offline139181157v1 completed233seconds/zero fallback; sole owner20994
accepted code entry57300990 once and observes its still-unreported hidden result. No project S6 score
exists yet. A distinct eight-fitting-study CPU input bridge139165316v1 completed; CPU942782
collected once and verified5372windows/1,154,209,297bytes. Its new CPU classifier
dispatch actually completed942806/1197.87seconds/5372windows/finite/zero fallback;
CPU942957 then completed the label-free teacher comparison below.
[Actual T4 recovery](../initiative/audits/s6-original-t4-recovery-20261017-v1/README.md),
[fitting bridge](../initiative/audits/s6-fitting8-uid-view-20261017-v1/README.md).
[Actual acceptance and next gate](../initiative/audits/s6-source-tree-admission-20261017-v1/README.md). Prioritize unchanged reproduction and source-exposure
accounting before teacher combination. Head-only folds on an already exposed
encoder are not independent OOF; probability agreement is not clinical truth.
The assessment retains source-DICOM physical crops and geometry/query mechanisms
as separate, falsifiable proposals, without changing current full fits.


## Boundary-crossing evidence lowers its research priority

The existing within-series policy was screened on **all3887 fitting feature bags**,
with no clinical labels, reports, pixels, new CNN encoding or fitting. CPU942278
found mean8.0003 crossing windows/94, carrying **0.35–1.54% mean attention** across
12targets. Removal changed strict study-pair order **0.074–0.494%**, versus
**0.321–1.511%** for a count-matched interior removal. The parent already suppresses
these windows strongly. Deprioritize a boundary-only training arm; this screen
cannot establish disease-AUC gain and rare large changes remain. Its full target
aggregates, exact source/input bindings and numerical precision limits are in
[the original audit](../initiative/audits/frozen-series-boundary-screen-20261017-v1/README.md).
This builds on an existing policy and earlier anatomy audit, rather than claiming
a newly found source bug. Within-series and edge-repeat inference notebooks
already have public MRI acceptance but no competition scores.

The next evidence-dependent research question remains **second-lineage
complementarity**: reproduce S6 unchanged, then characterize where its ordering
and localization differ from CoAtNet before freezing a dual-teacher mechanism.
Teacher agreement, cached-feature folds on exposed encoders and this label-free
sensitivity screen are not independent clinical correctness. Preserve ongoing
matched fits and actual scored outcomes before declaring improvement.

The fixed50/50 CoAtNet/S6 source passed its original arithmetic/fallback and
AST checks. It is now **actually staged and owned**, with no derivative GPU
execution/entry yet, as described in the new deployment section. No weights
were selected from leaderboard feedback or exposed-encoder head folds.
[Actual source and finite owner](../initiative/audits/s6-equalrank-deployment-20261017-v1/README.md).
[Exact preparation](../initiative/audits/s6-equalrank-preparation-20261017-v1/README.md).

Earlier verified private milestone: source**c1354577** at
HF**b67956eda6cda255da7c190c3eda5806d3238d63**, all58objects/238462bytes,
CPU942819/canonical shared lock/full re-download-size-SHA. This includes actual
S6 T4 acceptance, corrected grant accounting, fitting8 input acceptance and CPU
classifier dispatch, fixed blend preparation and new original epoch proofs.
No private pixels/case IDs/clinical arrays/teacher predictions/weights.
[One-time receipt](../initiative/audits/hf-s6-actual-t4-fitting8-20261017-v1/README.md).
Never repeat this or earlier verified prefixes; receipt bookkeeping follows the
mirrored source. Numerical fitting/inference code in rsna/scripts/tests is unchanged.

## Current0.97-target mechanism: transfer ordering, not confidence gaps

[Actual eight-study evidence](../initiative/audits/s6-fitting8-cross-lineage-20261017-v1/README.md):
S6/CoAtNet mean raw gaps8–39probability points coexist with rank correlations
0.8095–0.9762,1–5reversed pairs out of28, and no strong0.8/0.2opposites.
No labels/Gold/fitting/newCNN were used in that comparison. Both teachers exposed
the cohort; eight deterministic cost quantiles cannot prove accuracy or prevalence.
We therefore reject a raw-gap confidence gate for now.

The new [ordinal-transport contrast](../initiative/experiments/competitive-v1/s6-rank-transport-v1/README.md)
uses fixed equal teacher ranks, then maps their consensus order onto CoAtNet's
probability scale while preserving anatomical-view relative log-odds. It feeds
the existing25%source/75%teacher objective in a fully trainable single CNN.
Parent-only control preserves original target bytes; same input/order/compute,
fixed3epochs, no Gold/per-target tuning. Nineteen target-builder checks pass. A separate target overlay now passes17actual
synthetic CPU checks in the unchanged real full-encoder engine: parent-only
numerical identity and both-role full/stop/resume complete checkpoint/CSV equality.
No full MRI/GPU fitting or gain is claimed.
[Explicit data-path evidence](../initiative/experiments/competitive-v1/s6-rank-transport-v1/data-path/README.md).

Sole scout26454/exec91663 owns a staged32study B-only throughput/CPU-parity check.
It waits original S6 offline acceptance and fresh actual two-session/VM admission,
maxone1800second session/zeroentries. Complete3887 teacher acquisition and a new
trainer require actual full input/state/funding acceptance. Fixed equal-rank
inference is already prepared separately. Continue healthy training and the
original S6 entry independently. **0.97 is the goal;0.949 remains the result.**

## Untuned deployment and remaining evidence question

The fixed CoAtNet/S6 equal-rank derivative is now actually staged by CPU943330
and owned by27596/exec58044, not GPU-executed. It waits the original S6 scored
hidden result and fresh full resource/daily admission, then makes at most one
exact comparison entry. No fitted per-target weights or independent OOF claim.
[Actual preparation and ownership](../initiative/audits/s6-equalrank-deployment-20261017-v1/README.md).

The complete released S6v3 file list contains no fitting predictions, so those
outputs still need bounded original inference. CPU fixture preparation includes
compression/full-byte verification; CPU inference time and three-study startup
time must not be blindly multiplied into a full T4/hidden runtime estimate.
The32-study scout remains the actual throughput/parity observation.

The student and S6 see different physical representations. Larger B window
counts do not prove A misses clinical evidence, but rank distillation cannot
create information absent from student inputs. If the matched transport does
not help, examine source-position/evidence coverage and the already-running
observation-sampling contrast before attributing failure to insufficient epochs
or adding another generic backbone. This is an open mechanism, not a new gate
that blocks the currently ready comparisons or healthy training.

Latest privately preserved source is **9ef73d69**, HF**956d308744db79371ae02dc03a9cf43971c36d21**,
all60small objects/312744bytes fully re-downloaded/size/SHA verified by CPU943866
under the canonical shared lock. Never repeat. This guide/receipt bookkeeping is
later than that mirrored source. [Exact receipt](../initiative/audits/hf-r7-s6-progress-20261017-v1/README.md).
