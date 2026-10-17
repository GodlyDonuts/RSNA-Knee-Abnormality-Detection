# Research brief

**October17,10:55UTC.** Target0.97; actual best0.950. See
[current state](current-state.md) for measured scores and original live owners.

## What we are testing now

First reproduce the exact stronger public reference before changing it. The
public0.958 ensemble completed our unchanged8cell interactive two-T4 run:
CoAtNet384/224, four-foldR6v1 with physical/extent/state information and fluid/
state stackers, then a three-checkpoint2.5D reader. FinalCSV checks passed;
no committed offline version or hidden score exists. These are jointly different
mechanisms, not evidence that resolution or a particular fold alone causes a gain.
[Actual qualification](../initiative/audits/public0958-reference-20261017-v1/interactive-v1/README.md).

Second isolate **source supervision**. The newly released S6v5 labels differ
from our Stevenv6 targets on all3887fitting studies; Fracture ranks correlate
only0.51446. Neither is ground truth. The fixed next contrast uses the original
uniformIID16 full-encoder engine and all original inputs/random draws/views/
parent supervision/optimizer/1458updates, replacing only source labels. It
matches the existing uniform control, avoiding a slower second whole94 arm.
Real full-array acceptance/private Dataset/full-byte proof and32synthetic CPU
engine checks passed; actual treatment GPU recovery/fitting remain absent.
[Protocol and original evidence](../initiative/experiments/competitive-v1/s6-source-label-contrast-v1/README.md).

Third finish evaluating the owned learned checkpoints. Whole94 completed all
3epochs/1458updates and privately verified all epochs. FinalE3 serving/entry
remain owned. Uniform16 final is running, influence16 final remains owned.
Their actual scores may be positive or negative; completion is not improvement.

## Why the score gap is not just compute

The independent N3 model trains all73,136,096parameters, not a small frozen-DINO
head. It scored0.894. Reproduced MRI-trained CoAtNet and S6 weights score0.949
and0.950. The gap confounds fitting scope, initialization, supervision, physical
preprocessing, windows, optimization and serving. It cannot be assigned to rank,
blur or epochs from those comparisons.

Our parent adaptation uses3887fit/all94windows/effective8/3epochs/1458updates,
encoder/headLR3e-6/3e-5,AdamW0.02,10%warmup/cosine-to1%,clip1,FP16/scaler with
FP32loss, fixedBN buffers/learned affine parameters. Its25%source/75%detached
parent objective can preserve teacher mistakes. E1 scored0.949; finalE3 is
unscored. S6's released source instead uses4263fit/16epochs/64windows/denseBCE,
noEMA/SWA/fixedE16. Its86missing studies were local deletions. S6 has noOAI;
the separate CoAtNet author's data provenance includes OAI. These are not the
same recipe.

## Findings and useful negatives

| Mechanism | Actual result | Decision |
|---|---|---|
| Neighbor channels in independent N3 | Weak delta−0.00572762, interval crosses zero | No promotion |
| Positive BCE weights | Weak delta+0.00298799, interval crosses zero | No established primary gain |
| Late parameter averaging | Raw-serving delta+0.00031564, interval crosses zero | Keep rawE16 |
| Expanded N3,2592fit | Weak0.91743158 vs0.92092535; interval crosses zero | No established gain; data and updates both changed |
| Context / regional / full-field serving | Actual0.948 each vs0.949 parent | No demonstrated public gain |
| Historical DINO stratified sampling | +0.0009107, interval crosses zero | Keep uniform; different from current influence experiment |
| Dropping Selection_Tier | No changed series choice in all1156outer studies | Not the selection explanation; rendering remains separate |
| Frozen influence proposal | Teacher-approximation MSE−45.6%, all12targets | Fidelity result, not disease accuracy or measured speedup |
| Crossing-series windows | Only0.35–1.54%mean attention; deleting them changes fewer pairs than matched interiors | Deprioritize a boundary-only training arm |
| S6/CoAtNet fitting8 comparison | Raw confidence differs greatly; ranks are closer | Do not call confidence gap clinical uncertainty |

Exact original evidence is mapped in the [research handoff](research-agent-handoff.md).
The N3 component bootstrap had2836defined/1164undefined draws and very sparse
rare-target support. Conditional intervals omit undefined draws; internal weak
agreement is not clinical accuracy. Released encoders saw internal studies, so
heads on their features cannot yield independent whole-model OOF. We have not
used protected clinical values to select these mechanisms.

The sampled experiment preserves common starting model/RNG, order/views,16IID
replacement draws, inverse-proposal attention correction and full94 inference.
Influence uses25%uniform coverage plus75%teacher influence. Finite-bag ratios
are biased, proposals become stale and teacher errors can amplify;94/16 is not
an observed speedup. The matched source-label arm keeps this uniform recipe and
changes only target values. No tiny engineering model is competition-eligible.

## Promotion and next decisions

Obtain the exact reference's committed offline acceptance and actual hidden
result. Inspect original finalE3/CPU32/scout outcomes without duplicating them.
Admit a distinct real-MRI GPU source-label recovery under fresh complete VM and
storage funding, then fit only after acceptance. Preserve fixed-final evidence,
all12targets, original checkpoint identities and negative outcomes. A useful
positive mechanism should survive a matched seed/fold or appropriate external
check; public-label agreement alone cannot justify clinical truth.

Original detailed mechanism/source/evidence discussion remains in
[the preceding Git snapshot](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/ca90fa5e347310dbbb35765e5dcc5c1b07e04910/docs/research-brief.md).
