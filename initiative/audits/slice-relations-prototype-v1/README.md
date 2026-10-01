# Unsigned within-series relation supervision: isolated prototype

The owner asks for creative independent research. This experiment borrows from
multi-sensor estimation: examination geometry can supply extra supervision
without assigning a disease label to every slice. It is a hypothesis, not an
accuracy improvement. Spatial self-supervision has prior work:
[Doersch et al., context prediction](https://arxiv.org/abs/1505.05192) and
[Zeng et al., positional contrastive learning](https://arxiv.org/abs/2106.09157).
Their abstracts were checked; neither establishes RSNA DINOv3 performance or
the novelty of this proposed adaptation.

## What exists

`rsna/competitive/relations.py` is isolated from the live trainer and model.
Its first version takes raw **global slice encoder features**, before physical
position or acquisition embeddings. It predicts unsigned physical separation
within the same study and series. Symmetric feature differences and pair means
feed a two-layer head; geometry supplies targets only. For DINOv3-L width1024
and hidden128, the head has262,401 parameters and requires no additional image
encoding. This does not implement patch correspondence or cross-plane fusion.

Targets are log1p(distance in millimeters)/log1p(100), without clipping to a
fixed range. Their sign is discarded because orientation-normal direction is a
coordinate convention. Translation and normal reflection preserve the targets.
The objective is smooth L1 prediction loss, rather than forced neighboring
feature equality that might erase focal abnormalities. Zero-pair batches have
differentiable zero loss even if their padding contains NaN.

Geometry must come from the verified loader's physical-coordinate representation.
Unknown positions, padding, excluded series and distributed tail duplicates
create no pairs. Known positions must be finite and strictly monotone. There
are at most32 uniformly sampled unordered pairs per series. Every represented
study gets equal loss weight regardless of its number of eligible pairs/series.
The caller still must enforce accepted training membership; a boolean sample
mask does not attest provenance or distinguish outer patients by itself.

Initialization and pair sampling leave the global CPU random stream unchanged.
Seeded target permutation within each series preserves its sampled target
distribution and uses a separate generator, so the negative control receives
identical pair indices even when the pair budget is capped. A permutation can
coincidentally leave individual labels unchanged, and equal-distance pairs have
no shuffle information. Quantify that before treating it as an informative
negative control. Caller seeds must be persisted/reconstructed on resume.

Ten CPU engineering tests passed: masking and study weighting, monotonic-geometry
refusal, translation/reflection invariance, pair swap symmetry, strict head reload,
unchanged global RNG and matched shuffled pair roster, finite empty loss, an
actual adapter optimizer update with frozen base and no excluded-input gradient,
and generalization to disjoint synthetic feature/coordinate pairs. The synthetic
features already encode their coordinates; this is a learnability check, not
evidence that MRI features encode anatomy or that classification improves.

## Proposed matched experiment, not dispatched

After training-only geometry coverage is measured and exact integration passes
pilot/recovery acceptance, use one fixed accepted fold and fixed final epoch:

1. Same solo encoder, adapters, study head, images and classification updates,
   with the relation head present but its auxiliary weight zero.
2. Identical initialization and presentation/update budgets, with physical
   relation loss. Pick a fixed weight using training-only loss/gradient scale;
   do not tune it against the outer holdout.
3. Same additional loss and pair roster with within-series shuffled targets.

All arms retain examination-level masked BCE; there is no per-slice disease
supervision. The relation head is discarded for MRI-only classification inference.
No external cohort or teacher is needed. Extra cost is at most192 small pair-head
evaluations per six-series study plus target construction. Actual throughput,
peak memory and encoder gradient costs remain unmeasured; budget them in a pilot.
Keep at most two GPU nodes and fund persistent snapshots before dispatch.

Primary evidence is fixed-final all12-target outer AUC with leakage-component
uncertainty, comparing the same accepted patients. Repeat any promising result
with a seed/fold before promotion. Also monitor feature variance, rare-positive
ranking, missing-series stability and acquisition sensitivity. Reject this
mechanism if physical targets do not beat both controls, or if localization-
sensitive performance declines. A lower geometry loss could mean learning
scanner/protocol or intensity shortcuts rather than useful anatomy.

Before real-MRI use: verify exact2247 training membership against the actual run
binding; quantify eligible physical series and pair-gap distribution under the
actual six-series/sixteen-slice selection; admit no outer/gold/confirmation data;
wire raw features explicitly without passing positions into the predictor;
preserve classification initialization and strict inference/delta interfaces;
test accumulation, optimizer groups, checkpoint/RNG restoration and target
head removal; require bounded H100 acceptance. None of these remaining full
integration gates is claimed passed by the isolated prototype.

## Current training, separate from this prototype

`sampling-live-read-only.json` records actual job/run/input bindings and recent
finite updates: stratified866716 at170, recovered uniform866719 at17. Both were
RUNNING, with no completed epoch sidecar at this check. These full ten-epoch
runs use unchanged scientific source7ac7a97, accepted data and pinned official
weights. No live job was altered, no third GPU reserved and no Kaggle submission
made. Preserve each completed epoch once using the existing pinned coordinator.
