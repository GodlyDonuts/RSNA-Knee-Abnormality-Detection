# Establish a full-length CLS plus patch-mean baseline

This conventional feature readout is motivated by the captured public DINO
implementation's concatenation of CLS and mean patch tokens before
target-specific slot pooling. It is not claimed as a novel architecture or an
established gain. Our existing baseline already has target-specific slice and
sequence pooling; its established encoder summary was CLS alone.

The new optional `patch_summary` policy uses the same DINOv3-L forward call,
original register-excluded spatial tokens and identical MRI normalization.
Before the existing projection LayerNorm/GELU it adds

`W_CLS CLS + weight * W_patch mean(spatial_patches)`.

This is the linear projection of concatenated CLS/patch-mean features,
implemented as two matrix blocks. `W_patch` is initialized to zero without
consuming RNG. Both arms have the same learned graph, values and post-factory
RNG; control uses weight0, candidate1. Initial predictions equal the legacy
CLS baseline. The extra131,072 parameters are a readout block, not an extra
encoder/model or teacher. All10,223,616 original encoder adapter parameters
remain trainable and the official pretrained base remains frozen.

## Predeclared full-training comparison

The prepared configurations differ only in readout weight. Each has10 fixed
epochs/2,810 optimizer updates, original2247 fitting and1156 disjoint
weak-label development studies, native384 inputs,16 uniform ordered slices
and at most6 selected sequences, rank32/all120 projections, effective8,
encoder/head LR1e-4/3e-4,281-update warmup/cosine and clip1. The current
FP16/scaler128/chunk8 numerical route remains explicit. No short-run result
selects an epoch; final epoch10 is the primary all12 macro endpoint, with
all12 target results and paired whole-component uncertainty/undefined draws.
Protected confirmation and gold remain unused.

No loss balancing, anatomy-sharing head, neighbor-channel, spatial witness,
sequence-memory, report overlay or new data is mixed into this baseline.
Those are separate hypotheses. The logical anatomy-support audit remains
useful evidence, not a current training change. Fully trained performance
must be measured; additional features/parameters and software tests are not
evidence of clinical accuracy.

## Actual implementation status

Canonical train/infer, frozen-base delta packaging and resident inference
reconstruct the optional readout. DefaultNone keeps the old model state/RNG
and omits the new configuration key from historical serialization. The
already-staged scientific sources and live875858/queued877134 are untouched.
Do not replace any source/control or duplicate existing jobs.

Tests validate common initialization/zero-start predictions, nonzero candidate
and zero control readout gradients, concatenated-feature algebra, NaN padding,
isolated policy and actual synthetic trainer full/stop1/resume model/base/
AdamW/LR/counters/allRNG/AMP-scaler and CSV equality in FP32/FP16 and both fitting
scopes. This is software acceptance, not real-MRI/GPU or clinical acceptance.

**Not staged remotely, GPU-admitted, funded or dispatched.** Before launch,
bind this exact source/config, reuse original cache/tensor/whole-byte proofs
without duplicate bulk transfer, check actual official initialized graphs and
real GPU recovery, and explicitly fund all20 immutable epochs plus live/
atomic-write/download obligations on private node-local storage. Fresh quota
and original job review govern dispatch; no existing reservation is repurposed
by this preparation. Preserve the20GiB shared/local and5000 shared-file reserve,
zero new shared growth where required, at most two GPU nodes/minimum CPUs.

Finish the healthy existing comparisons and fullscale runtime test. The next
allocation priority is establishing this full-length documented baseline,
not adding more speculative heads. Deployment still needs exact trained
source/base provenance, actual offline two-T4 runtime/RAM acceptance and the
single canonical Newton Kaggle coordinator with fresh account allowance.
