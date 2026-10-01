# Full-input geometry/classification gradient diagnostic prepared

`scripts/newton/measure_relation_gradients.py` is prepared and CPU-tested. No
H100 diagnostic or relation production run has been staged or submitted by this
milestone. Both original sampling GPU roles remain unchanged. Their actual
progress and private checkpoints do not establish this objective's accuracy.

The earlier official-weight real-MRI CPU probe passed geometry-only optimization
and exact recovery at32-pixel FP32 input with every disease cell masked. It could
not answer whether the geometry objective complements disease learning. This
new probe addresses that specific gap at the actual224-pixel BF16 observations,
six-series/sixteen-slice policy and eight-study effective batch.

## Actual diagnostic contract

Use the prepared zero/physical/shuffled pilot config triplet. All sixteen probe
studies are the first sorted members of the accepted2247-study fold0 optimization
pool; the original training UID hash, accepted release/series/fold overlay and
whole-byte source-cache proof must match the existing run. No outer-fold image
or disease target enters this diagnostic. It validates local metadata byte for
byte against the original verified TAR, and each local volume against the
original metadata's size/SHA. A replacement cache with a self-consistent new
envelope is refused. All alternate-series metadata needed by the dataset must
also be materialized from the original source.

Instantiate the pinned official DINOv3-L and the common rank32 adapter/head
once. Without taking an optimizer step, replay the same initial parameter and
dropout streams for each control. Each of the two eight-study effective batches
uses the actual trainer's per-target classification normalization and separate
global represented-study relation normalization. Compute their encoder
gradients independently, accumulate before measuring, and report both norms,
weighted ratio, cosine and combined norm. Exact classification-gradient/logit
and pair-roster digests must agree across controls. Shuffled targets retain the
same pairs and original per-series distance distributions.

The unweighted cosine is undefined when either norm is zero; report JSON null,
not NaN. Opposing gradients can cancel, so combined norm must use their dot
product rather than adding magnitudes. The reported clip multiplier concerns
the encoder alone; actual trainer clipping includes both heads. Diagnostic
timing includes two separate gradient passes and CPU hashing; it is not normal
training throughput. Measurements concern initial parameters and these two
training batches, not a trained checkpoint or the population distribution.

All frozen tensors and learned parameters must remain unchanged. No `.grad`
buffers may be contaminated. The normal MRI-only twelve-output classification
path must work with auxiliary forward explicitly forbidden. The report binds
source, weights, exact configs, input bytes and both full/probe training UID
hashes. It contains no competitive metric, model checkpoint or submission CSV.

## Local evidence and next acceptance

Twenty tests compare the accumulated classification norm with a direct full
effective-batch derivative, including masked tail/unknown geometry; replay real
dropout across all three policies; detect cancellation and invalid gradients;
refuse config drift and reject substituted source/cache envelopes. These use a
small synthetic encoder and prove software contracts only. They do not prove
official-weight BF16/CUDA behavior, throughput, recovery or clinical benefit.

Before actual execution: freeze an exact new source pack including this script,
bind the original CPU proofs, materialize all needed original cache members and
accepted node-local runtime, review fresh quota including future native/sampling
debts, and use one H100/two CPUs only when the two-node limit permits. Never
change live7ac7a97 source or reserve a third GPU. This probe writes one small
exclusive JSON file; an existing intent/output is reconciled read-only.

After gradient measurement, run the actual matched sixteen-study/two-epoch
trainer pilots with interrupted/reconstructed optimizer/RNG recovery and finite
updates. This probe's zero optimizer updates cannot replace those gates. If
initial geometry gradients overwhelm disease gradients or strongly oppose them,
use a separately recorded training-only calibration decision and apply the same
weight to physical/shuffled arms; do not quietly tune on outer scores. Neither
a favorable cosine nor a decreasing auxiliary loss proves usefulness. Full
matched fixed-epoch held-out experiments and independent deployment acceptance
remain required before promoting a relation model or spending a Kaggle slot.
