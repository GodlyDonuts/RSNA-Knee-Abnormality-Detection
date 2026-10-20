# Actual encoder path for cached full-context correction

The label-free linearized cached-attention screen passed, but it did not train an
encoder or establish accuracy. This scope implements the next actual model path,
not another unchanged synthetic-drift screen. Original healthy975297/source-zero
training and all old numerical sources stay unchanged.

`cached_attention_training.py` adds a parameter-free readout mixin with explicit
`reference` and `linear` branches. Sampled training uses full parent score/vote
moments, exact duplicate-preserving indices/proposals and both views. Its missing-
proposal path delegates to the original full94 inference, with NO reference cache
needed at test time. All encoder/head parameters remain trainable. Both new arms
use deterministic eval mode WITH gradients and an FP32 MIL head; this is a matched
new comparison, not a causal comparison with older stochastic/precision protocols.

The new engine is derived from the exact originale024a8ee optimizer/recovery source.
It retains actual AdamW/counters/RNG/scaler/LR/history/snapshots. Only the explicit
readout/reference bindings, paired per-view gradients and common deterministic
mode change. One rank only until an actual separate multi-rank protocol passes.
No live trainer is patched. Engineering17-study/two-epoch/six-update recovery
exercises the final-one-study batch. Tiny CPU encoder tests pass complete
full/stop1/resume decoded state, every epoch snapshot, updated encoder/head and
unchanged original inference. These are NOT real73M/CUDA acceptance or accuracy.

## Missing original reference moments

Original fitting feature/head records already exist and are byte-admitted on33.
One CPU materializes only the missing unnormalized parent attention scores and
per-target votes for all3887/two-view/94windows/12targets. It retains original
UID ordering and parent/source/rights proofs, compares the moment reduction with
the original FP32 head and original cached FP16 probabilities, and opens no
clinical label/report/protected cohort. No CNN inference or weights/pixels are
transferred. Arrays/IDs go only through a private cluster pipe from33 to48, never
through this Mac, Git or source-milestone HF. Receiver checks exact byte/SHA/EOF,
array type/shape/finiteness and complete roster. Resulting cache remains private.
Existing feature extraction is not repeated; original source/model files stay.

Two source/funding phases precede exactly one transfer. Sender controls use256KiB
inside the current8MiB on33 while preserving another2MiB future fit/serving margin.
Destination has a new128MiB reference cap,8MiB controls and12GiB engineering-model/
recovery cap, after all prior future upper bounds and20GiB/5000file reserves.
This does NOT fund a new full production fit or reserve a GPU before prerequisites
exist. Actual source/cache/device/recovery receipts must admit any GPU dispatch.
Existing intent, partial or uncertainty means read-only reconciliation, not retry.

Before full fitting, the exact actual73M encoder must pass complete numerical
recovery, original inference and cache/input/mode/gradient/precision checks on
fitting-only engineering studies. Measure actual teacher/student drift, not the
old synthetic perturbations. Retain both arms/negative outcomes and use fixed
final evaluation; no tiny engineering score or proxy fidelity selects a winner.
