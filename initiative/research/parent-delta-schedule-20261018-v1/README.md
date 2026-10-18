# Compact same-bag parent correction for the actual training schedule

Materialize only the deterministic parent sampled logits for the original
influence sampler:3epochs/3887fitting studies/16windows, original seed20261016,
exact duplicates/proposals/mirror per study and epoch. No new image forward or
labels are needed: use the already admitted parent feature/readout cache.

The future training correction is original case.parent_logits minus this exact
same-bag parent prediction. It leaves all source labels, masks, parent anchor,
optimizer, input schedule and full94inference unchanged. No second encoder is
needed at deployment. The schedule can be looked up from the immutable bag
identity without mutable last-case state or consuming global RNG.

This namespace admits neither training nor GPU inference. Cached predictions are
deterministic FP32 readout values on accepted parent features; they do not cancel
all FP16 error, active dropout or stochastic-depth in the actual training graph.
A bounded real-MRI recovery and matched comparison remain necessary. Public
fitting matrices, teacher logits, UIDs and private per-case corrections stay on
the owning node; only aggregate/source/hash receipts may enter Git/source mirrors.

Materialization uses oneCPU/zeroGPU/4GiB/15minutes. New2MiB lives inside the
existing original32MiB node-control allowance, after fresh global/future-epoch/
shared byte/file reserve checks. No new checkpoint, Dataset or GPU is funded.
Save uint8positions/boolmirrors/FP32logits, re-open and independently replay every
scheduled bag, verify feature/target teacher equality and unchanged global RNG.
Original baseline targets are opened for UIDs and parent_logits only; no clinical
labels, masks, reports, Gold or confirmation are loaded.

Ten synthetic checks verify exact original sampling and FP32 teacher readout,
alignment, wrong bags/dtypes/NaNs/order refusal, immutable lookup and RNG. Actual
CPU terminal receipt, not those checks, must establish completed materialization.

Actual materialization955015 FAILED1:0 in7seconds at the global free-space
guard, before reading cached MRI features or creating the private schedule root.
Read-only955035 verified absence and a59.5MB reserve shortfall. Preserve that
original attempt; do not retry it. A distinct funded recovery is required.

The isolated training loss hook is implemented: exactly one FP32 logit addition
before the original anchored loss, with the complete original optimizer engine
source pinned.19focused CPU cases pass, including zero-correction control equality
and full/stop1/resume state/RNG/epoch equality for both modes on synthetic images.
These are not actual MRI/GPU acceptance. No live scientific source was changed.
