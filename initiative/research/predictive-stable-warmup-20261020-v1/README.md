# Correct the initial predictive-density mismatch

Original cached diagnostic 978554 completed exact B/C/C0 recovery. It also showed
an important negative result: C and C0 both produced exam loss near 10 and head
gradient norms near 4,400 after only a few updates, while B remained stable.
The density coefficient cannot explain the immediate failure because C0 has none.

The complete-support derivative accumulates regional contributions through the
same decoder. Random negative means initially create a coherent residual against
small fixed descriptors, amplified by full-acquisition supports. Global gradient
clipping alone does not bound Adam's first parameter displacement.

This distinct engineering run initializes C/C0 negative means and B generic
reconstruction means to zero. It retains fixed unit variance, all original support
priors, the exact likelihood, both masks, full trainable encoder, and accepted
labels. The small diagnostic uses the actual proposed production schedule horizon
of 2,916 updates, instead of compressing warmup into fewer than one update:
first LR 2.057613e-7, update-24 LR 4.938272e-6. This is not a new loss temperature,
support truncation, or a clinical finding.

Twelve fitting studies, 24 updates per B/C/C0 arm, effective batch eight including
the four-study tail, first three updates head-only, and exact full/stop1/resume
state/RNG/CSV equality. The four disjoint fitting engineering prediction rows
test recovery only; their labels do not select this recipe. Training-only means,
shifts, evidence, losses and gradients are recorded. This partial production
warmup does not establish stability throughout a full run or clinical accuracy.

The original 978554 inputs and all old checkpoints are reused in place and
preserved. A distinct 12-GiB engineering checkpoint and 4-MiB control reservation
is required in addition to all previous liabilities and the 20-GiB reserve. One
new H100 job may start only after actual CPU input/source admission and fresh
shared quota. No full-cohort fit or competition entry is authorized by a passing
engineering receipt.
