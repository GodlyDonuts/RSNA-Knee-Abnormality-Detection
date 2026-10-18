# Full encoder gradient noise measurement core

The positive head-only masked-loss result does not establish encoder gradient
quality. The actual new two-T4 recovery tests optimizer continuity and stochastic
execution, not the proposed noise-reduction hypothesis. No full fit is admitted
by recovery alone.

New `paired_encoder_noise.py` streams every parameter's dense-reference mean
and trace variance, then compares identical sampled draws in ordinary and paired
arms against that fixed reference. Encoder and head statistics stay separate;
all-parameter results cannot be replaced by a convenient head-only projection.
A stochastic full-bag reference has Monte Carlo uncertainty: report its mean's
squared standard error, raw MSE, and noise-subtracted MSE without clipping
negative values. Empirical mean-gradient error is not proof of gradient bias.
No draw vectors or MRI identifiers are exported by this numerical core.

Nine new synthetic cases pass, including analytic Welford uncertainty, exact
paired MSE ratios, zero-denominator/missing-precision outcomes, complete-group
validation before statistics change, reference freezing and RNG preservation.
Scoped Ruff and mypy pass. This is not a real encoder/GPU execution or gain.

The actual probe still needs a source-bound wrapper using the trained original
influence-E3 encoder, fitting-only original pixels/targets and frozen parent
features. Dense full94 references must use independent stochastic draws;
ordinary/controlled sampled forwards must share actual FP16/dropout RNG and
identical unscaled gradients. Record complete parameter counts, reference
uncertainty, both-view/target bindings, actual forward-backward timing separate
from statistic collection and memory. Use fixed fitting engineering studies;
no held-out/gold/confirmation tuning or best-epoch selection. Only a sufficiently
precise matched result can support a later full corrected fit, subject to fresh
compute/checkpoint/publication admission. No such probe was dispatched here.

## Trained-encoder probe implementation

The separate `paired_encoder_probe.py` and Kaggle compiler
`scripts/kaggle/paired_encoder_noise_probe.py` now implement that experiment.
They reuse the preserved influence epoch-3 model and existing fitting-only MRI
inputs; they do not fit another model or create a weight/pixel Dataset. Four
fixed fitting cases are split across two T4s. Each view has 16 independent
full-94 reference draws and 32 paired sampled draws. Each loss uses one view,
matching training; the sampled arms share windows, proposal and dropout seed.
Every learned encoder/head gradient is measured before clipping. Reference
uncertainty, forward/backward timing and peak memory remain explicit; neither
optimizer-step timing nor clinical accuracy is inferred.

Synthetic autograd checks establish exact zero-correction gradient agreement,
including the encoder, and preservation of RNG, model modes and BatchNorm
buffers. Compiler checks reject mismatched scientific inputs and ensure the
new entry point measures gradients rather than replaying a recovery or fit.
These checks are software evidence only. The actual original recovery notebook
139417155 has no observed terminal acceptance yet. No new noise GPU probe has
been dispatched; its later admission needs that outcome and fresh capacity and
runtime funding. No production or accuracy claim is made.

Actual CPU staging **958257** completed successfully from source `e442961f`.
Exact new notebook source is `78c4b58bad6bb2ba58bebedc259b19b85b006c32813d69a63e1b1c5212ea6dfb`;
worker `5bd7a81a253ebc771f3e5710c3cbb5f6907d0f6a35a04c57130499c520240700`.
The private child consumes 522,417 bytes within the existing 32 MiB controls
cap, with 4,401,102 parent-cap bytes still free at admission. Fresh owning-node
and shared reserves passed. No model/pixel transfer, GPU push or entry occurred.
All 17 focused numerical/compiler tests, scoped Ruff and two-module mypy passed.
See `staging/actual-staged.json`. This is prepared execution, not GPU acceptance.

## Finite execution owner

Original process **68344** now owns at most one new two-T4 noise measurement,
with a 12-hour bound and no competition-entry authority. Actual CPU958304
confirmed it is waiting for original recovery139417155; zero new GPUs were
pushed. It stops on a failed original recovery, binds the exact staged source,
checks fresh full future VM budgets/capacity and registers an explicit
capacity-only alias atomically so legacy scanners count its diagnostic.
Three new execution simulations confirm that missing prerequisites wait without
credentials/API writes, failed recovery stops and changed source stops, all
without changing source files or creating an intent. Never launch another owner
or restart this one after uncertainty. Actual numerical results remain absent.
