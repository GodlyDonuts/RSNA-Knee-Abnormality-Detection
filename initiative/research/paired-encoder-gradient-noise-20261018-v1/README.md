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
