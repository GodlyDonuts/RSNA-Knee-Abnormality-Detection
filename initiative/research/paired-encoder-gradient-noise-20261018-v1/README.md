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
