# Cached full-context attention correction: falsifiable next mechanism

The current IID16 importance readout normalizes over sampled windows. Original
CPU936087 reduced full-parent probability approximation MSE from0.00653638 to
0.00355590 with influence proposals, but did not remove it. Those are teacher
approximation metrics, not disease accuracy. Current whole94 and sampled fits
both scored0.949, so sampling bias is not established as the sole bottleneck.
The healthy current pointwise/ranking/source-zero fits are unchanged.

## Mechanism and exact limits

For each target/view let cached normalized attention be w0, per-window vote v0,
current score change ds and vote change dv. The complete reference numerator is
N0=sum(w0*v0) and denominator1. With K IID draws from fixed positive q:

```
a_i = w0_i * exp(ds_i)
Nhat = N0 + mean((a_i*(v0_i+dv_i) - w0_i*v0_i) / q_i)
Dhat = 1  + mean((a_i-w0_i) / q_i)
z = Nhat/Dhat + current_bias
```

Thus all cached original windows participate as a reference, while the CNN
computes changes only on selected windows. When the current deterministic model
exactly equals the reference, every sampled forward returns the full logit;
expected first derivatives match the full-bag loss gradient. Complete population
enumeration gives exact current attention after arbitrary finite change. The
moments are unbiased; their ratio/nonlinear-loss gradient generally are NOT
unbiased away from the reference. Negative denominator estimates are possible.
The research module exposes/refuses these rather than clipping or resampling.

This adapts established ideas, not a claim of a newly invented control variate.
[Chen et al., ICML2018](https://proceedings.mlr.press/v80/chen18p.html) use
historical activations to reduce graph-neighbor sampling variance.
[Zheng et al.,2023](https://arxiv.org/abs/2302.04542) analyze/control attention
approximation through control variates; their random-feature construction is
not this cached MRI ratio estimator. [End-to-end MIL gradient accumulation](https://arxiv.org/abs/2203.03981)
provides a relevant full-bag memory baseline. None establishes a benefit for
this competition. The proposed adaptation uses original full-context moments
and estimates only encoder/readout drift.

## Predeclared cheap rejection screen

One CPU/zeroGPU, original cached fitting-only feature population/head/proposals;
32 evenly spaced fitting indices, both original94-window views,128 common IID
16-window draws, seed20261020 plus original index. No clinical labels/reports,
protected cohorts, new CNN inference, training, transfer or competition entry.
Current moments are synthetically perturbed at eps0,.05,.2,.5 using fixed
window/class/view sinusoidal functions and within-target vote standard deviation.
These are stress inputs, NOT observed student/encoder updates or plausible
clinical changes. Compare original SNI and cached correction against the exact
full current moments on identical draws. Report all12 targets/eps, full MSE,
conditional common-valid-draw MSE and the invalid-denominator count; never conceal
failures or call oracle fidelity accuracy. All invalid denominators are failures
for promotion, even if conditional MSE improves.

Gate for considering a separately funded real GPU diagnostic: zero invalid
ratios at all predeclared levels, global MSE ratio<=.8 and each target<=1.2 at
all nonzero levels, exact reference/dense identities. A failed gate rejects the
unmodified estimator. No actual GPU/optimizer/production is authorized by this
CPU screen. Dropout, mixed precision, stale-reference drift, exact teacher feature
alignment and variance after actual learning remain unaccepted. Runtime serving
must ultimately retain a single complete CNN/full94 path, with no reference cache
required at test time; changing training to use cached teacher statistics does
not itself prove that train/serve path or an accuracy benefit.
