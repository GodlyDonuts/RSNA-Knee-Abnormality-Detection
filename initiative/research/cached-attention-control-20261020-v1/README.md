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


Actual original CPU976785 refused during its three-second preflight. Read-only
976787 proved all original code hashes equal and no screen directory/intent/
progress/result exists; no numerical readout experiment ran. The historical
607,456,016,323-byte free threshold failed against548,980,051,968 actual free;
the old32MiB controls also lacked the proposed2MiB headroom. Originals remain
unchanged. A distinct v2 screen uses256KiB within the currently admitted8MiB
node33 production-control scope, retaining another2MiB for original future fit/
serving controls and the full20GiB node/shared plus5000file reserves. It reads
old features/head only; all numerical loops/stress levels/proposals are unchanged.
This is explicit control relocation and fresh small-budget admission, not a
lowered fidelity threshold, new model funding or retry of the original job.


## Actual ratio result and distinct linearized hypothesis

CPU976806 completed the original numerical screen once in1.153seconds. At eps0,
reference error was3.9e-16; eps.05/.2 MSE ratios were.001789/.028719 with no
invalid denominators. At eps.5 the conditional ratio was.18950, but TWO of49,152
target/draw combinations had nonpositive denominators (min-.1669). The original
predeclared gate FAILED. Do not promote or GPU-train that unmodified estimator.

A distinct `linear_readout` removes the random denominator by explicitly using
only the full-attention first-order change:

```
mu0 = sum(w0*v0)
z_linear = mu0 + mean(w0_i/q_i * (dv_i + (v0_i-mu0)*ds_i)) + current_bias
```

It is unbiased for this linearized logit, not for the nonlinear current model;
finite drift adds Taylor error. Exact-reference forward/expected gradients still
match the full bag; complete enumeration after finite drift is deliberately NOT
claimed to equal the full current attention. This is a new hypothesis, not a
post-hoc relaxation of the failed ratio gate. A separate CPU v3 screen retains
all32fitting indices/levels/draws/proposals/functions/seeds/thresholds, but measures
this linear estimator on ALL draws, with no conditional omissions or denominator
clipping. It reads only the same existing fitting features/head. Another256KiB
inside the same8MiB controls is admitted only after current paid controls plus
2MiB original future headroom and all current20GiB/5000file reserves pass again.
No GPU, optimizer, new MRI/feature transfer, source-label change or entry.


Actual distinct CPU976828 completed the linear screen in1.042seconds; all
predeclared levels and all12 targets passed. All-draw MSE ratios to original SNI
were.001779 at.05,.027942 at.2 and.162722 at.5; worst target ratio at.5 was.260906.
There were zero invalid target/draw outputs, and reference error stayed near
floating-point rounding. These are synthetic-drift approximation metrics using
real cached MRI feature inputs, NOT measured student drift, patient truth or
accuracy. Source6dec668b; actual-CPU-linear-screen-v3.json preserves all levels/
targets. The original ratio failure remains rejected, without changing its gate.
Agent engineering decision accepts only a separately funded actual encoder
protocol with matched mode, source/moments alignment, actual recovery and measured
staleness. No new GPU training/production or entry has been dispatched for it.
