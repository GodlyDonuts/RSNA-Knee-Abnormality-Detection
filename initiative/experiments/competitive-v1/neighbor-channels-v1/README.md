# Neighboring anatomy in one encoder view

The canonical opt-in input policy encodes through-slice changes in DINO's three input
channels while retaining the center slice's grayscale mean. It changes neither
resolution nor encoder token/view count. It is integrated into training/checkpoint/inference and CPU-tested on original
MRI. Actual original-weight H100 recovery872200 passed all three arms. It is **not
fully fitted or demonstrated to improve accuracy**.
The completed supervision experiments and their frozen sources are unchanged.

The measured starting point is that adapted global slice features and gated
study attention do not explicitly encode immediately neighboring-slice anatomy.
The uniform/stratified coverage and224/384 comparisons did not establish gains;
this proposal tests a different representation, not another coverage claim.
The proposed primary is Medial Meniscus, whose original outer observations are
far less sparse than Effusion. Original accepted masks/folds remain fixed.

## Mechanism and competing explanations

For aligned, equally normalized center C and neighbors P,N in[0,1], define
m=min(C,1-C), dP=P-C, dN=N-C and amplitude a=λm. Return three channels:

```
R = C + a*dP
G = C - a*(dP+dN)
B = C + a*dN
```

In real arithmetic their mean is exactly C. For0≤λ≤1/2 every channel stays in
[0,1] without clipping, because|dP|,|dN|≤1 and the center-channel perturbation
is at most2λm≤m. The FP32 prototype uses FP64 intermediates, including for
subnormal inputs; output mean preservation has FP32 rounding tolerance.
λ=0 returns the exact repeated center bytes, including signed zeros. Existing
DINO normalization must subsequently be applied unchanged; the raw-channel mean
property does not imply equal pretrained convolution responses.

The hypothesis is that pretrained color-sensitive filters and adapted features
can learn local anatomical change without more encoder passes or a new 3D model.
That is an inference to test. Unnatural channel statistics could instead harm
transfer, suppress useful context at intensity extremes, or help only through
regularization. The center is recoverable by the channel mean, but this does not
guarantee the classifier preserves every lesion or uses neighboring information.
This is local context, not the entire volume inside one image.

## Related research and the limits of the analogy

Adjacent-slice context is established2.5D research, not a novelty claim. An
[empirical comparison of2.5D segmentation approaches](https://arxiv.org/abs/2010.06163)
found task-dependent benefits. [CSA-Net](https://arxiv.org/abs/2405.00130) uses
cross-slice attention for MRI segmentation; [LambdaUNet](https://arxiv.org/abs/2104.13917)
addresses discontinuous anisotropic MRI observations. These motivate testing
local relationships, but do not validate this channel formula, DINOv3, knee
classification, or any claimed score. Our mean-preserving bounded construction
is a proposed experiment; its novelty has not been established.

## Fixed comparison and admission

See protocol.json. Proposed arms are grayscaleλ0, physical-neighborλ0.25, and
same-series distant-contextλ0.25. The last arm distinguishes ordered local
anatomy from additional context/chromatic regularization. Keep the center views,
model initialization within a seed, encoder budget, labels, optimizer and final
endpoint identical. Raw neighbor decoding/CPU cost and unique observations must
be measured separately: equal encoder passes are not equal information budgets.
No hyperparameter/epoch selection from the outer pool; no protected data.

The prototype only accepts arrays; it cannot admit adjacency. Before actual use,
require exact original cache/source bindings, same series and orientation/grid,
monotone physical slice positions and bounded spacing, plus consistent intensity
normalization. Missing/unverified neighbors replicate the center. Never take
neighbors from consecutive *sampled slots*, cross series, or infer physical
continuity from array shape. A distant-context control must be deterministic,
within the same series, and retain its actual donor indices privately.

Canonical original-MRI input admission, loader/config/checkpoint/inference
integration and zero-arm identity now pass the CPU checks described below.
Real official-weight GPU recovery and separately funded fitting remain outstanding.
Do not reuse an old pilot to accept these changes. Source arithmetic:
rsna/competitive/neighbor_channels.py; canonical input: neighbor_input.py. No generative upscaling, external data or new
clinical report labels are used.

## Standalone geometry indices

`rsna/competitive/neighbor_geometry.py` now supplies independently tested donor
indices. Physical edges retain original order, require positive spacing at most
1.5 times the series median and in-plane origin displacement at most0.001mm.
Missing geometry, boundaries, non-monotone positions or invalid edges replicate
the center; malformed claimed geometry raises. Original decoder/cache proofs
must still establish consistent per-slice orientation and intensity normalization.
The distant control selects two farthest distinct same-volume indices, each at
least max(2,ceil(depth/4)) from the center, with deterministic ties and center
fallback for insufficient depth. These fixed engineering choices use no outer
labels. The original endpoint-heavy control is preserved as historical evidence; the
canonical path uses the matched rotation described below.

A separate one-CPU original-MRI probe is prepared in
`initiative/audits/neighbor-channels-real-input-v1`: all fitting-only metadata
and sixteen predeclared original fitting volumes, zero shared file writes,
no labels, outer pixels, model or GPU. CPU871757 completed this probe:199,274/199,306 uniform fitting centers have
physical context and zero strength is byte-identical on255 actual views. The
distant control exposes fewer unique donors (275 vs419), so revise its coverage
before fitting. This is input feasibility, not model acceptance or accuracy.

## Canonical integration and information-matched control

CPU872158 admitted the canonical loader on1,343original selected views, including
exact zero bytes against frozen C9. New distant donor-pair rotations preserve
the physical donor multisets/unique information budget (419vs419on the bounded
pixel probe); retain the older275vs419confound in the original audit. The
canonical policy isolates uniform/no-augmentation/global-attention arms and
binds one/three channels in encoder/checkpoint provenance. Synthetic actual
trainer recovery and checkpoint restoration pass; actual official-weight GPU
recovery/funding admission remains next. See neighbor-channels-integration-v1.

Donor-population matching does not match perturbation norms: distant anatomy
can produce larger channel residuals than physical neighbors. Interpret any
physical-versus-distant difference alongside gray and measured residual
statistics; it alone cannot prove an adjacency-specific mechanism. Direct actual
initial learned-state equality is also not yet collected, despite identical
official learned topology and matched seeded factory paths.
