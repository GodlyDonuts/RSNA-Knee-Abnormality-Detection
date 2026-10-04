# Neighboring anatomy in one encoder view

A standalone CPU prototype encodes through-slice changes in DINO's three input
channels while retaining the center slice's grayscale mean. It changes neither
resolution nor encoder token/view count. It is **not integrated into training or
inference, MRI/GPU accepted, dispatched, or demonstrated to improve accuracy**.
The live supervision experiment and its frozen numerical source are unchanged.

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

Actual original-MRI input admission, canonical loader/config/checkpoint/inference
integration, zero-arm identity, real official-weight GPU recovery and separately
funded fixed-budget fitting remain absent. Do not reserve a GPU or reuse an old
pilot to accept these changes. Existing engineering tests establish arithmetic
invariants only. Source: rsna/competitive/neighbor_channels.py; tests:
tests/test_neighbor_channels.py. No generative upscaling, external data or new
clinical report labels are used.
