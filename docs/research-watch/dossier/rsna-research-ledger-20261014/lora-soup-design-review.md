# Rank32 late-checkpoint soup: what is genuinely different

Read-only review, 14 October 2026. Repository snapshot: `1f05aa1df34bd92212ce92633b99366f98804e3c`. Proposal: [issue #89](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/issues/89). No weights, case records or predictions were accessed; no model, training, submission or repository mutation was executed.

## Bottom line

The earlier negative averaging result was already gauge-correct. The proposed E5–10 rank32 soup is a different, unproven extension: different training population, one extra epoch, and lossy rank compression. Gauge theory supplies a correctness requirement, not a reason to expect a better score.

The most useful additional check is numerical: compare the rank32 result with the exact, uncompressed E5–10 effective-weight mean using the identical averaged head. This separates compression error from the averaging operation. It requires no clinical labels and is not a new accuracy control or authorization to execute checkpoint/prediction work.

## What the completed experiment actually tested

The original fold-0 native-public E6–10 average achieved **0.8990962409**, versus its own E10 **0.8997840648**: delta **−0.000687824**, conditional paired 95% interval **[−0.004788950, +0.003119967]**, with 2,836 defined and 1,164 undefined bootstrap draws. E10 predictions reproduced the original CSV exactly. No gain was established; E10 remained the default. These are weak-development AUCs, not public leaderboard scores. [Terminal receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/epoch-weight-average-fixed-final-20261014-v1/README.md)

Its implementation concatenated five rank32 factor pairs into a rank160 representation with adjusted scaling. It averaged the effective products exactly in real arithmetic, never the A/B factors independently, and explicitly excluded SVD truncation. Its actual candidate contained 51,118,080 residual parameters. [Protocol](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/experiments/competitive-v1/epoch-weight-average-v1/README.md), [implementation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/competitive/weight_average.py#L219-L237), [build receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/epoch-weight-average-CPU-built-20261014-v1/built.json)

Issue #89 instead fixes equal weights over E5–10 from the completed **all-3,403-study** run, then compresses each mean update back to rank32. That run has **no internal outer holdout**. The old 1,156-study fold-0 development pool is part of its fitting population and cannot become held-out evidence for this soup. The correct final model comparator is unchanged E10 from this same all-data run. E5/E6 public scores of 0.872 are contextual observations, not E10's score; no all-data E10 public score is established in the reviewed snapshot. [All-data terminal receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/final-public-source-fixed10-20261014-v1/terminal.json)

## Arithmetic and compression requirements

For each projection, the local model is W0 + sBA, with **s = alpha / sqrt(32)**. The correct mean update is s times the mean of the six BA products. The transformation B → BG, A → G⁻¹A preserves each product for invertible G. Thus effective-product averaging already removes latent factor-gauge dependence; a separate factor-alignment step is unnecessary. Averaging A/B independently introduces cross terms. [Actual ResidualLinear code](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/competitive/adapters.py#L62-L82)

The exact six-parent mean has rank at most192. Thin QR on the concatenated B factors and transposed concatenated A factors reduces its SVD to a core no larger than192×192. Truncating that core to32 is optimal for matrix Frobenius error, not for AUC or downstream logits. Preserve the original scaling exactly once; changing to alpha/192 or applying s both during compression and inference would be wrong.

Useful falsifiers before any score claim:

1. Independently verify small synthetic dense means against QR/core-SVD, including nonunit scaling, zero/rank-deficient updates, reordered parents and well-conditioned invertible gauge transforms
2. Report all120 projections: absolute and relative post-FP32-cast residual, retained singular energy, and the rank32 cutoff gap. Report zero-mean cases explicitly rather than divide by zero or hide cancellation
3. Separate the theoretical discarded singular-value tail from additional FP32 refactorization/casting error. Near-tied singular values at the cutoff can make the optimal rank32 matrix nonunique; raw SVD-factor equality is not a sound invariance test
4. Keep two identity checks distinct: unchanged E10 loaded through the new artifact path should preserve its existing inference behavior; six identical E10 parents refactored through QR/SVD need product/forward tolerances, because algebraic identity does not guarantee bitwise floating-point equality

These are proposed numerical checks, not results. Small Frobenius error cannot certify unchanged ranking: small score gaps and downstream sensitivity can still matter. Any original-model or fitting-only prediction comparison needs separate execution authority; protected cohorts remain excluded.

## Compatibility and single-model deployment

Require identical original source, base hashes, tensor membership/shapes/dtypes, adapter targets/scaling, masks, fitting roster, input normalization, target order and head architecture. Learned head values need not be numerically identical: average all compatible learned head/pooling/LayerNorm parameters consistently. Frozen base weights, biases, registers and encoder normalization remain exact. The canonical head uses LayerNorm; the existing averager explicitly rejects BatchNorm running statistics and unsupported buffers. Do not silently add BN recalibration or mix a changed head. [Model graph](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/competitive/model.py), [loading guards](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/competitive/weight_average.py#L368-L379)

A compressed rank32 artifact preserves the original residual parameter count, 10,223,616. With the 203,020-parameter head, FP32 learned tensors total **41,706,544 bytes**, excluding base weights, serialization and runtime buffers. The exact six-parent representation would use **246,178,864 bytes** for those tensors. Both are single-model forwards; the exact representation has sixfold residual linear work, while rank32 compression restores the original adapter shapes. Neither establishes actual RAM, runtime or offline acceptance. Parent checkpoint loading and QR/SVD workspace also need a funded CPU memory envelope.

Keep the derived artifact inference-only, with explicit six-parent provenance and training-resume refusal. Do not impersonate a scheduled E10 checkpoint or copy its optimizer into averaged weights. No inference/TTA/input changes should accompany the comparison.

## Interpretation

The model-soups paper supports weight averaging as a plausible idea in pretrained-model fine-tuning; it does not establish this same-run, rank-compressed MRI result. PEFT documents concatenation and SVD merging, including output-rank control and rank-dependent memory. Neither substitutes for this repository's scaling and provenance checks. [Model soups](https://arxiv.org/abs/2203.05482), [PEFT LoRA reference](https://huggingface.co/docs/peft/main/en/package_reference/lora)

Continue the existing #89 proposal rather than introduce a duplicate branch. Preserve the old negative result, fix membership/weights now, and describe any eventual outcome as the E5–10 compressed all-data recipe. It cannot isolate gauge correction, added E5, training-population change or compression as the cause of an improvement.
