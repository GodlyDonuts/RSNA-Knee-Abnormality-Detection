# Knee MRI crop pipeline audit

Read-only review on October 14, 2026, pinned to GitHub main `1f05aa1df34bd92212ce92633b99366f98804e3c`.

## Bottom line

Cropping acquired margins is a reasonable hypothesis for an isolated experiment. Existing evidence does not establish that artificial resize padding is a major problem or that cropping improves performance. Cropping and resizing back to 384×384 does **not** reduce the spatial token count or encoder presentation count.

## Actual native path

- No anatomical/spatial crop: whole-volume min/max normalization, physical-aspect-preserving resize, centered zero letterbox, then uint16 storage
- Native384 cache stores **all original slices at 384×384**, retaining the full acquired field. It does not retain original matrix resolution
- Training selects up to 16 uniformly spaced slices and uses the full cached square. Native inference uses the same sampling/resize path
- Missing slice/series slots are excluded before encoding. Spatial background within a valid slice remains part of its input

Sources: [cache construction](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/dicom.py#L118-L181), [dataset](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/competitive/data.py#L200-L223), [full-square sampling](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/competitive/neighbor_input.py#L35-L51), [native inference](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/competitive/infer.py#L152-L171), [encoder masking](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/rsna/competitive/model.py#L184-L216).

## Padding is not acquired background

The fitting-only metadata audit covered 2,247 studies and 12,476 series. Wholly artificial letterbox cells averaged **0.9247756%** of each 24×24 patch grid; 607 series had any such cells. Partial boundary cells averaged 0.5346933% and contain some acquired content.

The reported average is an unweighted mean of per-series fractions. Because every grid has 576 cells, it is also the equal-series aggregate cell fraction, but it is not slice-/presentation-weighted and is not measured wall-time waste. The calculation uses exact resized-content rectangles, not an intensity threshold. It does not quantify acquired air/background or identify dispensable anatomy. Black pixels cannot automatically be equated with artificial padding.

Sources: [aggregate receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/patch-geometry-research-v1/actual-aggregate.json), [audit calculation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/patch-geometry-research-v1/audit-wrapper.py.txt), [geometry helper](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/scripts/research/cache_patch_geometry.py).

## Cache-derived crop versus original-pixel crop

A derived crop can be generated on load from unchanged cache bytes using `original_shape`, `pixel_spacing`, `resized_shape`, and the letterbox offsets. Rebuilding the cache is unnecessary for a framing experiment. Resizing that crop back to 384×384 cannot recover detail lost during the earlier resize or establish exact raw-DICOM/public-renderer pixel equivalence. A genuine higher-detail experiment must crop before the original downsampling.

The prior bounded 16-study/84-series engineering sample had median maximum FOV 165.50 mm versus the public renderer's 130 mm crop. This small sorted sample is not representative validation or proof of universally excessive margins. Smaller acquired fields can also need padding under a fixed physical crop.

Source: [FOV and contrast audit, including limitations](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/fitting-intensity-range-20261007-v1/README.md).

## Prior results do not isolate cropping

No completed crop-only comparison was found in the reviewed research brief, continuation, native code, and relevant experiment records.

- Source-faithful 130 mm triplet/head recipe: fixed-final internal macro AUC **0.8782871**, versus **0.8997841** for the native-public reference. Input representation, slots, intensity handling, laterality, windows, pooling, and augmentation differ. This is a compound negative result, not a causal estimate of cropping
- Selected-stack 1st/99th-percentile contrast experiment: **0.8786706 → 0.8537006**. It retained the full field and tests contrast, not spatial cropping

Sources: [source-recipe result](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/source-slot-fixed-final-20261014-v1/README.md), [compound input recipe](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/experiments/competitive-v1/public-input-recipe-v1/README.md), [contrast decision](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/cache-contrast-production-20261008-v1/fixed-final/decision.json).

## Conditional matched experiment

If authorized, compare the unchanged full-FOV384 path with **one predeclared conservative physical crop, resized to384**. Freeze crop choice using fitting-only information; use consistent coordinates across a series, preserve physical aspect, and specify missing-geometry/small-FOV fallbacks. First verify that peripheral evidence is not systematically truncated; do not assume central anatomy covers all twelve targets.

Keep series selection, slice indices, normalization, fitting labels/masks, accepted split boundaries, model/head, initial learned values/RNG, optimizer, seed, effective batch, update count, and fixed-final endpoint identical. Apply the same crop policy during candidate training and inference. Require tensor/provenance and full-stop-resume checks before any training. Reuse an existing control only if full experimental compatibility is actually verified.

Evaluate paired fixed-final macro AUC on the existing development cohort, per-target outcomes/supports, and whole-leakage-component uncertainty with undefined bootstrap draws retained. Do not tune on protected gold/confirmation data or promote from training loss. A promising result needs a matched repeat.

At unchanged384 dimensions, DINOv3-L/16 still sees **24×24 = 576 spatial patches per slice**, so this is a framing/detail-allocation test. Actual compute savings require a separate reduced-dimension or pre-encoder token-reduction comparison with measured end-to-end throughput, memory, and all-target performance. No speedup or accuracy gain is predicted here.

## Scope

Only repository code, documentation, directory listings, and aggregate receipts were read. No clinical rows/images, protected inputs, weights or data downloads were accessed; no training, inference, remote job, or repository mutation was performed. This file is a standalone audit deliverable.
