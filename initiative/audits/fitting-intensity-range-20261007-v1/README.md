# Actual bounded MRI contrast and anatomical scale comparison

The captured public Pilkwang renderer crops a130mm field, takes samples from a20–80% slice band, then normalizes the sampled/cropped stack using its1st/99th percentiles. Our native cache preserves full physical field of view and normalizes against the **entire original volume's minimum/maximum** before resampling/letterboxing. These are separate input differences, not an established reason for low scores.

An original oneCPU/zeroGPU reader inspected the first16 sorted fitting examinations from the exact active2247-study roster, selecting the original84 series and16 uniform slices per series. It used original retained uint16 MRI arrays and excluded artificial letterbox padding from its statistics. Only aggregate measurements were exported; no pixels, case IDs, reports, clinical targets, protected label rows, model weights or predictions were exported or fitted. Neither live trainer's preprocessing changed.

Actual sampled measurements:

- Median content1st–99th-percentile span is **0.48049** of our full normalized input range (interquartile0.36401–0.58849; range0.20955–0.86827).
- **47/84** series span less than half that range; only **2/84** span less than a quarter. The sample does not establish universal severe contrast collapse.
- Median maximum physical field of view is **165.50mm** (interquartile160.00–180.00; range99.99–220.02), versus the public recipe's130mm crop. A square image-side count alone does not make the two inputs anatomically equivalent.

[actual-receipt.json](actual-receipt.json) records the actual roster binding, sampling rule, denominators and limitations. These16 sorted studies are a bounded engineering sample, not a representative random prevalence or validation cohort. Percentiles here are measured on our reconstructed selected content, **not** the public renderer's original raw cropped pixel samples. There is no exact raw-input parity claim and no clinical accuracy measurement.

## Falsifiable follow-up

Prioritize an isolated robust-contrast comparison before combining changes: same slices, field of view, series, labels, masks, adapter recipe and budget; current min/max normalization versus1st/99th-percentile normalization on selected valid content. Clipping can discard informative bright or dark findings, so its public popularity is not enough to promote it. Require unchanged original-cache bytes, CPU pixel/tensor bindings and actual trainer recovery before a funded matched fitting run.

A130mm physical crop is a distinct subsequent axis. It magnifies anatomy but may remove peripheral evidence or require padding for fields smaller than130mm. Test it separately, preserving physical aspect, documented missing geometry and accepted study boundaries. Do not mislabel interpolation of our reconstructed cache as raw-DICOM fidelity, AI enhancement, new acquired resolution or exact reproduction of public pixels.

This audit is completed observation only. No normalizer/crop was integrated, no new GPU experiment was dispatched, and no accuracy gain is established. Both healthy original ten-epoch comparisons continue. Original probe/receipt retained under `/tmp/rsna-fitting-intensity-range-20261007-v1` on evc31; small controls capped16MiB, zero shared artifact growth,53GiB existing node future liabilities and20GiB free floor retained.
