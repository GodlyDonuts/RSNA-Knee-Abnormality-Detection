# Research update: mechanisms already implemented versus genuinely new

14 October 2026, 19:00 UTC review. Historical ledger remains pinned to 8300cfd20c2acaeef9d684a81dbc7fe977282a62. New implementation receipt pinned to e4f288d6d5cf605aa79935cd456359bd7eb498e1.

## Important live reconciliation

The repository now contains PR38's standalone bounded focal-window module and four-arm experiment design. The review records 127 focused passing tests plus targeted lint/type checks. It is not wired into training and has no GPU or accuracy result. Consequently, separating across-window anatomical variation from same-window prediction variation, adding disease-specific focal pooling, and per-window teacher BCE are **already implemented proposals**, not new discoveries to propose again.

[Exact integration](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/commit/e4f288d6d5cf605aa79935cd456359bd7eb498e1) · [Review](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/e4f288d6d5cf605aa79935cd456359bd7eb498e1/initiative/audits/focal-window-PR38-root-review-20261014-v1/README.md)

The focal head does not start as the same function as ordinary pooling, despite equal parameter count. Its pooling-only control is necessary. Teacher confidence is decisiveness, not verified reliability, and the teacher training roster remains unknown. Passing synthetic tests establishes arithmetic and contracts, not lesion localization or an accuracy gain.

The current-state page also records epoch 9 privately preserved and epoch 10 underway in the existing all-development fit. Best verified public score remains .872. No new public result follows from checkpoint preservation. These are observations of existing work, not actions performed in this research.

## Where the beyond-parity search now stands

1. **Cheapest unresolved training hypothesis:** bounded source-distance weighting, conditional on auditing the added cells. The original source's midpoint convention cannot safely be recovered from a blended .5. Compare identical images, masks, updates and target gradient budgets; use shuffled-weight and norm controls. No result exists.
2. **Highest-evidence unresolved supervision mechanism:** genuinely fold-safe image-derived supervision. Current report-derived probabilities are not such an experiment. Preserve total effective loss weight and compare against target-softening controls. Teacher construction is a real compute cost, not free labels.
3. **Distinct focal hypothesis under review:** relative window evidence transfer while leaving the bag head unchanged. This is different from PR38's head swap plus absolute window BCE. Single-teacher order is not epistemic uncertainty; multiple independent, eligible out-of-fold members are required to test the latter.
4. **Deprioritized for the current deadline:** a new physical-extent/severity full fit. Native attention already receives positions, and through-plane extent is not cartilage depth or surface size. A narrow continuity readout with a geometry-preserving shuffled control is the only inexpensive version worth considering if compatible cached features already exist.

## What the synthetic checks establish

The existing aggregate teacher-confidence formula can assign zero weight to heterogeneous windows and positive weight to homogeneous windows with the same mean probability. This is a property of that proposed weighting rule, not evidence of a measured clinical error or an explanation of the current .872 score. The canonical fit has not used that auxiliary objective.

A separate hypergeometric example shows how sampling few windows can omit hypothetical focal evidence. Actual lesions, sampling policy and useful-window counts were not measured. It is not a competitor error rate.

## No false promise from extra complexity

Per-target monotone calibration of final probabilities does not improve ROC AUC unless it changes ties/order. A second BCE on the same logit is equivalent to a weighted target mixture when normalization matches. Equal parameter counts do not guarantee identical initial functions, and equal nominal loss weights do not guarantee equal shared-gradient norms. These facts prevent attributing gains to a mechanism that was not isolated.

## Avoid a needless distillation bottleneck

The deployment objective is one strong eligible MRI-only model, not preservation of a particular backbone. A compact teacher/challenger that already outperforms the incumbent on a matched, admissible evaluation is itself a deployment candidate. Distilling it into DINOv3 adds cost and can lose information. Evaluate the direct model before paying for that transfer. Conversely, published .953 ConvNeXt-T and .951 DINOv2-S reports do not establish that the backbone alone causes their performance; preprocessing, adaptation, supervision and inference remain recipe factors.

## Why self-distillation is promising without being a source of clinical truth

Let weak report target be Y, image X, and a teacher approximate m(X)=E[Y|X]. Blending Y with m(X) preserves the conditional mean: E[(1−a)Y+a m(X)|X]=m(X). Its potential gain is finite-sample denoising, useful model priors, better image coverage or optimization. It does not, by itself, correct a systematic report-to-clinical severity bias. Distinct label sources or independently supported morphology concepts may add information, but correlated teachers trained on the same biased labels can confidently agree. This is a modeling argument, not a finding about measured RSNA outcomes.
