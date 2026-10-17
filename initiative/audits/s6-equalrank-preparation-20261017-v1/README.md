# Fixed CoAtNet/S6 rank blend: prepared, not dispatched

The next complementary-predictor contrast is fixed **50% CoAtNet /50% S6**
per-target rank averaging. The weight is chosen now, without public-score,
protected clinical or exposed-feature-fold tuning. No score is predicted.

The original S6 notebook already executes both predictors, including at its
original0/1 weights. This candidate changes only the weight assignment and a
correctly recomputed build-ID assignment. Every other AST node is identical:
encoder weights, physical preprocessing, spatial/window pooling, TTA, data,
fallbacks, correlation guard, offline environment and time limits are unchanged.
It retains the original failed-S6-row rule (CoAtNet rank only), rather than
silently claiming an always-pure two-model result.

Eight CPU contract checks pass: exact rank formula, symmetry, monotonic-scale
invariance, ties, finite twelve-target range, no-failure equivalence, failed-row
fallback and other-row mean-fill behavior. These test output arithmetic, not
GPU operability or generalization. [Checks](local-rank-contract-checked.json),
[exact source identity](prepared.json), [builder](builder.py.txt).
The full derivative source remains private; original upstream bodies, weights
and pixels are not newly copied into Git or HF.

There is **no new GPU admission, job, Dataset, notebook or competition entry**.
The candidate must first use fresh resource admission and exact offline public
acceptance; deploy after the original S6 hidden outcome is known. Its original
predictors are exposed to internal fitting studies, so this is a fixed diagnostic
blend, not independent OOF-selected ensemble training. It establishes neither
the author's shared0.956 nor a first-place result. A genuinely new adaptation
still needs measured complementary errors and a matched control.
