# Whole-series implementation and workload evidence

September 30, 2026. The accepted observed-development cohort has 3,403 studies
and 643,501 cached source slices. The existing deterministic six-series/16-slice
sampler exposes 282,326 slices, or 43.8734% of that inventory. This is a count of
source exposure, not a measurement of missed disease or anatomical coverage.
The median study has 166 slices; the maximum is 632. Full padded FP32 image
input peaks at 509,788,160 bytes across this cohort. Cache side remains 256 and
model side 224; no native-resolution recovery is claimed.

The immutable input hashes and per-study coverage table are archived here.
The resource calculation uses image metadata only; it does not inspect target
labels to choose the five largest fold-0 training workloads. Reserved
confirmation/gold and outer studies are excluded from optimizer/profile probes.

The implementation adds variable-depth/series batching and masked neighboring
feature messages. Local tests cover exact shared initial state/RNG, zero residual
prediction equality, padding and series isolation, gradients, position rejection,
variable full-coverage inference and exact model/optimizer/RNG/prediction recovery
for both outer-fold and all-development engine modes. No clinical efficacy,
H100/T4 acceptance or Kaggle score is claimed by software tests.

See [the experiment](../../experiments/competitive-v1/whole-series-context-v1/).
The bounded real-data pilot, largest-examination profile, production dispatch and
private compact preservation remain separate measured milestones.
