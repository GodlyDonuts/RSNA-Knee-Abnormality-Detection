# Rank32 slice coverage at a fixed compute budget

The existing sampled rank32 recipe uses the same uniform sixteen source slices
from each selected MRI series every training epoch (`augment=false`). The new
`slice_sampling="stratified"` policy chooses one real slice from each of sixteen
disjoint ordered integer bins. Short series use every available slice once and
retain masked padding. Series selection, plane/protocol features, images,
geometry/missingness and study-level labels/masks keep their existing policies.
A study's weak label still supervises the aggregated study output; no diagnosis
is assigned to individual slices.

The existing EffectiveBatchSampler supplies a CPU seed from `seed:epoch:index`
and SeededDataset restores model/Python/NumPy RNG states after reading a study.
This gives epoch variation and exact interrupted-epoch replay without changing
model dropout RNG or depending on worker count or distributed rank. A slice is
not guaranteed to change in every bin each epoch, and this does not guarantee
that the model sees a lesion. General series augmentation must remain disabled
for this separate sampling comparison.

## Measured metadata exposure

`source-exposure.json` replays the actual frozen 2,247-study training membership,
series preferences and all source slice counts. Each of the 75 input-plan files
and the accepted series table was size/hash bound; reports, target values and
MRI pixels were not read. It uses the implemented bin sampler and the same
lexically sorted training-study indices/seeds as the trainer.

| Exposure over ten epochs | Fixed uniform | Epoch-varying stratified |
| --- | ---: | ---: |
| Source slice presentations | 1,859,290 | 1,859,290 |
| Distinct source slices in selected series | 185,929 | 364,647 |
| Fraction of available slices in selected series | 48.06% | 94.26% |

Selected series contain 386,846 slices across 11,639 series. The full training
cohort has 421,333 slices across 12,476 series; the unchanged six-series cap omits
837 series. Stratification would expose 86.55% of all training source slices in
this replay. These are metadata exposure counts, not a trained model's observed
attention, clinical coverage, independent patient count or accuracy. The local
CPU Torch version is2.14.0; actual Newton CUDA runtime and recovery acceptance
remain required. Replay from the repository root to a new output path:

```sh
PYTHONNOUSERSITE=1 artifacts/qa-env/bin/python \
  initiative/experiments/competitive-v1/stratified-slices-v1/measure-exposure.py.txt \
  --output /tmp/rsna-stratified-exposure-new.json
```

The script requires a new output path and refuses overwrite. The second exact
replay also preserves analysis/loader/sampler source byte hashes and the actual
Torch version.

## Fixed evaluation and deployment

Outer-fold evaluation and offline Kaggle inference map the bound training
stratified policy to the original uniform sixteen-slice grid. The inference
receipt explicitly records both training and evaluation policies. A stochastic
`stratified` dataset is refused for `test_series`; standalone evaluation must
use `evaluation_slice_sampling`. The actual adapter-pilot reload verifier uses
the same mapping, and Kaggle bundle metadata declares the actual uniform
inference policy. No inference-time augmentation, extra test slices, teacher,
ensemble, threshold or public-score adaptation is introduced.

## Separate matched experiment

`control-config.json` preserves the existing sampled rank32 delta recipe.
`candidate-config.json` differs only in `slice_sampling`. Both use224 inputs,
256cache, six selected series, sixteen slices, ten epochs/2810 effective updates,
2247training/1156heldout studies, rank32/alpha8 and the same seed/optimizer.
The old rank32 measured internal0.8620940794159951 and Kaggle0.769 are contextual
references. A fresh uniform control on the same new source/runtime is the causal
comparison; do not claim attribution from unmatched historical runs. Record
initial model/head hashes and zero-adapter evaluation logits equality before
paired optimization, and keep all twelve target metrics and component-level
uncertainty. Use the fixed final epoch, without protected confirmation/gold
selection. A second seed/fold is needed before a broad conclusion.

The two pilot configs use16studies/twoepochs only for bounded engineering and
exact checkpoint/reload acceptance; their metrics are not competitive evidence.
Actual official-weight GPU acceptance, clean pinned source checkout, versioned
launch/output/private publisher, quota and full private verification remain
required before production. No new training or Kaggle job is dispatched here.
The next committed native384 resolution experiment stays on its C9 source,
uniform policy and exact config; do not change or restart its existing CPU jobs.
Run this separate coverage comparison after native readiness/result review and
fresh shared quota funding. At most two GPU nodes/minimum CPUs.

Fourteen new checks exercise real synthetic cache decoding, pixel-to-position
alignment, bin uniqueness/short-series padding, fixed series/masks, label-blind
sampling, greater epoch exposure, preserved RNG, two-worker/two-rank partition
and resume invariance, actual cached-image training mid-epoch recovery in both
epochs, and deterministic offline inference. These tests use synthetic data and
a diagnostic encoder; they establish mechanics, not accuracy on clinical MRI.

The complete repository suite passed 1,427 tests in 117.60 seconds after fixing
two test-fixture call regressions. The focused deployment/sampling suite passed
73 cases. `implementation-acceptance.json` pins config/source/log hashes and the
exact configuration difference. `network-readonly.json` records the subsequent
local connection observation; it does not certify a remote job outcome.

More exposure alone may not improve prediction: the historical all-slice rank32
run measured internal macro AUC0.8603399208109389 versus0.8620940794159951 for
sampled rank32. Those runs motivate the matched stratification test rather than
establishing that more slices are better.
