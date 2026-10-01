# Whole-series rank32 fold-0 result

The pinned C9 whole-series rank32 ten-epoch job 865357 completed 0:0. Its fixed-final held-out macro AUC is **0.8603399208109389** on the same 1,156-study outer holdout as sampled rank32, whose fixed-final result was **0.8620940794159951**. The whole-series difference is -0.0017541586050562. `outer_predictions.csv`, `result.json`, the preserved run binding and `comparison.json` permit independent replay of the masked per-target results.

This control used every eligible cached series/slice at 224 pixels from the existing 256 cache; it did not use native384 or adjacent-slice context. All ten immutable checkpoints are privately published and fully download/hash verified. This single-fold comparison does not establish that more coverage is harmful: target changes go in both directions and the rare targets have very limited held-out positives. No Kaggle score exists for this model, and no hidden inference submission is implied by internal completion.
