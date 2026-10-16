# Matched-center source-neighbor CoAtNet contrast

Declared October16 before any CoAtNet outer accuracy or Kaggle score. Original N3 source dcfff319/job930744 remains healthy and unchanged; its E4 competition entry57256756 is still unreported. This is a separate executable input policy, not a replacement of live files or an accuracy result.

## Measured question

Actual fitting-only geometry CPU931871 passed2247studies/11639selectedseries, zero unverified geometry/short series/studies without a physical triple. Only12421/162651 (~7.64%) old candidate triples are consecutive physical source neighbors. Original uniform source-index step median2,p95=3,max22. Full-source bags median150/p95=280/max609; simply replacing the old84-window cap with94 full-source windows changes coverage as well as adjacency.

Test whether local donor slices help while retaining EXACT original centers/window identities, counts, training12-window RNG and deterministic full inference bag (<=84). For the old triplet of uniformly selected samples (a,c,b), substitute original source(c-1,c,c+1) only where BOTH source physical edges pass original neighbor_geometry's positive<=1.5median spacing/inplane<=.001mm rule. Otherwise retain the original triplet explicitly and count that fallback. Keep middle channel byte-identical. No cross-series donor, invented position, extra source acquisition, raw cache rebuild, percentile crop, class weighting, augmentation, batch/BN, model or population change.

## Executable contract

compact_source_neighbors.py provides channel_plan, one-pass original verified native-volume loading and MatchedSourceNeighborDataset. It does not load both legacy and new views during training. Original metadata/cache byte checks and label masks remain intact. compact_train.py has an explicit --input-policy matched-source-neighbors registry; default n3 retains original behavior and binding. New state has separate engine native_coatnet_bag12_matched_source_neighbors_v1 and exact policy binding; original production inference rejects it. Old snapshots cannot silently resume under this new input policy. Frozen running source on Newton is never edited.

Same official CoAtNet initialization/73M full trainable parameters, original2247/1156/16epochs/4496updates, originalAdamW/OneCycle/clip3/BF16/FP32 loss and seed. Primary fixed-final comparison on the same original weak-label outer masks, all12 targets/paired whole-component uncertainty; no interim outer/Gold selection. Initial learned-state equality must be verified in actual GPU admission. Evaluate richer full-source94 coverage separately (Devin issue94), not as an adjacency-only result.

## Acceptance so far

87 focused compact family tests/Ruff/scoped mypy passed. Tests prove original source donor indices and exact unchanged middle channels; <=16-source-slice identity; explicit missing/gapped-geometry fallback; unchanged bag sampling; and actual numerical-engine synthetic full/stop1/resume complete model/BN/AdamW/OneCycle/allRNG/history/counter equality and identical prediction CSVs in the identity case. Legacy/new bindings refuse cross-policy resume. This is engineering evidence, not clinical improvement. Real MRI tensor/source/roster admission and real H100 recovery are still required before a full challenger fit. Budget/source/device/publication obligations remain separately admitted; no new GPU dispatched by this source commit.
