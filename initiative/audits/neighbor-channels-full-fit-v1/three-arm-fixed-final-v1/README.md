# Completed three-arm neighboring-slice experiment

Original job872279 completed0:0 in12:16:18 on oneH100/twoCPUs. All three
arms completed exactly three epochs/843 contiguous finite updates, using2247
original fitting and1156 disjoint weak-label development studies. All nine
immutable checkpoints were privately preserved once with five full-download/
size/SHA verified objects each. Final distant epoch3 revision is
`a7949b846ff30c7cce38e4b3a027183570183ee2`, owning-nodeCPU872979.

The primary Medial Meniscus effect and anatomical specificity are unestablished.
Do not promote a recipe, extend epochs or select a strength from this holdout.
The distant arm has a larger secondary macro point estimate, but its interval
also includes zero. These are weak-label internal results, not Kaggle scores.

| Contrast | Primary meniscus delta | Paired95% | Macro delta | Conditional95% |
| --- | ---: | --- | ---: | --- |
| distant_minus_gray | -0.000163 | [-0.004721, 0.004302] | +0.011565 | [-0.007365, 0.031157] |
| physical_minus_distant | -0.000785 | [-0.005257, 0.003646] | -0.009107 | [-0.024043, 0.004628] |
| physical_minus_gray | -0.000948 | [-0.006783, 0.004753] | +0.002457 | [-0.021886, 0.028624] |

All4000 primary draws are defined. Macro intervals condition on2836 defined
draws, retaining1164 undefined draws explicitly. Whole leakage components are
resampled jointly across all three arms with4000 draws/seed20261001.
One fitted seed and sparse endpoints limit conclusions; distant perturbation
magnitudes are unmatched, so this cannot isolate anatomical adjacency alone.

| Target | Observed positives | Observed negatives | Gray AUC | Physical AUC | Distant AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACL | 85 | 879 | 0.776062 | 0.789052 | 0.787506 |
| MCL | 2 | 946 | 0.528805 | 0.581924 | 0.566596 |
| Medial Meniscus | 412 | 580 | 0.791632 | 0.790685 | 0.791469 |
| Lateral Meniscus | 149 | 821 | 0.677497 | 0.678568 | 0.683276 |
| Medial OA | 23 | 499 | 0.964581 | 0.958787 | 0.961096 |
| Lateral OA | 9 | 531 | 0.739694 | 0.752982 | 0.746181 |
| PF OA | 19 | 445 | 0.783323 | 0.780189 | 0.775163 |
| Effusion | 9 | 671 | 0.853370 | 0.840702 | 0.861318 |
| Synovitis | 121 | 5 | 0.961157 | 0.961157 | 0.960331 |
| Baker's | 3 | 382 | 0.750000 | 0.864747 | 0.808028 |
| Contusion | 61 | 371 | 0.781671 | 0.794861 | 0.788233 |
| Fracture | 2 | 259 | 0.825290 | 0.668919 | 0.842664 |

Baker's cyst and Fracture have only three and two observed positives. Their
large secondary swings cannot select a recipe. The primary meniscus endpoint
has412 positives and580 negatives; its near-null effects are not explained
by those sparse macro endpoints.

Read `recompute.py`, `comparison.json`, `collection.json` and
`terminal-collection.json` for exact source/roster/base/config/result/prediction
bindings, all12 uncertainty records and actual terminals. Independent replay
uses only the admitted outer roster; protected gold/confirmation remain unused.
Original gray prediction bytes equal the historical control, but checkpoint
serialization includes different source/config provenance. The replay binds the
actual new gray final run/result and original publication proof rather than
substituting the historical control checkpoint. This metadata distinction does
not change any prediction or learned-state claim.

## Next engineering question

Test the already implemented signed patch-witness branch beside the unchanged
global context: does preserving small target-specific local evidence help when
neighbor context did not? Follow the existing patch-witness-v1 fixed-budget
protocol, including equal encoder observations, zero/one branch weight controls,
real-MRI full/stop1/resume recovery, gradient/stress inspection and fresh separate
resource/checkpoint funding. Do not conflate this with neighbor inputs, changed
labels, external weights or protected evaluation. No new GPU pilot or fitting
job is admitted or dispatched by this decision.
