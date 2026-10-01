# Matched sampling: fixed-final result

Both original one-H100/two-CPU jobs completed0:0 after10 epochs/2810 updates:
uniform866719 and stratified866716. Exact scientific source7ac7a97, equal learned
initial tensors/logits, accepted source/runtime/weight/release/rosters, fixed
2247 training and1156 disjoint development studies, effective batch8 and all
281 updates in each epoch passed the independent replay. Only training
slice_sampling differed; evaluation used the same uniform policy. All12 targets
stay included, with the accepted masks. No protected gold/confirmation or Kaggle
score was used. Original terminal/run/invocation/prediction/source/checkpoint
sidecars are preserved here; no large checkpoint was downloaded to this Mac.

Uniform macro AUC **0.8606037925461784**; stratified **0.8615144870268626**.
Difference **+0.0009106944806842598**. A paired whole-leakage-component bootstrap
(4000 draws,seed20261001) gives a conditional95% interval **[-0.0090501,0.0099533]**.
2836 draws define all12 AUCs;1164 do not because a sparse class disappears.
Undefined draws are recorded, never repaired by dropping targets or redrawing.
MCL and fracture each have only2 positives, Baker's3; synovitis only5 negatives.
This pool therefore cannot precisely resolve several target AUCs.

The greater slice exposure did **not establish an accuracy improvement**. This
is an inconclusive/near-null result on one seed/fold, not evidence of equivalence
or a promoted recipe. Preserve both results and retain uniform as the matched
control. A further sampling claim requires another predetermined matched
seed/fold; do not select a best epoch or tune on this holdout. Prioritize the
already staged native384 detail test and the training-only relation acceptance
instead of scaling this small difference into an all-development fit.

| Target | Uniform AUC | Stratified AUC | Delta | Positive/negative observations |
|---|---:|---:|---:|---:|
| ACL | 0.883832 | 0.886422 | +0.002590 | 85/879 |
| MCL | 0.641121 | 0.622093 | -0.019027 | 2/946 |
| Medial Meniscus | 0.863234 | 0.866404 | +0.003170 | 412/580 |
| Lateral Meniscus | 0.735145 | 0.744558 | +0.009413 | 149/821 |
| Medial OA | 0.976388 | 0.970376 | -0.006012 | 23/499 |
| Lateral OA | 0.770036 | 0.799958 | +0.029923 | 9/531 |
| PF OA | 0.865109 | 0.868185 | +0.003075 | 19/445 |
| Effusion | 0.884501 | 0.889303 | +0.004802 | 9/671 |
| Synovitis | 0.980165 | 0.983471 | +0.003306 | 121/5 |
| Baker's | 0.891361 | 0.853403 | -0.037958 | 3/382 |
| Contusion | 0.904887 | 0.918674 | +0.013786 | 61/371 |
| Fracture | 0.931467 | 0.935328 | +0.003861 | 2/259 |

Replay: `PYTHONPATH=. PYTHONNOUSERSITE=1 artifacts/qa-env/bin/python -B
initiative/audits/sampling-fixed-final-v1/recompute.py`.
The replay rejects modified original bytes, nonfinite predictions/updates,
different source/config/roster/initialization, wrong fixed-final step/checkpoint,
and scores inconsistent with accepted masked labels. `comparison.json` includes
all target counts/component uncertainty and exact prediction/label hashes.

## Separate gradient diagnostic failure

Original866846 FAILED1:0 in11:34 on evc32. Its actual stderr shows Triton's CUDA
helper C compilation failing at `#include <Python.h>`. The distinct relation
launcher omitted the accepted CPATH/header guard used by the sampling scripts;
this is an observed runtime failure, not a bad clinical result. Zero optimizer
updates and no completed gradient-measurement.json. Preserve original source,
CPU16 acceptance and runtime/error receipts in gradient-failure/. The correction
must use a distinct control and exact admitted3.12.13 headers/ABI guard; do not
alter scientific4acd6fa source or claim gradient/trainer acceptance until a new
actual receipt passes. CPU32 remains an input-only acceptance.
