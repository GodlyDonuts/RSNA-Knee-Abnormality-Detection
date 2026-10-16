# Research brief — October 16, 2026

Read [current state](current-state.md) first. It contains timestamped scores,
current run phases, the actual training recipe, population counts and deployment
status. This brief records conclusions; it is not another execution log.

## What the evidence supports

Fresh authenticated [public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
reference on October16 is **0.968** at first place. Our licensed0.949 baseline
is still0.019 below that public reference; own0.894 is0.074 below. Neither is
a winning result or a private-leaderboard guarantee. The new context path aims
to improve the strong foundation; its engineering pass is not an accuracy gain.

The attributed author-trained model scored **0.949** on Kaggle. Our own CoAtNet
epoch 16 scored **0.894**, following epochs 4/8/12 at **0.821/0.869/0.890** and
prior own DINO best **0.872**. The final increment is +0.004, compared with
+0.048 and +0.021 earlier; continued improvement is not guaranteed. These competition
results remain distinct from the completed fixed-16 internal macro AUC of
**0.920925**, independently replayed on 1,156 weak-label holdout studies.

That internal endpoint has only two MCL/fracture positives and three Baker
positives. It is not a clinical score or evidence of parity with the external
competition result. Exact per-target support and scored receipts are linked in
current state.

The new N3 model fully trains a 73M-parameter CoAtNet. It is not the historical
frozen-encoder/small-head recipe. It already uses dense public soft fitting labels,
384-pixel inputs, finding-specific MIL, 12 training windows and a fixed 16 epochs.
The remaining differences from the released reference include population,
observation/rendering, loss, optimization, late-state averaging and inference.
Their combined score gap does not identify any one causal ingredient.

Current N3 triples are consecutive **selected slots**, not guaranteed consecutive
source slices. The source-neighbor challenger preserves original centers and all
other registered settings. Its full fit and a separate positive-weight full fit
were submitted after actual H100 recovery. Neighbor932957 is now actually training;
weighted933001 is RUNNING; epoch1 privately verified.
Neither has demonstrated an accuracy gain. Public one-study gradient accumulation
also means mixed-study BatchNorm is not a known missing ingredient.

Additional 345 fitting-only caches and private preservation are complete.
Expanded 2,592-study inputs passed actual zero-GPU admission933324 with original
image/label/mask tensor parity; no expanded model is trained yet. The old 3,904/441 release counts describe a historical release;
current correspondence-filtered capacities are 2,592 fitting / 3,887 train /
4,327 potential frozen-final. Do not equate these with the active 2,247-study fit.

## Findings that should guide the remaining work

- Held-out weak-label AUC measures agreement with weak supervision. It is not a
  substitute for clinical truth or a numerical prediction of Kaggle performance.
- Higher rank, resolution and more parameters have not by themselves established
  a robust gain. Completed negative/null comparisons remain in the linked history.
- Physical source adjacency and denser coverage are different interventions.
  A neutral adjacency result would not rule out a coverage benefit.
- Cached rewindowing cannot restore spatial detail removed before caching.
  Raw physical cropping, crop-after-resize and intensity normalization are separate.
- Our read-only 31-second training sample showed bursty GPU utilization (22.4%
  mean); it did not isolate a cause. Source reads/decompresses whole NPZ volumes.
  Checkpointing occurs at epoch/stop boundaries, **not every update**.
- A public model trained on all RSNA studies cannot provide independent OOF
  predictions for our held-out cohort. Do not fit blend weights on such leakage.
- Source-trained Stanford results are separate three-task research, not an RSNA
  transfer result. Neither diagnostic recovery nor cache completion establishes
  accuracy. No model is described as clinically perfect.

## Active experiments and decision rules

The original fixed N3 fit and predeclared E4/E8/E12/E16 submission curve are complete;
all four exact hidden outcomes are scored; never repeat those submissions.
Compare source-neighbor and positive-weight fits separately against the same
2,247/1,156 baseline, fixed seed and 4,496-update schedule. Require per-target
results, paired whole-component uncertainty, runtime and failure accounting.
A promising result needs replication where time permits; no best-epoch picking
on protected cohorts.

Admit expanded fitting inputs without changing the outer roster. Freeze the
selected recipe before a larger final refit. Record that more examples at the same
number of passes also cost more updates. Do not label such a result data-only.

Investigate dense inference, explicit anatomical metadata, exact anatomical TTA
and late-epoch parameter averaging as separate candidates. BN buffer policy and
training-serving geometry must be explicit. OrthoFoundation/domain adaptation
remain research directions until actual checkpoint/input compatibility and
appropriate use are established; upstream task scores are not competition gains.

## Evidence and historical findings

- [Actual current runs and score receipts](current-state.md)
- [Exact public recipe correction](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md)
- [Paper 2 assessment and feedback](../initiative/audits/research-paper2-assessment-20261016-v1/README.md)
- [Neighbor full-fit admission](../initiative/audits/compact-source-neighbor-full-fit-20261016-v1/README.md)
- [Positive-weight full-fit admission](../initiative/audits/compact-positive-weight-full-fit-20261016-v1/README.md)
- [All additional-cache preservation](../initiative/audits/additional345-cache-terminal-20261016-v1/README.md)
- [Historical research brief, including earlier results and negatives](research-brief.history-20261016.md)

Historical findings are preserved, not erased. Dated statements such as “pending,”
“no score” or “not implemented” in that history do not describe the current state.

The registered equal learned-parameter average ofE12/E14/E16 is now constructed
and privately fully verified, with all savedE16buffers retained. Actual CPU MRI
serving acceptance and the actual paired two-T4 evaluation passed; its point
gain is small and not established by the uncertainty analysis. The first worker's
nonpersistent-buffer check failed before output; corrected verification leaves
selection and numerical averaging unchanged. No derived gain is claimed.

The fixed average and E16 were evaluated together on the same raw-serving
outer MRI windows in Kaggle, with clinical labels retained on the cluster. This
separates an averaging gain from a changed serving reference; neither is assumed.
A metadata-only replay already rules out Selection_Tier alone changing the held-out
series selection (zero changes/1,156 studies), but other metadata/rendering and
BF16-versus-FP16 differences remain. The preregistered comparison and negatives
are in [the original audit](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md).

A fitting-only CPU profile compares materializing all windows versus only a fixed
12-window subset. Code inspection confirms that **production N3 already selects
before materialization**, so this is not a new N3 optimization. Selected tensors and global RNG stayed exact;
window materialization summed 2.1092→0.9034 seconds over 16 studies, with another
5.3107 seconds spent loading. It is a counterfactual substep saving, not a new training speed/accuracy
gain; no live trainer was changed. The neighbor dataset still eagerly constructs
all donor windows, a separate future optimization. [Original measurement](../initiative/audits/compact-selected-window-cost-20261016-v1/README.md).

A new licensed inference experiment targets eight cross-series triples in the
attributed 0.949 model. Primary edge-repeat retains all centers; secondary
within-series removes the mixed windows. Both exact private v1 candidates
completed public MRI/CSV acceptance without changing learned weights; neither
has a competition submission/hidden score yet. This is an
unproven mechanism and external adaptation, not an own-training improvement or
independent internal validation. [Protocol and interpretation](../initiative/experiments/competitive-v1/released-series-boundary-v1/README.md).

The next learned mechanism preserves the licensed strong encoder and adds a small
zero-start, bounded contextual head over its per-window features. It tests whether
relations across slices/series help beyond independent window MIL, while avoiding
retraining a73M CNN every head epoch. It is implemented; the exact
label-free32-MRI feature probe COMPLETED successfully. The distinct full3,887
feature pass hit actual memory exhaustion.832records are byte-recovered; bounded suffix recovery uses a private Dataset after Kaggle omitted failed-notebook inputs. Parent exposure
precludes independent internal OOF. The complete production head engine passed actual32-MRI full/stop/resume; full fitting awaits complete features.
[Mechanism, literature, fixed proposal and rejection criteria](../initiative/experiments/competitive-v1/released-context-v1/README.md).

The registered averaging experiment is now complete: raw-servingE16 0.92068844
versus fixed average0.92100408 internal macroAUC, delta+.00031564 with paired
whole-component conditional95%[-.00214034,.00293081]. No established gain; retain
E16 as own-model default. Raw/cachedE16 aggregate difference-.00023691 also has
interval crossing zero; no large aggregate serving collapse was observed, but
this is not proof of pixel parity or an isolated decoder/precision experiment.

The context feature probe ACTUALLY passed32 MRI studies in60.166s on both T4s,
with exact original pixels/cache-head probabilities/zero-start logits. Features
are FP32, so no quantization is assumed. A separately funded full3,887-study
frozen feature pass requires bounded suffix recovery, reusing832verified records. The exact full FP32head engine is accepted on real features; no full fit or accuracy result yet.
The actual cache enables repeated context-head experiments without repeating
expensive CNN inference. Raw/index/features remain private.

## October16 feature-cache memory negative

The label-free32-MRI feature pipeline passed, but the scaled3887pass139034277/v1 hit actual system-memory exhaustion at970.3s. Partial832records have0recorded per-study failures; they are not a complete accepted cache. Bound producer submissions as well as the consumer queue, then verify/reuse existing private feature bytes. This engineering failure is not a disease-classification result. The complete FP32context head engine source2ee0cffe passed actual32-MRI full/stop1/resume with complete state/prediction equality before full fitting.

The dependency-corrected bounded suffix139044146 is now actually RUNNING. Its49.5s gate verified all832saved feature records and4privateDataset metadata files before new MRI, so only3055cases require new CNN inference. The finite owner advances to the exact accepted FP32head engine after full feature acceptance. No new disease score exists.
