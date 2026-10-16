# Current project state

**Checked October 16, 2026, 11:40 UTC.** This is the current research entry point.
Scores and job states below are verified observations; original receipts, Slurm
and authenticated Kaggle resolve later changes. Historical observations are in
[current-state history](current-state.history-20261016.md).

## Actual competition results

| Candidate | Submission | Public score | Interpretation |
|---|---:|---:|---|
| Released author-trained CoAtNet, reproduced unchanged | 57258866 | **0.949** | Attributed external baseline; not our training improvement |
| Own N3 CoAtNet, epoch 4 | 57256756 | **0.821** | Successful hidden execution |
| Own N3 CoAtNet, epoch 8 | 57259376 | **0.869** | Up .048 from epoch 4 |
| Own N3 CoAtNet, epoch 12 | 57262392 | **0.890** | New verified own-trained best; up .021 from epoch 8 |
| Own N3 CoAtNet, epoch 16 | 57265916 | **Unreported** | Submitted once; no score/error at the 11:40 UTC check |
| Prior own DINO best | 57182737 / 57182547 | **0.872** | Historical own-trained reference, now exceeded |

Epoch 12 is COMPLETE, no error, 381,602 output bytes and canonically reconciled.
[Original scored receipt](../initiative/audits/compact-production-e012-kaggle-20261016-v1/actual-scored-canonical.json).
The .890 does not establish parity with the external .949 or a hidden-private score.
The authoritative 11:22 UTC allowance is **five submissions used, zero available**.
The original curve controller finished all three scheduled E8/E12/E16 submissions;
never restart it, resubmit those exact versions or upload a public three-row CSV.
A null score/zero bytes on an unreported submission does not establish failure.

## What is actually training

| Experiment | Original/current Slurm | Observed state | Isolated change |
|---|---|---|---|
| N3 selected-slot triples | 930744 → 932869 | **COMPLETED 0:0**, evc33 | Original fixed 16 epochs / 4,496 updates |
| Source-neighbor challenger | **932957** | **RUNNING**, evc33 | Geometry-valid source-neighbor donor channels, same centers |
| Positive-weight challenger | **933001** | **PENDING**, evc33 | Fitting-derived positive BCE term weights |
| Expanded 2,592-study recovery | **933382** | **PENDING**, evc33 | Engineering diagnostic, not the expanded full fit |

Neighbor 932957 has **740 finite updates, epoch 3**, empty stderr; epochs 1–3
are privately preserved, all eight objects per epoch fully re-downloaded/size/SHA
verified. Leave it healthy and unchanged. Both challengers passed actual H100
full/stop/resume checks of model, BatchNorm, optimizer, scheduler, RNG and predictions.
Those are engineering results, not challenger accuracy gains. The weighted fit
has not started. [Neighbor](../initiative/audits/compact-source-neighbor-full-fit-20261016-v1/README.md),
[weighted](../initiative/audits/compact-positive-weight-full-fit-20261016-v1/README.md).

## Exact own N3 model and completed internal endpoint

This is a **fully fine-tuned 73,136,096-parameter CoAtNet**, not the old frozen-DINO
small head. Backbone `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`; finding-specific MIL
attention over 1,024-dimensional image embeddings. Frozen numerical source
`dcfff319cb7bd7bbae679a20da2b1b9268e8d680`.

- 2,247 fitting studies and 1,156 disjoint weak-label development studies.
- Six series maximum, 16 uniformly selected slices per series, 384-pixel inputs.
- Three-channel windows form **after slice selection**; at most 84 windows/study.
  Training samples 12 windows; inference aggregates the full selected bag.
- Fixed 16 epochs / 4,496 updates, effective batch eight, one-study forwards.
- AdamW encoder LR 3e-5/head LR 1e-3, weight decay .02, OneCycle, clip 3.
- BF16 forward/FP32 loss; no augmentation in this registered run.
- Dense public soft fitting supervision; accepted release/folds and protected
  evaluation remain unchanged. Gold/confirmation are unused for tuning.

All 16 immutable original epochs are privately fully download/size/SHA verified.
CPU 933901 independently matched the complete final model to immutable E16,
source/rosters/masks/boundary and all masked AUCs: **0.9209253491 internal macro AUC**.
This measures weak-label agreement, not a Kaggle or clinical score. Several targets
are unstable: MCL/fracture each two positives, Baker's three, effusion/lateral OA
nine each, synovitis five negatives. [Final evidence](../initiative/audits/compact-native-n3-fixed-final-20261016-v1/README.md).

No single established bug explains the remaining competition gap. The external
author also describes one-study forwards with accumulation; microbatch BatchNorm
is not a demonstrated difference. [Public recipe contract](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md).

## Current model comparisons

The predeclared equal learned-parameter average of E12/E14/E16 is **built**, with
all saved E16 buffers retained. Original CPU 933573 failed before output because a
nonpersistent cache was treated as a saved buffer; distinct corrected CPU 933892
passed. Private HF `4791c7e74af82220d521d0ecc1097172253aeea5` has all nine objects
fully re-downloaded/size/SHA verified. CPU 933958 accepted the strict complete-model
serving loader and anonymous 70-window reference. No averaged-model accuracy or
competition score exists. [Evidence](../initiative/audits/compact-late-average-v2-20261016-v1/README.md).

The next experiment compares **that fixed average versus unchanged E16 on the
same 1,156 raw-MRI outer studies in Kaggle**. One decode supplies identical windows
to both models; both T4s must first match the original CPU references. Clinical
labels stay on the cluster; only private image identities and models go to Kaggle.
Original CPU 934300 bound the complete outer roster, and private Dataset **12630620
version 1** was created once. All ten objects are fully re-downloaded/size/SHA verified. Exact private offline
two-T4 notebook `godlydonuts/rsna-fixedavg-e16-outer-5cc0e5bffb05` v1 was pushed
once (kernel 139017805); it is actually RUNNING, no evaluation result yet. A transient GetDatasetStatus 403 was reconciled through the
exact authenticated private Dataset metadata; no duplicate upload was made.
[Registered comparison](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md).

A metadata-only counterfactual found **zero selected-series changes on all 1,156
outer studies** when dropping Selection_Tier. Thus this field alone cannot explain
a held-out cache/serving gap. It does not establish complete official metadata or
pixel-rendering parity. [Scope and receipt](../initiative/audits/compact-serving-tier-counterfactual-20261016-v1/README.md).

Dense source coverage is separately CPU-admitted: 2,345 original windows on 32 MRI
studies remain bit-equal inside 4,188 dense windows. Both T4s actually accepted
precision/timing with four-update diagnostic weights: 84→180 windows costs median
1.32→2.86 seconds per GPU, peak 2.91 GB. This is not a dense clinical result, full-fit
weight qualification or hidden nine-hour end-to-end claim. Root implements directly;
**Devin is unavailable**. [Dense proof](../initiative/audits/compact-dense30-T4-diagnostic-20261016-v1/README.md).

## Completed assets and available compute

Additional **345 fitting-only studies / 1,769 series / 54,562 slices** have complete
paired 256/384 caches; all seven shards / 42 private objects are fully verified.
The expanded 2,592-study CPU gate 933324 passed original tensor/label/mask parity.
Its distinct H100 recovery is queued; full expanded training is not dispatched.
Sixteen expanded passes mean 5,184 updates, so a later result is not data-only.
[Expansion proof](../initiative/audits/compact-expanded2592-inputs-20261016-v1/README.md).
Correspondence-filtered capacities are 2,592 development fitting / 3,887 train /
4,327 potential frozen-final; these are not the counts already used in N3.

The evc39 endpoint now has **all 37,812 native cache members plus 55,969 runtime
members fully verified**. Suffix-only TLS reused 143.69 GB and transferred only the
remaining 9.36 GB. CPU 934232 reproduced the exact original and neighbor 32-MRI
tensors, scientific bindings and initial head. No second neighbor fit was created.
Other users occupy evc39 GPUs; an earlier free-GPU observation is not a reservation.
[Transport negatives and actual acceptance](../initiative/audits/compact-neighbor-evc39-relocation-20261016-v1/README.md).

Kaggle raw training-image access was already proved by the original 64-study T4
benchmark (318.497 seconds, no unusable fallbacks, about 11.61 GB process RAM).
A separate CPU-only probe measured four cores and 20.94 GB scratch. Bounded selected
cache shards could support an independent training lane; **no cache shard or full
Kaggle fit has run**. [Capacity receipt](../initiative/audits/kaggle-CPU-training-capacity-20261016-v1/README.md).

## Documentation and decisions

Use [research brief](research-brief.md) for conclusions and limitations and
[execution handoff](codex-continuation.md) for exact jobs/paths/intents. Preserve
negative results and originals; never interpret dated historical “pending” or
“unsubmitted” snapshots as live state. Keep healthy training running and judge
isolated mechanisms by fixed-final held-out evidence and actual Kaggle outcomes.
