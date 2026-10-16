# Current project state

**Checked October 16, 2026, 14:56 UTC.** This is the current research entry point.
Scores and job states below are verified observations; original receipts, Slurm
and authenticated Kaggle resolve later changes. Historical observations are in
[current-state history](current-state.history-20261016.md).

## Actual competition results

| Candidate | Submission | Public score | Interpretation |
|---|---:|---:|---|
| Released author-trained CoAtNet, reproduced unchanged | 57258866 | **0.949** | Attributed external baseline; not our training improvement |
| Own N3 CoAtNet, epoch 4 | 57256756 | **0.821** | Successful hidden execution |
| Own N3 CoAtNet, epoch 8 | 57259376 | **0.869** | Up .048 from epoch 4 |
| Own N3 CoAtNet, epoch 12 | 57262392 | **0.890** | Up .021 from epoch 8 |
| Own N3 CoAtNet, epoch 16 | 57265916 | **0.894** | New verified own-trained best; up .004 from epoch 12 |
| Prior own DINO best | 57182737 / 57182547 | **0.872** | Historical own-trained reference, now exceeded |

Epoch 12 is COMPLETE, no error, 381,602 output bytes and canonically reconciled.
[Original scored receipt](../initiative/audits/compact-production-e012-kaggle-20261016-v1/actual-scored-canonical.json).
Epoch 16 is also COMPLETE, no error, 382,473 bytes and canonically reconciled at
12:36 UTC. [Scored E16 receipt](../initiative/audits/compact-production-e016-kaggle-20261016-v1/actual-scored-canonical.json).
The .894 does not establish parity with the external .949 or a hidden-private score.
The authoritative 12:36 UTC allowance is **five submissions used, zero available**.
The original curve controller finished all three scheduled E8/E12/E16 submissions;
never restart it, resubmit those exact versions or upload a public three-row CSV.
A null score/zero bytes on an unreported submission does not establish failure.

Fresh authenticated public leaderboard reference on October16 is **0.968**.
Our external0.949/own0.894 results are distinct and below that reference.
[Recorded public-only snapshot](../initiative/audits/released-context-feature-stage-20261016-v1/public-leaderboard-reference.json).

## What is actually training

| Experiment | Original/current Slurm | Observed state | Isolated change |
|---|---|---|---|
| N3 selected-slot triples | 930744 → 932869 | **COMPLETED 0:0**, evc33 | Original fixed 16 epochs / 4,496 updates |
| Source-neighbor challenger | **932957** | **RUNNING**, evc33 | Geometry-valid source-neighbor donor channels, same centers |
| Positive-weight challenger | **933001** | **RUNNING**, evc33 | Fitting-derived positive BCE term weights |
| Expanded 2,592-study recovery | **933382** | **PENDING**, evc33 | Engineering diagnostic, not the expanded full fit |

Neighbor 932957 has **nine completed epochs**, empty stderr; epochs 1–9
are privately preserved, all eight objects per epoch fully re-downloaded/size/SHA
verified. Leave it healthy and unchanged. Both challengers passed actual H100
full/stop/resume checks of model, BatchNorm, optimizer, scheduler, RNG and predictions.
Those are engineering results, not challenger accuracy gains. The weighted fit is RUNNING and epoch1 is privately verified; no challenger final result yet. [Neighbor](../initiative/audits/compact-source-neighbor-full-fit-20261016-v1/README.md),
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
serving loader and anonymous 70-window reference. No averaged-model competition
score exists; its actual paired weak-label endpoint is now available below. [Evidence](../initiative/audits/compact-late-average-v2-20261016-v1/README.md).

The actual paired run139017805 COMPLETED all1,156 raw-MRI outer studies once
in6,000.47seconds, with zero unusable studies and12.22GB processRSS. E16 raw
macroAUC **0.92068844**, fixed average **0.92100408**, delta+0.00031564;
whole-component conditional95%[-0.00214034,+0.00293081], 2,836defined/1,164undefined
draws. **No established averaging gain; keep unchangedE16 as default.**
Secondary cached/BF16E16 versus raw/FP16E16 changed aggregateAUC by-0.00023691,
also with interval crossing zero. This does not prove exact pixel parity or rule
out offsetting per-target effects. Clinical labels/CSV rows stayed on the cluster.
Read-only HTTP503 and a collector CSV-schema assumption were recovered without
new inference or duplicate downloads. [Actual aggregate and failures](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md).

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

Source `92c558979d4dee6a59ca4e431af8724d7ce2e1ec` and 63 distinct small objects
are privately fully download/size/SHA verified at HF
`1ab4b530bbe7c70b3c29db64da384e04f0294332`. [Exact preservation receipt](../initiative/audits/hf-E12-paired-research-20261016-v1/verified.json).

## New licensed inference comparison

Two actual CPU-staged candidates use the attributed 0.949 reference, with unchanged
weights. Its 96-slice concatenation has eight cross-series triples: primary
edge-repeat repairs their donor indices while retaining all 94 centers; secondary
within-series retains 86 windows. This is an unproven input-context hypothesis.
No independent internal validation is claimed for an externally trained parent
that used our holdout/Gold58. Both exact private v1 notebooks were pushed once and COMPLETED public MRI/CSV
acceptance, including both-T4 pixel checks. Neither has a hidden score or competition
submission. No Dataset/weight upload was repeated. [Protocol](../initiative/experiments/competitive-v1/released-series-boundary-v1/README.md),
[actual CPU stage and finite queue](../initiative/audits/released-series-boundary-stage-20261016-v1/README.md).

The earlier window-materialization profile is correctly scoped: baseline N3
already selects 12 windows before materialization. The measured counterfactual
saving is not a new N3 speedup. Neighbor donor loading remains eager and unchanged.

Source32e52e3d and24distinct small objects/121965bytes are now privately fully
re-downloaded/size/SHA verified at HF7b6632ee82456ae9d76f073fcc59ed07b7fc0294
byCPU934786 under the canonical lock. Never repeat; see
[verified E16/boundary preservation](../initiative/audits/hf-E16-boundary-stage-20261016-v1/verified.json).
No private clinical/identity/image/weight bytes in this milestone.

## Sequence-context experiment: real engine accepted, cache recovery running

A bounded zero-start context readout is implemented over the licensed0.949
model's frozen encoder/head. It compares slices/series rather than treating
windows independently. The parent was developed using our internal cohorts and
Gold58, so this experiment cannot claim independent internal OOF. No new accuracy
result exists. [Protocol](../initiative/experiments/competitive-v1/released-context-v1/README.md).

The original32-MRI two-T4 probe passed exact pixels/cache-head/zero-start identity.
Its scaled pass exhausted RAM after832saved records. All32original plus800new
journaled records were privately recovered and byte/FP32NPY verified. A bounded
attempt then stopped before encoding because Kaggle omitted its failed-notebook
input. One private Dataset12634368v1 bridges those800records; no Dataset creation
or failed kernel was repeated. Kaggle expanded its tar; an offline mounted-byte
check now verifies all800feature+4metadata files before new MRI. External download
checks were rate-limited and are not called successful full-download proofs.

The distinct dependency-corrected **139044146/v1 is RUNNING** on twoT4s, source
6c6ed23f5e7d32d824d22bf2bf282a319c4526efe7794589ce73127700c9e575. Its actual49.5s
log confirms all832saved input records passed before any new MRI encoding; only
3,055new studies remain. Pending CPU results are capped at4 to enforce RAM
backpressure. This is a frozen feature pass, not fitted-head accuracy.
[Actual source, failures and finite owner](../initiative/audits/released-context-full-cache-dataset-recovery-20261016-v3/README.md).

The complete FP32context production trainer **passed actual CPU935437 recovery**:
32realMRI tensors, three fresh full/stop1/resume processes,2epochs/4updates;
complete head/AdamW/LR/order/counters/allRNG/history and prediction bytes matched.
The branch learned nonzero weights. Fixed full recipe:402,956parameters,
3,887safe training studies,16study batches/two views,10epochs/2,430updates,
AdamW3e-4/WD.01, one-epoch warmup/cosine,clip1,25%source+75%parent anchoring.
The once-only finite owner will collect the accepted suffix and start that
same full engine after the feature pass completes. **Full fitting has not started.**
[Real recovery receipt](../initiative/audits/released-context-head-engine-20261016-v1/README.md).
Final learned-head T4 serving qualification and actual competition outcomes remain
required; a software or recovery pass is not an improvement in score.

Sourcea2558281 and53smallobjects/462927B were privately fully re-downloaded/size/SHA
verified at HFaf4bf28765f81aaf5819cb8ca4b8f3726b66d7c8 byCPU935355. Never repeat.
[Exact preservation](../initiative/audits/hf-context-features-paired-20261016-v1/verified.json).
Independent Kaggle work uses actual account quota under the owner's later removal
of the old research cap; healthy Slurm jobs/holds and daily competition limits
remain intact. No paid upgrade or private clinical/feature publication occurred.
