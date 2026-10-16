# Current project state

**Checked October 16, 2026, 19:02 UTC.** This is the current research entry point.
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
The authoritative 16:10 UTC allowance is **five submissions used, zero available**.
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
| Source-neighbor challenger | **932957** | **COMPLETED 0:0**, evc33 | Geometry-valid source-neighbor donor channels, same centers |
| Positive-weight challenger | **933001** | **RUNNING**, evc33 | Fitting-derived positive BCE term weights |
| Expanded 2,592-study recovery | **933382** | **PENDING**, evc33 | Engineering diagnostic, not the expanded full fit |

Neighbor932957 completed16epochs/4,496updates; all16epochs are privately fully
8-objectdownload/size/SHAverified. CPU936773 verified the complete fixed-final
model/source/roster/predictions. Internal macro0.91519773versus original0.92092535,
delta-0.00572762, whole-component conditional95%[-0.02522802,+0.01337630],
2,836defined/1,164undefined draws. No established primary gain; keep originalN3.
[Fixed-final result](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md). Both challengers passed actual H100
full/stop/resume checks of model, BatchNorm, optimizer, scheduler, RNG and predictions.
Those are engineering results, not challenger accuracy gains. The weighted fit is RUNNING and epochs1–16 are privately verified; no weighted final result yet. [Neighbor](../initiative/audits/compact-source-neighbor-full-fit-20261016-v1/README.md),
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
Its original H100 recovery933382 is queued. New source0d1adeaa production/preservation controls passed30targeted tests/Ruff/Mypy and actual CPU936315 staging/funding review. Finiteowner51853 waits for the exact original actual acceptance, then dispatches one full fit/preserves all16epochs; it reserves no waiting GPU. Full expanded training is not yet dispatched. [Conditional route](../initiative/audits/compact-expanded2592-production-20261016-v1/README.md).
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

## Sequence context: completed fitting and serving acceptance

The exact bounded two-T4 feature pass139044146/v1 **COMPLETED all3,887studies** in3,519.157seconds. It reused832verified originals and encoded only3,055newcases, with3.90GBparentRSS and3.19GBpeakGPU/device. CPU936070 collected the new2.37GBarchive once, verified every member and admitted the full fitting-only source boundary. Original arrays/UID/index stay private. [Full cache proof](../initiative/audits/released-context-full-feature-accepted-20261016-v1/README.md).

The exact full engine2ee0cffe **COMPLETED as CPU936070**: all10epochs/2,430updates on3,887studies in430.943seconds,402,956FP32context parameters, batch16studies/two views. It preserves the licensed0.949parent and learns a bounded±0.5logit sequence correction;25%source/75%detached-parent anchoring. This is completed training, not a new accuracy result. The attributed parent was developed using our internal cohort/Gold58, so no independent internal OOF is claimed. Protected clinical values remain unused by us. [Exact engine/admission](../initiative/audits/released-context-head-engine-20261016-v1/README.md).

The strict final exporter/serving compiler is implemented. CPU935892 passed actual32MRI nonzero-head serving math (maximum probability difference5.96e-8) and correctly rejected the diagnostic checkpoint as a deployablefinal. Actual final export/CPUreference passed. All10epochs+final+slimcomponent/license/source/runtime are now privately fully download/hash verified at HF81d542359d3c06435367c173a2b9a2947f561b1f (34objects/57.00MB). A pre-intent publication path-type error was corrected without repeating the fit; the original courier misleading success is preserved. Private final-head Dataset12636011v1 is fully6-object verified. Exact privateoffline qualification139054096/v1 **COMPLETED on bothT4s**, maximumCPU/GPU difference1.192e-7. FullMRI/CSV notebook139054287/v1 **COMPLETED in22.319seconds**,3pairs/zerohead fallbacks/zeroempty studies;13columns/470CSVbytes fully verified. Existingfinite62742 is terminalsuccess. One new finiteowner33015 waits for the next authoritative allowance inspection atOctober17 00:03UTC, then submits the exact acceptedversion ONCE and observes its actual hidden outcome. It occupies no waiting cluster/GPU resource and retries no mutation. No context entry/score exists at16:12UTC. [Actual deployment/nextentry](../initiative/audits/released-context-deployment-20261016-v1/README.md). [Fixed-final proof](../initiative/audits/released-context-fixed-final-20261016-v1/README.md). [Serving gate](../initiative/audits/released-context-serving-preparation-20261016-v1/README.md).

## New measured sampling mechanism

A label-free importance proposal targets windows that stabilize the strong frozen teacher's full-study MIL readout. Fixed25%uniformcoverage/75%normalizedinfluence,16IIDwindows with inverse-proposal correction,128repeats. The32-study screen reduced teacher-approximation MSE42.0%; the unchanged full3,887study repeat **reduced it45.6%**, ratio0.54401685, with all12targets improving. It took31.3seconds/oneCPU and reused the existing feature cache. No labels/reports/Gold/confirmation or newCNN inference were used. [Exact results/limits](../initiative/audits/licensed-MIL-sampling-20261016-v1/README.md).

This is a concrete training-exposure hypothesis, **not disease accuracy or an independent validation result**. Teacher attention is not verified anatomy, and computing its proposal still needs the full teacher cache. No guided CNN training has been dispatched; matched same-input/initialization/compute evidence remains necessary before promotion.

Latest prior small milestone sourceccb85232 is privately fully verified at HF3b6ccde8c36723627c9de1796137b59907ea10ce (36objects/165,794bytes). Never repeat that or earlier mirrors. New serving/sampling/full-feature/final-head/T4/publicMRI evidence and the single next-allowance owner are now separately private-HF verified at e9d200fbbc9d6fcc9d12838279ceaf28bd75a35a:64objects/269,278bytes from source4428247b, all fully re-downloaded/size/SHA under the canonical lock. [Verified small milestone](../initiative/audits/hf-context-final-sampling-20261016-v1/verified.json). Never repeat.

Actual label-free post-fit comparison on3,887fitting studies confirms the context head reverses0.935%–3.829%of strict study-pair orders across targets. It is not only a monotone class calibration. These are behavioral changes, **not proof that they are correct or that accuracy improved**. [Actual displacement](../initiative/audits/context-rank-displacement-20261016-v1/README.md).

## Spatial context: matched full fits actually training

The corrected probe **139066640/v1 completed all 32 MRI studies on both T4s** in
54.176 seconds. The observer left original features/logits bit-equal; CPU/GPU
regional reduction error was 1.526e-5 and FP16 storage error 4.879e-4. CPU936644
collected the 111.05 MB archive once and verified every member, shape, dtype,
finite value and original window metadata. Earlier cache mismatches and the failed
first probe remain in the [failure audit](../initiative/audits/regional-feature-probe-20261016-v1/README.md).
The corrected input stores global features, spatial summaries and parent logits
from the **same forward pass**. It does not combine old embeddings with new regions.
[Actual acceptance](../initiative/audits/regional-feature-probe-20261016-v2/README.md).

Full private offline two-T4 pass **139068442/v1 COMPLETED all3,887studies**
in4,382.071seconds, reusing32 and encoding3,855new cases. BothT4peak3.185GB;
original pixels/observer outputs passed exact checks. Coherent global/regional/
parent outputs come from one forward; no model fitting occurred. Its13.378GB
archive transfer was interrupted twice; received prefixes were preserved. Distinct
bounded-suffix CPU937898 recovered only missing bytes and verified whole SHA,
every3,855new member and exact3,887membership. No GPU/kernel/push was repeated.
**Full cache collection is accepted; no spatial accuracy result exists yet.**
[Full source, funding and owner](../initiative/audits/regional-feature-full-20261016-v1/README.md).

Neighbor932957 is COMPLETED and independently verified:0.91519773internal macro,
originalN3control0.92092535; no established primary gain (interval includes zero).
Weighted933001 has16privately verified epochs and remains RUNNING, no final result.
Original expanded recovery933382 remains PENDING; conditional owner51853 and
context-entry owner33015 stay alive. Root implements directly; Devin is unavailable.
[Neighbor full paired result](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md).

Verified private preservation: sourcee2fd3911/all51smallobjects235,995bytes at
HF9d593e484d0fde760381e12a7c606e2f84fb32a4, fully re-downloaded/size/SHAverified
once under the canonical lock byCPU936730. No private arrays/images/identities/
weights in the small mirror. [Exact receipt](../initiative/audits/hf-regional-expanded-20261016-v1/verified.json).
Do not repeat this prefix or earlier verified milestones.

CPU936870 verified the actual final context head on old/new globals from32MRIcases:
max mean-TTA probability drift0.0001293 (parent0.0001238), within the existing
0.003precision bound. No newCNN/GPU/labels/fitting; cause of cross-run feature drift
and full-population bound remain unproven. Scheduled accepted context entry is
unchanged. [Actual targeted check](../initiative/audits/context-cross-cache-numerics-20261016-v1/README.md).
Neighbor fixed-final sourcea44f476a and10smallobjects are private-HF fully verified
ONCE at fd2b776e0bc977cdac4dc7730d2e894d1bfbfb32 (CPU936842).
[Verified preservation](../initiative/audits/hf-neighbor-fixed-final-20261016-v1/verified.json).

The matched spatial-head engine passed actual CPU937321 **COMPLETED0:0**:
six fresh full/stop1/resume processes, four updates per arm on32real MRI caches,
complete model/AdamW/LR/counters/allRNG/history and32x12prediction bytes exact.
Both roles learn nonzero changes. Same1,334,796-parameter8192-input graph and
initial state; control repeats globals8times, candidate uses globals+7regions.
The original pre-update verifier refusal is preserved; same-batch CPU oracle
corrected shape-dependent5.96e-8rounding without changing numerical training.
[Actual recovery](../initiative/audits/regional-context-engine-20261016-v2/README.md).

CPU937917 admitted all3,887typed inputs and fresh storage. Full10epoch/2430update
jobs **global_repeat937942** and **regional937975** are actually RUNNING on evc33,
oneCPU/zeroGPU each. Read-onlyCPU938007 saw175finite updates/empty stderr for both.
Finiteowner34894 preserves all10immutable epochs per successful role privately once.
The accepted source46f5014d and unchanged5addea82numerical loop are frozen.
512MiBtotal head/recovery cap and all prior resource liabilities are retained.
The separate typed final exporter/FP32readout now passes33focused tests/Ruff/Mypy;
the separate compiler now passes42focused tests and compiles the actual licensed
parent for both roles/modes. Actual final export and T4spatial execution remain. No independent
internalOOF is claimed for the externally trained parent.
[Prospective full controls](../initiative/audits/regional-context-production-20261016-v1/README.md).

Verified latest small preservation: source6c06674d/all49objects236002bytes,
privateHFc6e848a5e21090b2cf3b462680bf8166a064aee0. Full re-download/size/SHA
under the canonical lock byCPU938041; no private arrays/images/identities/weights.
[Exact receipt](../initiative/audits/hf-regional-actual-full-head-20261016-v1/verified.json).
Never repeat this prefix or preceding verified mirrors.

Actual spatial-readout CPU938126 passed both original32MRIfeature roles against
original learned engineering predictions (max1.192e-7/5.96e-8); diagnostic export
correctly refused. New finite owner92853 waits for original full-fit/private20epoch
completion before actual final export, combined Dataset and exact T4/publicMRI
qualification. No waitingGPU, public push or hidden entry is implied.
[Original conditional route](../initiative/audits/regional-context-deployment-20261016-v1/README.md).
Weighted933001 has all16privately verified epochs; one independent fixed-final
comparison observer waits for its actual terminal result, not a checkpoint-only
score. [Original protocol](../initiative/audits/compact-positive-weight-fixed-final-20261016-v1/README.md).
