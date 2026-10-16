# Current project state

**Checked October 16, 2026, 23:48 UTC.** This is the current research entry point.
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
The authoritative 20:31 UTC allowance is **five submissions used, zero available**.
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
| Positive-weight challenger | **933001** | **COMPLETED 0:0**, evc33 | Same16epochs/4,496updates; positive BCE weights |
| Expanded 2,592-study full fit | **938284** | **PENDING Priority**, evc33 | +345fitting cases,16epochs/5,184updates |
| Spatial global-repeat control | **937942** | **COMPLETED 0:0**, evc33 | Same1.33Mhead,8global copies |
| Spatial regional candidate | **937975** | **COMPLETED 0:0**, evc33 | Same1.33Mhead,global+7summaries |

Neighbor932957 completed16epochs/4,496updates; all16epochs are privately fully
8-objectdownload/size/SHAverified. CPU936773 verified the complete fixed-final
model/source/roster/predictions. Internal macro0.91519773versus original0.92092535,
delta-0.00572762, whole-component conditional95%[-0.02522802,+0.01337630],
2,836defined/1,164undefined draws. No established primary gain; keep originalN3.
[Fixed-final result](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md). Both challengers passed actual H100
full/stop/resume checks of model, BatchNorm, optimizer, scheduler, RNG and predictions.
Weighted933001completed5:32:28; all16epochs are privately verified. CPU938273
replayed exact final/model/source/rosters/predictions and all12targets: internal
macro0.92391334vs0.92092535, delta+.00298799, whole-component conditional95%
[-.00999470,+.01656811],2,836defined/1,164undefined draws. **No established primary
gain; keep ownN3E16.** MCLpoint gain has only2positives; several other targets
regressed. No weighted Kaggle score. [Actual fixed-final result](../initiative/audits/compact-positive-weight-fixed-final-20261016-v1/README.md). [Neighbor](../initiative/audits/compact-source-neighbor-full-fit-20261016-v1/README.md),
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
Original H100recovery933382 COMPLETED0:0/7:56 with complete real model/BN/optimizer/RNG/prediction recovery. Existing finiteowner51853 submitted full938284 ONCE after exact acceptance and fresh funding; it is PENDINGPriority. The same pending job now has an8h scheduler limit based on actual5:32:28same-graph runtime scaled for5,184updates; source/recipe/output unchanged. Its existing watcher preserves all16epochs. No full expanded updates or result yet. [Conditional route](../initiative/audits/compact-expanded2592-production-20261016-v1/README.md).
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
A separate CPU-only probe measured four cores and20.94GBscratch. All four full-population CPU cache shards have completed; actual full-parent139111324v1 is training on bothT4s with43finite updates observed. [Capacity receipt](../initiative/audits/kaggle-CPU-training-capacity-20261016-v1/README.md).

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

## Spatial context: full fits completed and actual T4/public paths accepted

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
Weighted933001has its verified fixed-final result above, no established primary
gain. Expanded full938284is PENDINGafter actual933382acceptance; its original
owner51853 and context-entry owner33015 remain alive. Root implements directly; Devin is unavailable.
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

CPU937917admitted all3,887typed inputs and fresh storage. Both fixed10epoch/
2,430update jobs completed in31:41/31:55. **All20immutable epochs plus final/
source/license/runtime are privately fully download/size/SHAverified**: control
HFfc5424032b222c52304bdc91b83bee514bff6931 and regional
HF133aa9f672ee5546265c8455fa7dc2d5a5feebb3,38objects per role. Original34894is
terminalsuccess; do not rerun it. [Actual full receipts](../initiative/audits/regional-context-production-20261016-v1/README.md).

Strict actual final CPUexports produced two5.354MBcomponents and tiny32x12
references, each matching its complete batched learned graph within5.96e-8.
One combined private Dataset **12639590v1** is fully9-object download/hashverified.
Both final heads passed **actual twoT4 precision**, max probability error1.192e-7,
then their exact private offlinev1 raw-MRI public-example notebooks completed:
regional28.051s and global-repeat25.964s,3studies/zero fallback/zeroempty/13columns,
CSV+test.csv roster/bytes/hash verified. This does not establish full hidden
runtime or an accuracy improvement. **Both remain unsubmitted/unscored.**
[Actual serving and receipts](../initiative/audits/regional-context-deployment-20261016-v1/README.md).

The separate entry owner49213/PID69887 is live: retain old context33015's exact
next-allowance entry; require its actual scored hidden outcome, then regional no
earlierOctober17 02:00UTC and global-repeat no earlier06:00UTC, each subject to
fresh allowance and the prior actual score. Stop on actual hidden failure; max2
new entries and no duplicate versions. These are spacing windows, not a claimed
server reset. Remaining daily capacity is available for separately justified
candidates. No waits occupy a Slurm/GPU allocation. Sources/policies stay frozen.

Verified latest small preservation: source6c06674d/all49objects236002bytes,
privateHFc6e848a5e21090b2cf3b462680bf8166a064aee0. Full re-download/size/SHA
under the canonical lock byCPU938041; no private arrays/images/identities/weights.
[Exact receipt](../initiative/audits/hf-regional-actual-full-head-20261016-v1/verified.json).
Never repeat this prefix or preceding verified mirrors.

Latest verified small milestone: source16dc318e/all59objects242,393bytes at
privateHF56671edc96e98252a5183f6cfaefcb955916017a, fully re-downloaded/size/SHA
under the canonical lock byCPU938456. No private arrays/images/identities/weights.
[Exact receipt](../initiative/audits/hf-spatial-T4-weighted-final-20261016-v1/verified.json).
Never repeat this or earlier verified prefixes.

## Source-confidence contrast — actual completed20:23UTC

OriginalCPU938522completed both3887-study/10-epoch/2430-update global-context
fits in444seconds per role. Same402,956parameters/data/seed/initialization/
teacher/compute, only source influence changes. Seven actual fresh32-MRI
processes passed recovery. The complete full uniform state/history/prediction
bytes match the original control exactly, so there will be no duplicate uniform
competition entry. All20epochs/finals privately fully verified: uniform
HF775e9f5db25f9063f1c4662d2d4f1c3df8b903cc; confidence
HFf326b093db6b9733a51f04d16319ffe5fbce1f5b.

Measured current source semantics correct a tempting explanation: only28/46644
cells are exactly.5, not the historical source-v1's25.4%. Mean source influence
changes.25→.19512, Synovitis→.13581. Fitting pair-order changes.633%–2.472%
are behavior, not accuracy. **No confidence Kaggle/clinical improvement exists.**
[Full actual receipts, negative finding and limits](../initiative/audits/source-confidence-context-20261016-v1/README.md).

Actual final export/CPU math passed with zero probability error; component
d215637cb2e4is privately full verifiedHF28f29480. A local import-path failure
stopped the original controller BEFORE any Dataset attempt. Its distinct
continuation corrected only the environment and is now TERMINALsuccess. Private
Dataset12640289v1 is all6object verified; exact twoT4qualification139097480v1
passed max5.96e-8; publicMRI139097695v1 passed3studies/zero fallback/13columns
in22.381seconds. **No hidden entry/score exists.** [Original serving/failure/
acceptance receipts](../initiative/audits/source-confidence-deployment-20261016-v1/README.md).

New sole finite72900/PID72998 owns this fourth candidate: no earlierOctober17
10:00UTC, after all three original context/spatial actual scored outcomes, fresh
authoritative allowance and canonical claim; at most one exact code entry. Entry
deadline17 23:15UTC, hidden observation deadline18 10:00UTC, no mutation retry.
The20:31UTCaccount/ledger confirms5used/0live slots and all four unsubmitted.
Existing next-window entry owners remain
unchanged. [Next larger initialization/input-cache direction](../initiative/experiments/competitive-v1/licensed-parent-end-to-end-v1/README.md)
now has actual exact-input and full-model T4 engineering acceptance; see the
current whole-study adaptation section below. That20:35UTC snapshot predates the actual full-parent GPU launch described below.

Sourcea823bebe and36small source/result/ownership objects (251,672bytes) are
privately fully re-downloaded/hash verified at HF29421f9826d34cdf3ddcdaccc1c1d8ac33bcc252.
[Original preservation proof](../initiative/audits/hf-confidence-full-acceptance-20261016-v1/verified.json).
No original model/image/feature publication was repeated. Actual20:35UTC Slurm
confirms confidence938522COMPLETED0:0/16:27; expanded938284stillPENDINGPriority.
All original next-entry and expanded-preservation owners are alive.

## Current exact-parent whole-study adaptation

The stronger direction starts from the attributed MRI-trained parent that actually
scored0.949; our own ImageNet-initialized N3 best remains0.894. No new leaderboard
or clinical gain has yet been established by this adaptation.

Actual32MRI input gate139099964v1/CPU938918 passed: exact original96x384uint8
volumes/masks and both94-window FP32 tensors,326.80MBarchive,149.16seconds,
1.90GBRSS. It preserves the scored model's inputs without additional quality loss;
it does not preserve every raw DICOM slice. Two secondary slow-reader cases also
matched exactly. [Original proof](../initiative/audits/licensed-parent-pixels32-20261016-v1/README.md).

Both real twoT4 numerical recoveries are now COMPLETE:

| Path | Exact notebook | Actual result | Peak allocated/GPU |
|---|---:|---|---:|
| Original12sampled-window training |139104577v1| Full/stop1/resume exact |7.604GB|
| Whole94-window checkpointed training |139105348v1| Full/stop1/resume exact |5.973GB|

Both used16fitting+16disjoint engineering MRI cases,2epochs/4updates,
all73,136,096parameters trainable, complete model/base/AdamW/LR/cursors/allrankRNG/
GradScaler equality and identical16engineering two-view prediction bytes. Parent
BN buffers stayed unchanged. These prove engineering behavior, not accuracy.
The whole94path consumed416.87seconds for all three process flows; its full arm
168.15seconds includes engineering evaluation and cannot be treated as pure
training time. [12window receipt](../initiative/audits/licensed-parent-trainer-recovery-20261016-v1/actual-twoT4-accepted.json),
[whole-study source/proofs](../initiative/audits/licensed-parent-whole94-recovery-20261016-v1/README.md).

Whole-study MIL keeps all94original windows in each bag, checkpoint-recomputing
8-window encoder chunks to bound activation memory. Every encoder/head parameter
still trains. Inference uses the unchanged parent encoder and94two-view path.
The full production recipe is now running:3epochs/3887eligible fitting studies/
1458updates/effective8, low encoder/headLR3e-6/3e-5, AdamW/10%warmup-cosine/clip1,
.25original source+.75view-specific full-parent teacher, frozen parent BN buffers.
Protected Gold/confirmation remain unused. No independent localOOF is possible
for this exposed parent. CPU939476 staged the unchanged production sourcec68d8752;
CPU939655 joined all3887ordered fitting UIDs to the exact accepted target bundle.
Original **139111324v1**,godlydonuts/rsna-parent-whole94-full-e1-50b5a0d9c138,
was pushed ONCE and is actually RUNNING. Authenticated live logs show BOTH Tesla
T4s, all73,136,096parameters trainable and **43finite optimizer updates** in epoch1
at23:10UTC. No epoch is complete and no new accuracy/competition score exists.
[Actual live optimizer proof](../initiative/audits/licensed-parent-whole94-production-20261016-v1/actual-live-training-UI-43.json).

Live finiteowner80039/exec6555 continues up to3checkpoint-continuous seven-hour
sessions at486/972/1458steps, preserving one complete LR/optimizer/RNG schedule.
All four original private CPU cache notebooks139103622/139103665/139103730/
139103781 v1 are COMPLETE:972/972/972/971studies, zero decode errors/empty cases,
38,870,763,520new archive bytes plus326,799,360reused original32bytes. All3887
ordered fitting studies and both94-window input-view hashes passed. The trainer
mounts those immutable originals read-only without a second40GBworking tree.
Independent Newton backup/replay is now complete for ALL4shards via
CPU939550/939676/939777/939861, each original archive/member/both-view hash
verified once. Collector80537 completed successfully; do not restart it or repeat
any original producer or transfer. The complete private registry SHA is8cc5c194.
[Actual four sources/funding](../initiative/audits/licensed-parent-pixels-full-20261016-v1/README.md).

CPU939271 compacted all3887original labels/masks and both teacher-logit views
bit-exactly into483656bytes. CPU939291 created ONE private training Dataset
**12641441v1**,godlydonuts/rsna-parent-fit-targets-48b4485f03b6, all5objects fully
download/size/SHAverified. No encoder or optimizer ran in that preparation. The
private UID/label/logit arrays stay on cluster/Kaggle, not Git/smallHF mirrors.
[Full input bridge](../initiative/audits/licensed-parent-production-inputs-20261016-v1/README.md).

Original numerical/input engineering checks:52passed,Ruff/mypy clean, including fresh
2rank CPU recovery, exact whole94gradient/RNG parity and input/target roundtrips.
The new loader also passed actualCPU939362: all32original MRI cases/64views,
all94window inputs, source labels/masks and both teacher views matched bit-exactly
in37.01seconds, reading directly inside the original tar without a second pixel
tree. Full production additionally passed all4actual original Kaggle producer
proofs, exact weight/source/label bindings, accepted whole94GPU receipt and fresh
resource state. Its fixed-schedule sessions retain the original LR/optimizer
schedule across the three segments.

Latest exact-parent source and actual engineering/input receipts are privately
fully verified at HFa879e194d2646971b0b9238ce588abce64c2d03f:71small objects,
419747bytes, CPU939371/canonical shared lock/source9adce769. No MRI/UID/label/
teacher arrays/prediction rows/parent source body/weights. Do not repeat this or
prior verified prefixes. [Original proof](../initiative/audits/hf-licensed-parent-whole94-20261016-v1/verified.json).

Original-byte readiness and Newton backup are now separate: the full original
Kaggle receipts cover every newly encoded case and both94-window tensor hashes,
complete roster, zero decoding errors/empty cases, original source and whole-byte
hashes. The production wrapper joins all3887indices to the exact target bundle
before any GPU push; its loader also checks every accessed immutable member and
rendered view. This is not a claim that the Newton replica has already passed.
That independent backup continues in parallel and cannot hold the submission lock.

Production preservation watcher80311/exec52344 is LIVE: each actual completed
session triggers oneCPU/zeroGPU collection, independent full snapshot/model/
AdamW/history/two-rank RNG/scaler checks and private HF publication with complete
re-download/size/SHA/license/provenance. No production epoch exists yet. The new
full-population/session/epoch/range paths pass48focused checks plus7existing
loader checks, Ruff/mypy. These are engineering checks, not an accuracy gain.
[Exact full-production source, stage and ownership](../initiative/audits/licensed-parent-whole94-production-20261016-v1/README.md).

Latest full-production control/source milestone: sourcebe105c27, privateHF
b08293736906370bfc050aef5509c721a457e703,35objects/247805bytes fully re-downloaded/
size/SHAverified under the shared lock byCPU939577. No MRI/identities/labels/
teachers/prediction rows/parent body/weights mirrored. Never repeat this or old
prefixes. [Original preservation receipt](../initiative/audits/hf-licensed-parent-full-production-20261016-v1/verified.json).


## Exact learned-epoch serving and single comparison owners

CPU939778 actually staged the separate inference controls from source095d3a7d.
Live82392/exec1021 waits for the first actual completed, independently preserved
full epoch, then checks exact trained-wrapper/original-class state,32MRI cases,
all94two-view tensors and probabilities on bothT4s. It subsequently qualifies the
exact private offline public MRI/CSV notebook, reusing the immutable original
training output directly; no new model Dataset or repeated weight upload. No
serving GPU has yet been pushed because epoch1 is still training. Three actual
epochs/six notebooks maximum, quota waits reserve no GPU and create no intent.

Live82509 is a separate finite ONE-entry epoch1 owner. It waits for actual
private complete-model preservation and bothT4/publicMRI/CSV acceptance, the
original context comparison's actual scored outcome, no unresolved entries and
fresh authoritative daily allowance. It sends the exact code notebook ONCE,
then observes its actual hidden outcome. Maximum one entry; no three-row CSV
upload, diagnostic model or uncertain-mutation retry. The existing spaced head
comparisons remain unchanged. Inspect each original intent/result before action.
[Actual stage, owners and source](../initiative/audits/licensed-parent-epoch-serving-20261016-v1/README.md).


Actual full-model training and learned-epoch inference/entry milestone source
63952698 is privately fully re-downloaded/size/SHA verified at HF
5c7d47eb24b32ff07d342622733c303fadf06634:30objects/217900bytes,CPU939848,
canonical shared lock. No MRI/identifiers/labels/teachers/prediction rows/parent
body/weights mirrored. Never repeat this or earlier prefixes.
[Verified preservation](../initiative/audits/hf-licensed-parent-live-training-serving-20261016-v1/verified.json).


## New study-pair objective: complete fixed-final model

OriginalCPU939891 COMPLETED0:0/8:13 on evc33. Its actual32MRI-feature seven-process
recovery and uniform frozen-engine numerical equality passed. The distinct ranking
candidate completed all3887studies/10epochs/2430finite updates in417.42seconds,
402956parameters; no CNN was re-encoded or fitted. The original full uniform fit
was reused, not repeated. Half of25%source BCE becomes study-pair logistic risk
on the actual mean-of-two-view probabilities; the75%teacher contribution remains.
Same graph/features/labels/masks/seed/order/update budget. No clinical or
independent internal accuracy claim follows from that completed training.

All TEN immutable epochs were privately fully re-downloaded/size/SHA verified
ONCE at HF9f27825ef383b30bf5c77486e09ada62165be991:40objects/55,403,596bytes.
Final checkpoint e0fc7c891d2d9f00ad9fb37aa4de27c239e1ef6f41c849e5cea77fde9e768a05.
ActualCPU939970 accepted strict fixed-final export, matching the trained graph
with zero probability error on the original32MRI feature cases. The1,626,295byte
component6e0894d90343 is separately private/full verified at
HFca34866401e2e68a87dd8b1f0dcf517d0d14856b:19objects/1,723,620bytes.
Never repeat this fit, either publication, acquisition or full uniform control.

Private Dataset12642673v1,godlydonuts/rsna-ranking-head-e10-6e0894d90343, is fully
6object download/hash verified. Exact private offline precision139119567v1,
godlydonuts/rsna-ranking-precision-6e0894d90343, actually COMPLETED on BOTH T4s:
all32MRI feature cases, maximum CPU/GPU probability error1.1920928955078125e-7,
no new CNN encodings. This is engineering acceptance, not disease accuracy.

Original84212 was stopped ONLY while idle in its confirmed pre-public quota wait;
no public notebook intent/push existed. A SaveKernel response alone does not
prove an entire future timeout is reserved by account stats. Distinct finite
public-only owner84548/exec49246 therefore retains7hours for EVERY actual-unfinished
original full-model epoch plus1hour reserve before its optional9hour public push.
It reuses the exact original Dataset/qualified version/model/source; maximum
ONE public notebook,zero newDatasets/precision pushes/competition entries.
No mutation intent on quota waits; never restart the old84212 or repeat qualified
139119567. Public MRI/CSV and a ranking competition result do not yet exist.
29focused math/recovery/export/compiler checks and Ruff/mypy6sources passed. [Protocol and primary papers](../initiative/experiments/competitive-v1/source-ranking-context-v1/README.md),
[actual original proofs/owners/funding](../initiative/audits/source-ranking-context-20261016-v1/README.md).

The original full CoAtNet139111324v1 independently continues on bothT4s,113finite
updates last verified in epoch1. All four original lossless pixel caches are now
also independently collected/replayed on Newton:3887studies,3.204MBprivate
registry SHA8cc5c194d4795f6ab574c28b2cf8eed510c5f92fc6b5baff63049d16323fb1df.
Original collector80537 is TERMINALsuccess; never restart it or repeat transfers.
Total archive bytes39,197,562,880including the unchanged original32. Zero decode
errors/empty cases; all pixels/masks/two-view94-window hashes match.

The rank experiment actually funded all prior607095306179required bytes plus
new256MiBcheckpoint/publication+32MiBcontrols, leaving25.494GB after liabilities.
Raw retained checkpoints138,599,668bytes stay private. No deletion credit or
shared growth. Both existing main training and all prior entry owners remain
unchanged; choose future exact ranking entries only after actual acceptance and
relevant outcomes/fresh daily allowance.


Latest ranking/full-pixel/source milestone4952238a is privately fully re-downloaded/
size/SHA verified at HFa9e666390b030b2b2c42c93f024115ed8395c73e:48objects302252bytes,
CPU940064/canonical shared lock. No MRI/UIDs/labels/features/weights/prediction rows;
never repeat prior prefixes. [Exact verification](../initiative/audits/hf-ranking-full-fit-T4-pixels-20261016-v1/verified.json).
