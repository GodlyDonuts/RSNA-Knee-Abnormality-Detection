# Current project state

**Observed through October 17, 2026, 05:55 UTC.** This is the live-state entry point.
Read the [research brief](research-brief.md), [research-agent handoff](research-agent-handoff.md)
and [execution handoff](codex-continuation.md). Original receipts and fresh actual
state override dated narratives. Historical evidence remains in Git and the
[earlier snapshot](current-state.history-20261017-0146.md).

## Actual competition results

| Candidate | Submission | Public AUC | What it establishes |
|---|---:|---:|---|
| Author-trained nartaa CoAtNet, unchanged inference | 57258866 | **0.949** | Reproduced external weights/inference, not our training |
| Our independently MRI-trained N3, E4 / E8 / E12 / E16 | 57256756 / 57259376 / 57262392 / 57265916 | 0.821 / 0.869 / 0.890 / **0.894** | Own MRI training improved with epochs |
| Licensed parent + our sequence-context E10 head | 57287572 | 0.948 | No gain over unchanged parent |
| Unchanged licensed parent, full384 field | 57289392 | 0.948 | More inference field alone did not help |
| Licensed parent + regional spatial E10 head | 57291182 | 0.948 | No demonstrated gain |
| Our learned whole94 parent, E1 | **57294905** | **0.949** | Actual hidden completion; same public score as parent, no demonstrated gain |
| Earlier own DINO | 57182737 / 57182547 | 0.872 | Historical own-model reference |

These are public leaderboard AUCs, not clinical accuracy or private-final scores.
**We did not train the 0.949 model from scratch.** Our N3 started from general
ImageNet pretraining; learned-parent adaptation is a distinct third category.
The latest saved official Public leaderboard is **0.969 first**, leaving a
0.020 gap. Shared0.950 solos/0.956 blends are catch-up targets, not the frontier;
the leader's method/private score is unknown.
[Leaderboard observation](../initiative/audits/github-current-state-refresh-20261017-v2/actual-public-leaderboard.json).

New E1 entry57294905 is exact private offline139154038v1,
`godlydonuts/rsna-parent-adapted-e1-offline-b29b8a1d08f6-g2`, modelSHAa8ed4a84.
Its public MRI run passed three studies/13columns/zero failures on both T4s.
The public3row CSV was **not uploaded**. It matches the original parent's three
rank rows, which neither proves identical learned weights nor predicts AUC.
Original owner97560 observed its actual hidden terminal: **COMPLETE/0.949/noerror**,
247103outputbytes. This matches the parent without establishing a gain. Never repeat this entry
or the quarantined unsuffixed capacity-rejected slug.
[Actual public acceptance, entry and hidden scored terminal](../initiative/audits/research-progress-20261017-0448-v1/README.md).
The dated earlier3today/2slots observation preceded this entry; obtain live
allowance before another mutation, maximum five/day. Remaining final E3 is owned.

Original scored receipts: [external baseline](../initiative/audits/released0949-baseline-20261016-v1/actual-scored-canonical.json),
[own N3E16](../initiative/audits/compact-production-e016-kaggle-20261016-v1/actual-scored-canonical.json),
[context](../initiative/audits/released-context-deployment-20261016-v1/actual-scored-recovered.json),
[field](../initiative/audits/licensed-native-field-20261017-v1/actual-scored-result.json).
Do not resubmit those exact versions.

## Active fitting and actual completed epochs

| Work | Original identity | Latest verified state |
|---|---|---|
| Whole94 licensed-parent adaptation | E1 139111324v1; E2 **139137922v1** | E1 complete486, all5private objects verified; E2 RUNNING, last visible722/1458 finite updates, bothT4s |
| Expanded own N3 | Slurm**938284**, evc33 | RUNNING1H100/2CPU/8h limit; **3458/5184 finite updates, epoch11**; **epochs1–10 privately verified** |
| Uniform16 full-encoder control | E1**139137040v1** | **E1 complete486**, whole state verified; all5private objects HFc19961db0f377086b4a2fd12f6ec477213a1da7d |
| Influence16 full-encoder contrast | E1**139154664v1** | **E1 complete486/all5private objects verified**; original owner97530 now runs uniformE2**139167076v1** |
| Original patch recovery | Slurm929938 | COMPLETED0:0, full/stop1/resume engineering equality; not accuracy |
| S6 unchanged-source reproduction | Qualification**139165932v1**; owner**20978** | **COMPLETE156.6s/twoT4/threeMRI/13columns**; remaining offline run and sole entry20994 wait capacity |

[New original epoch/publication/push evidence](../initiative/audits/research-progress-20261017-0448-v1/README.md),
[earlier optimizer and patch terminals](../initiative/audits/research-progress-20261017-0401-v1/README.md).
Actual CPU942744 now observed3458/5184 finite updates/epoch11. E9/E10 publication
proofs establish completed immutable epochs1–10, each privately fully verified.
InfluenceE1 completed486 and all5objects are independently private verified;
uniformE2 is now API RUNNING under the resumed same owner97530.
[New actual evidence](../initiative/audits/research-progress-20261017-0550-v1/README.md).
The original GPU and watcher95408 remain healthy. Never repeat publications.

Whole94 E1 was independently complete-state verified and privately preserved
at HF**d436986137570a9a4259a4b59981d261cffd2f20**, all5objects full download/size/SHA.
Learned-state qualification139136981v1 passed32originalMRIs on **both T4s**, exact
input/model equality and maximum prediction difference0. Its original worker
resumed E2 unchanged after a metadata-JSON key-order-only compiler repair.
[Actual worker continuity](../initiative/audits/whole94-original-worker-continuation-20261017-v1/README.md).

The full-parent recipe is **3887 fitting studies/all94windows**, all73,136,096
parameters, effective8, **3epochs/1458updates**, encoder/headLR3e-6/3e-5,
AdamW0.02,10%warmup/cosine-to1%,clip1,FP16/scaler/FP32loss. The objective is
25%source supervision/75%detached-parent anchoring. BN running buffers stay
fixed; affine parameters learn. [Exact recipe](../initiative/audits/licensed-parent-whole94-production-20261016-v1/README.md).
The E1 entry is a trajectory check, not best-epoch selection.

Expanded N3 adds345 fitting studies to **2592 fit/1156 disjoint outer**, keeping
the original recipe and16epochs/**5184updates**. Data and update count both
change, so it is not a controlled data-only contrast. No fixed-final result exists.
[Original recovery/production](../initiative/audits/compact-expanded2592-production-20261016-v1/README.md).

## Matched importance-guided observations

The original uniform16 versus influence16 comparison matches initial full model,
all-rank RNG, study order/views, targets, optimizer, draws and fixed1458updates.
It uses16IID windows with replacement/common variates, inverse-proposal attention
correction and **full94 inference**. The fixed influence proposal combines25%
uniform coverage with75%teacher influence. On all3887 fitting cases, it reduced
frozen-teacher readout approximation MSE **45.6%**, every target improving. That
is teacher fidelity, not disease accuracy or a measured training speedup.

Actual two-role GPU recovery139130771v1 passed complete model/AdamW/LR/counters/
all-rank RNG/scaler/history and16×12 full94 CSV equality. CPU940795 matched32
originalMRI inputs/both views/repeated draws; proposal Dataset12644218v1/all5
objects was privately fully verified once. UniformE1 is now actually complete
and preserved; influenceE1 runs under the same original comparison owner.
The original influenceE1 now also completed486 and all5objects are private verified
at HFb09816181e2e79eb24563884af5c7ed0e1af40a5; uniformE2 runs.
Sampled-model serving remains synthetic preparation, with **no actual learned
sampled-checkpoint serving T4/public acceptance or score**.
[Protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md),
[GPU recovery](../initiative/audits/licensed-importance-encoder-20261017-v1/actual-terminal.json),
[production](../initiative/audits/licensed-importance-full-comparison-20261017-v1/README.md),
[serving preparation](../initiative/audits/licensed-importance-serving-preparation-20261017-v1/README.md).

Kaggle actually permits **two batch GPU sessions**. Main continuity retains one
lane; sampling and qualified inference share the other. Existing canonical
capacity/VM admission preserves healthy work and all unfinished liabilities.
No apparent spare competition slot creates a spare GPU lane.

## New research results: useful negatives, not score gains

CPU942278 screened the **existing** within-series policy on3887 fitting feature
bags without labels/reports/pixels/new CNN/GPU/fitting. All cases had mean8.0003
crossing windows of94. Across12targets these windows carry only **0.35–1.54%**
mean attention; removing them reverses **0.074–0.494%** of49822 deterministic
strict study pairs. Count-matched ordinary interior removal changes **0.321–1.511%**.
This deprioritizes a new boundary-only training arm: the strong parent already
suppresses these windows. It does **not** prove a clinical gain or rule out rare
large effects. Existing within-series/edge-repeat public notebooks are already
accepted but unscored; do not recreate them.
[Protocol, all-target result and limits](../initiative/audits/frozen-series-boundary-screen-20261017-v1/README.md).

The earlier label-free teacher-view screen had mean probability gaps1.29–2.71
points and strict-pair discordance2.75–7.41%, at most4.37%cases/target above10
points. Its predeclared triage was **intermediate**; no new GPU arm or target
change followed. Those teacher pairs are not clinical positive-negative AUC errors.
[Actual screen](../initiative/audits/teacher-view-consistency-screen-20261017-v1/README.md).

## S6 second lineage: actual T4 public acceptance, no hidden score

Exact author138285808v2/sourceSHA6e87350d, Dataset12589750v3/modelSHA0c7297d5,
293-member source tree/anatomy and CPU classifier references remain pinned.
Original qualification **139165932v1** actually COMPLETED on **both T4s in156.6s**,
three public MRIs, strict50M graph, zero S6 fallback and a finite13-column CSV
(626bytes/SHA00031c53). Both original CoAtNet and S6 paths executed. The public
CSV was not uploaded, and this is not a hidden runtime or score.

Our observer stopped because the unchanged author's baked build ID is stale.
Independent execution of its build-ID function on the exact original source
reproduced the mismatch. The repaired observer requires exact source SHA plus
that exact baked/derived pair and all original device/model/CSV checks; no GPU
rerun or numerical source change occurred. Canonical acceptance is reconciled.

S6 admission also now counts the actual already-granted45h account allowance,
not just its30h minimum. Used/reserved time, all future training, serving reserve
and actual two-session limit still govern. No billing/account setting changed.
Remaining offline-run owner **20978/44859** waits the auxiliary lane; sole entry
owner **20994/20982** waits actual offline acceptance and fresh allowance. Neither
has submitted S6. Original H100942047 remains pending.
[Exact acceptance, quota correction and owners](../initiative/audits/s6-original-t4-recovery-20261017-v1/README.md).

A separate CPU-only eight-fitting-study input bridge139165316v1 now COMPLETED;
CPU942782 collected once and verified all5372windows/1,154,209,297compressed bytes.
Original owner20144 is terminal success. A distinct twoCPU/zeroGPU classifier
dispatch reuses them, with no actual terminal or accuracy result yet. It uses a UID-only roster view and unchanged physical/anatomical
functions after a preserved first guard-contract failure139162434v1. No labels,
classifier predictions or clinical pass is claimed; fixtures remain private in
the existing2GiB pool. This prepares a cross-lineage comparison without waiting
for a GPU lane. [Protocol, failure and nine checks](../initiative/audits/s6-fitting8-uid-view-20261017-v1/README.md).
The author's0.950 remains external; no project S6 score exists yet.

## Completed fixed-final own-model comparisons

All original N3, neighbor and positive-weight checkpoints are privately fully
verified. Same1156 weak-development studies:

| Fixed-final comparison | Macro AUC | Delta | Conditional95% whole-component interval |
|---|---:|---:|---|
| Original own N3E16 | 0.92092535 | Reference | — |
| Physical source-neighbor channels | 0.91519773 | -0.00572762 | [-0.02522802,+0.01337630] |
| Positive BCE weighting | 0.92391334 | +0.00298799 | [-0.00999470,+0.01656811] |
| E12/E14/E16 parameter average, raw serving | 0.92100408 | +0.00031564 vs raw E16 | [-0.00214034,+0.00293081] |

No primary improvement established; keep original own N3E16. Averaging reference
is raw E16**0.92068844**, not differently served0.92092535. These comparisons had
2836defined/1164undefined draws; MCL/fracture have2positives each, Baker's3,
effusion/lateralOA9each, synovitis5negatives. Internal metrics are weak-label
agreement, not clinical validation or predicted Kaggle AUC.
[Neighbor](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md),
[weighting](../initiative/audits/compact-positive-weight-fixed-final-20261016-v1/README.md),
[averaging](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md).
RemovingSelection_Tier changed zero series choices on all1156 outer cases;
complete rendering parity remains separate. Context/regional/full-field0.948
also fail to improve0.949. Confidence/ranking-head experiments are mapped in the
[research-agent handoff](research-agent-handoff.md).

## Provenance, exposure and operations

Own N3 is fully fine-tuned CoAtNetRMLP2, not a frozen DINO head. Original2247fit,
384pixels/up-to6series/16selected-slices,12training windows,16epochs4496updates,
encoder/headLR3e-5/1e-3,AdamW0.02/OneCycle/clip3/no augmentation.
Source**dcfff319cb7bd7bbae679a20da2b1b9268e8d680**.

Released parent Dataset12431568v1/modelSHA7e5315da uses96slices in26/22/18/12/18
slots,94triples,140mmcrop,384preparation/320execution,mirrorTTA/per-target ranks.
Author reports4349RSNA/blended labels/2399OAIknees/selected averaging. We did not
reproduce that full training or acquire controlled OAI. Permission/attribution
remain bound. [Exact contract](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md).
Both released encoders exposed internal studies; head-only folds do not create
independent whole-model OOF. Protected clinical values remain unused by us.
Stanford/KneeCoT remain separate private research, not the eligible scored weights.

All3887 original fitting pixel producers/lossless archives/node39 replica are
already verified. Do not rebuild/transfer them. Preserve accepted masks/folds,
original reports and provenance. Keep private rows/pixels/weights on authorized
storage, credential stdin and20GiB/5000file reserves after liabilities.
Final-serving17301 and final-entry17317 replaced only terminal observer dependencies;
training/preservation and numerical source are unchanged. E2's unnecessary
inference-only runs are omitted; E3 remains the final planned learned entry.
[Exact recovery and evidence](../initiative/audits/research-progress-20261017-0448-v1/README.md).

Original broad QA actually passed**4537tests/8skipped/2warnings/exit0** on numerical
source that remains byte-identical. Full mypy108 passed. Known unchanged global
Ruff1327/format57 debt is preserved; archive/currentv2 checks passed.
[Original terminal](../initiative/audits/github-current-state-refresh-20261017-v2/full-regression-terminal.json).
These are engineering results, not accuracy. Current archive/index checks follow
new receipts; [prior private sourceb9dabf26 mirror](../initiative/audits/hf-s6-original-mri-20261017-v1/verified.json)
is fully verified at HF0917971d/all40objects268191bytes and must not be repeated.
The newer milestone has its own verified receipt below; it is not that old mirror.


Earlier distinct private preservation: source**5b55c961** is fully verified at
HF**feb3100aa7b418db9bfed48cc869c710288d1e38**, all46smallobjects/207863bytes,
CPU942389/canonical shared lock/full re-download-size-SHA. This includes the
actual learnedE1 **0.949**, new boundary/S6 CPU results, original epoch proofs and
sole final-serving owners. No private pixels/clinical arrays/features/weights.
[Exact one-time mirror](../initiative/audits/hf-parent0949-boundary-s6cpu-20261017-v1/README.md).
Never repeat this or earlier prefixes; this receipt bookkeeping is later than
that mirrored source.

A fixed50/50 CoAtNet/S6 per-target rank contrast is now **prepared only**. It
changes only two AST assignments (weights/build metadata), passes eight CPU rank/
fallback contracts, and preserves original inference. No GPU/entry/funding is
claimed. Evaluate after the original S6 hidden outcome and fresh resource/public
acceptance; no leaderboard-tuned weights or exposed-encoder OOF claim.
[Exact preparation](../initiative/audits/s6-equalrank-preparation-20261017-v1/README.md).

Later05:55UTC input milestone: fitting8CPU139165316v1 completed and private
collection942782 verified all5372windows/1,154,209,297bytes. Its CPU classifier
was dispatched once (CPU942806/exec77108), with no terminal/score yet. Original input owner
20144 is terminal success. [Exact later proof](../initiative/audits/s6-fitting8-uid-view-20261017-v1/actual-inputs-private-verified.json).

Latest verified private milestone: source**c1354577** at
HF**b67956eda6cda255da7c190c3eda5806d3238d63**, all58objects/238462bytes,
CPU942819/canonical shared lock/full re-download-size-SHA. This includes actual
S6 T4 acceptance, corrected grant accounting, fitting8 input acceptance and CPU
classifier dispatch, fixed blend preparation and new original epoch proofs.
No private pixels/case IDs/clinical arrays/teacher predictions/weights.
[One-time receipt](../initiative/audits/hf-s6-actual-t4-fitting8-20261017-v1/README.md).
Never repeat this or earlier verified prefixes; receipt bookkeeping follows the
mirrored source. Numerical fitting/inference code in rsna/scripts/tests is unchanged.
