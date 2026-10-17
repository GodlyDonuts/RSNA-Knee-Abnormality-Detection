# Current project state

**Observed through October 17, 2026, 04:15 UTC.** This is the current-state entry point.
Read the [research brief](research-brief.md), [research-agent handoff](research-agent-handoff.md)
and [execution handoff](codex-continuation.md) next. Dated documents and
[previous snapshots](current-state.history-20261017-0146.md) preserve history;
their pending/running/next-action statements are not current instructions.

## Actual competition results

| Candidate | Submission | Public AUC | What it establishes |
|---|---:|---:|---|
| Author-trained nartaa CoAtNet, unchanged inference | 57258866 | **0.949** | Reproduced external weights and inference; not our training |
| Our N3 CoAtNet, epoch 4 | 57256756 | 0.821 | Successful hidden execution |
| Our N3, epoch 8 | 57259376 | 0.869 | +0.048 from epoch 4 |
| Our N3, epoch 12 | 57262392 | 0.890 | +0.021 from epoch 8 |
| Our N3, epoch 16 | 57265916 | **0.894** | Best verified independently MRI-trained model |
| Licensed parent + our sequence-context head, epoch 10 | 57287572 | 0.948 | No demonstrated gain over the unchanged parent |
| Unchanged licensed parent, full 384-pixel field | 57289392 | 0.948 | No demonstrated gain over its original 320-pixel field |
| Licensed parent + regional spatial head, epoch 10 | 57291182 | **0.948** | Actual hidden completion; no gain over unchanged parent |
| Prior own DINO | 57182737 / 57182547 | 0.872 | Historical own-model reference |

These are public leaderboard AUCs, not clinical accuracy or final private scores.
**We did not train the 0.949 model from scratch.** N3 started from general ImageNet
pretraining and was independently MRI-finetuned by us. Current licensed-parent
adaptation is a third category, with no scored learned checkpoint yet. The
fresh live Public leaderboard check now shows **0.969 first place**, versus our
0.949: a **0.020 public-AUC gap**. A shared 0.956 blend is not the leaderboard
frontier. This does not reveal the leading team's method or final private score.
[Actual visible leaderboard observation](../initiative/audits/github-current-state-refresh-20261017-v2/actual-public-leaderboard.json).

The [fresh authenticated check](../initiative/audits/whole94-original-worker-continuation-20261017-v1/actual-current-observation.json)
confirms these recent outcomes and **3 submissions today / 2 live slots remaining**
at the latest saved account check. That allowance expires as an observation; query it again before a
mutation. The canonical ledger agrees on the completed context result. The regional
entry receipt and fresh API agree on submission 57291182: COMPLETE, score 0.948, no error.
The later **owning-node observation in this update** confirms expanded N3 at **2,166
finite updates / epoch 7**, with empty stderr and six completed checkpoints
privately fully verified. Epoch 4 is HF `e183899c0b2ea38adfe66c287713a872ce0effc9`,
all nine objects. [Earlier progress](../initiative/audits/github-current-state-refresh-20261017-v2/expanded-actual-progress.json),
[new actual progress and epoch5/6 preservation](../initiative/audits/research-progress-20261017-0401-v1/README.md).

Original receipts: [external baseline](../initiative/audits/released0949-baseline-20261016-v1/actual-scored-canonical.json),
[own N3 epoch 16](../initiative/audits/compact-production-e016-kaggle-20261016-v1/actual-scored-canonical.json),
[context](../initiative/audits/released-context-deployment-20261016-v1/actual-scored-recovered.json),
[full field](../initiative/audits/licensed-native-field-20261017-v1/actual-scored-result.json).
Do not resubmit their exact versions.

## What is running, and what is only prepared

| Work | Original identity | Latest actual evidence |
|---|---|---|
| Full licensed-parent adaptation | E1 139111324 v1; E2 **139137922 v1** | **E1 complete**, 486 updates and private verification; E2 **682/1458 actual finite updates**, both T4s; exact worker/checkpoint resumed |
| Expanded own N3 fitting | Slurm 938284, evc33 | **RUNNING**, one H100 / two CPUs; 2,166 of 5,184 finite updates / epoch 7; epochs 1–6 privately verified |
| Uniform-16 versus influence-16 full-encoder comparison | Owner **97530**; uniform E1 **139137040 v1** | Original uniform **367/1458 actual finite updates**, both T4s; influence remains unpushed |
| Patch-header research | Original Slurm 929938 | **COMPLETED0:0**, exact full/stop1/resume engineering recovery passed; no accuracy result |
| Sampled-epoch serving compiler | Committed preparation | Synthetic CPU/regression checks passed; **no actual serving GPU/public-MRI acceptance** |

[Fresh Slurm/account/ledger observation](../initiative/audits/whole94-original-worker-continuation-20261017-v1/actual-current-observation.json).
The first full-parent epoch completed all **486** finite updates on both T4s,
with 73,136,096 optimizer parameters and unchanged BN running buffers. It was
independently CPU-verified and privately published at HF
`d436986137570a9a4259a4b59981d261cffd2f20` (all five objects fully downloaded,
size- and SHA-verified). Its learned-state inference qualification passed on
32 original MRI studies on **both T4s**, exact input/model equality and **zero
prediction difference**. This is serving acceptance, not a competition score.
[Actual first epoch, continuation and qualification](../initiative/audits/whole94-original-worker-continuation-20261017-v1/README.md).

The first continuation preparation exposed a proof-JSON key-order mismatch.
Actual CPU941151 proved every metadata value/binding equal and replayed the
**exact original worker bytes**, without changing numerical code, recipe or
optimizer/RNG state. The one actual epoch-2 push is139137922v1/source7cbc68dad857.
No completed GPU run was restarted. The later04:01 logs now show both allocated T4s and **682** main / **367** uniform
finite updates. This supersedes the earlier blank-log observation.
[Actual visible records](../initiative/audits/research-progress-20261017-0401-v1/actual-Kaggle-progress.json).

The parent adaptation trains **all 73,136,096 parameters**, on **3,887 fitting
studies**, with all **94 original windows**, effective batch 8 and a fixed
3-epoch / 1,458-update schedule. Encoder/head learning rates are 3e-6 / 3e-5;
AdamW weight decay 0.02, 10% warmup, cosine decay to 1%, clipping at 1, FP16/scaler
with FP32 loss. The objective is 25% source supervision and 75% detached-parent
anchoring. BN running buffers are frozen; affine parameters learn. Three bounded,
checkpoint-continuous sessions stop at 486 / 972 / 1,458 updates. Epoch 1 is complete;
there is no learned-parent competition score yet.
[Exact training audit](../initiative/audits/licensed-parent-whole94-production-20261016-v1/README.md).

Expanded N3 reuses the original recipe and disjoint outer roster, adding 345
fitting studies: **2,592 fitting / 1,156 development**, 16 epochs / 5,184 updates.
Original H100 recovery 933382 passed before its one production dispatch. This is
not a controlled data-only contrast because the update count increases. No final
metric or Kaggle score exists yet.
[Expanded fitting audit](../initiative/audits/compact-expanded2592-production-20261016-v1/README.md).

## New observation-sampling experiment

A fixed teacher-influence proposal combines 25% uniform coverage with 75% influence.
Both roles sample 16 IID windows with replacement, preserve repeated draws and
use inverse-proposal attention correction. Their initial full model, all-rank RNG,
study order/view, targets, optimizer and fixed 3-epoch schedule are matched.
**Inference still uses the original full 94 windows.**

On all 3,887 fitting studies, influence sampling reduced frozen-teacher readout
approximation MSE by **45.6%**, with every target improving. This uses no clinical
labels or new CNN inference. It is teacher fidelity, not disease accuracy or a
measured training-speed gain.

Actual two-role GPU recovery **139130771 v1** completed in 476.790 seconds: fresh
full / stop-after-one / resume runs matched the complete learned model, AdamW,
LR/cursors, all-rank RNG/scaler, history, snapshots and 16×12 full-94 prediction
bytes. Peak allocation was 5.87 GB per T4. CPU 940795 also matched the 32 original
MRI cases, both views, both sampled roles and 180 repeated draws. The frozen
3,887×94 proposal table and private Dataset **12644218 v1** were each created
once; all five Dataset objects were fully downloaded, size- and SHA-verified.

CPU 940785 funded 12 GiB of node-local checkpoints/publication plus 16 MiB of
controls after prior liabilities. Once the original main epoch completed, the
first uniform session passed the fresh all-future GPU funding check and was
pushed once: **139137040 v1**, source8155b3803c5a. Its source was independently
reconstructed from persisted inputs byte-for-byte, excluding the main compiler's
initial proof-order problem. It continues unchanged under the recovered observer.

Kaggle's actual **two batch GPU session** limit is now enforced before auxiliary
push intents. Main continuation retains one lane; sampling and public inference
share the other. The already qualified learned epoch-1 public run has next-lane
priority when funded. Its first attempted public launch was explicitly capacity-
rejected; the ambiguous old identity is permanently quarantined, never retried
or submitted. A distinct `-g2` identity will be created once after capacity opens.
No old three-row CSV, rejected notebook, model weights or Dataset is resubmitted.
[Capacity admission and exact identities](../initiative/audits/kaggle-two-session-capacity-20261017-v1/README.md).
[Protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md),
[actual recovery](../initiative/audits/licensed-importance-encoder-20261017-v1/actual-terminal.json),
[full-comparison admission and failures](../initiative/audits/licensed-importance-full-comparison-20261017-v1/README.md).

The [new serving preparation](../initiative/audits/licensed-importance-serving-preparation-20261017-v1/README.md)
keeps role/proposal/source/checkpoint identities distinct and retains full-94
inference. Forty focused synthetic/regression tests, Ruff on four files and mypy
on two source files passed. Those checks are **not** actual sampled-model serving
or accuracy acceptance. No such checkpoint has completed yet.

## Completed experiments and negative findings

All original N3, neighbor and positive-weight 16-epoch checkpoints are privately
preserved and fully re-downloaded/hash verified. Same 1,156 weak-development studies:

| Fixed-final comparison | Macro AUC | Difference from matched reference | Conditional 95% component interval |
|---|---:|---:|---|
| Original own N3 epoch 16 | 0.92092535 | Reference | — |
| Physical source-neighbor channels | 0.91519773 | -0.00572762 | [-0.02522802, +0.01337630] |
| Positive BCE weighting | 0.92391334 | +0.00298799 | [-0.00999470, +0.01656811] |
| E12/E14/E16 parameter average, raw serving | 0.92100408 | +0.00031564 versus raw E16 | [-0.00214034, +0.00293081] |

None established a primary improvement; unchanged own N3 epoch 16 remains its
default. These comparisons used 2,836 defined and 1,164 undefined whole-component
bootstrap draws. The averaging reference uses raw-serving E16 AUC 0.92068844,
not the differently served 0.92092535. Sparse support matters: MCL/fracture have
two positives each, Baker's three, effusion/lateral OA nine each, synovitis five
negatives. These values measure weak-label agreement, not clinical validation.
[Neighbor](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md),
[weighting](../initiative/audits/compact-positive-weight-fixed-final-20261016-v1/README.md),
[averaging](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md).

Removing `Selection_Tier` changed zero series selections on all 1,156 outer
studies; that does not prove complete rendering parity. Context and full-field
Kaggle scores of 0.948 do not improve the unchanged 0.949 parent. Matched regional
and global-repeat heads completed ten epochs each; all 20 checkpoints are privately
verified. Regional **57291182** now actually completed at **0.948**, without error; no improvement over 0.949 is established.
The [research-agent handoff](research-agent-handoff.md) maps other completed and
historical evidence without treating every proposal as an executed experiment.

## New fitting-only research evidence

CPU941335 screened the original 3,887 paired-view teacher predictions without
reading source labels, masks, reports, pixels, weights or protected clinical values.
Mean probability gaps are 0.01287–0.02707; strict-pair ordering disagreement is
2.75–7.41%, highest for Baker's. At most 4.37% of studies per target had gaps
above 0.10. The predeclared result is **intermediate**: it does not admit another
GPU arm or prove a source of disease error. These are teacher ordering pairs,
not positive-negative AUC errors. Current training and target definitions stay
unchanged. [Protocol and actual all-target result](../initiative/audits/teacher-view-consistency-screen-20261017-v1/README.md).

## Newly supplied research and next parent

The [owner's deadline program and assessment](../initiative/research/owner-deadline-program-20261017-v1/README.md)
are archived verbatim with provenance. We verified the public S6 notebook
**138285808 v2**, pinned its source and actually downloaded the two small
checkpoint/code manifests from the public weight Dataset **12589750 v3**.
The original **200,185,003-byte S6 checkpoint is now privately acquired once**
and SHA-bound, with 362 finite tensors and a 12-target header independently
reconciled. No accepted S6 MRI terminal, GPU execution or competition entry is claimed. Its
reported **0.950 belongs to its author**, not this project's score table.
[Actual source-only admission](../initiative/audits/s6-public-source-20261017-v1/README.md).
[Actual checkpoint admission and preserved checker failure](../initiative/audits/s6-checkpoint-admission-20261017-v1/README.md).

Prioritize an unchanged S6 reproduction as a second model lineage, then measure
complementarity before a matched two-teacher experiment. Preserve healthy current
fits and the learned-E1 serving priority. A new head's five-fold split cannot
provide independent OOF for an encoder trained on those same studies. Raw teacher
agreement is not calibrated correctness; proposed gains and runtimes are untested.

## Exact provenance and data limits

Our own N3 is fully fine-tuned **CoAtNet RMLP2**, not a frozen DINO head. Its
finding-specific MIL pools 1,024-dimensional embeddings. Original fitting used
2,247 studies, up to six series and 16 selected slices per series at 384 pixels,
12 training windows, 16 epochs / 4,496 updates, encoder/head LR 3e-5 / 1e-3,
AdamW 0.02, OneCycle and clipping at 3, without augmentation. Original source:
`dcfff319cb7bd7bbae679a20da2b1b9268e8d680`.

The external parent is nartaa's Dataset **12431568 v1**, checkpoint
`7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`.
It selects 96 slices in slots 26/22/18/12/18, uses 94 triple windows, a 140 mm
physical crop, 384-pixel preparation / 320-pixel execution, anatomical mirror TTA
and per-target ranks. The author reports 4,349 RSNA studies, blended labels,
2,399 OAI knees for three tasks and selected snapshot averaging. We did not
reproduce that full training recipe or acquire controlled OAI data. Explicit
author permission covers competition use/adaptation/noncommercial redistribution;
retain it with attribution.
[Exact source admission](../initiative/audits/released0949-baseline-20261016-v1/source-manifest.json),
[public training-contract correction](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md).

The parent's development exposed our internal cohort/Gold58; its derivatives
cannot claim independent internal OOF there. Our protected clinical values remain
unused. All four original full-population pixel producers finished once: 3,855
new cases plus 32 reused cases, 3,887 total, zero decode/empty cases. Original
pixel/mask/view hashes and all four lossless archives and node39 replica are
verified. Do not rebuild or transfer them again. Stanford/KneeCoT remain separate
private research, with no demonstrated eligible RSNA-transfer benefit. Accepted
masks/folds are unchanged.

## Immediate operations and preservation

The regional comparison scored 0.948. The two remaining planned entries
are whole-parent epoch 1 and fixed-final epoch 3, subject to actual completion,
precision/public MRI acceptance and fresh allowance. Do not upload public
three-row CSVs. Read exact live owners before changing this plan or dispatching.

SSH observation loss stopped local observers, not the original GPU jobs. Distinct
reconciled owners now watch those same originals; the [execution handoff](codex-continuation.md)
records identities and intents. Preserve each completed epoch once. Keep large
private assets on authorized cluster/Kaggle storage; credentials stay ignored and
are streamed through stdin. Respect the funded 20 GiB / 5,000 shared-file reserves.

Earlier verified small mirrors: source `3d83115d` at private HF
`5656db3fe39639bff57d7fcfc29d778c4022915f` (65 objects / 360,309 bytes), and the
cleaned guide source `805974c0` at `6eaf98ff679cb4e9fc975c8b11c488a4ddb4990f`
(4 objects / 28,890 bytes). Both were fully re-downloaded, size- and SHA-verified.
They contain sanitized source/receipts, not private pixels, clinical arrays,
features, proposal tables or weights. This documentation refresh is later;
[its audit](../initiative/audits/github-research-handoff-20261017-v1/README.md)
separates fresh observations and source preparation from prior private mirrors.

### Earlier preservation recovery and verified mirrors

The final check caught expanded preservation watcher 93375 failing on a local
module import **before any publication intent or RPC**. Training and its snapshot
were intact. Replacement **95408 / exec49302** fixes only the child import
environment; original numerical source, publisher and GPU job are unchanged.
Its first original publication, **CPU941066, completed successfully**: expanded
epoch 1 is privately preserved at HF `490960b8a7be8114607da040028e1ffe1a19ae54`,
all **nine objects** fully re-downloaded/size/SHA verified. Epochs 2, 3 and 4 are
now also fully verified through the same original watcher; never repeat 1–4.
The watcher continues on the same original job; this is preservation, not accuracy.
[Failure and recovery](../initiative/audits/compact-expanded2592-publication-observer-recovery-20261017-v1/README.md),
[actual epoch receipt](../initiative/audits/compact-expanded2592-publication-observer-recovery-20261017-v1/actual-epoch1-private-verified.json).

| Earlier verified private small mirror | Verified scope |
|---|---|
| `b34d69e5d815ec2bd8e799d1c52f91a7c9778b7b` | Source `1a2e8204`, 24 objects / 322,098 bytes: research handoff, fresh observations and serving preparation |
| `b2ecca99b9b8ac2b0ac9e039eff99a922510ae5d` | Source `9008eced`, eight objects / 35,644 bytes: observer recovery and preceding mirror receipt |

Both used the canonical shared lock and full re-download/size/SHA checks, with
no private pixels/clinical arrays/features/proposals/weights. The subsequent
expanded epoch receipt above is covered by its original private checkpoint
publication. This final receipt bookkeeping follows those verified sources.

## Latest verified private research preservation

Source `60ed92d3e3876a9e996076d6fff2451bcb1726ed` is privately mirrored at HF
`da34234c32cae097e6a04cd392a88f6e2c594373`: **102 objects / 648,152 bytes**,
all fully re-downloaded/size/SHA verified under the canonical shared lock. It
covers the new current guides, original paper/assessment, source/checkpoint
admission and actual run receipts. Large private weights/pixels/clinical arrays
are excluded; the global archive manifest has an exact Git-blob/SHA binding.
[Actual mirror receipt](../initiative/audits/hf-whole94-e1-capacity-screen-20261017-v1/verified.json).
Never repeat this or earlier verified prefixes. Final receipt/QA bookkeeping is
later than the source revision mirrored.

## Actual S6 source/stack acceptance and bounded MRI gate

CPU941872 completed all293 exact original source members and the original anatomy
checkpoint. CPU941960 passed the author's unchanged B_PROBE, using a separately
installed, hash-pinned OpenCV sidecar and the existing admitted timm overlay.
The anatomy forward is finite and the full50,018,284-parameter classifier loads
strictly. No active training environment or source bytes changed.
[Actual source, runtime and stack proofs](../initiative/audits/s6-source-tree-admission-20261017-v1/README.md).

A distinct private **CPU-only** Kaggle notebook139151928v1,
`godlydonuts/rsna-s6-original-cpu-mri-e0e75ecc1202`, actually COMPLETED successfully. It
reused unchanged original preparation/geometry/anatomy/window functions on all
three public MRIs, checking1586window/metadata round trips with zero failures.
No GPU, classifier predictions, new Dataset or competition entry was requested. One SaveKernel was
accepted; an immediate missing-status observer error was reconciled read-only
against its exact source/metadata/canonical row. Never repeat the push. Actual
CPU942028 then collected348312520compressed bytes once and verified every
file/tensor SHA privately. [Actual input proof](../initiative/audits/s6-source-tree-admission-20261017-v1/CPU-MRI-collected-verified.json).
Bounded H100 FP32/FP16 classifier job942047 is submitted once and PENDING for
resources on evc33. Both-T4/full-runtime and competition comparisons remain ahead.

The original broad QA run has now actually finished: **4,537 passed, 8 skipped,
2 warnings, exit0**,32:47. Code under rsna/scripts/tests is byte-identical to the
tested source. [Original terminal proof](../initiative/audits/github-current-state-refresh-20261017-v2/full-regression-terminal.json).
Known unchanged archived/global lint debt remains recorded; no accuracy follows QA.
