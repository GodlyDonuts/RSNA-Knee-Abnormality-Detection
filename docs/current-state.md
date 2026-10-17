# Current project state

**Observed through October 17, 2026, 07:41 UTC.** This is the live-state entry point.
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
Authenticated GetSubmissionLimits at06:50UTC reported four used/one available.
The subsequent S6 exact code entry57300990 was accepted once; its hidden result
is still unreported. That dated allowance is not an available-slot guarantee. Obtain fresh allowance before every
entry, maximum five/day. Original final E3 is owned; a guarded next-allowance
standby cannot act while its first owner is live or repeat an entry.
[Exact allowance and staged comparison](../initiative/audits/s6-equalrank-deployment-20261017-v1/actual-staged.json).

Original scored receipts: [external baseline](../initiative/audits/released0949-baseline-20261016-v1/actual-scored-canonical.json),
[own N3E16](../initiative/audits/compact-production-e016-kaggle-20261016-v1/actual-scored-canonical.json),
[context](../initiative/audits/released-context-deployment-20261016-v1/actual-scored-recovered.json),
[field](../initiative/audits/licensed-native-field-20261017-v1/actual-scored-result.json).
Do not resubmit those exact versions.

## Active fitting and actual completed epochs

| Work | Original identity | Latest verified state |
|---|---|---|
| Whole94 licensed-parent adaptation | E2 **139137922v1**; final E3 **139172537v1** | E2 COMPLETE972/all5private objects verified; original E3 API RUNNING/configured twoT4s |
| Expanded own N3 | Slurm**938284**, evc33 | RUNNING1H100/2CPU/8h limit; **epochs1–15 privately verified**, at least4860/5184 completed updates |
| Uniform16 full-encoder control | E2**139167076v1** | **E2 complete972/all5private objects verified**, HFea38a60272ff39f86e298f5765c7c45466ad2c89 |
| Influence16 full-encoder contrast | E1**139154664v1** | **E1 complete486/all5private objects verified**; original owner97530 now runs influenceE2**139181903v1** |
| Original patch recovery | Slurm929938 | COMPLETED0:0, full/stop1/resume engineering equality; not accuracy |
| S6 unchanged-source reproduction | Offline**139181157v1**, code entry**57300990** | **Public run COMPLETE233s/twoT4/zero fallback**; exact entry accepted, hidden score pending |

[New original epoch/publication/push evidence](../initiative/audits/research-progress-20261017-0448-v1/README.md),
[earlier optimizer and patch terminals](../initiative/audits/research-progress-20261017-0401-v1/README.md).
E2 now actually completed972updates and CPU942967 verified all5private objects
at HF51d5efae30b4efe820723c6147054fc90ac094c0. Original E3 is RUNNING.
Expanded epochs1–15 are now privately fully9object verified each, establishing
at least4860completed updates; the older3458cursor is superseded.
[New actual terminals and originals](../initiative/audits/research-progress-20261017-0625-v1/README.md).
InfluenceE1 completed486 and all5objects are independently private verified;
uniformE2 completed972/all5private objects and the same owner97530 now runs
influenceE2 139181903v1.
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
objects was privately fully verified once. Both E1 roles are complete and privately preserved under the same original
comparison owner; uniformE2 also completed972/private verified.
The original influenceE1 now also completed486 and all5objects are private verified
at HFb09816181e2e79eb24563884af5c7ed0e1af40a5; uniformE2 completed972/private verified; influenceE2 runs.
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
Original offline-run owner **20978/44859 reached terminal success** after exact
notebook139181157v1 completed233seconds/threeMRI/zero fallback. Sole entry owner
**20994/20982 accepted exact code submission57300990 once** and continues its
hidden observation. Its status/score are still null at the latest receipt. The
public CSV was not uploaded. Original H100942047 remains pending; never duplicate
S6 or infer its hidden score from either public run.
[Exact acceptance, quota correction and owners](../initiative/audits/s6-original-t4-recovery-20261017-v1/README.md).

CPU-only input139165316v1 completed5372original fitting windows, all compressed/
decoded bytes privately verified once. **CPU942806 now COMPLETED0:0**, scoring
all eight/5372windows in1197.87seconds, peak6.62GBRSS, finite logits/zero fallback.
CPU942957 reused those outputs and existing parent arrays without labels/masks/
reports/Gold/newCNN/GPU/fitting: raw gaps8.04–39.09points, rank correlations
0.8095–0.9762 and1–5reversed strict pairs/28per target. No strong0.8/0.2 opposites.
These are descriptive exposed-cohort cost quantiles, not clinical AUC or OOF.
**Do not use raw probability gaps as a calibrated clinical uncertainty gate.**
[Actual classifier and all12targets](../initiative/audits/s6-fitting8-cross-lineage-20261017-v1/README.md).

**Target0.97; verified best0.949.** The next fixed mechanism transports equal
teacher ranks onto the parent score scale and shifts its two view logits together,
then reuses the full-encoder source/teacher loss. The matched control keeps parent
targets bit-identical. Nineteen target-builder contracts pass. The new explicit data overlay additionally
passed17actual synthetic CPU checks using the unchanged real training engine,
including both-role full/stop1/fresh-model/resume complete state and9x12CSV
equality. Parent-only numerical state matches the original except explicit input
binding metadata. No full MRI/GPU training or accuracy result exists.
[Actual data path/recovery scope](../initiative/experiments/competitive-v1/s6-rank-transport-v1/data-path/README.md). Full3887 S6 targets and a separately admitted complete-
state recovery/funding are still required. The original target bundle stays
immutable. [Matched protocol/source/tests](../initiative/experiments/competitive-v1/s6-rank-transport-v1/README.md).

CPU942920 staged a distinct32fitting-study B-only scout, source0573a2a1, in
existing controls. Sole finite PID26454/exec91663 waits original S6 offline
acceptance and fresh canonical capacity/budget; **no GPU push yet at this
observation**, zero entries. It contains the original eight and checks actual
T4/CPU parity, full preparation/inference time and diversity before any full
teacher acquisition. Never restart its original owner or stager.
[Exact source, ten contracts and five owner paths](../initiative/audits/s6-fitting32-scout-preparation-20261017-v1/README.md).
The author's0.950 remains external; no project S6 hidden score exists yet.
A later CPU-only handoff26454/v4 repaired a post-entry state wait without
changing the staged scientific source or repeating a push.
[Actual no-mutation reconciliation and14checks](../initiative/audits/s6-scout-monotone-handoff-20261017-v1/README.md).

## Fixed blend deployment and next-allowance readiness

CPU943330 actually staged and registered the untuned equal-rank0.5/0.5 source
46a96883, with only weights/build ID changed. Sole **27596/exec58044** waits the
actual original S6 hidden score, then admits at most one9h private offline T4 run
with fresh full future VM funding/capacity and one exact accepted code entry
under live max5daily allowance. It stops on an original hidden failure or any
uncertain mutation. No GPU push/entry occurred at this observation.
[Exact source, allowance,17contracts/seven paths](../initiative/audits/s6-equalrank-deployment-20261017-v1/README.md).

FinalE3 standby **28090/exec30473** waits only original17317's actual terminal
zero-entry window expiry before using the unchanged original entry phase through
the next allowance window. It exits if the original submits and never creates a
GPU/Dataset/weight copy. [Exact ownership and seven tests](../initiative/audits/final-epoch3-next-allowance-20261017-v1/README.md).

The complete public S6v3 package has seven files/200500299bytes and no cached
fitting predictions; its extras manifest is source/runtime only. This rules out
a shortcut **in that release**, not all external sources. No weights or pixels
were acquired again. [Actual versioned complete listing](../initiative/audits/s6-public-teacher-cache-survey-20261017-v1/README.md).

## Public0.956 catch-up reference: actual source and weights

The exact Helios138222340v1 and extra224 variant138462884v1 both display0.956,
not a measured0.97. The source review identifies four R7 ConvNeXt folds,384/352
physical anatomical inputs,176training windows, state/extent and ordered-pair
supervision, EMA, laterality and state/fluid stackers, blended with the existing
CoAtNet. Seven embedded module SHAs match the exact manifest. CPU943797 acquired
all8 model/sidecar objects886809309bytes once; CPU943811 reconciled an observer-only
duplicated-reserve assertion using existing full-byte proofs, without repeating
any download/hash/header read. Originals remain private. CPU943843 staged the original Notebook/asset binding in the canonical ledger;
no R7 GPU/push/entry or new project score exists. Reproduce the unchanged complete reference next under
fresh actual session/future funding, while original work continues.
[Actual pins, mechanisms and limits](../initiative/audits/r7-public-reference-admission-20261017-v1/README.md),
[latest entry/training proofs](../initiative/audits/research-progress-20261017-0735-v1/README.md).

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

The fixed50/50 CoAtNet/S6 source passed its original arithmetic/fallback and
AST checks. It is now **actually staged and owned**, with no derivative GPU
execution/entry yet, as described in the new deployment section. No weights
were selected from leaderboard feedback or exposed-encoder head folds.
[Actual source and finite owner](../initiative/audits/s6-equalrank-deployment-20261017-v1/README.md).
[Exact preparation](../initiative/audits/s6-equalrank-preparation-20261017-v1/README.md).

Later05:55UTC input milestone: fitting8CPU139165316v1 completed and private
collection942782 verified all5372windows/1,154,209,297bytes. Its CPU classifier
was dispatched once (CPU942806/exec77108), with no terminal/score yet. Original input owner
20144 is terminal success. [Exact later proof](../initiative/audits/s6-fitting8-uid-view-20261017-v1/actual-inputs-private-verified.json).

Earlier verified private milestone: source**c1354577** at
HF**b67956eda6cda255da7c190c3eda5806d3238d63**, all58objects/238462bytes,
CPU942819/canonical shared lock/full re-download-size-SHA. This includes actual
S6 T4 acceptance, corrected grant accounting, fitting8 input acceptance and CPU
classifier dispatch, fixed blend preparation and new original epoch proofs.
No private pixels/case IDs/clinical arrays/teacher predictions/weights.
[One-time receipt](../initiative/audits/hf-s6-actual-t4-fitting8-20261017-v1/README.md).
Never repeat this or earlier verified prefixes; receipt bookkeeping follows the
mirrored source. Numerical fitting/inference code in rsna/scripts/tests is unchanged.

Latest privately preserved source is **9ef73d69**, HF**956d308744db79371ae02dc03a9cf43971c36d21**,
all60small objects/312744bytes fully re-downloaded/size/SHA verified by CPU943866
under the canonical shared lock. Never repeat. This guide/receipt bookkeeping is
later than that mirrored source. [Exact receipt](../initiative/audits/hf-r7-s6-progress-20261017-v1/README.md).
