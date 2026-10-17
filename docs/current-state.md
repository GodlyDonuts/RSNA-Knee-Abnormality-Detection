# Current project state

**Checked October 17, 2026, 01:46 UTC.** Actual submission receipts, Slurm and
publication journals resolve later changes. Use this entry point, the
[research brief](research-brief.md) and [execution handoff](codex-continuation.md).
[Earlier detailed snapshots](current-state.history-20261017-0146.md) are historical.

## Actual competition results

| Candidate | Submission | Public AUC | What the result establishes |
|---|---:|---:|---|
| Author-trained nartaa CoAtNet, unchanged inference | 57258866 | **0.949** | Reproduced external weights/inference; not our training |
| Our N3 CoAtNet E4 | 57256756 | 0.821 | Successful hidden execution |
| Our N3 CoAtNet E8 | 57259376 | 0.869 | +0.048 from E4 |
| Our N3 CoAtNet E12 | 57262392 | 0.890 | +0.021 from E8 |
| Our N3 CoAtNet E16 | 57265916 | **0.894** | Best verified independently MRI-trained model |
| Licensed parent + our sequence-context E10 | 57287572 | 0.948 | No demonstrated gain over unchanged parent |
| Unchanged parent, full384 field | 57289392 | 0.948 | No demonstrated gain over original320 field |
| Prior own DINO | 57182737 / 57182547 | 0.872 | Historical own-model reference |

These are public leaderboard AUCs, not clinical accuracy or final private scores.
The externally MRI-trained0.949 model was not trained from scratch by us. Our N3
started from general ImageNet pretraining. Current whole-parent adaptation is a
third claim, with no scored learned checkpoint yet. The October16 public-only
leaderboard reference was0.968; it is not a fresh leaderboard observation.

[External scored receipt](../initiative/audits/released0949-baseline-20261016-v1/actual-scored-canonical.json),
[own E16](../initiative/audits/compact-production-e016-kaggle-20261016-v1/actual-scored-canonical.json),
[context](../initiative/audits/released-context-deployment-20261016-v1/actual-scored-recovered.json),
[full field](../initiative/audits/licensed-native-field-20261017-v1/actual-scored-result.json).
Do not resubmit these exact versions. A current allowance requires a fresh account
query; historical counts do not authorize a new entry.

## Live training and next comparison

| Work | Original identity | Actual latest observation |
|---|---|---|
| Full licensed-parent adaptation | Kaggle139111324v1 | **RUNNING**, bothT4s,415/1458updates; epoch1 |
| Expanded own N3 fitting | Slurm938284,evc33 | **RUNNING**,1H100/2CPUs,45updates; empty stderr |
| Uniform16 vs influence16 full-encoder comparison | finite owner93305 | Inputs/storage accepted; uniformE1 prepared; **GPU budget wait**, no push |
| Patch-header research | Slurm929938 | OriginalPENDINGPriority; do not duplicate |

The parent adaptation trains **all73,136,096 parameters** on3,887 fitting studies,
all94 original windows, effective batch8, fixed3epochs/1458updates. Encoder/head
LR3e-6/3e-5, AdamW.02,10%warmup/cosinefinal1%,clip1, FP16/scaler/FP32loss,
25%source+75%detached-parent anchoring. BN running buffers are frozen; affine
parameters learn. Three exact checkpoint-continuous7hour sessions stop at
486/972/1458updates. Its numerical source and original E1 push remain unchanged.
[Actual415-update proof](../initiative/audits/licensed-parent-whole94-production-20261016-v1/actual-live-training-415.json).

Expanded N3 reuses the accepted original recipe and disjoint outer roster, adding
345 fitting studies:2,592fitting/1,156development,16epochs/5,184updates. Original
H100 recovery933382 passed before the one production dispatch. This is not a
controlled data-only contrast because update count increases. It has no final
metric or competition score yet. [Actual allocation/progress](../initiative/audits/compact-expanded2592-production-20261016-v1/actual-running-sanitized-observation.json).

The new matched sampling mechanism uses a fixed teacher-influence proposal with
25%uniform coverage. Both roles see16 IID windows with replacement, preserve
repeated draws and use inverse-proposal attention correction. Their full73.1M
initial model, all-rankRNG, study order/view, targets,3epochs/1458updates and
optimization are matched. Full94 inference stays original. It is a specific
hypothesis about useful observations per update, not a demonstrated accuracy gain.

Actual both-role GPU recovery139130771v1 completed in476.790s: fresh full/stop1/
resume learned model, AdamW/LR/cursors/allrankRNG/scaler/history/snapshots and
16x12 full94 MRI prediction bytes match exactly. Peak5.87GB/T4. CPU940795 matched
all32 original studies/two94-view hashes and both sampled roles, including180
repeated draws. Fixed3887x94 proposals were produced once from existing teacher
features without clinical labels or newCNN inference. Private Dataset12644218v1
has all5objects fully downloaded/size/SHA verified once. CPU940785 admitted12GiB
node39 checkpoints/publication plus16MiB controls after all prior liabilities.

The finite comparison owner is live, but **no production GPU was pushed**. Fresh
quota107092.916s was below115200s required to cover all unfinished original7hour
epochs, all6new100minute sessions and1hour reserve. It rechecks every5minutes,
without reserving a GPU or creating a mutation intent while insufficient.
[Protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md),
[actual recovery](../initiative/audits/licensed-importance-encoder-20261017-v1/actual-terminal.json),
[full comparison admission/negatives](../initiative/audits/licensed-importance-full-comparison-20261017-v1/README.md).

## What the completed experiments taught us

All original N3, neighbor and positive-weight16epoch checkpoints are privately
fully download/hash verified. On the same1,156 weak-label development studies:

| Fixed-final comparison | Macro AUC | Difference from its matched reference | Conditional95% component interval |
|---|---:|---:|---|
| Original own N3 E16 | 0.92092535 | Reference | — |
| Source-neighbor channels | 0.91519773 | -0.00572762 | [-0.02522802,+0.01337630] |
| Positive BCE weights | 0.92391334 | +0.00298799 | [-0.00999470,+0.01656811] |
| E12/E14/E16 parameter average, raw serving | 0.92100408 | +0.00031564 vs raw E16 | [-0.00214034,+0.00293081] |

None established a primary improvement; unchanged ownN3E16 remains its default.
These intervals used2,836 defined/1,164 undefined whole-component draws. Weak
holdout support is sparse: MCL/fracture each2positives, Baker3, effusion/lateralOA
9 each, synovitis5negatives. These are not clinical validation or Kaggle scores.
[Neighbor](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md),
[weighting](../initiative/audits/compact-positive-weight-fixed-final-20261016-v1/README.md),
[raw averaging](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md).

Teacher influence reduced full-readout approximation MSE45.6% on all3,887 fitting
studies, every target improving. This is teacher fidelity, not disease accuracy.
Context and full-field Kaggle results0.948 do not improve unchanged0.949. Spatial
regional/global-repeat heads completed matched10epoch fits and all20 epochs are
privately verified; regional remains unsubmitted at this check. The sequence head
actually relates the windows, but that alone did not improve the public score.

## Data and model provenance

Our own N3 is a fully fine-tuned CoAtNet RMLP2,73,136,096parameters, finding-specific
MIL over1,024-dimensional embeddings. It used2,247 fitting/1,156disjoint development
studies, up to6series/16selected slices per series at384pixels,12training windows,
16epochs/4,496updates, encoder/headLR3e-5/1e-3. Original source
`dcfff319cb7bd7bbae679a20da2b1b9268e8d680` remains immutable.

Released parent: nartaa Dataset12431568v1, checkpoint
`7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`.
Author permission explicitly covers competition use/adaptation/noncommercial
redistribution and must remain attached. Author reports4,349RSNA studies,
blended labels,2,399OAI knees for3tasks and selected snapshot averaging. We did
not reproduce that complete training recipe or acquire controlled OAI data.
Its parent development exposed our internal cohort/Gold58; derivatives cannot
claim independent internalOOF. Our protected clinical values remain unused.
[Exact source/admission](../initiative/audits/released0949-baseline-20261016-v1/source-manifest.json),
[public recipe correction](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md).

All original4full pixel producers completed once. Original32 cases were reused;
3,855new cases decoded,3,887total/zeroerrors/zeroempty; all uint8 pixels/masks/
both94-window view hashes matched. Four lossless archives and their node39 replica
are fully verified. Do not repeat cache extraction, transfers or Dataset uploads.
Stanford/KneeCoT work is separate private research, with no demonstrated eligible
RSNA-transfer gain; source-task scores are not RSNA/Kaggle results. Accepted masks,
folds and protected evaluation are unchanged.

## Deployment, preservation and authority

Today's existing comparison plan: context(actual0.948), full field(actual0.948),
regional(after02:00UTC), learned whole94E1 and fixed-finalE3. Regional owner87869
and distinct reconciled E1/E3 owners each allow at most one exact entry through
the canonical ledger after actual precision/public acceptance and fresh allowance.
No public three-row CSV upload. Other public-accepted boundary/global-repeat/
confidence candidates are unsubmitted and have no active entry owner.

Socket loss stopped local observers, not the original GPU runs. They were reconciled
read-only and distinct finite observers now preserve/qualify the same original
jobs and receipts. Latest handles and exact paths are in the execution handoff.
Never infer a failed training job from a dead observer or duplicate an uncertain
mutation. Keep credentials in ignored0600.env/RAM/stdin; original large assets
stay on private authorized cluster/Kaggle storage. Publication uses the canonical
shared lock and full re-download/size/SHA checks. Preserve20GiB/5000shared-file
reserves after all liabilities, deducting paid verified bytes only once.

Latest distinct small mirror: source3d83115d, **65objects/360309bytes** fully
re-downloaded/hash verified at private HF
`5656db3fe39639bff57d7fcfc29d778c4022915f`. It includes actual GPU recovery,
proposal Dataset/input/storage proofs and original training continuation. No
MRI/labels/UIDs/features/proposals/weights/prediction rows were mirrored.

The cleaned current guides at source805974c0 are separately private-HF verified
ONCE at `6eaf98ff679cb4e9fc975c8b11c488a4ddb4990f` (4objects/28890bytes), including
the preceding65-object receipt. [Guide preservation receipt](../initiative/audits/hf-current-guides-20261017-v1/verified.json).
