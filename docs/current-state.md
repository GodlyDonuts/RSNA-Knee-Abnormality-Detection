# Current RSNA Knee research state

Updated October19,2026,23:45UTC. [Agent guide](../AGENTS.md) · [Continuation](codex-continuation.md). Goal active;0.97/first place unachieved. Original receipts and actual live handles supersede historical snapshots.

| Model | Actual Kaggle score | Entry |
| --- | --- | --- |
| External R6 reproduction | **0.958** |57333150|
| Released S6 reproduction |0.950|57300990|
| Own whole94 full encoder |0.949|57336190|
| Own influence IID16 full encoder |0.949|57332456|
| Own paired / higher-LR adaptations |0.949 /0.949|57392062 /57394237|
| Own uniform / source-label adaptations |0.947 /0.947|57329796 /57331631|
| R6 with own paired member |0.957|57397292|

No original score improvement is established.0.958 is a reproduction. Full94 already completed1458updates; repeating it unchanged is not new research. Tests, training loss and more parameters are not accuracy.

## Full changed-target fitting is now running on Kaggle

**Kernel139699006v1 [rsna-s6-cross-full-e1-85ad0c706996](https://www.kaggle.com/code/godlydonuts/rsna-s6-cross-full-e1-85ad0c706996/log) is RUNNING.** This is full3887-study E1, target486updates, of a fixed3epoch/1458update experiment. The full73M CoAtNet encoder/head learns from complete cross-lineage equal-rank teacher targets; original MRI, source labels/masks, proposals/IID16/effective8, optimizer/LR and inference remain fixed. Actual live logs confirm finite updates3/7/10, both T4s, original input admission and all73136096trainable parameters; last observed loss0.52659324/gradient norm7.5594883. This is a lower bound from visible logs, not the latest completed count or an accuracy result. No completed epoch/new score yet. [Actual source and push](../initiative/research/S6-Kaggle-production-route-20261019-v1/README.md).

The previously completed original influence control139220032v1/0.949 is reused. Its entire teacher tensor is unchanged in the parent-only overlay. CPU974263 verified identical original numerical modules/non-main worker functions; actual new engineering prediction/proposal hashes and both-rank initial model/RNG identities match the earlier original control. This avoids repeating roughly3.5hours of baseline training. It is a matched recipe reference, not a claim of unrun final-weight identity or independent OOF performance.

Fresh pre-push quota34391.7526641GPU VM seconds covers at most18000seconds for the three changed epochs plus6000serving allowance. Private target Dataset12718845v1 was fully downloaded/hash verified; all original MRI/weight inputs already existed, so no40GB upload. Current E1 observer is recorded in actual-epoch1-observer.json. It only follows this immutable version, cannot submit/restart, and does not yet continue E2/E3. Privately preserve each completed epoch before its exact optimizer continuation.

## The diagnostic failure was reporting, not a repeated training attempt

Original139695847v1 remains **ERROR** after514.3seconds. Its six full/stop/resume GPU subprocesses all returned0. The final collector then used a stale `roles['uniform']` key after both complete-state equality loops. The failure is preserved.

CPU974246 downloaded the65 existing outputs/5856573000bytes once to the private owning node and independently executed the original collector tail, correcting only two role keys and the local output path. Complete learned/base/AdamW/LR/cursors/all-rank RNG/scaler/history/snapshots and16-case full94 prediction equality passed for both roles; both T4s and411different learned tensors confirmed. No GPU work was repeated. This is a separate numerical engineering acceptance; the original notebook has not been relabeled COMPLETE. [Actual CPU acceptance](../initiative/research/S6-Kaggle-existing-output-replay-20261019-v1/actual-accepted.json).

## Queue, data and next work

All3887teacher outputs and complete targets finished, with parent-only targets bit-identical and accepted labels/masks untouched. Handoff/source compilation/bridge all completed once. Newton973904 is held PENDING/zero runtime, and only its proven idle old production/packaging/scoring followers97036/97047/97060 were retired. No running fit stopped. The separate novel paired-ranking scout973601 and its observers remain pending/untouched.

Next: follow actual E1 optimizer/terminal progress, independently verify and privately preserve its immutable snapshot, then continue the same candidate through E2/E3 and qualify genuine full model checkpoints for useful Kaggle evaluation. Finish the source/provenance/runtime serving adaptation while training runs. Do not duplicate completed controls or healthy fits. Record negative outcomes as carefully as gains.

Model retention is funded within the original12GiB node48 recovery scope: paid replay plus at most4GiB for three production epochs/checkpoints and one verification temporary, with original87GiB liability/20GiB physical/5000shared-file reserves retained. Credentials and raw clinical/weight/source arrays stay private. New own-source/sanitized Kaggle milestones are on GitHub; one distinct private source mirror remains due. Protected cohorts remain unused by adaptations; public teachers/base exposure are not independent OOF validation. Stanford-derived weights private/noncompetition; no newly admitted KneeCoT or unreliable MedGemma labels.
