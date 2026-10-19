# Current RSNA Knee research state

Updated October 19, 2026, 15:05 UTC. **Best score remains 0.958; no new original accuracy gain is established.** Target 0.97 remains unachieved. [Operational guide](../AGENTS.md), [continuation](codex-continuation.md), [previous snapshot](current-state.history-20261019-before-resident-owner.md).

| Model | Actual Kaggle public score | Submission |
| --- | --- | --- |
| External R6 ensemble reproduction | 0.958 | 57333150 |
| Released S6 reproduction | 0.950 | 57300990 |
| Our whole94 full-encoder adaptation | 0.949 | 57336190 |
| Our influence IID16 full-encoder adaptation | 0.949 | 57332456 |
| Our paired fixed-final adaptation | 0.949 | 57392062 |
| Our higher-encoder-LR adaptation | 0.949 | 57394237 |
| Our uniform IID16 adaptation | 0.947 | 57329796 |
| Our source-label adaptation | 0.947 | 57331631 |
| R6 with our qualified paired member | **Hidden outcome pending** | **57397292** |

Current adapted encoder is full 73,136,096-parameter CoAtNet. The highest score is a reproduction, not a from-scratch breakthrough. Public results do not establish independent/private generalization. Paired supervision and raising encoder LR from3e-6 to3e-5 both returned the same displayed0.949; stop unchanged sweeps and retain these negatives.

## Work actually running

**Complementary supervision:** phase25/job970171 completed0:0,64new/1,533total in690.74seconds. Same resident owner2320176 advanced to phase26/job970271 RUNNING on39 with23workers/48allocatedCPU/256GiB. Original source/model/renderer/FP32 math and fitting population are unchanged. The resident can finish all3,887 and construct full targets without the laptop/VPN. This is teacher inference, not new student training or an accuracy gain. [Sources and receipts](../initiative/research/S6-cluster-resident-owner-20261019-v1/README.md).

**Kaggle comparison:** the exact qualified paired member replaced cell0 in the original eight-cell R6, leaving the other seven and blend weights unchanged. Native private notebook139613596v2/script360061652 completed its public MRI run. Its paired-member CSV exactly equals the earlier accepted solo public CSV; final output has13columns/three public rows with finite probabilities. Code submission57397292 was accepted once at13:10UTC. Actual authenticated hidden status/score are still null, not success or failure. [Public acceptance and submission](../initiative/research/R6-postscore-member-execution-20261019-v1/README.md).

**Next original student:** the exact source package and original32MRI inputs were actually admitted on Newton by CPU969738:26source files/18old-control modules equal, no new pixel/weight/feature transfer. Local follower55175 waits the exact original full-target terminal and can make one direct private39→33 handoff. Full-size synthetic transport and10refusal checks passed; real targets, GPU recovery and new student fitting remain outstanding. Both target roles preserve original labels/masks and full73Mencoder. [Execution handoff](../initiative/research/S6-target-execution-inputs-20261019-v1/README.md).

**Multimodal experiment:** original GPU968562 passed CUDA initialization/model load, then failed in its blank-image control because Triton could not compile without Python.h. No MRI-pixel review occurred. Matching signed RPM headers and the exact installed configuration header now form a private overlay; CPU970014 compiled the exact extension. No system/venv package changed or model/image reacquisition occurred. One corrected GPU970072 is queued on33,1H100/2CPU/64GiB/30min. Model/prompt/processor/all195images and functions remain fixed; it must pass blank abstention before the MRI chunks. No clinical/student accuracy gain is established. [Original failure and correction](../initiative/research/medgemma-runtime-header-recovery-20261019-v1/README.md).

**Observed Kaggle capacity:**14:08UTC actual total162000s,used127093.9703359,reserved0:34906.03s (~9.7h) remaining under the current total allowance.3submissions today/2available. The observer's incorrect minimum-floor arithmetic is explicitly corrected in the receipt; no account/billing change. This does not fund two new3epoch T4 fits plus serving. Fresh execution-budget/runtime decisions remain necessary.

## Limits and preservation

Accepted labels/masks/folds/base/protected gold/confirmation are unchanged. Stanford-derived weights remain private and excluded from competition. Clinical pixels/reports/IDs/teacher arrays stay private. Useful accessible physical1/2-H100 nodes are allowed; physical4/8-GPU nodes are forbidden. Fresh quota and exact immutable handles govern dispatch; no blind retries or deletions. Current CPU/RAM scope defers unstarted legacy bulk reservations and preserves20GiB physical shared/node free space,5000shared files and32MiB metadata headroom after actual obligations. Resident controls have a4MiB/256file cap; other-user allocation is preserved.

Original immutable checkpoints and earlier selected-source private mirrors remain verified; do not repeat them. New current sources/receipts must be distinguished from earlier HF07425a3a milestone rather than claiming all history is mirrored. GitHub guides now replace stale active-owner snapshots; the previous full state is archived unchanged above. Tests, backup completeness and worker count are not accuracy evidence.

Latest selected resident-teacher/R6 source snapshot30228a2e is privately preserved at HF17e5b642f049666adb8742577d940818d52c50e2 viaCPU969435:75small objects/312998bytes, all full download/size/SHA verified under the canonical shared lock. No clinical data/model weights; do not repeat. Later live status is separate from this immutable source snapshot.

Follow-up source preparation: local finite CPU-compile follower56997 waits original handoff55175, then compiles the original recovery source once on33; no GPU/Dataset/entry action. Existing12MiB control cap applies. Actual GPU student recovery/fit remains absent. Selected earlier snapshot4c52dcf6 is privately preserved at HF8e5e15d3a65994bb0b40bcdb49e2d8ef7b813dd3 throughCPU970316, all34objects/155131bytes fully re-downloaded/size/SHA verified under the canonical shared lock. Later compiler-follower additions are not part of that snapshot.
