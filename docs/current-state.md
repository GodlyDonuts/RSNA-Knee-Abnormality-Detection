# Current RSNA Knee research state

Updated October 19, 2026, 13:49 UTC. **Best score remains 0.958; no new original accuracy gain is established.** Target 0.97 remains unachieved. [Operational guide](../AGENTS.md), [continuation](codex-continuation.md), [previous snapshot](current-state.history-20261019-before-resident-owner.md).

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

**Complementary supervision:** phase18/job969332 completed0:0,64new predictions in711.23seconds,1,085total. Cluster-resident owner2320176 completed phase19/969475 with64more/1,149total in687.84seconds and automatically started phase20/job969585 RUNNING on evc39 with23workers/48allocatedCPU/256GiB;at least1,149/3,887durable rows preserved. Original source/model/renderer/FP32 math and population remain fixed. Source429/interrupted-render recovery reused original bytes and prior predictions. Newton now orchestrates later batches without the laptop/VPN; retired local owners stay stopped. Exact complete-population target construction follows only after all3,887 and actual terminal admission. This is ongoing teacher production, not student fitting or improved accuracy. [Source, actual start and phase18 completion](../initiative/research/S6-cluster-resident-owner-20261019-v1/README.md).

**Kaggle comparison:** the exact qualified paired member replaced cell0 in the original eight-cell R6, leaving the other seven and blend weights unchanged. Native private notebook139613596v2/script360061652 completed its public MRI run. Its paired-member CSV exactly equals the earlier accepted solo public CSV; final output has13columns/three public rows with finite probabilities. Code submission57397292 was accepted once at13:10UTC. Actual authenticated hidden status/score are still null, not success or failure. [Public acceptance and submission](../initiative/research/R6-postscore-member-execution-20261019-v1/README.md).

**Next original student:** production TeacherOverlay now compiles into the original real-MRI recovery engine, both parent-only and complete-cohort cross-lineage target arms, with original full model/init/sampler/optimizer/precision held fixed. Seven compiler/refusal contracts and actual emitted-adapter synthetic execution passed;11 existing recovery tests passed. This is preparation, not actual GPU recovery, student fitting or improved accuracy. Full teacher completion precedes exact targets, separately funded GPU recovery, matched fitting and MRI-only deployment. Partial ranks are prohibited. [Mechanism and remaining execution](../initiative/research/S6-target-real-MRI-recovery-20261019-v1/README.md).

**Multimodal evidence:** official MedGemma1.5-4B and one195-slice native packet are verified privately. Job968562 remains pending on physical2GPU evc33 under maintenance/reservation. Earlier26 attempts failed CUDA101 before model inference. No actual medical/model result yet. Existing Daenerys annotations remain candidates, never accepted-label changes.

## Limits and preservation

Accepted labels/masks/folds/base/protected gold/confirmation are unchanged. Stanford-derived weights remain private and excluded from competition. Clinical pixels/reports/IDs/teacher arrays stay private. Useful accessible physical1/2-H100 nodes are allowed; physical4/8-GPU nodes are forbidden. Fresh quota and exact immutable handles govern dispatch; no blind retries or deletions. Current CPU/RAM scope defers unstarted legacy bulk reservations and preserves20GiB physical shared/node free space,5000shared files and32MiB metadata headroom after actual obligations. Resident controls have a4MiB/256file cap; other-user allocation is preserved.

Original immutable checkpoints and earlier selected-source private mirrors remain verified; do not repeat them. New current sources/receipts must be distinguished from earlier HF07425a3a milestone rather than claiming all history is mirrored. GitHub guides now replace stale active-owner snapshots; the previous full state is archived unchanged above. Tests, backup completeness and worker count are not accuracy evidence.

Latest selected resident-teacher/R6 source snapshot30228a2e is privately preserved at HF17e5b642f049666adb8742577d940818d52c50e2 viaCPU969435:75small objects/312998bytes, all full download/size/SHA verified under the canonical shared lock. No clinical data/model weights; do not repeat. Later live status is separate from this immutable source snapshot.
