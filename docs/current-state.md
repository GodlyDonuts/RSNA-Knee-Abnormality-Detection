# Current RSNA Knee research state

Updated October19,2026,22:14UTC. [Agent guide](../AGENTS.md) · [Continuation](codex-continuation.md). Goal active;0.97/first place unachieved. This page supersedes historical snapshots retained in Git and original audits.

| Model | Actual Kaggle score | Entry |
| --- | --- | --- |
| External R6 reproduction | **0.958** | 57333150 |
| Released S6 reproduction | 0.950 | 57300990 |
| Our whole94 full encoder | 0.949 | 57336190 |
| Our influence IID16 full encoder | 0.949 | 57332456 |
| Our paired adaptation | 0.949 | 57392062 |
| Our higher encoderLR adaptation | 0.949 | 57394237 |
| Our uniform / source-label adaptations | 0.947 / 0.947 | 57329796 / 57331631 |
| R6 with our paired member | 0.957 | 57397292 |

The adapted backbone is full73M CoAtNet, not a small frozen head.0.958 is a reproduction; no original breakthrough/new student improvement is established. R6 member substitution underperformed its unchanged baseline. Full94 training already completed1458updates; an unchanged repeat was withdrawn as a duplicate. Neither extra compute nor passing engineering checks proves accuracy.

## Main improvement in progress

Matched parent-only versus complementary S6 cross-lineage equal-rank supervision, same MRI/labels/masks/full encoder/optimizer/LR/IID16/effective8/three epochs. Full3887teacher outputs are required before ranking. **3837/3887 (98.7%) completed** at actual phase61/973554; original final phase62/973659 is RUNNING on39,23workers/two-thread FP32. No partial ranks or new accuracy result. Public teachers are not independent OOF evidence; all-fitting training is not internal validation. [Original teacher receipts](../initiative/research/S6-cluster-resident-owner-20261019-v1/README.md).

Original follow-through owners remain live:55175 full-target handoff39→33;56997 exact source compiler;67258 actual48 recovery;67269 matched full fitting/six private epoch publications;71311 CPU packaging;72340 canonical actual-T4/Kaggle qualification and outcome handling. No main student GPU dispatched yet. Metadata89734 promotes only their exact future unstarted GPU JobIds to normal. Do not duplicate/restart/edit frozen owners. [Actual execution](../initiative/research/S6-earlier-H100-execution-20261019-v1/README.md), [packaging](../initiative/research/S6-fixed-final-packaging-20261019-v1/README.md), [Kaggle execution](../initiative/research/S6-final-Kaggle-execution-20261019-v1/README.md).

Node48 already has the original full38.87GB MRI inputs and678.56MB auxiliary packet, independently SHA verified once, matching runtime/strict73M FP32 model/full3887roster/eight94view tensors. Originals remain private/intact on33; no raw data/weights crossed Mac/Git/HF. [Actual admission](../initiative/research/S6-earlier-H100-route-20261019-v1/README.md).

## New objective and compute setback

A distinct [paired-view teacher ordering objective](../initiative/research/teacher-pair-full-encoder-20261019-v1/README.md) preserves source-label BCE and replaces half of teacher BCE by between-study ordering of mean two-view probabilities through the full encoder. It changes no serving capacity. Earlier frozen-head weak-source ranking is a different experiment.4942library tests and complete CPU recovery evidence pass; no clinical benefit yet.

Scout972928 failed in8seconds at CUDA admission on26, before MRI/gradient phases. Assigned-device diagnostics973573/973577 observed zero Torch devices and no NVML devices despite Slurm assignment; physical cause unproven. [Preserved negative](../initiative/research/teacher-pair-earlier26-20261019-v1/README.md). No isolation bypass or blind retry.

Distinct unchanged scout**973601** is **PENDING Priority on48**, oneH100/twoCPU/32GiB/20minutes/normal. CPU973600 admitted its source/original32 packet without a new bulk transfer. Observer**94271** follows its actual terminal only. [Exact fallback](../initiative/research/teacher-pair-admitted48-fallback-20261019-v1/README.md).

Coherent48 optimizer-recovery source**24e96a77** is privately preserved/download-hash verified at HF**1f8155b37096dcc58af3329c6aa8b73b23045252**,25objects142028bytes viaCPU973647. Live recovery**95036** and collector**95037** wait only actual973601 positive gradients; no recovery GPU/production/accuracy admission yet. New controls correct an unlaunched26/48 host mismatch; numerical worker/decoder functions remain AST-identical, seven focused tests pass. [Source and owners](../initiative/research/teacher-pair-real-recovery48-20261019-v2/README.md).

An alternative ordinary2V100 node11 had one unallocated32GB GPU. Its bounded assigned-driver probe973661 timed out87seconds without a cuInit/count result. No runtime, model, MRI transfer or fitting occurred; that alternative remains unadmitted. CUDA12.6 support exists in official PyTorch2.14 wheels, but current CUDA13 binaries cannot run on Volta. This is a resource observation, not an accuracy experiment.

## Evidence and next action

Finish original final teacher phase and follow the existing matched training pipeline. Separately collect actual973601 gradients; only a positive result can dispatch the prepared real optimizer recovery. Require actual full/stop/resume state/CSV equality before any accuracy experiment. Queue predictions are not reliable ETAs. Protected gold/confirmation and accepted masks/labels remain untouched.

MedGemma native likelihood probes completed but failed letter/order/Unknown reliability criteria; no teacher labels adopted. Stanford weights remain private/noncompetition. No newly admitted KneeCoT. All raw clinical data/source bodies/arrays/weights stay private. GitHub contains own code and sanitized aggregate/hash receipts; private HF mirrors use the canonical shared lock and exact original proofs, never duplicate mutations.
