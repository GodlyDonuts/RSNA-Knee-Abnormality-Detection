# Solo DINOv3 A32 implementation and launch

September 29, 2026. The owner asked to implement the solo DINOv3 improvement and
get it running before adding teachers or ensembles. The active goal covers
implementation, real allocated-device acceptance, training and measured progress.

## Implemented

`rsna/competitive/adapters.py` adds rank-stabilized residual linear updates on
all 24 blocks' query/key/value/output projections and the final 12 blocks' MLP
projections. A32 has exactly 10,223,616 adapter parameters in 120 modules, plus
the existing 203,020-parameter study head. Original weight and bias state names
are preserved; A/B tensors are separately named. Base parameters remain frozen.

The trainer admits adapters to their own optimizer group even with the original
encoder frozen, leaves the adapted encoder in training mode, and checks nonzero
adapter gradients on the first update. Non-reentrant activation checkpointing
allows backpropagation with frozen input/base tensors. Checkpoint-only inference
reconstructs the exact adapter graph before strict state loading. A separately
recorded warmup/cosine schedule uses absolute update counters during resume.

Local tests cover zero-residual exact equality on a real HF DINOv3 graph, frozen
base tensor identity after optimization, finite adapter gradients and updates,
production-width/depth module enumeration and count, checkpointed/direct gradient
equality, exact optimizer/dropout resume, complete checkpoint-only restore and
rejection of missing adapters or invalid policies. These are implementation tests,
not evidence of MRI classification improvement. See [local-acceptance.json](local-acceptance.json).

## First experiment

- Fold 0, accepted masks/folds and official pinned DINOv3-L source unchanged.
- Initial A32 production comparison: ten epochs, 2,247 eligible training studies,
  effective batch eight, 16 slices per series, six series, 224 inputs from the
  existing cache, global-CLS study aggregation, no data augmentation.
- Adapter LR 1e-4, head LR 3e-4, alpha 8/sqrt(32), dropout 0.05, warmup 141
  updates then cosine, BF16 only after an actual H100 pilot passes.
- One H100/two CPUs for the bounded pilot. It uses 16 training and 16 outer
  studies for two epochs. Compare uninterrupted versus stop-after-one-update
  plus resume, then compare trained base tensors against the exact official file.
  Reload a complete trained model from its checkpoint and reproduce a real MRI
  prediction. Its tiny holdout metric is diagnostic, never competitive evidence.
- Production dispatch requires terminal pilot success and fresh job/quota checks.
  No duplicate output paths or dispatches. At most two GPU nodes total.
- A frozen-encoder control with the same BF16/head schedule is prepared, so changes
  in precision and schedule do not get misattributed to adapters. Launch it only
  after the first pilot passes and resource/storage acceptance allows useful work.

The first comparison isolates the new training recipe using existing images and
supervision. Native-detail inputs and depth context follow only after the adapter
produces measurable held-out evidence. No teacher predictions, public labels or
ensemble outputs enter this run. Development folds are already used for development;
the reserved 441-study partition remains confirmation-only.

## Storage and provenance

Newton was authenticated and had no active account jobs at first inspection.
Only about 7 GiB remained under its Lustre soft byte quota. Twenty-five redundant
old immutable snapshots were matched to prior full-download receipts and freshly
queried private HF LFS hashes/sizes; a one-CPU job verified every local file's full
SHA before deleting any. All final recovery checkpoints, fold-0 epoch 8, original
data, masks, folds, sidecars, run records and private HF objects were retained.
The first CPU invocation failed before any mutation because the system Python
lacked optional imports; a distinct standalone standard-library script completed
after read-only confirmation that neither retention receipt nor intent existed.
The [plan](storage-retention-plan.json) and [receipt](storage-retention-receipt.json)
preserve exact retained-copy lineage. It reclaimed 30,397,314,545 bytes.

Fresh quota/job inspection, exact source/config/script hashes, dispatch intent,
Slurm identity, actual logs and GPU acceptance receipts must be inspected before
claiming a launched or successful training run. Every completed production epoch
is to be privately preserved and fully downloaded/hash verified once. The active
goal is not complete merely because code or a pilot passes.

## Allocated pilot findings and correction

The first two distinct pilots, Slurm 864755 on evc45 and 864757 on evc22,
terminated during bounded allocated-driver checks before model construction.
They are terminal and their dispatches are preserved. Pilot 864759 on evc32
passed both allocated NVML/CUDA checks and a real torch CUDA context, staged the
verified environment/cache and reached its first MRI forward. It failed before
any optimizer update with an exact BF16 indexed-write dtype mismatch in
`CompetitiveStudyNet.series_features`. CUDA autocast keeps some normalization
outputs FP32 while linear output is BF16; an indexed destination requires matching
dtypes. The correction explicitly converts the series embedding to the destination
buffer dtype. CPU autocast regression tests reproduce that mixed dtype boundary,
check all three aggregation modes, padding invariance and finite gradients.

See [pilot-three-terminal.txt](pilot-three-terminal.txt). This is an evidenced
code correction, not an inference about prior Kaggle failures. A new pinned source
and distinct pilot output are required; no production run has been submitted.

Three additional redundant baseline fold-2 snapshots (epochs 2/4/6, retaining
best epoch 8 and final) were freshly checked against private HF metadata and prior
full-download receipts, then fully hashed locally before removal. See the second
[plan](storage-retention-plan2.json) and [receipt](storage-retention-receipt2.json).
Current soft-byte reserve is about 37.5 GiB and file reserve about 8,500, subject
to a fresh check before dispatch. Old source data, masks/folds and final checkpoints
remain intact. No competition submission was made in this implementation milestone.

## Corrected pilot dispatch

Corrected training source is Git `6f1ee6d332de29a8d6e5807aeac4777620865933`.
It is staged as a clean sparse shared checkout at
`/home/sa305415/rsna-knee/code-adapter-a32-6f1ee6d`. The corrected distinct
pilot is Slurm **864825**, submitted once, RUNNING on evc32 at its initial check.
Its private control directory is
`/home/sa305415/rsna-knee/production-launch-adapter-a32-6f1ee6d-v1` and unique
output root is `runs/dinov3-l16-adapter-a32-fold0-6f1ee6d-pilot-v1`.
The [launch plan](bf16-fixed-launch-plan.json), script, precheck, intent and
result preserve exact dispatch lineage. Source acceptance passed 1,080 tests
and repository source/archive/coverage checks. This dispatch is a diagnostic
pilot, not completed training or competitive holdout evidence.

Corrected production has a distinct prepared script/config, remains unsubmitted,
and requires the pilot's terminal `pilot-verified.json` and fresh quota/job checks.
The A32 epoch preservation coordinator and publisher are bound to this corrected
Git, fold-0 schedule and unique private HF prefix. It requires each immutable
completed epoch and exact adapter/base/source lineage before upload, then verifies
all four objects by downloading all bytes. No upload is repeated on uncertainty.
