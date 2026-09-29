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
