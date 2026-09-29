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

The source/dispatch milestone was privately mirrored at HF revision
`07af9ff00b5cb9475bb7a7005c3009f080a0c907`, with all 44 objects fully
re-downloaded and byte/SHA verified. This preserves implementation and receipts,
not a newly trained checkpoint or score. The later optional retention precheck
for five baseline fold-2 odd snapshots found those files already absent; it
refused before an intent or deletion. That unexecuted proposal is ignored local
evidence only and is not counted as reclaimed space. Production storage budgeting
will use the actual pilot checkpoint size and hard-link snapshot behavior.

## Intentional-stop recovery

Corrected pilot 864825 completed its uninterrupted two-epoch diagnostic run
(16 training and 16 outer studies, four finite optimizer updates), with exactly
10,426,636 trainable parameters including 10,223,616 in the encoder. It then
completed the intentional one-update stop and wrote a valid recovery checkpoint.
The shell harness expected raw exit 75; torchrun wraps that worker exit as its
own exit 1, so the harness ended the allocation (terminal FAILED 2:0) without
resuming. This is not a model optimizer failure.

The [terminal receipt](pilot-four-terminal.txt) and
[full diagnostic summary](pilot-full-diagnostic-summary.json) are preserved.
A CPU-only one-core Python 3.12 check validated the stopped checkpoint's entire
SHA and exact binding against the trainer's stop receipt. A prior read-only check
using the system Python failed on its older syntax support before any mutation.
Distinct **resume-only Slurm 864840** was dispatched once on evc32. It preserves
the original complete full-run output and resumes only the existing valid partial
output under unchanged source/config. It repeats neither the full run nor the
stopped update. It must finish and write terminal `pilot-verified.json` before
production. The existing supervisor remains deliberately conservative about
unsolicited worker exits; launch-script handling of this deliberate stop is corrected
by validating the exact receipt, rather than accepting a bare launcher code.

The actual pilot checkpoint is 1,338,164,716 bytes. Production worst-case budgeting
allows eleven unique checkpoint inodes (ten immutable snapshots and a transient
recovery save) plus the two pilot checkpoints. Against the original 37.54 GiB
soft-byte reserve, projected remaining space is 21.34 GiB; use a fresh quota check
with overhead before dispatch. The prepared production script now selects the
same CUDA-accepted evc32 node, one H100 and two CPUs, and is still unsubmitted.

## Terminal acceptance and primary production

Resume-only job **864840 completed 0:0**. The
[pilot verification](pilot-verified.json) confirms full model/optimizer/RNG payload
equality, byte-identical outer predictions, all 120 residual modules changed, all
original official base tensors unchanged and checkpoint-only real MRI prediction
reproduction. Four-update/16-study diagnostic results are not accuracy evidence.

After a fresh clean-source/config/script/publisher/Slurm/output/quota check, primary
production **864856** was submitted once and is RUNNING on evc32 at dispatch.
It trains fold 0 for ten epochs on 2,247 studies, 281 effective-eight updates per
epoch, with immutable snapshots every epoch. Its unique output is
`/home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a32-fold0-6f1ee6d-production-v1`;
control is `production-launch-adapter-a32-6f1ee6d-v1`, source is the clean
`code-adapter-a32-6f1ee6d` checkout. See
[the intent](production-dispatch-intent.json),
[result](production-dispatch-result.json) and quota precheck. One H100/two CPUs,
12-hour bounded allocation; inspect exact stop/final state before resuming.

The publisher/coordinator preserve completed epochs under the private prefix
`experiments/dinov3-v1/training-epochs/adapter-a32-fold0-6f1ee6d-seed20260922`.
Inspect each immutable sidecar and existing local/remote HF intent before action.
Never repeat an uncertain upload; validate all downloaded bytes and exact license,
source, fold, official weight and encoder-adapter lineage.

At dispatch the remaining soft-byte space was 35.04 GiB after the two pilot
checkpoints; worst-case production checkpoints leave a projected 21.33 GiB.
Do not start the prepared control or any additional GPU work without fresh storage
and allocation acceptance. No Kaggle candidate was submitted or score obtained
for this adapter. Training code is also available as a 52-file, 696,320-byte exact
Git deployment projection in ignored local artifacts; all members are SHA checked
and include adapters, with no weights/data/credentials or notebook execution.

## Real production learning acceptance

The [production acceptance](production-progress-acceptance.json) records ten
actual optimizer updates on the full accepted 2,247-study training pool with a
1,156-study outer pool, finite loss/gradient events, nonzero encoder-adapter
gradients and zero stderr bytes. Early steady throughput was 9.19 seconds per
effective-eight update, approximately 43 minutes per 281-update epoch before
evaluation and variable case/checkpoint overhead. This is a small timing sample;
no new competitive metric or epoch publication exists at this acceptance point.

The terminal pilot and primary dispatch milestone was privately preserved at HF
revision `bf7bc42ebfd6c38555dd1a77b73511a1145d47eb` with all 57 objects fully
re-downloaded and hash verified. Active goal: obtain honest held-out evidence,
continue solo DINOv3 improvements (native detail and slice context next), and
measure competition performance before introducing other models/ensembles.
Do not call the goal achieved based on this launch or its diagnostic pilot.

Real production progress and its exact source/acceptance were privately mirrored
at HF revision `530e692aee440bbf94e9b6ee1902f2fbe436f711`, all 58 objects
fully re-downloaded and size/SHA verified. A later live read still showed job
864856 RUNNING on evc32, 27 completed optimizer updates, finite losses/gradients
and zero stderr bytes. This observation does not claim a completed epoch or score.
