# Stanford full-source adapter fitting preparation

This follows the owner's instruction to use Stanford. The original 3750-volume
acquisition and eight-exam CPU acceptance are complete. Original source labels
remain separate from RSNA targets and original validation never enters fitting.

`rsna/competitive/mrnet_training.py` provides the full-source numerical engine:
1129 exams, three planes and ordered 16-slice inputs at the original 256-pixel
resolution, pinned official DINOv3-L rank32, separate three-task exam attention,
ten epochs, effective batch eight, 142 updates per epoch/1420 total. The smaller
one-exam final batch is included and normalized by its actual size. Encoder/head
learning rates are 1e-4/3e-4, AdamW decay1e-4, one-epoch linear warm-up followed by
cosine decay, global clipping1, BF16 forward/FP32 loss. Only the fixed final epoch
is evaluated on all119 source-validation exams; this is not RSNA or Kaggle AUC.

The engine retains every immutable completed epoch, exact base provenance,
learned tensors, AdamW, update histories, cursor and all RNG streams. The runner
`scripts/newton/train_mrnet_private.py` exercises that same engine in separate
full/stop-after-one/resume GPU processes before permitting full-source fitting.
Its receipt binds code/config/input/acquisition bytes, not just a success string.
The earlier pending four-update pilot tests the same model/data but a different
training loop and cannot silently stand in for this engine's GPU recovery.

Six CPU tests check full exam coverage, the last smaller batch, immutable base,
adapter updates, complete optimizer/RNG/order/dropout recovery and identical
validation predictions, plus refusal of changed membership/recipe/publication
scope or validation fitting. These are engineering evidence only. No full-source
GPU wrapper, dispatch, training result or accepted learned weights exist yet.

Keep source images, label values and all Stanford-derived weights private under
the previously recorded research scope. Competition/weight redistribution rights
remain unresolved. Before attributing a clean RSNA gain to source warm-up, screen
cross-source overlap and compare adapter transfer with a matched extra-compute
RSNA-only control on unchanged development membership. Protected gold and
confirmation remain unused. Source-task accuracy is not evidence of transfer.

## Earlier-node restoration preparation

The original native867367 and Stanford867419 were both pending on their owning
cache nodes at19:23UTC. A scheduler-only `sbatch --test-only` query for the original
Stanford batch on evc24 predicted17:18:40 Eastern; the original evc32 job forecast
was22:10:33 Eastern. Test-only did not dispatch a job or prove CUDA health.

`restore_mrnet_private.py` is prepared for a single private intra-cluster copy,
oneCPU/zeroGPU at each end. It streams only the exact original metadata/volume
whitelist through RAM/stdin, verifies every copied volume against the original
SHA journal, and reproduces the original eight image/label tensor records with
the original accepted CPU code/runtime. Raw pixels are never staged on shared
storage or downloaded again from Stanford. Both originals are preserved.

`launch_mrnet_restored_probe.py` changes only data locality admission; the
original immutable F2d scientific archive/model/worker/config are reused. It
requires the original acquisition/input and actual new restoration hashes and
owning node. Only after that CPU restoration passes may the original still-
pending pilot be cancelled, its actual terminal state reconciled, and a distinct
new-node pilot dispatched once. No duplicate equivalent GPU jobs or new source
recipe are authorized by a scheduler forecast. The native production stays intact.

## Actual pilot and restoration outcomes supersede the forecasts

Original867419 ran through Slurm backfill on evc32 and completed0:0 in2:38.
It made four finite updates in each full/resumed trajectory; full/stop1/resume
matched complete learned/base/AdamW/LR/counter/allRNG and prediction bytes.
Actual loss and gradient records are preserved in original-pilot-terminal/.
Terminal checkpoint bytes were reverified on the owning node with CPU-only
reader867473. No diagnostic weights were republished and no accuracy is claimed.

One private restoration also completed: sender867456 on evc32 and receiver867457
on evc24, both COMPLETED0:0 (2:16/2:39). All3750 original volumes/7740871424bytes
passed exact SHA checks; eight original image/label tensors were reproduced
exactly. Receipt7367116f9f3775860404aed880d0e658bfa1b3633d33e5401466d8af1d1c8cdc
binds retained /tmp/rsna-mrnet-restored-0nri0agx on evc24. Read restoration/.
The SSH response was lost, but original terminal/receipt read-only reconciliation
proved success; no transfer was repeated. Both original and restored copies remain.

The proposed replacement stopped at its read-only preflight because the original
pilot had already finished. It was never cancelled, replaced or repeated. The
new full-source engine's immutable phase wrapper is now prepared for a distinct
actual GPU recovery, then separately funded full private fitting. It borrows the
accepted allocated-device/header/runtime checks and original evc32 data/input
proofs. Each phase has a unique control/output and cannot accept an old pilot as
a new engine pass. Full-source snapshots/weights remain private on the owning
node; no new RSNA transfer or competitive score exists.

## Exact full-engine recovery dispatched once

New sourceF12a319 full-engine recovery867478 was submitted ONCE on evc32,
oneH100/twoCPUs. Read full-engine-dispatch/. It runs the actual full-fitting
engine with sixteen source-training exams/two epochs/four updates, plus four
validation engineering outputs, comparing complete fresh full/stop1/resume
envelopes and exact prediction bytes. It is not full1129-exam fitting.

The original scientific/config/code remain unchanged. While still pending, its
wall cap was reduced from60 to10minutes in a recorded scheduler update, using the
original measured158-second whole-job profile with over3x allowance. This avoids
requesting an unnecessarily large backfill window; it is the same job, no restart
or duplicate. Funded3GiB node-local retained states plus pinned runtime and20GiB
reserve; all existing shared liabilities plus32MiB/96small objects leave20GiB/
5000files. Only after actual exact-engine recovery acceptance may full source
fitting be dispatched, with separately reviewed wall/runtime/checkpoint capacity.
