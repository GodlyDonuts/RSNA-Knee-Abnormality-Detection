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
