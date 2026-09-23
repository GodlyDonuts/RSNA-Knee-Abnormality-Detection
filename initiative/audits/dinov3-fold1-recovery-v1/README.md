# Fold-1 allocated-device failure and bounded retry

Newton allocated the first fold-1 training job **843914** on evc43 and its
one-CPU watcher **843923** on evc5. The allocated-device inventory and full
Python-header and runtime/cache preflight passed. At the start of training,
PyTorch rank 1 failed in `torch.cuda.set_device` with
`cudaErrorDevicesUnavailable` / `cuDevicePrimaryCtxRetain` error 999. Torchrun
stopped rank 0. Slurm recorded `FAILED 1:0` for job 843914 after 6:03.
The only run output was a zero-byte `.run.lock`: there is no `run.json`,
optimizer update, checkpoint, epoch, fold-1 score or Kaggle submission. The
dependent watcher correctly exited `FAILED 1:0` after the primary job ended
without an epoch. Its lack of snapshots is expected for this attempt.

The original stdout/stderr, verified node-header receipt and scheduler states
are preserved under `failed-843914/` and `scheduler-reconciliation.txt`.
This is an allocated CUDA-context failure on evc43. The inventory passing
does not establish that both GPUs can create a context. The exact cause on
that node has not been determined; the evidence does not implicate the
accepted MRI data or training labels.

One retry training job **843935** was submitted at 2026-09-23 18:39:54 UTC,
with a new, distinct output directory. It keeps the identical fold-1 config,
accepted data and Git `b74060b98f01bff1b924070ccba5ce44dd002f40`.
The reviewed Slurm script excludes evc43 in addition to the previously
excluded nodes and runs a two-rank CUDA-context plus NCCL collective probe
after extracting the pinned environment but before unpacking the 82.8-GB
cache. This fails quickly if an allocated GPU cannot actually be used.
Only those lines, the job/control/output names and the new probe differ from
the original script; `bash -n`, source hashes, unchanged config hash and the
fresh output path passed before the single `sbatch` call. CPU-only watcher
**843936** was submitted once with dependency `after:843935`, using the same
verified odd-epoch archive code but the new job log and output path. Both
jobs were `PENDING` at the first post-dispatch check. The exact plans,
scripts, probes, intents and accepted scheduler receipts are under
`retry-843935/`.

Reconcile jobs 843935 and 843936 read-only before any further scheduler
mutation. The user still expects every completed epoch privately preserved;
the watcher captures odd epochs 1/3/5/7/9, and the trainer saves even
epochs 2/4/6/8/10. A snapshot is not a private Hugging Face publication
until its full upload and download/hash verification receipt exists.
