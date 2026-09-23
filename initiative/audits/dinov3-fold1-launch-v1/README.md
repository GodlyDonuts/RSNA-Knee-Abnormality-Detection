# DINOv3-L fold-1 launch

The owner asked to keep training after the completed ten-epoch fold-0 run.
Newton job **843914** was submitted once at 2026-09-23 18:13:04 UTC. It was
`PENDING (Resources)` at the first read-only scheduler check, so no fold-1
optimizer update or score was yet established.

The job uses the exact fold-0 training source Git
`b74060b98f01bff1b924070ccba5ce44dd002f40`, accepted report-supervision
release, folds, MRI cache, official pinned DINOv3-L weights, environment and
two-H100/four-CPU configuration. A parsed comparison confirms the sole training
configuration change is `fold: 0` to `fold: 1`; all other configuration values
are identical. The reviewed Slurm script retains the prior allocated-device,
Python-header, cache/environment, source and NCCL guards, and removes only the
fold-0 recovery proof and resume argument. It excludes the previously failing
nodes including evc27. `bash -n`, exact file hashes, a distinct absent output
path and an empty owner job queue passed before the single `sbatch` call.

The private Newton output path is
`/home/sa305415/rsna-knee/runs/dinov3-l16-production-fold1-b74060b-v1`.
The immutable launch inputs, intent and accepted scheduler receipt are preserved
here. Do not submit the same script again until job 843914 has been reconciled
read-only through Slurm, logs and run records. The cluster reported 931.8 GiB
used of a 1,000 GiB soft quota and 989,142 files of a 1,000,000-file soft quota
before dispatch; recheck quotas before another fold or checkpoint transfer.

The user's request to preserve every completed epoch privately still applies.
The trainer creates its scheduled even-epoch snapshots. CPU-only watcher
**843923** was submitted once with Slurm dependency `after:843914` and was
`PENDING (Dependency)` at the first check. It requests one CPU, no GPU, and
checks the training log and exact checkpoint binding at odd completed epochs
1, 3, 5, 7 and 9. Its step count is derived from the fold-1 training-study
count and DDP effective batch size, rather than the fold-0 count. A synthetic
epoch-1 capture and idempotent verification passed before staging. The exact
watcher code, job script and dispatch receipts are preserved here and hashed
against the private cluster copies. Reconcile job 843923 read-only before any
retry. Publishing to the private Hugging Face dataset still requires separate
per-epoch byte and SHA-256 verification.
