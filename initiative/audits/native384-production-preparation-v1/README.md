# Native384 rank32 production admission and preservation

This implements the full-resolution fold-0 experiment's CPU/GPU admission,
initial production launch and private epoch preservation. The matched scientific
config remains SHA `e7ab923745a22995470ff68d98b46212acd1f4d12c10b07004851027ad488788`,
and training remains the unchanged C9 checkout. This is not a trained model,
score or completed full-cache claim. Prepared/staged receipts are separate from
actual scheduler dispatch and optimizer evidence.

## Exact roles

- Assembly: existing CPU-only job **866458**, after source lanes 866426/866427.
  Require terminal `COMPLETED 0:0`, exact assembly launch, all 75 private source
  receipts, 3,403 studies, 18,906 series and 37,812 restored cache files.
- Control: `/home/sa305415/rsna-knee/production-launch-native384-a32-c9-v1`.
- New initial output:
  `/home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a32-native384-fold0-c9fcc7e-production-v1`.
- Native CPU preflight: one CPU, zero GPUs on evc31. Check every restored byte,
  exact 2,247/1,156 train/holdout membership, six frozen input artifacts, official
  model/config and environment archive hashes, source identity and prior native
  deterministic-recovery proof. Retain all file stat signatures.
- Fresh CPU-only recheck on evc31 precedes any initial GPU dispatch. It checks
  exact inventory, ownership, inode/device/size/mtime/ctime and the bound assets.
  Missing node-local cache or changed bytes prevent GPU allocation. The earlier
  CPU byte preflight must itself be terminal successful.
- Production: one H100, two CPUs on evc31, ten epochs, the eight-study effective
  batch and exact rank32 recipe. Use the retained native384 cache, not the old
  256-pixel tar. Recheck files/device/runtime in the allocation, preserve all
  215 accepted Python-header hashes, and run through the accepted deterministic
  wrapper and checkpoint supervisor. Clean only the job-owned environment;
  retain the separate assembly cache for explicit recovery.

The intended scientific change is spatial resolution. The older sampled224
baseline did not use strict deterministic algorithms; reconstructed256 rounding
differences are bounded by the source-shard checks. Those are documented
comparison limitations, not proven explanations of model accuracy.

## Coordinator commands

Read-only inspection, safe while the source lanes are live:

```sh
PYTHONNOUSERSITE=1 artifacts/qa-env/bin/python \
  scripts/newton/dispatch_native384_production.py check --ssh-control SOCKET
```

Only when the check says `ready_for_cpu_preflight`, call the same coordinator
with `cpu`. This submits the one CPU byte audit once, recording local and remote
intents/results. After its actual terminal success, `gpu` performs the fresh
CPU-only node check, rechecks Slurm and funded quota, then submits one initial
production job. A local or remote existing intent/run blocks redispatch. Any
unknown mutation requires read-only reconciliation; no automatic retry. Prepared
payloads and a successful CPU receipt do not constitute a GPU dispatch.

The resource check funds the existing 5,246,914,048-byte shared obligation plus
12 MiB of native admission metadata, twenty future entries per uncompleted source
unit and 190 assembly/production entries, while retaining 20 GiB and 5,000 files.
Never exceed two GPU nodes. The scientific code, accepted masks and folds stay
unchanged. Initial dispatch does not implement automatic continuation: a terminal
interruption requires actual checkpoint/stop-receipt review before a distinct
explicit resume launch.

## Each epoch is a distinct private publication

All ten epochs use the trainer's scheduled immutable snapshots. The new
`scripts/hf/dispatch_native384_epoch.py` coordinator verifies source/helper pins,
actual completed sidecar, exact role and unattempted private path before streaming
the ignored local token through stdin to a one-CPU zero-GPU publisher.

`scripts/hf/publish_native384_epoch.py` refuses rank64, old cache, diagnostic,
different population, changed membership/runtime/source/projection or partial
epochs. Its 128 MiB checkpoint cap matches the funded rank32 delta bound. It binds
and preserves `native-input-provenance.json`: actual assembly, all-cache CPU
hash proof, unchanged GPU signatures, source/config/environment/wrapper and
real allocated CUDA evidence. Each epoch includes the checkpoint, snapshot,
official license, native provenance and manifest. Every object is fully
downloaded and size/SHA checked at the returned immutable private revision.

The publisher takes the same pinned atomic directory exclusion as the live
source lanes before reading the private parent revision or committing. This
supports Lustre's `localflock` setting. Existing dispatch/publication intent or
path requires read-only reconciliation. Never publish a diagnostic fixture,
overwrite an existing epoch or imply a Kaggle score.

## Verification scope

Seventy-one focused cases pass: real synthetic-DICOM-derived native cache bytes
and CPU/GPU handoff corruption, exact frozen full-cohort membership, terminal CPU
failures, role/provenance faults, already-attempted dispatches, quota/node limits,
actual factory configuration and source pins, and shell parsing. Small byte
fixtures are not full-cohort admission. Full-cohort registry fixtures test
membership with synthetic receipt placeholders and are never published or used
for training. Runtime/real CUDA acceptance still requires actual cluster execution.

See `production-files.json` for all staged source byte bindings. The baseline
comparison remains fixed-final internal macro AUC 0.8620940794159951 and scored
sampled224 Kaggle entry 56697342 at 0.769. Rank64 entry 56740377 still had no
reported outcome or score at the October 1 05:12 UTC read-only check. No new
Kaggle submission or external-data admission occurs in this milestone.

The full repository suite also passed 1,411 cases; Ruff, formatting, mypy, archive
integrity and complete v2 evidence coverage passed. Source shards0–10 are privately
preserved and canonically re-admitted:489 studies,2712 series,95034 slices.
The last read-only source jobs remain RUNNING and assembly remains PENDING for
Dependency; no native CPU-preflight or GPU-production job is dispatched. See
`qa.json`, `source-live-inspection.json`, and `kaggle-readonly-outcome.json`.
