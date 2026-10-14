# Offline private Kaggle T4 path for the whole-window bag (issue #55)

This is prepared only. Nothing was pushed, no Dataset was created, no GPU was allocated, no
competition code was submitted, no pixels were downloaded and no account setting was changed.
Root decides whether Kaggle is used only for independent engineering or also for production.

## Engineering notebook (first deliverable)

| Part | Binding |
|---|---|
| Numerical package | The 69 files of the science commit `971d7dac` are read from Git objects. Each one must match `source-window-bag-GPU-recovery-dispatch-20261013-v1/stage-intent.json`. The current-tree `train.py`/`throughput.py` changes are never shipped. The only current files are `scripts/kaggle/window_bag_t4.py` and `window_bag_plan.py`, recorded under `bound_current`. |
| MRI inputs | The 20 original completed private CPU shard kernels (v1, ids in the committed shard audits) are mounted as kernel sources and located by exact shard-manifest SHA. Each is checked with `kaggle_source_render.verify_shard` against cohort `3f265ddf`. The trainer index is rebuilt and must be byte-identical to the original finalized index `84985528` (3403 members, 27,667,165,884 B). Only the 32 engineering members are decoded into `/tmp`. Nothing is re-rendered, re-downloaded or re-uploaded. |
| Release/folds/series | Taken from the existing private render-capsule Dataset 12567458/v1, archive `b0899dea`. Re-verified by release binding, full 2247/1156 roster SHAs, fitting-overlay `c8fd2e86` receipt and leakage disjointness. |
| Official base / runtime | The existing pinned model bundle Dataset 12333543, the shared wheelhouse 12139697 and the pinned runtime image, through the unchanged `notebook_entry.py` bootstrap. A branch is inserted at a single exact anchor. |
| Fixtures | Original public-input fixture `adb931d5` (16+16 tensor SHAs, rosters, cohort, supervision) and whole-window CPU admission `7417e363` (common mean/query graph/values/RNG, configs, code hashes). Root embeds the original bytes in the private notebook only. |

Before any GPU optimizer update, the CPU checks re-observe all of the following:
- the actual restored cohort binding;
- the 16 fitting + 16 outer tensor/label SHAs against the fixture;
- the official initialization/RNG of both arms against `7417e363`.

Then both arms run the unchanged trainer under `enforce_deterministic_runtime.py`, in fresh full, stop-after-1 and resume processes (2 epochs/4 updates). The actual interrupted checkpoint and invocation are copied before resume replaces them. Acceptance requires all of:
- `compare_recovery`: decoded learned/base/AdamW/LR/counters/all RNG/scaler/history, plus exact 16x12 MRI-only prediction bytes;
- `validate_state` against the fixture;
- an identical first-update loss for mean and query.

The original recipe is unchanged: classifier/window/sampling/supervision, effective 8, one sigmoid, no gold or confirmation. The optional release-admission cache is not used.

Envelope:
- one visible T4, timeout of 2 hours or less, no internet;
- a 6 GiB diagnostic output cap, a 3 GiB scratch reserve and a 29 GiB RSS ceiling.

The quota charge counts both provisioned T4s for the full timeout. With the dated issue observation (108000 − 2464.964 − 0 = 105535.036 s), one engineering session charges 14400 s. Root must re-query quota inside the coordinator claim before any mutation.

## Conditional single coordinator

`scripts/kaggle/window_bag_coordinator.py:run_phase` is meant to run inside `coordinator.single_coordinator` with the canonical ledger, like the original T4 recoveries.
- `preflight` (read-only) checks the following:
  - all 20 shard kernels are private, CPU, offline v1 and COMPLETE;
  - both Datasets are private v1, and the wheelhouse passes verification;
  - fresh conservative quota;
  - authenticated target absence;
  - the source SHA.
- `push` writes an exclusive intent and makes exactly one SaveKernel call. An exception or a returned ref/version that differs is recorded as uncertain. That case is reconciled read-only and never retried.
- `status` reads back the source, settings and the exact dataset/kernel/competition source lists.

The kernel-source and Dataset-source counts (20 + 3 for engineering, 21 + 3 for a resumed segment) are refused unless root records an actual platform/API constraint in `platform_limits`. No limit is assumed here.

## Fixed ten-epoch production continuation (plan, not funded)

- **Recipe:** 2247 fitting / 1156 fixed-final outer, effective 8, 281 updates per epoch and 2810 per arm, the same mean/query pair. Only the encoder paths and `min_free_gib` differ from the committed production configs.
- **Sessions:** each session is one bounded `segment`.
  - It restores all 3403 members (27.7 GB) from the same mounted shards.
  - It resumes only from a previous-segment `checkpoint.pt` whose SHA the contract binds; that output is mounted read-only as an extra kernel source.
  - It stops at a scheduled optimizer boundary (`--stop-after-updates`).
  - Epoch snapshots stay immutable through the trainer's scheduled cadence.
  - Only the final segment reaches epoch 10, which is the trainer's only outer evaluation. No intermediate epoch is evaluated or selected.
- **Sizing:** `window_bag_plan.production_plan` sizes segments from actual measured T4 seconds per update, startup, checkpoint and final evaluation. It uses 80% of 9 h. The scientific hyperparameters never change.
- **Chaining:** `next_segment` chains contracts and refuses skipped, non-contiguous, evaluated or arm-swapped receipts.
- **Gates:** the plan reports whether quota (2 provisioned T4 × timeout × sessions) and scratch currently fit.
- **Dispatch rule:** no full-fit dispatch until all of these exist:
  - the real T4 engineering acceptance;
  - measured performance;
  - a fresh allowance;
  - explicit checkpoint/output/storage funding;
  - an actual check of `/tmp` capacity in the T4 runtime.
- **Expectation:** a rough range, inferred and not measured, is ~4x the one-window encoder views. That suggests production may not fit the weekly Kaggle allowance under conservative accounting, in which case Kaggle serves independent engineering only.

## Verification here

The CPU tests in `tests/test_kaggle_window_bag.py` cover:
- the real producer/shard code, rebuilding byte-identical index bytes from mounted shard outputs and decoding only the requested members;
- refusal of tampering;
- the immutable package;
- quota/limit/coordinator behaviour with fakes;
- complete recovery across an actual process and read-only session-output boundary for both arms on the canonical trainer, using a tiny engineering encoder.

Engineering tests are not scores, accuracy, eligibility or discovery claims.

The original H100 two-arm recovery 85546 is still running. Its final receipt is absent and overall acceptance is not claimed. It is unaffected.
