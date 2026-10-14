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

## Optional concurrent two-T4 engineering mode (prepared, not accepted)

Kaggle's "T4 x2" session provisions and bills two T4s. The sequential path above uses only device 0. `--parallel` is an engineering-only option that uses both devices:
- **Build switch:** `window_bag_notebook.build(parallel=True)` embeds `WINDOW_BAG_PARALLEL = True`. The coordinator contract carries `parallel_arms`, and `bound_in_source` refuses a source that does not embed the same value. `parallel_arms` is refused for segments.
- **Two isolated processes:** the mean and query arms each run as their own single-GPU chain of fresh trainer processes, on `CUDA_VISIBLE_DEVICES=0` (mean) and `1` (query).
  - Each process uses `OMP/MKL/OPENBLAS=2` threads.
  - Each needs at least 4 schedulable CPUs, and exactly 2 visible T4s are required.
  - This is not DDP: `WORLD_SIZE=1`. Config, dataset, RNG, loss, schedule and effective-8 recipe are identical to sequential mode.
- **CPU gate first:** the CPU input/initialization admission finishes before either optimizer starts. That admission covers cohort, 16+16 tensors, full rosters/folds, and official init/RNG. After it, `freeze_cache` makes the restored admitted cache read-only (0444/0555) and records its content digest. Both arms share that cache read-only. The digest must be unchanged after both arms finish.
- **Separate outputs:** each arm writes its own exclusive output directory `<out>/<arm>` and its own exclusive `O_EXCL` journal `<out>/<arm>-journal.jsonl`. The journal records start/exit/pid/device/returncode/seconds and is fsynced.
- **Independent recovery:** each arm runs the real full/stop1/resume sequence on its own. The actual interrupted checkpoint is copied before resume, and `compare_recovery`/`validate_state` are applied per arm. The first-update-loss equality check across arms is kept.
- **Failure handling:** `Supervisor` starts every trainer in its own process group. On the first failure of either arm (nonzero exit, cap, deadline or exception), `run_concurrent` stops the sibling: SIGTERM to the group, a 30 s grace period, then SIGKILL. It reaps every owned process, including grandchildren in the group. No new trainer process starts after a stop.
- **Combined caps:** every 2 s poll measures the coordinator RSS plus the RSS of all live owned process groups, against the 29 GiB ceiling. It also checks combined output+cache bytes under `<out>` (6 GiB cap), the 3 GiB scratch reserve and the session deadline. The receipt records peak combined RSS and output, devices, threads, the shared-cache digest and the journal SHAs.
- **Quota:** a session is charged `timeout × 2` exactly once, the same as the sequential session (`provisioned_GPUs_charged: 2`, `GPUs_used_concurrently: 1|2`). Both arms in one session therefore halve the GPU charge per matched pair.
- **Projection:** `production_plan(..., concurrent=True)` gives a projection only. It halves the session count and charge, and adds one more arm's checkpoints to the scratch and output caps. Segments and recipe stay unchanged. Its timings must come from an actual concurrent T4 engineering session. A paired production segment runner is not implemented, and production funding still requires a sequential `GPUs_used_concurrently == 1` engineering receipt.
- **Receipt status:** `source_window_bag_two_arm_actual_concurrent_two_T4_canonical_recovery_accepted`. It can only come from an actual Kaggle run. The CPU tests use real subprocesses and the canonical trainer on CPU, which is not parallel acceptance. Root decides after actual measurements.

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
  - It stops at a scheduled optimizer boundary (`--stop-after-updates`).
  - Epoch snapshots stay immutable through the trainer's scheduled cadence (`submission_every_epochs=1`).
- **Publication (`window_bag_t4.publish_segment`):** each segment writes a write-once private notebook output under `/kaggle/working/window-bag-segment/`, plus `window-bag-segment-accepted.json` beside it.
  - Exact allowlist: the continuation `checkpoint.pt`, `run.json`, `invocation.json`, every completed `epoch-checkpoints/epoch-NNN.pt` with its sidecar, and `config.json`/`segment.stdout`/`segment.stderr`. The final segment also writes `result.json`/`outer_predictions.csv`.
  - Any other file under the run directory is refused. The MRI cache, staged code, private fixtures, capsule and reports stay in `/tmp` and are never published.
  - Every file is created with `O_EXCL`, fsynced, re-hashed and made read-only. Each epoch file must match its sidecar SHA.
  - `segment-publication.json` lists each file's size and SHA. The total must fit the contract's `output_cap_bytes` and the API-observed `platform_limits.max_output_bytes`, with a 1 GiB free-space reserve.
  - The cap is (complete checkpoints the segment produces) × measured `checkpoint_bytes` + 256 MiB of sidecars. Epoch snapshots are hard links in scratch but separate files in the output.
- **Previous-output identity:** a resumed contract binds `previous_output` = {kernel slug, version 1, source SHA, receipt SHA, publication-manifest SHA, checkpoint SHA}. A free-floating checkpoint SHA is refused.
  - `kernel_request` attaches that kernel slug to `kernelDataSources`.
  - Before any intent, the coordinator reads back the previous kernel and checks all of the following:
    - it is private, offline, T4/GPU and COMPLETE;
    - its source SHA and exact input lists match, including its own previous kernel;
    - its downloaded receipt hashes to the bound SHA and is a non-evaluated segment for the same arm, contiguous and bound to the same engineering/config;
    - its publication and checkpoint SHAs match.
  - Inside the session, `restore_previous` re-verifies the mounted receipt, manifest and every member hash before resume.
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
- **Funding verification (`verify_engineering`):** a truthy or nonempty `production_funding` is not enough. Before any segment intent or SaveKernel, the coordinator checks all of the following:
  - The funding names the engineering kernel (slug, version 1, source SHA, receipt SHA).
  - That kernel is read back as private, offline, T4, COMPLETE, on the same runtime image, with the 20 shards and the same Datasets.
  - Its downloaded `window-bag-engineering-accepted.json` hashes to the bound SHA and reports `ENGINEERING_STATUS` for both arms.
  - The receipt carries the same science commit, package manifest, fixture, CPU admission, index and shard manifests.
  - The receipt's per-arm production config SHA equals the segment's.
  - The receipt's measured seconds/update, startup, checkpoint and final evaluation equal the funding's.
  - Recomputing `production_plan` from those measurements, fresh quota and the receipt's observed scratch reproduces the funded segment schedule. That plan must fit and stay within the authorized GPU seconds.
  - The contract's output cap equals the recomputed cap.
- **Expectation:** a rough range, inferred and not measured, is ~4x the one-window encoder views. That suggests production may not fit the weekly Kaggle allowance under conservative accounting, in which case Kaggle serves independent engineering only.

## Verification here

The CPU tests in `tests/test_kaggle_window_bag.py` cover:
- the real producer/shard code, rebuilding byte-identical index bytes from mounted shard outputs and decoding only the requested members;
- refusal of tampering;
- the immutable package;
- quota/limit/coordinator behaviour with fakes;
- complete recovery across an actual session-output boundary for both arms on the canonical trainer, using a tiny engineering encoder. Session 1 stops at the epoch-1 boundary and publishes. Its scratch is deleted. The publication is copied to a read-only mounted input and restored by `restore_previous`. A fresh process then resumes to bit-equal state, predictions and epoch-2 snapshot;
- write-once/capped/cache-free publication, and refusal of tampered members, receipt, publication or chain;
- that the resumed SaveKernel request mounts the verified previous kernel;
- the concurrent mode on CPU: both arms run the real full/stop1/resume concurrently through `Supervisor` over one read-only cache, and each reaches bit-equal state/predictions. A failing arm stops and reaps its sibling and grandchild. The combined output and RSS caps stop both arms. Cache drift is detected. The concurrent plan charges both T4s once per shared session;
- that forged funding (a bare nonempty dict, wrong receipt/measurements/schedule/source/image/config/package/GPU seconds/output cap) fails with no intent and no SaveKernel.

Engineering tests are not scores, accuracy, eligibility or discovery claims.

The original H100 two-arm recovery has since actually COMPLETED 0 for both arms (receipt `0700d08b`, preserved in `source-window-bag-GPU-terminal-20261014-v1`). That is H100 recovery, not T4 acceptance. The notebook's `CUDA_VISIBLE_DEVICES=0` intentionally selects one T4, while quota charges both provisioned GPUs.

## Actual input-shape bound, output cap and omitted repeated fields (issue #55 follow-up)

- **Input shape:** the only admitted `platform_limits` is `plan.admitted_platform_limits()`, bound to the actual private offline CPU kernel `138609817/v1` (`rsna-input-shape-cpu-21x3-20261014-v1`, source `e06b3842`). It COMPLETED with exactly 21 kernel inputs and the same 3 Datasets (ids `12333543`, `12139697`, `12567458`).
  - A test re-reads `kaggle-input-shape-CPU-20261014-v1/verified.json` and `contract.json`. It checks the slug, version, source, counts, the 20 shard-manifest SHAs and the Dataset ids.
  - The coordinator refuses any other proof, including a claimed limit above 21/3, and model/capsule Dataset ids other than the probe's.
  - This is a supported-shape bound, not a platform maximum. An engineering session mounts 20+3 and a resumed segment mounts 21+3.
- **Output cap:** Kaggle documents 20 GB of saved `/kaggle/working` output. Every segment is held to an explicit `OUTPUT_CAP_CEILING_BYTES` = 16 GiB.
  - `segment_contract`, `require_platform_limits` and `production_plan["output_caps_fit"]` enforce it, including the two-arm concurrent projection.
  - The engineering cap stays 6 GiB.
- **Omitted repeated fields:** GetKernel can omit an empty repeated field or return it as null. `repeated()` reads either as `[]` for `datasetDataSources`, `kernelDataSources` and `competitionDataSources`.
  - Any non-list value is still refused.
  - A nonempty competition source is still drift.
