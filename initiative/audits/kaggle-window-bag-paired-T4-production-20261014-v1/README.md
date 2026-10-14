# Paired two-T4 fixed-ten-epoch production continuation (issue #55)

Prepared only. No kernel was pushed and no Dataset was created. No GPU was allocated, no
cluster job was launched, no data was acquired and nothing was submitted. CPU stand-in tests
are not GPU acceptance. Root decides on dispatch after:
- the actual concurrent T4 engineering acceptance;
- a fresh quota observation;
- measured performance;
- explicit checkpoint/output/storage funding.

## What is new

| File | Role |
|---|---|
| `scripts/kaggle/window_bag_paired.py` | In-session runner. It also holds the paired plan, the contract and the publication/restore logic. |
| `scripts/kaggle/window_bag_paired_coordinator.py` | `preflight`/`status`/`push` phases for paired segments, run inside the canonical Newton coordinator claim. |
| `scripts/kaggle/window_bag_paired_notebook.py` | Builds the private paired segment notebook. It never pushes anything. |

The frozen engineering package is not changed: the `971d7dac` numerical files plus the
`window_bag_t4.py`/`window_bag_plan.py` bytes. The paired package is that package plus one
file, `window_bag_paired.py`. Its manifest embeds the exact engineering manifest SHA
(`engineering_package_manifest_sha256`); at current main it is `8d064f04`, the frozen package root recorded. The session recomputes that SHA from the shipped
manifest and refuses any mismatch with the contract. The coordinator refuses any mismatch
with the accepted concurrent engineering receipt. The mean/query configs, recipe, batch and
schedule are unchanged.

## Session

- **Processes:** one Kaggle "T4 x2" session runs two fresh canonical trainer processes, `mean` on `CUDA_VISIBLE_DEVICES=0` and `query` on `1`. Each process uses two CPU threads. They run under the existing `Supervisor`: separate process groups; the first failure stops the sibling and reaps its group; there is a combined RSS ceiling of 29 GiB and a combined scratch cap; deadlines apply.
- **Arm state:** each arm keeps its own model/base/AdamW/LR/cursors/all-RNG/scaler state and its own checkpoint. There is no DDP and no shared mutable state.
- **Cache:** the admitted cache is restored once from the 20 mounted original shards. It is then frozen read-only and shared by both arms. Its content digest must be unchanged after the session.
- **Previous output:** a resumed segment mounts exactly one previous paired kernel output. That single attachment carries both arms' continuations, so the request uses 20 shards + 1 previous output + 3 Datasets: the actual supported 21/3 shape, never 22.
- **Restore:** `restore_previous_paired` locates the single receipt whose canonical SHA is bound. It checks the receipt's status, chain position and engineering/package/config bindings. For each arm, it verifies the publication manifest SHA, every member's size/SHA and the continuation checkpoint SHA. Only then does it stage that arm's checkpoint write-once.
- **Stop and publish:** both arms stop at the same scheduled optimizer boundary (`--stop-after-updates`, exit 75). Each arm's checkpoint step and code binding are re-read. `publish_paired` then writes each arm's continuation checkpoint, newly completed immutable epoch snapshots and sidecars write-once to `/kaggle/working/window-bag-paired/<arm>/window-bag-segment/`. Writes stay within the per-arm cap and the combined cap (`2 x` per-arm cap, at most 16 GiB). Cache, MRI and reports are never published. The receipt `window-bag-paired-segment-accepted.json` is written last.
- **Evaluation:** the final segment ends at update 2810 and runs the trainer's only 1156-study outer evaluation (epoch 10). No intermediate epoch is selected.

## Funding and planning

`paired_plan` builds on `production_plan(..., concurrent=True)`:
- **Timings:** they must equal the `measured` block of the actual concurrent engineering receipt.
- **Session:** one session per segment, and every session runs both arms. Each session is charged `2 x timeout` for both provisioned T4s, once. The timeout is at most 9 h x 0.8.
- **Caps:** a per-arm output cap, a combined output cap, and a scratch cap equal to both arms' published state plus one transient checkpoint rewrite each.
- **Scratch:** a scratch requirement for the full member cache plus both arms.

The coordinator (`verify_engineering`) accepts only:
- `ENGINEERING_PARALLEL_STATUS` with `GPUs_used_concurrently == 2`;
- devices 0/1, two threads per process, a hex shared read-only cache digest, and a combined RSS no higher than 29 GiB;
- at least 4 CPUs and two visible GPUs;
- an engineering kernel source that embeds `WINDOW_BAG_PARALLEL = True` and the engineering manifest SHA;
- the exact source/image/inputs of the completed v1 kernel, read back;
- per-arm production config SHAs;
- funding whose segments, timeout, caps and authorization equal the recomputed paired plan.

Before any intent, `run_phase` refuses dispatch when the fresh conservative allowance, `min(total, minimum) − used − reserved`, is below the session charge. The previous paired output is read back from the completed kernel (status, source, mode, package, inputs, receipt) before any intent. There is at most one `SaveKernel`; an uncertain outcome is journaled for read-only reconciliation and is never retried.

## Tests (`tests/test_kaggle_window_bag_paired.py`)

- **Plan and contract:** whole schedule, one charge per session, cap doubling, starved quota and scratch.
- **Chain and contract refusals:** chain drift, cap/ceiling/scratch drift, identical or missing configs, a missing arm or a second previous kernel.
- **Coordinator:**
  - a push that mounts 20 + 1 sources;
  - exactly one SaveKernel, with an uncertain outcome not retried;
  - 21 forged funding/binding cases, including sequential engineering status, one GPU, wrong devices/RSS/CPUs, receipt/source/image/package/config/measured/segments/GPU/output-cap/scratch-cap/timeout/mode, all refused before any intent;
  - insufficient fresh quota;
  - seven previous-output tampers;
  - a read-only status check.
- **Package and notebook:** the package is the engineering package plus one file; the notebook branch runs only the paired runner on both devices.
- **Publication and restore:** write-once capped publication that excludes cache, and eight restore tampers.
- **Linux `/proc` supervision tests** (skipped on other OSes):
  - Two actual fresh-process paired sessions through the real `run_paired_segment`, Supervisor, publication and restore. Only the T4/shard admission is patched, with CPU synthetic data and the canonical trainer. Session 1 stops both arms at update 2 and publishes. Its scratch is deleted and its output is mounted read-only. Session 2 restores both arms and finishes. Each arm's final checkpoint, epoch-2 snapshot and prediction bytes equal an uninterrupted run.
  - One-arm failure reaps the sibling and publishes nothing.
  - A wrong engineering package or config fails before any trainer starts.
