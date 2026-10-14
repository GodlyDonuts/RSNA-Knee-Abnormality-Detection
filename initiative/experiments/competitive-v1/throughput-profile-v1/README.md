# Training throughput profile v1 (proposal only)

Status: code/tests/config proposal. Nothing here was dispatched, reserved, staged or
run on MRI. The live public-label comparison and original controls are unchanged.

Measured motivation (issue #25): original `899766` control fixed10 completed 2810
updates with the last update at 70589.364 s (~25 s per 8-study optimizer update) and
a 7,732,975,104 B allocated peak on an 80 GB H100.

## What may change

`settings.json` lists candidates that change only `batch_size`,
`accumulation_steps`, `encode_chunk_size` and `workers` (at most two loader
workers, default `prefetch_factor`, pinned memory as in the trainer).
`rsna.competitive.throughput.apply_setting` refuses every other recipe field and
any change of the global effective batch (8 studies), so label masks,
observed-count normalization, sampling order, AdamW groups, warmup/cosine LR,
FP16 with scale 128 and the rank-32/native-384 encoder are inherited unchanged.
A promoted setting must be applied with the same overrides to both
`public-label-source-v1` arms (tested).
The same overrides apply unchanged to the source-slot / window-bag arms
(`source-matched-slot-v1`, `source-window-bag-v1` mean/query; tested).

## Admitted recipes (issue #25 follow-up)

`build_profile_state` mirrors `train._run_locked`'s single-rank setup order for:

- the plain native-cache canonical classifier;
- the public ordered-triplet input (`public_input`), with or without the
  `source_matched_slot_v1` / `source_window_bag_v1` `classifier_strategy`
  (`SourceSlotStudyNet` / `WindowBagSlotStudyNet`, `frozen_base_delta` guard);
- `label_source_v1` fitting supervision, enabled or control, applied through
  `apply_fitting_supervision` with its exact overlay (`--fitting-supervision`) and
  soft fitting targets, and the public cohort bound by `verify_public_cohort`
  before any diagnostic subsetting.

Relation, witness, plane memory, neighbor channels, patch summary, cache contrast,
non-attention pooling, all-development scope and MRNet initialization are refused.
CPU tests compare against the unchanged trainer on substantive fixtures: initial
learned + frozen base parameters, torch/Python/NumPy RNG and AdamW groups are equal;
after the first epoch's two updates the delta model/storage, AdamW, GradScaler, RNG
and per-update loss/gradient norm/observed cells/optimizer steps are equal.

Slot-classifier window selection and augmentation run on CPU (`cpu_transform`)
before the copy (`host_to_device`). Window groups derive from (seed, epoch, update)
and augmentation from per-study generators, so they do not depend on
microbatch/chunk/workers. The head's dropout (0.2) does depend on microbatch shape.
`views` counts acquired slots x training windows (1 for slot, 4 for bag) on valid
samples. It is checked against `encoder_images`, the count the encoder actually
received (timed through an instance-level `forward_features` wrapper with CUDA
events; removed after profiling).

The receipt binds `input_format`, the public cache index/cohort receipt
(`--expected-public-index-sha256` is required for public recipes and refused for
native ones), the fitting-supervision receipt and the classifier strategy.
`input_decomposition` uses the matching reader: native volume read/checksum/
decompression/slice sampling, or public NPZ member read/SHA-256/decompression/
ten-window construction. A dataset whose format differs from the recipe is refused.

## Numerical audit

- Workers only: per-study RNG is forked by `SeededDataset` and the loader generator
  is separate from training RNG, so a bit-identical trajectory is *expected*; it must
  still be demonstrated with actual evidence.
- Batch or chunk changes: the same studies form each update and normalization is
  identical, but padded batched kernels, summation order, FP16 accumulation and
  per-call adapter dropout (p=0.05) masks differ. These are statistically equivalent
  recipes, **not** bit-identical; no bit equality is claimed.

## Profiler

`scripts/newton/profile_training_throughput.py` runs one setting per fresh
process, replays the trainer's first-epoch update arithmetic (tested equal to
`_run_locked` model state on CPU) and writes one new JSON receipt with:

- per-update phase seconds with device sync at boundaries: `data_wait`,
  `cpu_transform`, `host_to_device`, `forward`, `backward`, `optimizer`, plus CUDA-event
  `encoder_forward` and `encoder_backward` (backward includes checkpoint recompute);
- per-item read/verify/decompress/transform seconds (`item_seconds_sum`) and a
  separate post-update decomposition of read, checksum, decompression and CPU
  transform on the same studies (`--evict-page-cache` for cold reads);
- CUDA allocated and reserved peaks per update;
- true studies/views per second counted from valid samples and valid slices only;
- bindings for config, settings, manifest, series, source revision, torch version,
  device name, caps, code tree and profiled UIDs.

Hard caps: at most 12 measured plus 4 warmup updates (128 studies) and 7200 s
wall. No checkpoint, prediction or metric is written. Receipts always state
`accuracy_claim: false` and `production_acceptance: false`; CPU/diagnostic or
incomplete runs are marked `evidence_eligible: false`.

## Before any promotion

1. Actual original-MRI H100 receipts for each setting, each in a fresh process.
2. Within-setting actual full/stop/resume exactness with the real trainer.
3. Same setting applied to both matched arms.

The source-slot log observation (updates 1043->1044, ~14.78 s/update on a shared
H100 with other workloads) is contention-affected and is not isolated throughput
evidence. No worker/chunk/batch choice is made from it.
