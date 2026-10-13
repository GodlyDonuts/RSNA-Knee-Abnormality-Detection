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
  `host_to_device`, `forward`, `backward`, `optimizer`, plus CUDA-event
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
