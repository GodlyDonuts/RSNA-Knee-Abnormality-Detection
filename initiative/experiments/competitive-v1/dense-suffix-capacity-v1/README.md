# Matched frozen-prefix12 capacity experiment: dense suffix deltas vs suffix-only R32 (issue #43)

Status: **canonical integration, engineering only**. This recipe adds no runtime or GPU
reservation, transfer, upload or competition submission. It makes no accuracy claim.
Root admits fresh funding, then runs the actual recovery once before any production fit.
Running jobs 899766/908454 are not touched.

Mechanism source: the reviewed prototype `tools/research/dense_suffix_delta` (main
`600d332d`, root corrections). Input-contract acceptance is
`initiative/audits/dense-normalized-MRI-engineering-20261013-v2`. That actual H100 check
divided the first acquired uint8 ordered RGB triplet by 255 before ImageNet
normalization. It then passed zero-delta identity, two finite updates, exact
full/cached outputs and an unchanged official anchor. It covered one window only; it is
not trainer recovery or accuracy evidence. `dense-official-MRI-engineering-20261013-v1`
is preserved but superseded for the input contract, because its worker omitted the
/255 step.

Training keeps the canonical single path. `PublicTripletStudyDataset` executes the
hashed `dino_adapter.study_inputs`, which applies uint8 `/255` to `[0,1]`. Then
`_GrayEncoder._normalized_rgb` applies ImageNet normalization exactly once. Prefix
records accept only those float32 dataset windows and use the same encoder
normalization. The canonical code is `rsna/competitive/suffix_adapters.py`.
It is selected only by `encoder.suffix_adapter`. The default encoder and the all-block
`encoder.adapter` (`rs_lora_v1`) paths, their provenance and their configs are unchanged.

## Arms (predeclared; fixed-final; one seed each)

| | `suffix-r32-config.json` | `dense-config.json` |
| --- | --- | --- |
| method | `suffix_rs_lora_v1` rank32 alpha8 (alpha/sqrt(rank)) | `dense_suffix_delta_v1`, `y = W0x + b0 + D dropout(x)` |
| learned encoder tensors | 72 projections `model.layer.12..23.{q,k,v,o,up,down}_proj` | same 72 targets |
| learned encoder parameters | 7,077,888 | 150,994,944 |
| encoder LR | 1e-4 (live R32) | 1e-5 (repo full-encoder default) |

The two configs are `public-input-recipe-v1/public-triplet-config.json` with one change
each: `encoder.adapter` is replaced by `encoder.suffix_adapter`, and `encoder_lr` is
set as in the table. Everything else is shared:
- the pinned official strict-loaded DINOv3-L;
- the identical acquired336 public triplet source policy and cache index;
- fold0 2247/1156 roster, masks, folds and release;
- `label_source_v1` fitting supervision;
- seed, effective batch 8 (1x8), FP16 scale128, chunk8, cosine/warmup281, clip1,
  weight decay 1e-4;
- the unchanged 12-target attention head and labels;
- 10 epochs, fixed-final endpoint and frozen-base-delta checkpoints.

The LR difference is predeclared, not tuned. The contrast is "dense full-rank suffix
deltas at 1e-5 versus suffix R32 at 1e-4". It is **not** an isolated parameter-count
causal claim. Global clipping also covers about 21x more entries in the dense arm.

## Recipe change versus the live all-block R32 recipe (report it; do not compare arms against live)

In both arms the embeddings, RoPE and layers 0..11 form a frozen prefix. It always runs
in eval mode, without gradients and without autocast (FP32, the official tensor dtype).
The suffix runs under the recipe's FP16 autocast with non-reentrant gradient
checkpointing. RoPE is always the deterministic eval value.

The live trainer puts adapted encoders in train mode. There, Transformers applies the
stochastic `pos_embed_rescale=2.0` coordinate augmentation, so live RoPE differs.
Live R32 also adapts all 24 blocks, and its prefix runs under FP16. A comparison
against the live recipe is therefore contextual only.

The two arms share the same head initialisation. LoRA `A` initialisation runs in a
forked CPU RNG scope, so the dense and R32 arms start the head from identical values
and identical RNG streams.

`_GrayEncoder.requires_grad_(True)` and `configure_encoder_training` now treat
`DenseDeltaLinear` like `ResidualLinear`: official weights, biases, norms, LayerScale,
embeddings, registers and `mask_token` never unfreeze.

## Data path and storage

Training streams acquired windows (`images[slice_mask]`) online through the frozen
prefix. It needs no new feature cache and no new storage beyond the existing admitted
public triplet cache. `rsna/competitive/prefix_cache.py` defines an exact optional
record format, should a cache ever be funded. A record binds:
- official config/tensor SHA;
- prefix depth and execution;
- public policy, index and source revision;
- member path/bytes/SHA;
- slot mask and per-window `[slot, view]` layout;
- FP32 hidden states with RoPE. Any other dtype is refused, never cast.

Measure the real requirement read-only before claiming funding:

```bash
python -m rsna.competitive.prefix_cache --cache CACHE --train-config dense-config.json \
  --dinov3-config configs/competitive/weights/dinov3-large.config.json \
  --source-revision REV --output measured-storage.json
```

Arithmetic: 1,826,816 B per acquired window (446 x 1024 x FP32). The upper bound is
6x10 windows/study, i.e. 347.4 GiB for 3403 studies. The actual value depends on the
acquired-slot counts.

Checkpoint arithmetic (FP32 learned state only; the official anchor is never stored):

| arm | learned | + AdamW m/v |
| --- | --- | --- |
| dense | 603,979,776 B | 1,811,939,328 B |
| R32 | 28,311,552 B | 84,934,656 B |

Each figure adds a small head. With `submission_every_epochs=1` the dense arm keeps
10 epoch snapshots of about 1.8 GB each, plus `checkpoint.pt`. That needs new
explicit funding.

## Recovery (prepared, not run)

`scripts/newton/run_dense_suffix_recovery.py` runs each arm sequentially. It first
verifies an immutable source manifest that must bind exactly every `rsna/**/*.py`.
It then runs the unchanged `rsna.competitive.train` three times in distinct processes
under `enforce_deterministic_runtime.py`:
- full two-epoch, 16-study engineering fit;
- fresh `--stop-after-updates 1` (exit 75);
- resume.

It requires equality of all of the following between the full fit and the replay:
- learned model, frozen-base records and AdamW state/LR;
- counters, every RNG stream and the AMP scaler;
- numerical history and prediction bytes/metrics.

It also requires:
- the declared pinned graph: 7,077,888 / 150,994,944 learned encoder parameters with
  no official tensor learned;
- nonzero residual change;
- bindings that match across arms in everything except the suffix policy and LR.

It writes `accepted.json` with exclusive creation and never retries. CPU coverage of
the real trainer and wrapper is in `tests/test_dense_suffix_training.py`.

## Owning-node CPU admission (proposed, not staged)

`scripts/newton/admit_dense_suffix_inputs.py` is the separate one-CPU, zero-GPU gate
root runs before the unchanged full/stop1/resume wrapper. It reuses the original
public-input CPU receipt (`adb931d5`, CPU899766), root's complete-cache receipt
(`d2c0ad74`) and the whole-byte official-weight proof by fresh file stat only. The exact
`rsna/` delta from that receipt's code is pinned; the changed loader modules
(`competitive/train.py`, `release_cache.py`) are admitted only because the step replays
the complete 2247/1156 release/fold/supervision/cohort binding and the 16+16 public
tensor/label/mask hashes through the staged code and requires equality with the
original. Both arms are rebuilt in trainer seed/factory order: 7,077,888 / 150,994,944
learned suffix entries, zero initial residual, frozen official tensors equal to the
original base, identical head values and post-factory RNG. The receipt holds hashes and
counts only. It neither funds nor infers GPU recovery; production stays fixed10 and
streams triplets online (no prefix cache). See `cpu-admission-stage-manifest.json`.

## Fixed-ten production (prepared, not staged/funded/dispatched; issue #68)

`scripts/newton/run_dense_suffix_production.py` (staged by root as `ROOT/launch.py`,
root `/tmp/rsna-dense-suffix-fit-20261014-v1`) runs the unchanged canonical trainer from
the exactly admitted `0ecece12` tree on one assigned GPU: `suffix_r32` to completion,
then `dense`. Each arm fits 2,247 studies for 10 x 281 updates (effective batch 8) and
predicts the same 1,156 development studies once at fixed epoch 10. The production
configs are the unchanged recipes (`prepare`, CPU-only, materializes them for root to
hash). No best-epoch selection, protected gold, confirmation or external initialization.

Every launch rechecks, before any output:
- the actual CPU899766 receipt `158c242f…` and its pinned source manifest, wrapper,
  recipes, official base spec, overlay and fold file;
- the ACTUAL original GPU recovery `recovery/accepted.json`, whose SHA root binds in
  `production-funding.json`, plus its on-disk per-arm checkpoint/prediction hashes and
  CUDA/source binding (a prepared or invented receipt is refused);
- root's explicit node-local checkpoint/runtime/publication funding (the 12 GiB
  engineering envelope is refused), parent job, CPUs, assigned GPU, config SHAs,
  folds/release/supervision/overlay SHAs and the staged launcher/publisher bytes.

Each trainer invocation has an exclusive intent and terminal record. A trainer exit 75
(deadline SIGUSR1 or preemption) is resumed only by a later launch via `--resume` of the
bound checkpoint; an intent without terminal, a changed exit-75 checkpoint/binding or any
other failure is refused, never retried. Completion requires 2810 updates, 2247/1156,
fixed-final selection, ten snapshots and matched bindings; `terminal.json` claims no
accuracy. Paired fixed-final analysis remains a separate step, and a negative dense
result is reported as such.

`scripts/newton/dense_suffix_epoch_publisher.py` (`ROOT/publish.py`) preserves one
completed scheduled epoch per arm once, privately, under
`dense-suffix-capacity-fold0-0ecece12-fixed10-seed20260922/<arm>/epoch-NNN`. It
re-admits everything through the staged launcher and uses the pinned original transport
(`0f929dc1`) for lock, stdin token, private destination and complete forced
re-download/size/SHA. An existing journal/remote path or an uncertain commit is never
retried; `reconcile` is read-only. Tests: `tests/test_dense_suffix_production.py`.
