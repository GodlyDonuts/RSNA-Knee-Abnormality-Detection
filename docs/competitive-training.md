# Competitive training handoff

The accepted data release is unchanged. A separate experiment package now supplies
training-only folds, offline pretrained adapters, geometry-aware slice/series
pooling, distributed masked loss, recovery and competition-template inference.
Start with [Newton setup](newton-launch.md), [image/model contracts](competitive-image-model.md),
and the [research strategy](training-strategy.md).

The current tooling requires Python 3.11 or newer. Newton, Kaggle and CI use
Python 3.12; the existing local CPU verification environment uses Python 3.11.

This is the initial global-embedding experiment. The two-H100 100-study diagnostic
measured 26 optimizer updates; a real Slurm signal stop and separate resume
also recovered its exact checkpoint payload and predictions. See
`initiative/audits/dinov3-h100-pilot-v6/` and
`initiative/audits/newton-recovery-production-v1/`. Primary ten-epoch fold-0
job 840562 reached 22 epoch-0 optimizer updates on 2,247 eligible optimization
studies, then stalled in CUDA synchronization and ended without a checkpoint.
The same pinned configuration was relaunched once as job 840578 on evc40 after
excluding the stalled node. First-epoch throughput,
competitive accuracy and the hidden-test time budget remain unmeasured until
actual completed epochs and Kaggle notebook runs finish.

## Frozen development data

The accepted manifest remains
`2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7`.
The overlay under `initiative/experiments/competitive-v1/folds/` covers all 3,904
training studies and 3,761 leakage components. Its membership fingerprint is
`8699248177042dc2e95b1e1947e25cbe5bc39e42c43ee4919398afc219665e07`.
The three folds contain 1,301 / 1,302 / 1,301 studies. Every target has both
observed classes in each fold, but rare evidence remains extremely limited:

| Independent leakage components | Fold 0 | Fold 1 | Fold 2 |
| --- | ---: | ---: | ---: |
| Fracture positive | 2 | 1 | 2 |
| MCL positive | 2 | 3 | 2 |
| Baker's positive | 3 | 3 | 3 |
| Synovitis negative | 5 | 5 | 5 |

These are conservative leakage components, not verified patient counts. The
training loop selects only accepted train rows with at least one observed target.
It excludes the chosen outer fold from all optimization. The 441 reserved
confirmation studies and 58 official evaluation studies are never model-selection
inputs. Four quarantined studies remain excluded. Fully masked studies keep their
overlay membership but do not enter supervised training or its reported OOF pool.

Every training invocation verifies the complete accepted release before selecting
rows. A distributed job performs the expensive evidence replay on rank zero,
broadcasts its verified table, checks local manifest bytes on every rank, and
requires identical content/configuration bindings across ranks. There is no CLI
switch to bypass evidence validation.

## Initial experiment configurations

The owner selected DINOv3 on September 22. The successor recipe
`configs/competitive/dinov3-campaign.json` makes DINOv3-L/16 the primary encoder.
The earlier `campaign.json` and DINOv2/ConvNeXt configurations remain as historical
comparison plans; they are not authorized GPU work for the current starting run.
DINOv3-L has 1,024-dimensional features and 24 transformer blocks. Its official
weight revision, size and SHA are pinned. Meta approved account access; the
authenticated exact config and license bytes are verified. The executable source
manifest is `configs/competitive/weights/dinov3-large.json`. The campaign recipe
is combined with that manifest only after staged weights pass full verification.
For this exact official checkpoint, the offline adapter applies Transformers
5.6.1's documented `layer.` to `model.layer.` compatibility conversion. Original
weight hashes and tensor values stay unchanged, and full strict loading remains
required. Actual Newton CPU verification checked all 415 tensors, a 224-pixel
forward and exact checkpoint-only restoration; CUDA recovery is a separate gate.

Start with one Newton node, two H100s, four CPUs total and zero extra loader
workers. Use two-rank DDP with one study per microbatch and four accumulation
steps per rank, for an effective batch of eight. The initial model uses six
series, sixteen distinct slices per series, 224-pixel inputs and a frozen
DINOv3-L encoder with target-specific slice/series attention. This establishes a
measured reference before fine-tuning. A second node is permitted for a separate
fold only after the first node's pilot passes. Never reserve GPUs while required
weights, data or the environment are unavailable.

A 100-study, two-epoch diagnostic pilot checks real pretrained loading, memory,
throughput and same-world-size checkpoint recovery. Begin in FP32; BF16 requires
an explicitly recorded configuration after a numerical/hardware comparison.
The first production run is fold 0, seed 20260922, ten fixed epochs, and has
`diagnostic_study_limit: null`. The head starts at learning rate 1e-3. Later
fine-tuning is a separately recorded experiment; warm-starting a different stage
is not implemented by exact-resume. The tiny random encoder is diagnostic only.

The user also authorized a new submission policy: preserve and queue completed
epochs 2, 4, 6, 8 and 10 from the primary run. This supersedes the earlier
final-checkpoint-only deployment plan. Checkpoints are immutable hard links with
hash-bound sidecars; later recovery saves cannot change them. Diagnostic snapshots
are explicitly ineligible. The account-wide queue checks Kaggle's live allowance,
never exceeds five submissions per day, and waits when quota is unavailable.
Other folds share the same allowance and do not each receive five extra slots.
Repeated public leaderboard feedback is development evidence. The reserved 441
confirmation studies and 58 official studies remain untouched, and outer-fold
predictions are written only after the fixed final epoch.

## Train and recover

Run from the repository root, inside the prepared environment and allocation.
Stage weights with the pinned manifests described in the Newton guide first.
The following command illustrates the training arguments; the Newton launcher
wraps it in two-process torchrun for the initial DINOv3 job. Project variables
must identify the materialized configuration and approved storage locations:

```bash
python -m rsna.competitive.train \
  --config "$RSNA_TRAIN_CONFIG" \
  --studies initiative/releases/report-supervision-v1/studies.parquet \
  --series initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv \
  --cache "$RSNA_CACHE" --folds initiative/experiments/competitive-v1/folds \
  --artifact-root . --output "$RSNA_RUN" --device cuda
```

For two-GPU DDP, use `torchrun --standalone --nproc-per-node=2 --module
rsna.competitive.train` and the `2gpu` config. The Newton generator also supports
two independent one-GPU fold/seed runs on one allocated two-GPU node. Benchmark
both arrangements; inter-node InfiniBand alone does not establish a benefit from
multi-node training.

Unknown labels remain masked. Per-target observed counts are aggregated across
all ranks and accumulation steps before backward. Microbatch contributions sum to
the global effective-batch objective; there is no second accumulation division.
Padding rows receive zero supervision and are omitted from evaluation output.
An entirely masked effective batch skips optimizer updates, including weight decay.

Checkpoints are atomic and written at complete optimizer boundaries, periodically
and at epoch ends. They include optimizer state, exact progress, per-rank CPU/GPU,
Python and NumPy RNG states, deterministic sampling settings, weight/config hashes,
the fold overlay, source release, cache version and every runtime source file.
Random data augmentation uses per-study/epoch CPU seeds and cannot reset model
CUDA generators. Partial gradient accumulation is never serialized as a completed
optimizer update.

The Newton supervisor responds to `SIGUSR1` by forwarding it to workers and
writing an atomic, invocation-bound stop request beside the checkpoint. Both
training ranks check that request at optimizer boundaries and during final
prediction. This file fallback was added after actual Slurm job 839411 reported
worker signal forwarding but completed normally; see
`initiative/audits/newton-slurm-signal-v1/`. The local real-trainer stop/resume
test passes, while a new actual Newton stop and separate resume remain required
before production. The supervisor requires an explicit exit-75 stop receipt,
binding and checkpoint SHA; a periodic checkpoint plus an unrelated crash is
not successful recovery.

Resume by repeating the same command with `--resume "$RSNA_RUN/checkpoint.pt"`.
Changed world size, source, model files, code, device type or numerical settings
are rejected. A fresh invocation cannot overwrite a prior run without explicit
resume. Same-world-size GPU recovery still needs a Newton pilot; cross-world-size
recovery is deliberately unsupported. Keep enough free space for both the old
checkpoint and its temporary successor plus the configured reserve.

## Evaluation and offline prediction

Each completed fold writes `outer_predictions.csv`, `result.json`, `run.json` and
`checkpoint.pt`. Collect all three folds for the same seed before evaluation:

```bash
python -m rsna.competitive.evaluation \
  --studies initiative/releases/report-supervision-v1/studies.parquet \
  --artifact-root . --folds initiative/experiments/competitive-v1/folds \
  --predictions "$RUN_FOLD0/outer_predictions.csv" \
    "$RUN_FOLD1/outer_predictions.csv" "$RUN_FOLD2/outer_predictions.csv" \
  --seed 20260922 --membership observed_train --output "$RSNA_EVALUATION"
```

The evaluator rejects missing/duplicate/misassigned studies and mixed seeds. Its
primary summary computes each target's within-fold AUC weighted by observed
positive-negative pairs, then averages all twelve targets equally. Pooled OOF AUC
is separately named. Undefined classes never receive an invented AUC of 0.5.
Paired component-bootstrap and rare-positive-component sensitivity APIs are
available; seeds remain separate. These are measures of report agreement and
conditional sampling uncertainty, not clinical ground truth or independent proof
of a small model advantage.

`python -m rsna.competitive.infer` supports an exact public/hidden-test template,
local series metadata, and a local cache or `--dicom-root`. Production packaging
uses `--encoder-config` with the original official config and the trained
checkpoint. The helper reconstructs the exact architecture, checks config and
provenance, and strictly loads the complete encoder state from the trained
checkpoint; the full model is then strictly loaded, including the prediction
head. The original pretrained weights are not duplicated in the submission.
Their recorded SHA describes training ancestry, not a fresh inference-time read.
The package also needs the exact runtime source and offline Linux CUDA wheels.

Inference validates source binding, target order, cache transform and all runtime
code hashes. A final checkpoint is accepted by default. Intermediate deployment
requires `--scheduled-epoch` and a completed epoch matching the bound cadence;
partial-epoch recovery files cannot enter the submission queue. Kaggle deployment
uses explicit `--precision fp32` so a future BF16 H100 training run can execute
on a GPU without BF16. The metrics distinguish training and inference precision.
`--encoder-spec` remains available for original-weight and tiny diagnostic checks.
See [Kaggle submissions](kaggle-submissions.md) for exact notebook and quota rules.

Diagnostic sample inference requires `--allow-diagnostic` and at most three
template studies. It is not a submission candidate. Time the full decode-to-CSV
path on Kaggle-compatible hardware before selecting an ensemble; a fast H100 run
cannot certify Kaggle's nine-hour limit.

## Local verification and remaining experiments

`scripts/local_competitive_acceptance.py` replays full release validation, a real
16-study training / 16-study outer-fold diagnostic, within-epoch interruption and
resume, exact full checkpoint-tree comparison and the three public examples.
Actual one-process and two-process CPU/Gloo runs passed exact full checkpoint and
prediction equality after within-epoch recovery, in 170.25 and 173.99 seconds.
These are tiny diagnostic checks, not estimates of H100 epoch speed. The helper
also supports a configurable two-epoch CUDA diagnostic using `--config`,
`--device cuda` and `--skip-public-examples`; the Newton acceptance launch runs
it with the verified 100-training/100-outer-study pilot cache. It checks real
continuous/interrupted/resumed training before production. Acceptance records
and failed-attempt disclosures are archived separately from the immutable release.
The first actual two-H100 DINOv3-L diagnostic, Newton job 839321, completed two
epochs over 100 training studies and 100 outer studies. Its uninterrupted run
performed 26 optimizer updates in 109.06 seconds including prediction, with a
1,759,206,400-byte peak CUDA allocation on rank zero. The deliberate one-update
stop and same-world-size resume produced elementwise identical final model and
optimizer state and byte-identical outer predictions. Raw PyTorch checkpoint
file hashes differed despite exact decoded state equality. The original records
are preserved in `initiative/audits/dinov3-h100-pilot-v6/`. Its diagnostic
submission cadence is zero and it established no Kaggle score. Fold-0 production
has 2,247 eligible training studies and 281 effective batches per epoch; a linear
pilot extrapolation is roughly 14 minutes per epoch, pending a full-data measure.
The separate actual Slurm USR1 delivery and full cache staging remain open.
An actual pinned ConvNeXt CPU load/forward/head-backward check and DINOv2 checkpoint
header compatibility inspection are recorded under
`initiative/audits/competitive-pretrained-local-check-v1/`.

The initial implementation does not yet provide cached feature training,
fold-local MRI self-supervision, regional-token pooling, native 448-pixel caches,
2.5D/3D branches, teacher distillation or a fitted ensemble. Those are gated
experiments in the strategy. Adding all of them before measuring the reference
pipeline would not establish that they help. The next executable gate is the
actual Newton environment, pinned pretrained model, throughput/memory and recovery
pilot, the actual Slurm signal rehearsal and full-cache staging, followed by the
primary DINOv3-L run and measured fine-tuning experiments.
