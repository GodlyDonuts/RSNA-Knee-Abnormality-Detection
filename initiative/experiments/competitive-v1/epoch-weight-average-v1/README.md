# Epoch-weight average v1 (issue #53): exact effective-delta averaging of epochs 6-10

Status: **prepared only**. No training, GPU job, Kaggle kernel, data transfer,
label/fold/mask change or protected/gold evaluation was made. This is a new
experiment motivated by `initiative/research/public-high-score-update-20261013-v1`,
not a promised gain. Software tests establish arithmetic and provenance, never
clinical or competition gain.

## Predeclared contrast

- Source: ONE completed, non-diagnostic, fixed-ten-epoch `frozen_base_delta` run
  with `submission_every_epochs: 1` (every epoch snapshot kept).
- Candidate: uniform arithmetic average of epochs 6, 7, 8, 9, 10 (weights 1/5).
- Primary comparison: the same run's fixed-final epoch 10.
- No epoch set, weight or representation is chosen from labels, predictions or
  scores. Only two epoch sets exist in code: the average and an epoch-10-only
  identity control. Anything else fails.

## Code: `rsna/competitive/weight_average.py`

- `average_snapshots(run)` reads `run.json` (must be `completed`, not diagnostic,
  no gold) and `epoch-checkpoints/epoch-006..010.{pt,json}`. Every file must be a
  regular file whose bytes match the recorded SHA256/size. Each snapshot must be
  the completed `epoch, next_update=0` boundary of the bound schedule. All five
  must have the same binding as `run.json`: source, release, config, fold, seed,
  fitting supervision, encoder/base provenance and code. They must also have the
  same `model_storage`: frozen base tensor hashes and learned graph. Optimizer
  counters must equal `epoch * updates_per_epoch`. Any mismatch, missing file or
  tampering raises.
- Sources are read only. The candidate is written once to a new path through a
  hard-linked temp file and a JSON receipt. Afterwards the source bytes are
  checked again. An existing candidate is never replaced.
- Output `epoch_weight_average_inference_v1` has `optimizer_boundary: false` and no
  optimizer, RNG, AMP or `binding` key. Its `schema_version` is a string. Because
  of that, `load_checkpoint`, `restore_checkpoint_model`,
  `validate_inference_checkpoint` and `FrozenBaseDelta` (once applied, the model has
  no trainable tensors) all refuse it. It cannot resume training.

### Why LoRA factors are not averaged

`ResidualLinear` computes `W0 x + b0 + s B A x` with `s = alpha / sqrt(r)`.
Averaging `A` and `B` separately gives `s mean(B) mean(A)`, which is not equal to
`mean(s B_k A_k)`. A test shows the gap. The exact average is

```
mean_k(s B_k A_k) = (s / 5) [B_6 .. B_10] [A_6; ..; A_10]
```

This is stored as concatenated factors of rank `5r` with scaling `s/5`, using the
same `ResidualLinear` forward. There is no SVD and no truncation. A rank-32 SVD
would be an approximation and is not used. Dense suffix deltas
(`dense_suffix_delta_v1`, unit scaling) are linear, so they are averaged directly.
Learned head and pooling tensors (including source-slot/window-bag heads and head
LayerNorms) are averaged directly in float64 and cast back to their dtype.
Non-floating learned tensors must be bitwise identical across epochs.

The following stay exact: frozen projection weights and biases, register tokens,
encoder LayerNorms, and the `mean`/`std` normalization buffers. They are never
stored. They must reproduce the bound pretrained hashes before and after the
average is applied. Refused cases:
- any buffer outside the encoder base, including BatchNorm running statistics;
- any learned encoder tensor other than the bound residual targets;
- any adapter method other than `rs_lora_v1`, `suffix_rs_lora_v1` or
  `dense_suffix_delta_v1`;
- an encoder with more than one adapter.

### Rank, storage and runtime (canonical DINOv3-L, rank 32, last 12 FFN layers)

| | parameters | fp32 bytes |
|---|---:|---:|
| source residual (per epoch) | 10,223,616 | 40.9 MB |
| averaged concatenated factors (rank 160) | 51,118,080 | 204.5 MB |
| equivalent dense deltas (120 projections) | 201,326,592 | 805.3 MB |
| head (unchanged count) | 203,020 | 0.8 MB |

- Storage is 5 times one epoch's residual, about 25% of the dense-delta size.
- Runtime is still one backbone forward per view. There is no prediction
  ensemble.
- Adapter residual MACs per token go from about 10.2M to about 51.1M. The base
  ViT-L linear projections cost about 302M MACs per token, so residual linear
  work rises from about 3.4% to about 16.9% of base projection MACs. Attention
  and convolution costs are unchanged.
- Merging into `W0` would add no runtime cost, but it would change the immutable
  frozen base. It is not implemented.
- Actual MRI/GPU throughput and memory are root's measurements.

Averaging weights is not the same as averaging logits or probabilities. The tests
never assert that they are equal.

## Root admission and evaluation plan (no gold, no confirmation tuning)

1. Pick the single completed fixed-ten-epoch run. Do not look at any per-epoch
   score first. Keep its original snapshots in place. The averaging step needs
   only CPU and reads no labels:
   `python -m rsna.competitive.weight_average --run RUN --output OUT/average/candidate.pt`
   and `... --epoch10-identity-control --output OUT/identity/candidate.pt`.
2. Root owns the source-pinned outer predictor. It must:
   - run `verify_source_code(binding, current rsna/*.py hashes)`. Only this module
     may be new;
   - build the model exactly as `train.py` does for the bound config: encoder,
     `configure_encoder_training`, and the same `CompetitiveStudyNet` /
     source-slot / window-bag class with the same cache inputs;
   - call `load_candidate(path, receipt_sha)` and then `apply_candidate`;
   - run the unchanged `train.py` outer-fold prediction path: fold-0 1156-study
     outer pool, `EffectiveBatchSampler`/`SeededDataset`/`predict`, bound precision.
3. Gate: predictions from the epoch-10 identity control must reproduce the run's
   own `outer_predictions.csv` exactly: same 1156 UIDs, same 12 targets, same
   values. Any difference stops the experiment before the average is scored.
4. Comparison: control = epoch-10 `outer_predictions.csv`, candidate = averaged
   predictions, same 1156 studies. Run once with the existing paired
   whole-leakage-component bootstrap, `paired_metrics` in
   `scripts/research/compare_smooth_witness.py`: 4000 draws, seed 20261001,
   original masks, undefined draws counted and never re-drawn. The endpoint is
   all-12 macro AUC delta (candidate minus epoch 10) with its conditional 95%
   interval, as in the fixed-final paired analyses. Do not retry with other epoch
   sets, weights or seeds. Protected, gold and confirmation labels are not used.
5. If the interval excludes zero, that is weak-development evidence only. A
   matched replication and Kaggle acceptance remain root decisions. Without that,
   the plain epoch-10 default stays.

Note: `docs/codex-handoff-20261012.md`, named in the task automation, does not
exist on `main`. `docs/research-handoff-20261012-corrected.md` was used instead.
