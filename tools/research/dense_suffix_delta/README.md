# Frozen-prefix12 DINOv3-L: full-rank suffix deltas versus suffix R32 (issue #40)

Status: **standalone research prototype + precommitted protocol**. Nothing under
`rsna/` imports this directory: the live trainer, `rs_lora_v1`, every frozen config
and every `code_sha256` over `rsna/**/*.py` are unchanged. No job, data, weight,
cache or GPU work was performed. Synthetic CPU tests (`tests/test_dense_suffix_delta.py`)
prove mechanism properties only; official-weight GPU prefix/suffix equality, cache
generation, trainer integration/recovery, funding, fitting and accuracy remain root's.
No claim is made that more parameters improve accuracy.

## Mechanism (`dense_delta.py`)

* Backbone: pinned `facebook/dinov3-vitl16-pretrain-lvd1689m` config
  (`135ecd23…`), Transformers 5.6.1 `DINOv3ViTModel`: 24 layers, width 1024,
  MLP 4096, 16 heads, 4 registers, patch 16, non-gated GELU MLP.
* Frozen prefix: embeddings + `model.layer.0..11`; never receive gradients.
* Suffix targets (source-derived names, same as `rs_lora_v1`):
  `model.layer.{12..23}.attention.{q,k,v,o}_proj`, `model.layer.{12..23}.mlp.{up,down}_proj`
  = 72 `nn.Linear` modules.
* `dense_suffix_delta_v1`: `DenseDeltaLinear` aliases the official `weight`/`bias`
  (state keys unchanged, `requires_grad=False`) and adds `dense_delta`, zero-initialized,
  same shape as `weight`: `y = W0 x + b0 + D dropout(x)`. At `D=0` the output is
  bitwise identical (tested, including train-mode dropout).
* Biases, LayerNorms, LayerScale, embeddings, registers, `mask_token` and the final
  `norm` stay frozen. The first bounded version isolates "dense versus rank-32 weight
  residual on identical targets"; also unfreezing norms/biases would add a second
  difference with no rank-32 counterpart.
* Validated graph from the actual model (meta device, pinned shapes):
  72 tensors, 48 x 1024x1024 + 12 x 4096x1024 + 12 x 1024x4096 =
  **150,994,944** delta entries (12 x 12,582,912). Matched suffix R32 control:
  **7,077,888** (all-block live R32: 10,223,616). Classifier head (live
  attention pooling, 203,020) is trainable in every arm.
* Installation refuses already-adapted backbones, non-DINOv3 configs, gated MLPs,
  nonzero `drop_path_rate`/`attention_dropout` (a stochastic prefix cannot be cached),
  and any learned tensor sharing storage with an official tensor.
* Checkpoint format: only `learned_state` (the deltas) plus head/optimizer; the
  official anchor is re-loaded from the pinned SHA and `official_state_fingerprint`
  (SHA256 over every non-learned tensor) must match. `restore_learned_state` copies in
  place, so optimizer references and anchor aliases survive.
* Integration hazard for root: `backbones._GrayEncoder.requires_grad_(True)` only
  re-freezes the backbone for `ResidualLinear`; with `DenseDeltaLinear` it would
  unfreeze every official tensor. An explicit integrated version must extend that rule.

## Exact frozen-prefix cache (`prefix_cache.py`)

`run_prefix`/`run_suffix` split the pinned forward exactly: `embeddings(pixel_values)`
-> `[N, 1+4+441, 1024]` (CLS, 4 registers, 21x21 patches at 336) -> RoPE
`(cos, sin)` from `rope_embeddings(pixel_values)` -> layers 0..11 | layers 12..23 ->
`norm` -> `pooler_output = sequence[:, 0]`. Synthetic tests assert
`run_suffix(run_prefix(x)) == model(x)` bitwise in eval mode.

Record per study (all tensors required to reproduce the suffix):

| tensor | shape / dtype | note |
| --- | --- | --- |
| `hidden` | `[n_acquired, 446, 1024]` float32 | output of layer 11, acquired windows only |
| `slice_mask` | `[6, 10]` bool | live `CompetitiveStudyModel` already encodes only `images[slice_mask]`, so acquired-only is exact |
| `rope_cos`, `rope_sin` | `[441, 64]` float32 | eval-mode RoPE for the 21x21 grid; recomputed and compared on load |
| `identity` | json | method, prefix depth, config SHA, official-tensor SHA, source-contract SHA, token prefix (5), hidden dtype |
| `digests` | json | SHA256 of every tensor (dtype/shape/bytes) |

Labels, label masks, series masks, sequence features and slot positions are **not**
cached; they still come from the unchanged frames and the admitted public cache
(`public_ordered_triplet_v1`, renderer `pilkwang/rsna-knee-baseline-v1/v15`, uint8
`[6,12,336,336]` -> 10 overlapping triplets `/255` -> ImageNet normalization in the
encoder). The source contract hashed into `identity` must carry the existing
renderer/source/node-text/pixel-group/adapter SHAs from `public-triplet-config.json`
plus the per-member original byte SHA.

Rejections (tested): any changed tensor digest, FP16 or any non-prefix dtype, mask
count != hidden rows, wrong RoPE, wrong config, any changed official tensor, wrong
source contract, wrong prefix depth, `augment != False`, train-mode build. Digests detect
corruption only; a self-consistent forged record is caught only by
`verify_by_recompute` against the official prefix, which root must run on a precommitted
sample (and on every record if funded) on GPU before fitting.

Dtype: the official weights are float32 and the cache is built in eval mode without
autocast, so `hidden` is float32 and stored as float32. It is never cast to FP16/BF16 to
save space; a reduced-precision cache would be a different, separately named experiment.

Storage (exact from shapes, hidden only, FP32: 446 x 1024 x 4 = **1,826,816 B per window**):

| scope | dense 6x10 windows/study | acquired-only |
| --- | --- | --- |
| per study | 109,608,960 B (104.5 MiB) | 18,268,160 B per acquired series |
| fold0 fit 2247 | 246,291,333,120 B (229.4 GiB) | 18,268,160 B x acquired fitting series |
| fold0 dev 1156 | 126,707,957,760 B (118.0 GiB) | 18,268,160 B x acquired dev series |
| both 3403 | 372,999,290,880 B (347.4 GiB) | <= dense bound |

The acquired-series counts must be read from the admitted public index slot masks; they
are not asserted here. This is far beyond the current 72 GiB source-cache allowance and
requires NEW explicit node-local private funding; no originals may be deleted for credit.

### Augmentation policy (precommitted)

The live public-triplet recipe already sets `augment: false`. A cache is valid only for
that exact fixed, non-augmented input; `prefix_identity` refuses any other contract.
Cached features are never presented as augmented inputs; any future augmented arm must
run the prefix online and is a different experiment. Adapter dropout (in the suffix) is
the only stochasticity and is unaffected by caching.

### Semantics differences versus the live R32 recipe (confounds to report)

* Live R32 runs the whole backbone under FP16 autocast with gradient checkpointing;
  the cached arms run the prefix once in FP32 eval mode. Both matched arms share that
  prefix, so the arm contrast stays clean, but neither equals live R32 bitwise.
* RoPE: the pinned config has `pos_embed_rescale=2.0`. In Transformers this coordinate
  augmentation is applied only while the RoPE module is in training mode, so whenever
  the live trainer puts an adapted backbone in train mode its RoPE is stochastic. A
  cache can only reproduce eval-mode RoPE; `rope_context` refuses train-mode RoPE and
  the suffix must therefore run with the RoPE module in eval mode (root must state
  which mode the live R32 run uses before treating it as a reference).

## Matched protocol (precommitted, not funded)

Arms (all fold0 outer fit, same accepted public-triplet source pixels, unchanged
`label_source_v1` fitting supervision, label masks, 2247/1156 roster, seed 20260922,
fixed 10 epochs / 2810 updates, effective batch 8, attention head, head LR 3e-4,
cosine schedule, warmup 281, clip 1.0, adapter dropout 0.05, final endpoint fixed, no
early stopping or checkpoint selection):

1. **A: prefix12 + suffix R32** — `install_suffix_lora(rank=32, alpha=8)`, the existing
   `ResidualLinear` on the same 72 targets; encoder LR 1e-4 (live R32 value).
2. **B: prefix12 + dense deltas** — `install_dense_suffix_deltas(dropout=0.05)`;
   encoder LR 1e-5 (repo `TrainConfig` default full-encoder LR). Adam's per-element step
   is ~LR regardless of rank, so 1e-4 on every dense entry would be a far larger weight
   change than R32; 1e-5 is fixed a priori, not tuned. No LR/decay sweep on gold,
   confirmation, outer development or tiny pilots.
3. **C: all-block R32 (live recipe)** — contextual reference only. It adapts all 24
   attention blocks, runs train-mode RoPE/FP16/online prefix, and has a different target
   set, so A-vs-C is not a rank comparison and B-vs-C is not a pure rank comparison.

Regularization: weight decay 1e-4 (live value) in both arms. For B, decoupled AdamW
decay of `D` equals decay toward the official `W0`, not toward zero; at LR 1e-5 it is
very weak (cumulative shrink ~ 2810 x 1e-5 x 1e-4 ~ 3e-6), so the anchor is mostly
the zero initialization plus the fixed endpoint. Report `||D||_F / ||W0||_F` per module
at each saved epoch.

Confounds that remain and must be reported, not adjusted post hoc: the LR difference
(A 1e-4 vs B 1e-5), global grad-norm clipping covering ~21x more delta entries in B,
FP32 cached prefix vs live FP16, and seed-to-seed variance (one seed per arm; a gain
needs a matched replication before promotion). Primary metric and bootstrap follow the
existing outer development evaluation; gold remains evaluation-only and is not used.

Costs (FP32 learned state; frozen-base-delta checkpoints never store the anchor):

| item | A suffix R32 | B dense | C all-block R32 |
| --- | --- | --- | --- |
| learned params | 7,077,888 | 150,994,944 | 10,223,616 |
| weights/checkpoint | 27.0 MiB | 576.0 MiB | 39.0 MiB |
| grads | 27.0 MiB | 576.0 MiB | 39.0 MiB |
| Adam m+v | 54.0 MiB | 1,152.0 MiB | 78.0 MiB |
| suffix linear FLOPs | ~+6% | ~+100% (second full matmul) | ~+6% (all blocks) |

Activations: both cached arms back-propagate only through 12 suffix layers per acquired
window (446 tokens x 1024 x 4 B = 1.83 MB per saved layer input per window), and both
need the same layer inputs for their weight gradients, so activation memory is
essentially equal between A and B and smaller than online full-depth training. Ten
epoch checkpoints for B add ~5.6 GiB weights (+~11.3 GiB if Adam state is retained for
exact resume), which also needs new funding.

Inference: one solo DINOv3-L with its learned residual (B may be merged as
`W0 + D` after fitting only if merged-vs-unmerged equality is separately verified);
no teacher or ensemble is required.
