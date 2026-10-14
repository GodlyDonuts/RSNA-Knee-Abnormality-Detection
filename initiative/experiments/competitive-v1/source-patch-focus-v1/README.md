# Source patch focus v1 (issue #62): identity-preserving disease-specific patch queries

Status: **prepared only (code/tests/configs)**. No job, reservation, data or weight
transfer, upload or submission was made. No accuracy or score improvement is claimed.
Root reviews and admits any real MRI/GPU work.

## Question and basis

`SourceSlotStudyNet.slot_features` reduces the register-free patch tokens of each slot to
their mean before the per-target slot head. Disease-specific slot attention therefore
cannot recover spatial feature co-occurrence that the mean discards. Query2Label
([arXiv:2107.10834](https://arxiv.org/abs/2107.10834),
[official code](https://github.com/SlongLiu/query2labels)) motivates label-specific
spatial pooling. This is a hypothesis to test. It is not an established clinical bottleneck
and not a novel method. It is distinct from window pooling (#38/#49/#60) and from the
earlier replacement `patch_attention` head, which regressed. Here the reviewed source-slot
path stays in place and the focus is only a residual.

## Policy `source_patch_focus_v1` (`rsna/competitive/patch_focus.py`)

The policy reuses the reviewed `source_matched_slot_v1` components unchanged: inputs, the
one-group-per-update window, augmentation, the DINOv3 encoder/adapter with CLS plus
register-free patch mean, the 6-slot `SourceSlotHead` (hidden 256, dropout 0.2, no prior),
ten-window probability averaging and MRI-only deployment. `source_slot.py` is byte-unchanged
(test-bound SHA `3e4e1bb9…`). It adds `patch_query` `[12, D]` (D = 1024 for DINOv3-L)
initialized at zero.

For acquired slot `s`, target `t` and patch `p`, using the tokens from the single encoder
pass (no second backbone forward):

```
w[t,p]   = softmax_p(<patch_p, Q_t> / sqrt(D))         u = softmax_p(0)   (FP32)
focus[t] = mean_p(patch_p) + sum_p (w[t,p] - u[p]) * patch_p
x_Q[t]   = [CLS, focus[t]]                             x_0 = same with Q = 0 (== x)
logits   = head(x) + gate * (H(x_Q) - H(x_0))
```

`H` (`target_conditioned_head`) is the same `SourceSlotHead` applied with a separate slot
feature per target. It uses the same LayerNorm/Linear/GELU projection, slot embedding, slot
attention, per-target context dropout `[B,T,H]` and output row for each target.

- **Original path unchanged.** `head(x)` has the original GEMM/dropout shapes and RNG draws.
- **Shared dropout draws.** Both residual terms run on the same RNG state, which is reset
  between them. The global CPU/CUDA RNG is then restored, so the forward consumes exactly
  the old model's draws.
- **Exact cancellation.** At `Q = 0`, `w == u` bitwise, the deviation is exactly zero and
  both terms are identical operations on identical values. The logits and probabilities
  therefore equal the old model's bitwise, in training and evaluation, FP32 and BF16
  autocast (CPU tested).
- **Gradient.** `dQ` is generally nonzero at zero. Row `t` receives gradient only from
  target `t`, so masked targets give exactly zero.
- **Stateless head required.** A head with buffers is refused, so no state can be mutated.

| | `control-r32-config.json` | `candidate-r32-config.json` |
|---|---|---|
| `patch_focus` | `disabled` (gate 0) | `query_residual` (gate 1) |
| parameters / names / init values / factory RNG | old model + `patch_query` = 0 | identical |
| compute, branch, dropout draws, global RNG | identical (branch is computed in both) | identical |
| residual / query gradient | exactly 0 / exactly 0 | active / informative |
| non-query gradients | bitwise the old model (CPU test) | mathematically equal at Q=0, rounding-level differences from accumulation order |

Both configs equal `source-matched-slot-v1/r32-config.json` in every key except
`classifier_strategy`: accepted fitting supervision, official anchor, rank-32 adapter,
effective batch 8, FP16, warmup 281/cosine, fixed 10 × 281 updates, fixed-final, fold 0 /
1156 outer. There is no target-specific post-hoc selection, and no protected gold or
confirmation data is used for tuning.

Pooling is content-only over acquired slots; absent slots are never encoded, and registers
are excluded by the encoder's `forward_features`. No token coordinates are used. Token grid
positions are normalized indices of the resampled 336×336 window, not physical anatomy.

## Tests (`tests/test_patch_focus.py`, engineering only)

- Policy tampering is refused. The arms differ only in `patch_focus`, configs are
  manifest-bound, and the source-slot bytes are unchanged.
- Parameters and factory RNG equal the old model's, plus a zero query.
- Old-model, control and zero-query candidate logits, ten-window probabilities and
  post-forward RNG are bitwise equal in train and eval mode, FP32 and BF16.
- The two residual terms share dropout draws. `target_conditioned_head` reads each
  target's own row.
- Zero-query gradients are nonzero for observed targets and exactly zero for masked ones.
  Control gradients are bitwise the old model's.
- Large queries under BF16 keep values and gradients finite. Absent slots are never
  encoded and never contribute. Non-finite patches are refused.
- Expressivity only: a synthetic study pair with equal CLS and patch means but different
  patch co-occurrence. The mean head cannot separate it; a nonzero query can.
- MRI-only deployment equals ten-window evaluation, and training inputs are the source-slot
  inputs.
- Canonical trainer, diagnostic encoder, both arms: full versus stop1/resume complete state
  (model, AdamW, LR, counters, all RNG, scaler) and `outer_predictions.csv` bytes match
  exactly. The arms and the old recipe share the first-update loss.

Synthetic tests are not GPU acceptance or accuracy evidence.

## Proposed bounded admission (Root decides; nothing executed)

1. **CPU common initialization (one CPU, no GPU).** Use the official DINOv3 base and
   frozen-base provenance on real admitted rows. Check: identical learned
   graph/values/RNG for both arms, `patch_query` zero, first-update forward and loss
   bitwise equal, and nonzero candidate query gradient. This uses the canonical trainer,
   checkpoints and inference unchanged. Bound: ≤ 1 h, 16 MiB control files.
2. **GPU recovery (one assigned H100, two CPUs).** Run full versus stop1/resume, 4 updates
   per arm, the same protocol as the source-slot recovery. Accept only on exact
   complete-state/CSV equality, finite nonzero encoder and query gradients, and recorded
   peak memory/time. Bound: ≤ 2 h, about 6 GiB engineering checkpoints.
3. **Production (only after 1–2 pass and fresh funding).** Two fixed-final arms on the
   configs above, one H100 each, ≤ 48 h walltime, epoch checkpoints at the source-slot
   checkpoint size. Analysis is a paired whole-component weak-development macro AUC with a
   bootstrap CI, a single seed and no promotion without a matched repeat.

Expected cost relative to `source_matched_slot_v1` is not yet measured on GPU. The added
work is one 12-row pooling per acquired slot (`N×441×1024×12` MACs), two
target-conditioned head passes on `[B,12,6,2048]`, and an FP32 copy of the patches (about
1.8 MB per slot). Both are small next to the encoder, but GPU recovery must measure them.

The control is forward-identical to the live `source_matched_slot_v1` recipe. It is not
assumed to be a bitwise trajectory duplicate: it has a different code hash, and the
gradient-norm clip includes one extra zero-gradient tensor. Root may decide whether the
existing source-slot fixed-final fit can stand in for it. That substitution was not
validated here.

## Routing, serialization and gradient report (Root review follow-up)

Baseline context, kept as reported by Root: the actual original source-slot fixed10 independently
replayed macro `0.8782870961938172` on 1156 outer studies, below current native-public
`0.8997840648139764`. The recipes differ, so that gap is not evidence about patch pooling. This
experiment stays matched to its own source-slot control (`control-r32-config.json`), not to
native-public, and is not mixed with dense-versus-low-rank capacity factors.

- Model routing: `patch_focus.classifier_model(strategy)` is the single class route. The trainer
  builds every classifier-strategy model through it (`source_patch_focus_v1` ->
  `PatchFocusSlotStudyNet`, window bag -> `WindowBagSlotStudyNet`, otherwise
  `SourceSlotStudyNet`) after exact strategy validation. Trainer inputs and ten-window deployment
  go through `patch_focus.training_inputs` / `patch_focus.deployment_probabilities`, which require
  the patch-focus student and delegate to the reviewed source-slot window contract.
- Competition inference: `infer.validate_inference_checkpoint` rejects these public
  ordered-triplet research checkpoints (`no admitted deployment input`), exactly as for source
  slot. No submission route is added here; one needs Root admission.
- Serialization: `patch_query` is a registered parameter. Schema-1 checkpoints restore with
  `strict=True`; frozen-base delta (schema 2) restores require exact trainable membership, shape,
  dtype and finiteness before a strict load. Tests confirm a patch-focus state is rejected by the
  source-slot model (unexpected `patch_query`), a source-slot state is rejected by the patch-focus
  model (missing `patch_query`), a reshaped query is rejected, a control checkpoint cannot resume
  or load under the candidate binding, and trainer checkpoints rebuild bitwise through
  `classifier_model`.
- Gradient flow: CPU-autocast FP16 and BF16 tests confirm finite, nonzero gradients in every
  diagnostic-encoder parameter for both arms, control-equal encoder gradients at zero query, and
  changed encoder gradients once the query moves (the focus path backpropagates into the encoder).
  A tiny 20-layer DINOv3 with the r32 adapter (12 adapted FFN layers, gradient checkpointing)
  checks finite gradients reaching every adapted layer under FP16/BF16 autocast when
  `transformers` is installed. These are CPU autocast checks only; full-prefix gradient flow on
  the real DINOv3-L weights under CUDA BF16/FP16 has not been run and is part of the GPU
  admission step below, not claimed here.
