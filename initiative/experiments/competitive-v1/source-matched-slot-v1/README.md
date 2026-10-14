# Source-matched v15 slot classifier on one DINOv3 student (issue #47)

Status: **canonical integration, engineering only.** No training, job, GPU, transfer,
upload or submission happened in this change, and it makes no accuracy claim. Root
funds and runs the actual source/input/recovery/performance acceptance before any
production fit. Healthy 899766/908454 are not touched.

This tests the public solution's *classifier/training interface*, not its input (issue
#45) or encoder capacity (issue #43). The question is whether the original v15
window sampling, pooling and per-target slot aggregation help a solo DINOv3 student on
the same admitted pixels.

## 1. What the original v15 source actually does

Source: `initiative/audits/public-dinov2-source-handoff-20261013-v1/source.notebook.json`,
SHA `b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8`
(pilkwang/rsna-knee-baseline-v1 v15, by Pilkwang; private research use, redistribution
licensing unconfirmed). Read from cells 25/26/28/30/31/33 and `main()`.

| Step | Original | Where |
|---|---|---|
| Cache | uint8 `[N,6,12,336,336]`, 4 groups of 3 slices (`slices=12`, `group=3`) | `build_cache`, `adopt_config_globals` |
| Training window | **one** group `g = randint(N_GROUP)` per batch of `BATCH_STUDIES=8`, shared by every study and slot of that batch; `take_group` = slices `3g..3g+2` (non-overlapping: views 0/3/6/9) | `main()`, `take_group` |
| Augmentation | `augment`: per-slot rotation ±8°, zoom-in ≤8%, shift ±5%, intensity ±10%, `border` padding, on uint8 then truncated back to uint8; no flips | `augment` |
| Encoder input | `x/255`, ImageNet mean/std once, DINOv2 `last_hidden_state` | `Model.forward` |
| Pooling | `cls_mean`: `[out[:,0], out[:,1:].mean(1)]` (2·D) | `Model.forward`, `POOL_PARTS` |
| Head | `SlotHead(2D, 6, 12, hidden=256, p=0.2)`: `h=proj(x)+slot_emb`; per-target query softmax over slots, absent slots `-1e4`; `prior=False` for teacher members | `SlotHead` |
| Loss | `(BCE(logits) * w).mean()`, gold weight 3.0, derived 0.25+0.75·conf | `main()` |
| Optimizer | AdamW backbone 8e-6 (last 6 blocks + final layernorm) / head 1e-3, wd 0.02, OneCycle pct 0.15, holdout epoch selection | `build_model`, `main()` |
| Holdout scoring | mean **logits** over the 4 groups, then sigmoid | `predict` |
| Submission | `predict_member`: **10 overlapping** windows `0..9`, mean of **sigmoid** (`TTA_POOL="prob"`); members rank-averaged | `window_starts`, `predict_member` |

So the original does **not** train all ten windows jointly. Each update sees one
non-overlapping group; ten overlapping windows appear only at test time. The holdout
and submission aggregations also differ (logit mean over 4 versus probability mean over 10).

## 2. What is canonical now

`rsna/competitive/source_slot.py`. It is selected only by the new versioned
`classifier_strategy` (`kind: source_matched_slot_v1`). With no strategy, the default
config, every live recipe, the binding (the key is popped when absent) and
`CompetitiveStudyNet` are unchanged.

- `SourceSlotHead`: the original `SlotHead` (`prior=False`). It has the same parameter
  names and the same construction/RNG order, so a given seed gives the identical initial
  state. Tests prove bitwise equality with the compiled reviewed node (teacher review
  `4288d80a…`).
- `SourceSlotStudyNet`: the canonical DINOv3 encoder plus that head. **DINOv2→DINOv3 bridge
  (required):** CLS is DINOv3's final-normalized CLS (`pooler_output == last_hidden_state[:,0]`).
  The patch mean is over spatial tokens only. It **excludes the 4 register tokens**, via
  `_GrayEncoder.forward_features`. Feature width is 2048 for DINOv3-L.
- Training (`training_inputs`): the dataset still yields the admitted ten overlapping
  views /255. Per optimizer update (8 studies), group `g = sha256(kind, seed, epoch,
  update) % 4` selects view `3g` (equal to `take_group(row, g)/255`). Then `source_augment`
  replays the original `augment` draw order and uint8 arithmetic on CPU with a per-study
  generator `sha256(kind, seed, epoch, study_uid)`. One window per study is encoded, not ten.
- Evaluation and deployment: `window_probabilities` gives `sum_w sigmoid(z_w.float()) / 10`
  over windows 0..9 in the original order. `train.predict` uses it directly (no extra sigmoid).
  `deployment_probabilities(model, uint8_row, slot_mask)` is the MRI-only per-study interface.
- Normalization: once. uint8 → `/255` (hashed adapter, or `source_row_windows` at
  deployment) → `_GrayEncoder` ImageNet. Augmented pixels stay exact `k/255`.
- Kept from this repository, not the source: accepted `label_source_v1` fitting supervision,
  masks, folds and `effective_batch_loss`; all 12 targets in `rsna.schema.TARGETS` order;
  the fixed-final schedule; no gold, reports or teacher at inference. Study-level labels are
  never treated as evidence that the chosen window shows the finding.

### Declared differences from the original (not isolated causal effects)

- Window draw and augmentation use deterministic hashed generators, not global
  `torch.randint`/`torch.rand`. The distribution is the same; this makes resume exact and
  the pixels device independent.
- Absent slots are never encoded. In the original their attention weight is exactly 0.
- Loss, label weights, optimizer (canonical cosine/warmup281, head 3e-4, wd 1e-4, clip 1),
  FP16 scale128 and fixed 10 epochs / 2810 updates differ from the source's OneCycle,
  holdout-selected recipe.
- Encoder adaptation is DINOv3-L R32 or dense deltas, not full fine-tuning of the last
  6 DINOv2-small blocks plus the final layernorm.
- Teacher members' training roster is unknown. Nothing here gives a clean teacher/student comparison.

## 3. Arms (predeclared; fixed-final; one seed each)

| | `r32-config.json` | `dense-prefix18-config.json` |
| --- | --- | --- |
| base | `public-input-recipe-v1/public-triplet-config.json` + `classifier_strategy` | same + `suffix_adapter`, `encoder_lr` |
| encoder | ordinary live `rs_lora_v1` rank32 (all 24 blocks; FFN last 12) | `dense_suffix_delta_v1`, frozen prefix 18, last 6 blocks' 36 projections |
| learned encoder parameters | 10,223,616 R32 | **75,497,472** dense |
| encoder LR | 1e-4 | 1e-5 (as issue #43's dense arm; the source used 8e-6) |
| RoPE | live train-mode (stochastic `pos_embed_rescale`) | deterministic eval RoPE; FP32 frozen prefix |

Shared between the arms:
- pinned official DINOv3-L;
- admitted acquired336 public triplet rows and index;
- fold0 2247/1156 rosters, masks, folds and release;
- `label_source_v1` overlay `c8fd2e86`;
- seed 20260922;
- effective batch 8 (1×8);
- FP16 scale 128, chunk 8, cosine/warmup 281;
- 10 epochs / 2810 updates;
- the same strategy, with augmentation on.

The RoPE/prefix execution and the LR are declared recipe differences between the arms.
The issue #43 prefix12 configs are unchanged.

## 4. Before production (root)

- CPU input/recovery acceptance on the real rows: run the canonical trainer full /
  `--stop-after-updates 1` / resume, as in `run_dense_suffix_recovery.py`. Then H100 recovery
  and fresh funding. Cost estimate: training reads one window per study, so it encodes about
  10× fewer images than the live public-triplet recipe. Evaluation still reads 10 windows.
- Deployment is an interface only. `deployment_probabilities` needs the reviewed renderer's
  rows and slot mask. `rsna.competitive.infer` still refuses `public_input` checkpoints. Before
  any slot, a runtime must be admitted on offline 2×T4 / 4 CPU / ~29 GB / <9 h and write
  `submission.csv`. One DINOv3 model, never the public committee.
- No promise of 0.95 or of winning.

CPU coverage is in `tests/test_source_matched_slot.py`:
- original-node equality (head, ten-window aggregation, `take_group`, `augment`);
- register exclusion, masks, deterministic RNG and single normalization;
- canonical trainer full/stop1/resume for both arms, comparing model, frozen base, AdamW,
  LR, counters, all RNG streams, scaler, history and prediction bytes;
- deployment equal to trainer predictions;
- the legacy path unchanged.
