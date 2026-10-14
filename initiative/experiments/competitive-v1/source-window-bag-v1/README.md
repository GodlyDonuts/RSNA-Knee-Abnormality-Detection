# Source window-bag v1 (issue #49): plain mean versus learned disease-query weighting

Status: **prepared only**. No GPU job, data transfer, upload, label/fold change,
teacher collection, or submission was made. Root runs actual GPU recovery and
funds production separately. Nothing here claims novelty, clinical accuracy, or a
reproduction of the 0.949 CoAtNet pipeline described in
[Kaggle discussion 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792).
That discussion motivates the question. See
`initiative/research/public-high-score-update-20261013-v1`.

## Question

Does letting each disease pick evidence across a study's windows beat fixed
probability averaging? The two arms differ only in the window weighting. Both use
the same inputs, encoder, supervision and budget.

## Policy `source_window_bag_v1` (`rsna/competitive/window_bag.py`)

The policy reuses the reviewed `source_matched_slot_v1` components without change:
the DINOv3 encoder with CLS plus register-free patch mean, the 6-slot
`SourceSlotHead` (hidden 256, dropout 0.2, no prior), and the source augmentation.
`rsna/competitive/source_slot.py` is byte-unchanged (SHA `3e4e1bb9…`, checked by a
test), so the live source-slot recovery and its configs are untouched.

For each window `w` and target `t`, the head's own slot attention gives a
target-conditioned window feature `c[w,t]` and a window logit `u[w,t]`
(`target_window_features`, bitwise the head's logits). The pooled study output is:

```
beta[w,t] = 0.75 * softmax_w(selection * <c[w,t], q[t]> / 16) + 0.25 / W
p[t]      = sum_w beta[w,t] * sigmoid(u[w,t])          (returned as exact logit)
```

- `q` (`window_query`, `[12,256]`) starts at zero.
- `selection` is set by `window_weighting`:
  - `plain_mean`: `selection = 0`, so `beta = 1/W` and `p` is fixed probability
    averaging.
  - `disease_query`: `selection = 1`.
- The pool runs in FP32 log-space with masked windows. Each window weighs between
  `0.025` and `0.775`.
- The trainer applies its normal single study-level masked BCE to `logit(p)`.
  Prediction applies one sigmoid.

| | `mean-r32-config.json` | `query-r32-config.json` |
|---|---|---|
| parameters / names / init values | identical (`window_query` present, zero) | identical |
| RNG, inputs, augmentation draws, supervision, FP16, schedule, budget | identical | identical |
| floor 0.25, FP32 pool, gradient clip 1.0 | identical | identical |
| window weights | fixed `1/W` (query has no effect and gets zero gradient) | learned per disease |

Both configs equal `source-matched-slot-v1/r32-config.json`, the fixed-final
fold-0 recipe, in every key except `classifier_strategy`. That recipe uses accepted
fitting supervision, 10 epochs × 281 updates, effective batch 8 and warmup
281/cosine. The weights start equal, so the first update's forward and loss match
between arms. After that, the gradient-norm clip includes the query gradient, which
is zero in the mean arm.

## Differences from the one-window `source_matched_slot_v1` recipe

| | one-window v1 | window bag v1 (both arms) |
|---|---|---|
| training views per study per update | 1 group, `3g` for one hashed `g` per update | all 4 groups 0/3/6/9 every update |
| label assignment | study label on one arbitrary window | study label on the pooled bag (one loss) |
| augmentation | per-slot draw on the selected group | the same per-slot draw on all 12 slices; group `g` is bitwise the one-window augmented group `g` |
| evaluation | mean sigmoid over 10 windows | bag pool over the same 10 windows |
| extra parameters | — | `window_query` 3072 |

`plain_mean` evaluation equals v1's ten-window probability mean up to floating
rounding, but it is trained differently: on the four-group bag mean instead of one
window. It is therefore a new control, not v1.

## Cost

Encoder passes scale with views × acquired slots:

- **Training:** 4 views per study per update instead of 1, so about 4× encoder
  forward/backward compute and activation memory per micro-batch. With gradient
  checkpointing, `encode_chunk_size=8` bounds the peak per chunk, but the
  four-view graph is kept for one backward. For 2247 fitting studies × 10 epochs
  that is 89,880 study-bag passes, the same as v1's study count but with 4× the
  views.
- **Evaluation and deployment:** 10 views per study in both arms, the same as v1.
- **Pool:** negligible.

The real H100 time and memory are not measured here. The root GPU recovery must
admit them.

## Not claimed or interpreted

- Windows show different anatomy. Their disagreement is not epistemic
  uncertainty and is never used as a weight or as confidence.
- No teacher is used in this first contrast. PR #38's window-teacher loss and
  standalone `FocalWindowPool` are not imported or duplicated. Its bounded-pool
  tests are adapted in `tests/test_window_bag.py` with authorship preserved.
- Inference is MRI-only (`window_bag.deployment_probabilities`: uint8
  `[6,12,336,336]` row plus slot mask, giving `[12]`).

## Tests

- `tests/test_window_bag.py`: policy tampering, arm graph/initial equality,
  mean-is-mean, NaN-poisoned window masking, missing slots never encoded,
  extreme-logit gradients, single sigmoid, augmentation equality and MRI-only
  deployment. It also runs the canonical trainer on the engineering encoder:
  full/stop1/resume exact for both arms, and the arms share the first-update loss
  and RNG.
- `tests/test_window_bag_training.py`: canonical-trainer full/stop1/resume exact
  equality for both arms (model, base, AdamW, LR, counters, all RNG, scaler,
  history, prediction bytes), binding, four-group bags, and student restore.
