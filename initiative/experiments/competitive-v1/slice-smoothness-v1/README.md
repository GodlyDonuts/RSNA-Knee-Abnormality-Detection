# SmMIL adjacent-slice smoothness arm — experiment design + implementation

October 6, 2026. Roadmap step 6b (docs/single-model-roadmap-20261006.md):
slices are spatially ordered — a lesion occupies a contiguous run of
slices, not scattered singletons. SmMIL-style smoothness regularizes the
instance scores along the physical slice axis.

Status: **IMPLEMENTED** — `slice_smooth` config seam live in
`rsna/competitive/train.py`, loss in `rsna/competitive/loss.py`, model
exposes gated per-slice scores via `return_slice_scores=True`, and the
`slice-smooth-smmil` run config is emitted. Blocked on a GPU slot like
every other arm.

## Mechanism

- The attention pooling already computes per-slice gated scores
  (`slice_score(tanh·sigmoid gated features)` → `[study, series, slice,
  target]` logits driving the masked softmax over slices).
- Auxiliary loss = per-target mean squared difference between the scores
  of **physically adjacent slices** (`slice_score_tv`). Total variation on
  the score sequence rewards contiguous score mass — the tear/meniscus
  lesion prior — without hard-forcing any region.
- Adjacency is real: `data.py` verifies cache slice positions are strictly
  increasing along the series axis and `_indices`/`arange` preserve
  ascending order, so k-index adjacency of valid slices is physical
  adjacency. Padded slices are masked out of pairs; `adjacent_slice_pairs`
  counts pairs once per optimizer update (all-reduced under DDP), matching
  the effective-batch normalization of the other auxiliary losses.

## Isolation contract

`{"enabled": bool, "kind": "attention_score_tv", "weight": (0,1]}` —
config-bound like `masked_cell`. Isolated outer-fold arm only: no
relation/witness/fitting_supervision/neighbor_channels/canonical_sides/
masked_cell; requires `pooling="attention"` (per-slice scores only exist
there) and `slice_context=="none"`. Arm config: weight 0.1.

## Receipts

Per-update event fields: `slice_pairs`, `slice_smooth`,
`slice_smooth_weight`. Exact mid-epoch resume verified end-to-end on CPU
(no extra checkpointed state — the penalty is deterministic given the
student weights).

## Expected outcome range

Honest prior: attention MIL on MRI tends to spread mass diffusely;
smoothness priors (SmMIL, ordinal slice regression) help most on
contiguous-morphology targets (meniscus tears, ligament, effusion extent)
and least on point findings (small cysts). Target-level, not macro-level,
win expected — the holdout bootstrap decides per target.
