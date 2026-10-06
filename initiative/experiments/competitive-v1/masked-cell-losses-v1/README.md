# Masked-cell loss treatments — design for the sparse-target problem

**Status**: EM arm implemented (2026-10-06); consistency arm still design.
Frontier item 6 (`docs/research-frontier-20261006.md`).

## Why this experiment

Our fitting table is ~53% masked cells (uncertain + not_mentioned stay masked —
silence is not negative). That is a missing-label problem, and the literature
on single-positive/multi-label (SPML) and PU learning has spent five years on
exactly it:

- **Cole et al. 2021** ("Multi-label learning from single positive labels") —
  an EMA pseudo-label + consistency-regularization scheme that recovers most
  of the fully-supervised gap with one positive per sample.
- **MPC / IPC-FPC-BTC** (MICCAI W 2024) — masked-cell treatments benchmarked
  on medical multi-label tasks.
- **ESA** (arXiv:2412.02240) — entropy maximization on unlabeled cells to
  prevent collapse to all-negative.
- **NeurIPS 2025 PU survey** — generalized assume-negative variants.

The sparse targets are where this matters most: MCL has 7 observed positives,
fracture 5, Baker's 9 in the fitting roster. At those counts, how the ~95%
masked cells are handled IS the supervision signal.

## Design: three arms on the masked budget

Identical fit in every respect except the loss on masked cells:

| arm | masked-cell treatment | mechanism |
|---|---|---|
| control | ignore (current) | BCE on `mask==1` cells only |
| EM | entropy maximization | `+ λ Σ_{masked} H(ŷ)` — penalizes confident outputs where we have no information; generalized assume-negative breaks |
| consistency | EMA pseudo-label | teacher (EMA weights, weak aug) emits target for masked cell; student (strong aug) fits it with weight ρ(t) ramped over epochs |

Hyperparameters λ and the ρ ramp are the only new knobs; each arm needs one
fit under the same seed/folds/recipe.

## What it is NOT

- Not the graded overlay (`graded-targets-v1/`): that supervises `uncertain`
  cells at 0.5 through the label table; this attacks the same budget through
  the objective. They compose — consistency pseudo-labels on `not_mentioned`
  cells with graded 0.5 on `uncertain` cells is the natural combined arm once
  each wins or loses alone.
- Not assume-negative: the repo rule that silence is not negative stands. EM
  and consistency only shape the *distribution* over masked predictions; no
  masked cell ever receives a hard 0 target.

## Implementation surface (as built)

- `TrainConfig.masked_cell = {"enabled": bool, "kind": "entropy_max", "weight":
  float in (0,1]}` — config-bound (no artifact). `kind` is extensible; only
  `entropy_max` is implemented. `enabled: false` binds the arm while running
  the ignore control, same overlay convention as other arms.
- `masked_cell_entropy(logits, mask, masked_counts)` in `loss.py` mirrors
  `effective_batch_loss` accounting: per-target sums over `mask == 0` cells
  divided by effective-batch masked counts collected once per optimizer
  update (DDP all-reduced); the trainer subtracts `weight × entropy`
  (subtraction maximizes entropy — penalizes confident predictions where
  labels observe nothing).
- Isolated like every first contrast: outer-fold scope, no
  relation/witness/fitting_supervision/neighbor_channels/canonical_sides.
  Arm composes with the graded overlay only after winning alone — the
  validator forbids the combination today.
- Receipt fields: `masked_entropy`, `masked_cells`, `masked_weight` in the
  per-update event log.
- Consistency arm (EMA teacher pseudo-labels) still needs the teacher copy
  — ~30 lines, not yet built.
- Bound like the other experiments: config hash + receipt in the run record,
  same release/folds/outer fold, paired bootstrap on the derived holdout.

## Expected outcome range

Honest prior: SPML gains are largest when positives-per-class is small and
missingness is high — both true here — but the published magnitudes are on
natural-image benchmarks. Order them: consistency > EM > ignore is the
literature prior; measuring it on knee MRI is the point of the arm.
