# Graded-target overlay — supervising `uncertain` v2 cells at an intermediate value

**Status**: artifact bound and verified end-to-end against the accepted release;
the fit itself has not run.

Frontier item 4 (`docs/research-frontier-20261006.md`): soft-label learning
beats hard labels specifically under small-N and class imbalance (Machine
Learning 2025, doi:10.1007/s10994-025-06860-8). Our v2 release already carries
an ordinal `__state` per cell — this experiment tests whether spending that
third level as supervision beats masking it.

## The contrast

The binary baseline feeds each fitting cell as:

| v2 `__state` | baseline label | baseline mask |
|---|---|---|
| positive | 1.0 | 1 |
| negative | 0.0 | 1 |
| uncertain | 0.0 | **0** (masked) |
| not_mentioned | 0.0 | **0** (masked) |

The graded overlay changes exactly one row:

| v2 `__state` | overlay label | overlay mask |
|---|---|---|
| uncertain | **0.5** | **1** (supervised) |

`not_mentioned` stays masked — silence is still not negative. The overlay is a
`label_source_v1` artifact, so it rides the identical bound admission path as
the label-source ablation: sha256 + release/folds/roster binding, fitting-scope
only, evaluation frames untouched.

## Artifacts (all three outer folds)

- `graded-states-u05.fold0.overlay.json`, sha256
  `77b3533880bfe22d520c09631d15012a12d97485c4d9fe615355840a26cb887a` —
  2,247 fitting studies (`fitting_uid_sha256`
  `e9d38ee21a114973177b89f4602129c7dc047927b8499991d3be30fd89737f3a`),
  **5,688 uncertain cells** supervised at 0.5 (~21% of the fitting table);
  verified through `load_fitting_label_source` (6,516 cells bound and
  swapped, zero evaluation-mask changes).
- `graded-states-u05.fold1.overlay.json`, sha256 `14a50020…` — 2,278 fitting
  studies, 5,555 uncertain cells supervised.
- `graded-states-u05.fold2.overlay.json`, sha256 `17952028…` — 2,281 fitting
  studies, 5,643 uncertain cells supervised.

## Paired protocol

Two arms, identical in every respect except the supervision artifact:

| arm | config |
|---|---|
| control | current binary/masked fitting (no overlay) |
| graded | `fitting_supervision = {kind: "label_source_v1", path: <artifact>, sha256: "77b35338…"}` |

Same release, same folds, same outer fold (1,156 studies), same recipe, same
seed. Decision rule per the research-frontier doctrine: paired bootstrap
interval on the derived-label holdout as arbiter; public submission breaks
ties, never gold58.

## Interpretation caveats

- 0.5 is the agnostic choice: it says "the reviewer saw something but would
  not commit", not "50% prevalence". A per-target calibrated value (e.g. the
  positive rate among uncertain cells resolved by adjudication) is the
  principled follow-up variant; it needs the disagreement-queue adjudication
  results first, so u=0.5 is the honest first contrast.
- The risk is symmetric: if uncertain cells are mostly noise, supervising them
  at 0.5 injects a learnable bias toward the decision boundary — the ablation
  measures exactly this trade-off.
- Loss-side alternatives (entropy-maximization, consistency pseudo-labels on
  masked cells — frontier item 6) attack the same budget through the objective
  rather than the labels; they are complementary, not competitors.

## Rebuild

```
PYTHONPATH=. python scripts/research/build_graded_overlay.py \
  --studies initiative/releases/report-supervision-v1/studies.parquet \
  --folds initiative/experiments/competitive-v1/folds/fold.csv \
  --fold 0 --uncertain 0.5 \
  --output initiative/experiments/competitive-v1/graded-targets-v1/graded-states-u05.fold0.overlay.json
```

`--uncertain` sets the intermediate target; `--fold N` rebuilds for another
outer fold.
