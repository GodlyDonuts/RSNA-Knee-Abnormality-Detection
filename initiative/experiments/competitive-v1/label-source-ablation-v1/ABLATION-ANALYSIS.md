# Label-source ablation — preregistered analysis plan

Written before any GPU run of these arms. When results land, this document
is the decision rule — don't retrofit hypotheses after seeing scores.

## Arms under test (all same architecture, same folds, same recipe)

| arm | supervision delta vs v2 baseline | hypothesis class |
|-----|----------------------------------|------------------|
| `base-v2-control` | — | reference |
| `label-source-public` | replace v2 with public LLM labels | source swap |
| `label-source-blended` | v2 on 8 targets, public on 4 severity | oracle-ish upper bound |
| `label-source-v21hc` | +pass-2 positive flips only (gold-prec 0.75) | high-precision augmentation |
| `label-source-v21` | +all positive flips (mixed precision) | full severity-recall augmentation |
| `label-source-v21pn` | v21 + explicit-negative flips | augmentation + negation supervision |
| `graded-u05` | v2 `uncertain` cells supervised at 0.5 | soft-label alternative to masking |

## Predictions (recorded 2026-10-06)

1. **Per-target split on public**: public-arm beats v2 on Effusion/Baker's/OA
   (public's gold recall 0.82–1.0 vs v2 0.41–0.67) and loses on meniscus/
   synovitis/ligaments. If the split does NOT appear, the per-target blend
   rationale collapses — re-examine gold arbitration.
2. **v21hc ≥ v21 on severity targets**: pass-1/pass-3 flips are gold-measured
   noisy (0.46 / 0.08 precision). If v21 ≥ v21hc instead, severity-label noise
   is being tolerated well — argues BCE robustness makes label purity less
   critical than coverage.
3. **v21pn > v21 on Effusion/Baker's**: pool-3 produced 367 explicit
   negatives on fluid targets vs only 48 positives; negatives are the
   highest-precision additions (3/3 on gold). If v21pn ≤ v21, explicit
   negation supervision on contested cells is harmful — drop the arm.
4. **graded-u05 underperforms v21**: 0.5 supervision on uncertain cells
   injects label mass without commitment; predicts worse on high-signal
   targets. If graded-u05 ≥ v21, masking is throwing away usable signal —
   revisit masked-cell handling entirely.
5. **blended ≥ all arms on mean AUC** but is disqualifiable: it's
   gold-informed, so it only bounds how much a per-target policy could win.
   Any arm beating blended is evidence of extraction bias in the arbitration.

## Decision rules

- **Ship single-model recipe**: the arm maximizing mean outer-fold AUC with
  severity-target recall ≥ public baseline and meniscus/ligament recall
  within noise of v2 baseline.
- **If v21hc ≈ v21 ≈ base**: supervision additions had no effect → the
  bottleneck is elsewhere (architecture/capacity), pivot to the
  laterality/SSL arms before more label work.
- **If public wins broadly**: gold's severity convention dominates and v2's
  bar was the binding constraint → adopt public-on-severity policy (blend),
  keep v2 morphology for the rest.
- **Per-target inspection, not just mean**: record all 12 target AUCs per
  arm per fold; a +0.02 mean from losing meniscus is a different decision
  than +0.02 from winning severity uniformly.

## Known confounds

- Gold-58 held out of fitting → outer-fold AUC still measures report-label
  targets, not official gold semantics; public-LB confirmation needed for
  the final call.
- Pool-3 classifier is a research artifact (clause-level regex classifier),
  not a reread of images — its precision estimate (0.08 on gold positives)
  is the weakest measured link; treat v21/v21pn gains on severity targets
  with proportionally more skepticism than v21hc gains.
- All overlays are `label_source_experiment_only`; evaluation masks are
  untouched, so arm comparisons are apples-to-apples by construction.
