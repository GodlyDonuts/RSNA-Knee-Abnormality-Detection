# Breakthrough assessment: why 0.804 vs 0.94, and the highest-leverage moves

October 6, 2026. Read alongside [the research brief](research-brief.md),
[kaggle-submissions](kaggle-submissions.md) and
[the public-teacher audit](../initiative/audits/public-teacher-notebooks-v1/README.md).
Every score below is an actual public competition score recorded in the repo;
interpretations are labelled as inference where they are not directly proven.

## The score ledger

| Entry | Public score (mean of 12 ROC AUCs) |
| --- | ---: |
| Owner's copied OSS ensemble `rsna-knee-0942-restructured` (sub 56407271) | **0.942** |
| Owner's copied OSS ensemble `head-and-shoulders-knees-and-toes` (sub 55969300) | **0.936** |
| Our rank32 native384 all-development (best solo) | **0.804** |
| Our rank32 native384 fold0 | 0.785 |
| Our strict rank32 uniform224 fold0 | 0.770 |
| Our sampled rank64 fold0 | 0.778 |
| Our fold-2 global-CLS epochs 2/4/6/8/10 | 0.611/0.628/0.637/0.638/0.635 |

The gap between the best in-house solo model and the copied community ensembles
is ~0.14 mean AUC. Our fold-2 comparison shows the same recipe plateaus by
epoch 6-8: this is not something more epochs or adapter rank will close.

## What the 0.94 pipeline actually is

From the [static audit](../initiative/audits/public-teacher-notebooks-v1/sources.json)
of the three public notebooks (`pjmathematician/rsna-knee-d4-lite`,
`maverickss26/rsna-knee-restructured-version-3`,
`amanatar/rsna-knee-abnormality-detection`):

- **A large rank-fused ensemble**, not one model: ~14 attached datasets including
  `mattiaangeli/knee-mri-fold-weights`, `rsna-knee-coatnet-*` variants,
  `rsna-knee-coat-resgated-*`, `antoinegg1/rsna-knee-e11-diverse-heads-v20`,
  `e9-radimagenet-heads-v15`, `prvsiyan` RadImageNet heads,
  `marwanmath/resnet-50-radimagenet`, `pilkwang/rsna-knee-weights`, and
  `metaresearch/dinov2` small. Backbones span DINOv2-small, COATNet variants and
  ResNet-50/RadImageNet — decorrelated error profiles.
- **Trained on public LLM labels** (`pilkwang/rsna-knee-llm-labels` is an
  attached dataset), not on reviewed report labels.
- **2.5D inputs**: overlapping three-slice windows packed into three channels,
  enumerated over a cached slice selection, with per-target series attention —
  an MRI-MIL recipe proven to fit the T4/9 h budget.
- **Rank-fusion output**: each member's per-target column is percentile-ranked
  across the cohort and ranks are combined. For a per-target ROC-AUC metric
  this is free performance — rank transforms preserve individual AUC while
  decorrelated member errors average out.

## Why the gap is unlikely to be closed by more of the same recipe

1. **Label-source alignment.** Every public member trains on the same public
   LLM label tables (`pilkwang/rsna-knee-llm-labels`). If the hidden evaluation
   distribution is closer to those labels than to our conservative clinical
   review (unproven, but plausible for a rank metric), our cleaner-but-sparser
   labels could actually hurt a single model. This is a testable hypothesis,
   not an assumption — see direction 2 below.
2. **Our internal metric cannot steer.** The 1,156-study weak-label development
   set has **two** positive MCL components; 1,164 of 4,000 whole-component
   bootstrap draws are undefined for macro AUC. Internal deltas of ±0.001–0.04
   are unresolvable at this sensitivity. Mechanism experiments measured only on
   this set are hypothesis generation, not evidence — the public score is the
   only arbiter that matches the competition objective.
3. **Observed supervision is very sparse for rare targets.** In the 3,403-study
   loss: MCL 7 positives, fracture 5, Baker's 9, effusion 25, lateral OA 27.
   No architecture change manufactures positives.
4. **Input/recipe differences remain uncontrolled.** The OSS recipe uses
   overlapping three-slice 2.5D windows over cached slices with per-target
   series attention; ours encodes up to 16 independent slices per series and
   pools globally. Our earlier frozen patch-attention variant regressed
   (-0.016), but that was a different stage, not a controlled recipe
   comparison.

## Recommended directions — single model first, ensemble later

The owner's direction (October 6): build a **highly sophisticated single
model** first; treat ensembling only as a later step once a strong solo model
exists. Public single models reportedly reach ~0.95, so the solo-model ceiling
is not the binding constraint — the recipe and supervision are.

1. **Controlled label-source ablation — the single most important experiment.**
   Same architecture, same folds, same training budget: one member trained on
   the public LLM labels vs. one on the v2 reviewed labels, each scored
   publicly as a single model. This directly measures whether the label source
   is the main lever for the hidden score — everything else (recipe tuning,
   mechanism research, distillation) is downstream of that answer. If public
   LLM labels win, our v2 review effort needs different uses (mask policy,
   correction overlays, evaluation) rather than the primary training signal.
2. **Build the best single model on whichever labels win.**
   Adopt the proven input recipe — overlapping three-slice 2.5D windows,
   per-target series attention — with a strong backbone (our adapted DINOv3-L
   rank32 recipe or DINOv2-small as in the OSS members). This is a controlled
   recipe comparison against our current 16-slice/global-pooling recipe on the
   same labels and folds, not another untested variant.
3. **Densify rare-target supervision via teacher distillation** (plan already
   in [dinov3-adapter-and-supervision-plan](dinov3-adapter-and-supervision-plan.md)):
   outer-fold-safe teacher predictions on the masked fractions (fracture
   2,699/3,403 masked, MCL 7 observed positives) — admit per-target only;
   agreement is not truth.
4. **Keep mechanism research** (smooth witness, corroborating witnesses,
   neighbor context) as research, but promote nothing to competition on
   internal deltas alone; gate on paired public submissions.
5. **Ensemble only after a strong solo model exists.** `rsna/rank_fusion.py`
   (this change) already provides the canonical, tested per-column rank-fusion
   implementation matching the OSS mechanism, so the option is ready when
   warranted — it is not the current priority.

## Explicit non-recommendations

- More rank/epoch/resolution scaling of the same solo recipe: plateau measured
  (fold-2 e6→e10 flat at ~0.64; internal +0.0009 and +0.0025 deltas with
  intervals including zero).
- Reading official gold or protected labels for tuning — remains prohibited.
- Claiming the v2 labels are "better" for the competition before the paired
  label-source ablation (direction 1) is measured.

## Hard blockers observed in this session

- Kaggle API auth from this environment returns 401 (expired token); ensemble
  integration must run where account auth is live (Newton/codex workflows).
- The 1,300-study T4 inference benchmark completed in 18,303.6 s (~5.1 h),
  inside the 9 h budget — runtime is not the blocker for adding members.
