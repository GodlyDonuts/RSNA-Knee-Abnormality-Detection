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

## Why the gap is structural, not an encoder-quality problem

1. **Ensemble diversity.** ~10+ decorrelated members vs. one backbone, one
   recipe, one fold's weights. This alone is worth a large fraction of the gap
   on a rank metric; it does not require any member to be individually better
   than ours.
2. **Label-source alignment.** Every public member trains on the same public
   LLM label tables. If the hidden evaluation distribution is closer to those
   labels than to our conservative clinical review (unproven, but plausible for
   a rank metric), our cleaner-but-sparser labels could actually hurt. This is
   a testable hypothesis, not an assumption — see direction 3 below.
3. **Our internal metric cannot steer.** The 1,156-study weak-label development
   set has **two** positive MCL components; 1,164 of 4,000 whole-component
   bootstrap draws are undefined for macro AUC. Internal deltas of ±0.001–0.04
   are unresolvable at this sensitivity. Mechanism experiments measured only on
   this set are hypothesis generation, not evidence — the public score is the
   only arbiter that matches the competition objective.
4. **Observed supervision is very sparse for rare targets.** In the 3,403-study
   loss: MCL 7 positives, fracture 5, Baker's 9, effusion 25, lateral OA 27.
   No architecture change manufactures positives; supervision work and
   ensembling over members trained on denser public labels both address this.

## Recommended directions, ranked by expected value per unit cost

1. **Add our checkpoints as members of the proven ensemble recipe.**
   The 0.942 kernel is already on the account and scored. Adding our
   all-development native384 rank32 (0.804 solo) and fold-0/fold-2 checkpoints
   as rank-fused members is the cheapest real gain available: if our errors are
   even partially decorrelated from the community members, mean AUC should move
   above 0.942. Requires Kaggle-side work where account auth works.
   `rsna/rank_fusion.py` (this change) provides the canonical, tested
   per-column rank-fusion implementation matching the OSS mechanism.
2. **Train diverse OSS-family members on the v2 reviewed labels.** The proven
   winning recipe is DINOv2-small three-slice windows + per-target series
   attention (+ COATNet/ResNet-RadImageNet variants). Training 2–3 members of
   that family on our reviewed labels combines our label investment with their
   proven architecture and adds further ensemble diversity.
3. **Controlled label-source ablation.** Same architecture, same folds: one
   member on public LLM labels vs. one on v2 reviewed labels, submitted as
   paired ensemble members. This directly measures whether the label source
   matters for the hidden score — the single most important unresolved
   supervision question for this competition.
4. **Distill public teachers on masked cells** (plan already in
   [dinov3-adapter-and-supervision-plan](dinov3-adapter-and-supervision-plan.md)):
   outer-fold-safe teacher predictions on the large masked fractions (fracture
   2,699/3,403 masked) could densify supervision without corrupting observed
   labels. Agreement is not truth — admit per-target only.
5. **Keep mechanism research** (smooth witness, corroborating witnesses,
   neighbor context) as research, but promote nothing to competition on
   internal deltas alone; gate on paired public submissions.

## Explicit non-recommendations

- More rank/epoch/resolution scaling of the same solo recipe: plateau measured
  (fold-2 e6→e10 flat at ~0.64; internal +0.0009 and +0.0025 deltas with
  intervals including zero).
- Reading official gold or protected labels for tuning — remains prohibited.
- Claiming the v2 labels are "better" for the competition before the paired
  label-source ablation (direction 3) is measured.

## Hard blockers observed in this session

- Kaggle API auth from this environment returns 401 (expired token); ensemble
  integration must run where account auth is live (Newton/codex workflows).
- The 1,300-study T4 inference benchmark completed in 18,303.6 s (~5.1 h),
  inside the 9 h budget — runtime is not the blocker for adding members.
