# Research handoff — corrected replacement for PR #6 (2026-10-12)

Supersedes `docs/codex-handoff-20261012.md` on PR #6 where they disagree. PR #6
treated the label-source ablation as pending and proposed MRI-CORE as a one-config
drop-in; both are out of date or wrong.

## 1. Label-source experiments: complete

All on the same 1,156 weak-development studies, fixed-final, current rank-32 /
native-384 / fixed-10-epoch recipe. One seed each. Weak-development internal macro AUC,
**not Kaggle scores**, not clinical evidence.

| Fitting supervision | Macro AUC | Paired delta vs original (95% conditional) | Source |
|---|---|---|---|
| Original (accepted v2) | 0.8786705965 | — | `initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/` |
| Full public | 0.8997840648 | +0.0211134683 [0.0029817801, 0.0405243740] | same |
| Original masks, public values | 0.8536812809 | -0.0249893156 [-0.0439907607, -0.0054025504] | `initiative/audits/public-label-fixedmask-production-20261010-v1/fixed-final/` |
| Coverage only, original observed values | 0.8693023133 | -0.0093682832 [-0.0402732007, 0.0222779240] | `initiative/audits/public-label-coverage-production-20261011-v1/fixed-final/` |

- Neither follow-up was promoted. Value-only (fixed mask) regressed; coverage-only
  is not a supported explanation of the full-public gain. Joint value/coverage effects
  remain open.
- The full-public gain is a single-seed weak-development result; its medial-meniscus
  secondary regressed. It is not a promotion and not a target-source winner declaration.
- **Next empirical priority:** a separately matched repeat of full-public versus
  original supervision (fresh seed, same recipe/data/evaluation), before building any
  further label-source arm.
- Do not use gold subsets with inconsistent denominators (e.g. PR #4 gold-58
  arbitration) to declare per-target source winners, and do not use cross-notebook
  leaderboard comparisons (e.g. a copied public recipe at 0.942 vs our 0.804) to rule
  out encoder or recipe bottlenecks; those comparisons change data, labels, model and
  pipeline at once.

## 2. MRI-CORE is not a drop-in

PR #6 called MRI-CORE a "direct drop-in ... one config change". It is a DINOv2-teacher
ViT-B/16 (768 wide, 12 layers, fused QKV, no registers), not DINOv3-L (1024 wide, 24
layers, registers, RoPE, separate projections). It needs its own loader, key mapping,
position interpolation and adapter graph, and any comparison is a backbone-package
comparison, not initialization-only. Prototype, provenance status, costs and a bounded
experiment proposal: `docs/mri-core-backbone-prototype-20261012.md`.

## 3. Still valid from PR #6

The corrections it lists for PR #4 arms (entropy sign, blended-overlay cell drift,
non-physical sagittal flip, unfiltered SSL manifest, EMA touching the frozen base,
frozen no-adapter configs, gold-informed arms) remain blockers; see
`research-pr-review-20261006-v1`.

## 4. Order

1. Matched repeat: full-public vs original supervision.
2. Primary-source verification of MRI-CORE, then (if admitted) the matched backbone
   experiment in the prototype doc, on whichever supervision the repeat supports.
3. Remaining PR #4 arms only after their listed corrections.
4. Ensembles stay deferred.
