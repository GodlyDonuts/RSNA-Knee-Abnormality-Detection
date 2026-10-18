# No established overall gain from the dense spatial reader

Both original three-epoch jobs completed0:0 with843updates on2247fitting studies.
All six immutable checkpoints are privately verified. Distinct CPU959391 fixed
only the missing module-file binding after original analysis959352failed before
metric computation; original numerical functions, predictions and source stayed
unchanged. The new weighted-AUC wrapper regression passed.

On the same1156weak-label development studies/1106components, macro AUC is:

| Model | Macro AUC |
| --- | --- |
| Aligned path | 0.9212176612 |
| Marginal null path | 0.9205166745 |
| Simultaneous unchanged FP32 parent | 0.9205878230 |

Aligned minus null is+0.00070099, conditional95%interval[-0.00092435,+0.00236799].
Aligned minus parent is+0.00062984, interval[-0.00429430,+0.00486056]. Of4000whole-
component draws,1186were undefined for the all12-target macro. Keep those counts;
the intervals are conditional on defined draws, this fitted seed/fold and the
weak-label pool. They exclude training/search/clinical uncertainty. Direct full
production step-zero tensor equality was not separately dumped.

Lateral Meniscus improves against parent in BOTH aligned and null branches
(+0.010774/+0.010390), while their contrast is only+0.000384 with an interval
crossing zero. That pattern does not attribute the shared gain to anatomical
alignment. Do not cherry-pick target outputs or claim a clinical/Kaggle gain.

Stop promotion and expansion of this unchanged frozen-encoder version. Preserve
its source/results/negatives and all six checkpoints; keep the parent reference.
This is not evidence that anatomy-aware learning is generally ineffective.
Prioritize the actual encoder-learning/gradient/cost and higher-LR experiments.
The H100 alternative currently has availability evidence only; actual tensors,
immutable wrapper/runtime and resource admission remain prerequisites to a GPU.
