# Direct AUC optimization: a reserve hypothesis, not a drop-in fix

14 October 2026. Read-only literature and implementation review. No training or package installation.

## Why inspect this

The project correctly found zero positive/negative study pairs inside a single-study forward. Eight independent backward passes do not create a pairwise ranking graph. This does not rule out every AUC-oriented objective: moment/min-max formulations replace explicit pairs with class-conditional statistics and auxiliary variables.

The [ICCV 2021 Deep AUC paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) develops an AUC-margin surrogate and reports improvements on several medical-image tasks. Its evidence is relevant to the metric, but is neither a knee-MRI result nor a guarantee under report-dependent missingness. [Author implementation](https://github.com/Optimization-AI/ICCV2021_DeepAUC).

## The existing library is not compatible without careful adaptation

The [official MultiLabelAUCMLoss source](https://docs.libauc.org/_modules/libauc/losses/auc.html) checks targets for equality to 0 or 1. Continuous public probabilities would not be treated as intended. Its v1 default estimates prevalence from the current batch; with a single hard-labeled example this becomes 0 or 1. Its v2 conditional means can encounter empty-class denominators. Masked missing targets are also not the same as negatives. These are reasons not to paste the default loss into this trainer.

A valid alternative would need fixed fitting-only priors or an explicitly justified class-conditional sampling estimator, correct masks, stable rare-class statistics, and complete checkpointing of auxiliary parameters and optimizer state. Binarizing uncertain labels merely to satisfy the API would change the scientific question.

## A theoretical soft-target extension has an important limit

One can replace hard positive/negative indicators in expectation by q and 1−q, with p equal to the fitting-distribution mean of q. That defines a surrogate for the induced probabilistic label distribution. It does not establish that q is a calibrated disease probability, recover missing severity, or identify a real clinical class prior. The current report table mixes finding strength and uncertainty; optimizing its induced AUC more aggressively could amplify those mistakes.

The constant-memory sufficient-statistic formulation is therefore potentially practical, but the target-distribution problem remains. It also has a different objective from teacher pair-order transfer. Treat these as separate hypotheses.

## Minimal proposed comparison if supervision semantics are first resolved

- Start from the same immutable fitting-only checkpoint, never an all-development model when evaluating the outer split
- Prespecify a bounded continuation budget and identical images, augmentations, data order, target mask and endpoint
- Keep an unchanged BCE continuation control
- If optimizer, sampler or update schedule changes, add the corresponding BCE recipe control or label the result as a bundled loss/optimizer/sampler comparison
- Preserve all 12 targets, target supports and undefined bootstrap draws; do not select only classes where a pair happens to be available
- Verify on synthetic data that masked targets have no influence, single-class microbatches remain finite, gradients match the intended aggregate estimator, and resume restores all state
- Go only if the estimator is valid for the admitted labels, runtime remains comparable, and a fixed-final matched result warrants another seed
- Stop if success depends on treating missingness as negative, guessed clinical prevalence, unstable rare-class auxiliary state, or extra image exposure hidden inside resampling

## Priority

Below fold-safe image supervision, severity-aware target construction and unresolved existing adaptation/bag comparisons. Above an unexamined claim that direct AUC learning is impossible solely because the forward batch size is one. The strongest potential benefit is improving the ranking objective with no added inference cost; the largest risk is optimizing incorrect weak targets more efficiently.
