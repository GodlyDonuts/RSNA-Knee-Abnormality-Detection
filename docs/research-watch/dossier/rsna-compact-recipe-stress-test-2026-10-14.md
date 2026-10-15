# Compact CoAtNet recipe stress test

14 October 2026. Independent review of the single Native-CoAtNet-Bag12 proposal and final-week decision.

## Recommendation

**Keep the one compact comparison, with a more precise sampling caveat.** Two mechanisms survive scrutiny: training and inference can weight the available evidence differently, and single-study BatchNorm can change feature distributions at deployment. Neither demonstrates a lower knee-MRI AUC, justifies an architecture sweep, or establishes a fix that is better than the declared recipe. No new performance experiment is recommended here.

## 1 Sampling coverage and pooling consistency are different objectives

For fixed window embeddings, attention pools `z = sum(exp(a_i) h_i) / sum(exp(a_i))`. This is the same normalized weighted-average form in the original [attention MIL paper, equations 7–8](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf). Permutation invariance does not make predictions invariant to changing the number or composition of windows.

**Proven unequal-length effect.** Suppose six selected series have `[14,14,14,14,14,2]` candidate windows. The proposed sampler takes two from each. With equal attention scores and a constant embedding within each series, the short series receives exactly `1/6` of training pooling mass and `1/36` at all-window inference. The other series change from `1/6` each to `14/72` each. This sixfold short-series reduction is a real change in the pooling measure, not numerical noise.

This can change ranking, not only calibration: with projected evidence 1 in that short series and 0 elsewhere, the pooled value falls from `.1667` to `.0278`; a second study whose windows all project to `.1` would reverse order. This is a synthetic possibility, not an observed AUC loss. The prevalence of such short series is unknown. If all present series have the same length, candidate inclusion probabilities are equal; missing whole series alone does not create this particular defect.

**Why not reflexively switch to uniform sampling?** Uniform sampling of twelve from 84 gives each candidate equal inclusion probability and removes this unequal-length sampling preference. But it omits at least one of six complete series in **49.276%** of bags; omission of any particular series is **9.416%**. For a hypothetical three-window witness confined to one full series, two-per-series sampling captures it with probability `1 − C(11,2)/C(14,2) = 39.560%`, versus `1 − C(81,12)/C(84,12) = 37.408%` under uniform sampling. These are exact combinatorial calculations, not actual lesion statistics. Without prevalence or outcome evidence, reversing the coverage decision would trade one plausible risk for another.

Uniform sampling would not make sampled softmax pooling unbiased. For three synthetic instances with unnormalized attention weights `[9,1,1]` and scalar values `[1,0,0]`, the full weighted average is `9/11 = .8182`. Uniformly choosing two gives averages `.9, .9, 0`, whose expectation is `.6`. Equal inclusion probabilities do not commute with the ratio.

**The 12-to-84 change also matters with equal lengths.** Conditional on including one witness, a score advantage of `log(11)` over background assigns it mass `.5` among twelve windows and `.1170` among 84 if no further witnesses appear. Conversely, dense inference may discover witnesses absent from the training bag. The previously audited public dense-window result, `.940` with 94 windows versus `.934` with 24 for one checkpoint, is evidence against asserting inevitable harm; it does not validate this new recipe. [Previously verified author ablation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)

**Warranted correction:** keep the declared stratified sampler, explicitly describe the changed measure and conditional attention dilution, and avoid calling dense inference a guaranteed improvement. No hierarchical head, importance-weighting intervention, or extra sampler arm is justified by this analysis alone.

## 2 Single-study BatchNorm is a statistical risk beyond memory fit

Convolutional BatchNorm uses batch and spatial positions to calculate training statistics, then uses stored running statistics in evaluation. Eight-study gradient accumulation does not mix those normalization statistics. Spatial pixels mean this is not literally one scalar observation, but they do not eliminate shared study effects. [PyTorch BatchNorm documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)

A simple random-effects example makes the concern precise. Let one channel have activation `x_swp = μ_s + ε_swp`, with a study-wide shift `μ_s` and within-study residual variance `σ²`. Per-study training normalization approximately removes `μ_s`. Averaging those within-study variances stores approximately `σ²`, while evaluation produces `(μ_s − E[μ_s] + ε_swp)/σ`. It reintroduces a study-dependent component absent from the training representation. Larger spatial maps do not remove this component. This is an illustrative mathematical case, not a measurement of CoAtNet activations.

The original [Batch Renormalization paper, sections 4.2–4.3](https://papers.nips.cc/paper_files/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf) demonstrates degradation from small and dependent normalization batches on ImageNet even when gradients aggregate over many examples. That establishes a real failure mechanism, not its magnitude in this MRI recipe.

**Warranted correction:** preserve the explicit BN policy and its uncertainty. A successful finite-gradient/memory check or eval-mode FP16-versus-FP32 comparison cannot establish train/eval statistical equivalence. The public source also uses study-bag training; this is not evidence that the challenger uniquely introduced the risk. Freezing ImageNet moments, replacing normalization, or adding recalibration would each introduce a new domain-shift or implementation tradeoff. None is recommended solely from theory.

## Other concerns did not justify changing this candidate

No pixel augmentation may increase overfitting, but the inspected evidence does not establish that a particular knee-safe transform would improve this exact fit. Selected-slice triplets can skip physical neighbors and miss focal findings; the proposal already discloses that limitation and preserves an input recipe with stronger completed evidence. Neither objection supports inventing a new geometry, augmentation search, or extra control in the final week. There is no evidence here that this candidate will outperform the leaderboard, nor evidence that these choices make success impossible.

Research scope: local specification review, primary-source reading and exact synthetic arithmetic only. No MRI records, pixels, weights, training, repository changes, submissions or external publication were used. The sole specification edit adds the sampler/pooling caveat; the experiment itself is unchanged.
