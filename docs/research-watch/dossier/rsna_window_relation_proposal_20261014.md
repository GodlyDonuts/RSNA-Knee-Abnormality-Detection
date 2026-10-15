# Preserve useful window differences without calling them uncertainty

**Research proposal · 14 October 2026 · MRI-only, one deployed student**

## Recommendation

Test a **loss-only window-relation comparator** to the existing disease-query bag model. Teach the student which of its already encoded window bundles a permitted teacher rates more strongly, while leaving its study loss, attention head, labels, inputs and inference unchanged. If at least two eligible teachers exist, reduce supervision only when those teachers disagree about the **same window pair**. Never reduce supervision merely because different windows receive different predictions.

This separates two observable axes cheaply. It does **not** identify true pathology or calibrated epistemic uncertainty. The necessary teacher evidence is not currently established: the single-member packet cannot estimate between-model consistency, and an unknown training roster cannot support a clean protected-validation claim. A permitted one-teacher ordering ablation is the practical fallback, with its narrower interpretation stated explicitly.

## What is already implemented, and what remains distinct

The newly integrated PR38 standalone module already separates across-window variation from same-window transformation variability. It implements capped window logits, bounded disease-query pooling, and per-window soft-label BCE weighted by decisiveness, `|2p−1|`. Its root review reports 127 focused tests, but no trainer wiring, GPU run or accuracy result. Reproposing that mechanism would duplicate current work. Its pooling also differs from the existing probability-mixture bag head, so its four-arm comparison is necessary. Caps and a uniform floor bound influence; they do not establish robustness to a confidently wrong selected window. Pre-pooling saturation changes gradients and potentially pooled study ordering, not merely final calibration. The pooling-only C arm therefore remains essential. [PR38 root review](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/e4f288d6d5cf605aa79935cd456359bd7eb498e1/initiative/audits/focal-window-PR38-root-review-20261014-v1/README.md), [module](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/e4f288d6d5cf605aa79935cd456359bd7eb498e1/rsna/competitive/focal_window.py), [precommitted design](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/e4f288d6d5cf605aa79935cd456359bd7eb498e1/initiative/experiments/competitive-v1/focal-window-evidence-v1/design.json)

This proposal instead retains `WindowBagSlotStudyNet`: four training bundles at starts 0/3/6/9, ten evaluation bundles, unchanged target-conditioned window logits, and the existing probability mixture with uniform floor 0.25. It adds **relative window supervision**, without absolute per-window probability matching, new caps or attention-map copying. Its gradients reach the shared representation and window classifier; the window-selection queries continue learning through the unchanged study objective. [Existing bag implementation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/rsna/competitive/window_bag.py)

Earlier fixed-mask replacement, coverage-only expansion, patch means, contrastive objectives and checkpoint averaging do not test this loss. Historical cross-study ranking also had a different exposure problem: within-study comparisons supply six pairs even when the forward batch contains one study. This is not evidence that the new loss will succeed.

## The precise failure to avoid

The earlier proposed single-teacher confidence was:

`max(0, |2p_study−1| − range_w(p_window))`

At the same study probability 0.892, nine windows at 0.99 and one at 0.01 receive weight zero; ten windows at 0.892 receive weight 0.784. This proves suppression of observed window variation at a fixed study score. It does not prove that real focal lesions were suppressed, and canonical training has not consumed that proposed objective. The separate committee design already rejects using window spread as teacher uncertainty. [Single-teacher implementation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/rsna/competitive/teacher_evidence.py), [committee design](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/tools/research/public_teacher_evidence/committee_design.json)

For compatible predictions `p[m,w,t]`, measured on identical rendered inputs for every member, report:

`H[t] = Var_w(mean_m p[m,w,t])`

`E[t] = mean_w Var_m(p[m,w,t])`

With uniform finite-population weighting, `Var_(m,w)(p) = H + E`. This is a descriptive variance identity. `H` includes anatomy, artifacts, acquisition and model bias; it is not a pathology map. `E` measures disagreement among the chosen members, not all uncertainty. Correlated teachers can agree while wrong, and all teachers predicting 0.5 have zero disagreement despite high predictive ambiguity.

The same-input entropy difference `h(mean_m p) − mean_m h(p)` has a Bayesian information interpretation under appropriate posterior assumptions. Here it is only a committee-disagreement statistic. [Gal et al., ICML 2017, section 4](https://proceedings.mlr.press/v70/gal17a/gal17a.pdf)

## One concrete method

For target `t` and each of the six unordered pairs `(w,v)` among the four training bundles:

1. Compute each eligible teacher's soft preference `r_m = sigmoid((z_mwt−z_mvt)/T)`, with **T=1 fixed in advance**.
2. Set the target `q = mean_m r_m`. Ignore near ties, `|q−0.5| < 0.1`, using that fixed engineering threshold rather than tuning against evaluation labels.
3. With at least two eligible members, compute `d = h(q) − mean_m h(r_m)` and weight `c = clip(1−d/log(2), 0, 1)`. With one member, use `c=1` and make no uncertainty claim.
4. Predict the student preference `s = sigmoid((u_wt−u_vt)/T)` and minimize `c × BCE(q,s)`. Sum over pairs and divide by the **fixed six-pair denominator**, then average over the fixed 12 targets; skipped pairs contribute zero. Do not divide by each study-target's sum of confidence weights: that would cancel a uniformly low confidence and restore full auxiliary strength to an uncertain bag.
5. Add this term to the unchanged study loss. Predeclare its scale by matching its median shared-projection gradient norm to **10% of the study-loss norm** on a fixed fitting-only calibration batch set, then freeze the coefficient. Stop if the primary norm is negligible or the ratio is numerically unstable. This is an engineering starting budget, not an evidence-backed optimum.

A common teacher logit offset cancels in the difference, so exact absolute calibration is less binding. Logit scale and nonlinear calibration still matter: this is **soft relation distillation**, not full invariance to every monotonic transformation. Keep all twelve sigmoid targets separate; do not impose a softmax across mutually nonexclusive abnormalities.

The only inference network remains the student. No positive study label is copied onto every bundle. A bundle contains three selected slices from **each acquired slot**; the same index is not a co-registered anatomical point across MRI planes. Teacher scores therefore represent model evidence for a multisequence bundle, not local clinical probabilities. If the teacher was trained on whole bags, evaluating isolated bundles also changes its input distribution. Matching source-window preprocessing does not solve that conceptual problem.

Relation transfer is motivated by DIST's evidence that matching prediction relations can be more forgiving than matching exact teacher probabilities. Its published author code uses Pearson relations after class softmax, so the formula above is a proposed adaptation, not the published method or a validated knee result. [DIST paper](https://arxiv.org/abs/2205.10536), [author implementation inspected](https://github.com/hunto/image_classification_sota/blob/main/lib/models/losses/dist_kd.py)

## A falsifiable, finite comparison

Freeze the recipe before the first fit: same admitted fitting roster, labels, masks, initialization, augmentations, frame budget, optimizer updates, ten-epoch fixed endpoint and existing evaluation protocol. Bind teacher member, study, slot mask, window identity and canonical rendered-input hash. Every teacher member must see the same canonical input. The common student augmentation remains an explicit assumption: it should preserve the teacher's relative evidence; a crop that removes that evidence can violate it. This is not eliminated by hashing the pre-augmentation cache. Do not add a study-level distillation term only to the treatment.

| Arm | Only difference from baseline |
|---|---|
| A | None: existing disease-query bag and study objective |
| R | The window-relation term above |
| N | The same term with an anatomy-stratified **donor relation pattern** |
| U, only with multiple members | Identical targets and tie gate, but all eligible pair weights equal one |

**Strong negative control N:** assign each fitting study a different-leakage-component donor within the same acquired-slot mask and the same fixed window layout. Match prespecified bins of teacher study score and window-variation magnitude, using fitting-only quantities. Transfer the donor's complete target-specific pair-target/weight pattern while retaining the recipient's study labels and any common study-level teacher target. Use a deterministic, prespecified matching and fallback rule; report unmatched coverage. This preserves window positions and group-level evidence distributions better than arbitrary within-study shuffling while destroying study-specific alignment. It cannot perfectly match every recipient's anatomy or pair distribution, so quantify residual imbalance.

Calibrate N and U to the same initial shared-parameter auxiliary gradient budget and valid-cell exposure as R. Equal scalar coefficients or equal losses are insufficient. Log subsequent gradient norms, dot products with the actual primary update, clipping and optimizer effects. Gradient matching at initialization does not guarantee identical optimization later.

**Sequence:** fit A and R first, reusing A only if exactly matched. If R has no established gain, stop this branch. If promising, fit N; with multiple eligible members, fit U only if R also beats N. Do not tune parameters to rescue each failed comparison.

- **R versus A:** primary paired 12-target macro-AUC on the accepted 1,156-study weak-development frame, using the existing whole-leakage-component bootstrap. Report all targets and undefined draws; no Gold58 or reserved-confirmation selection.
- **R versus N:** if indistinguishable, the claimed benefit from study-specific window evidence is unsupported, even if both regularize usefully. An interval crossing zero is inconclusive, not proof of equivalence.
- **R versus U:** if indistinguishable, do not credit uncertainty weighting. Prefer the simpler unweighted relation loss if practical performance is comparable.
- **Stop immediately:** ancestry/input mismatch, protected-row teacher exposure, nonfinite loss, near-zero eligible-pair coverage, or an auxiliary objective that overwhelms the agreed gradient budget. If teachers give nearly identical relations everywhere, uncertainty weighting has little to add.

This cannot establish clinical localization without independent local reference labels. MIL instance predictions can change substantially despite stable bag performance. [Cheplygina et al., MICCAI 2015](https://arxiv.org/abs/1703.04986)

## Why deletion/complement is a diagnostic, not identified causal evidence

For a probability-mean teacher, `p_bag = mean_w p_w`, deleting window `w` gives:

`p_bag − p_without_w = (p_w−p_bag)/(W−1)`

Thus deletion is exactly a rescaled centered window score. It adds no new information or causal identification. Complement comparisons are similarly determined by the same window predictions.

For the existing query bag, cached window features allow inexpensive leave-one-out repooling. Changes also reflect attention renormalization and the uniform floor. Compare deletion of a selected bundle with same-size, position/slot-matched alternatives; all four-window deletions produce three-window bags. The four nonoverlapping training bundles avoid the obvious duplication of ten overlapping evaluation windows. Nevertheless, three-window inputs differ from the four-window training distribution, and removing evidence is not an intervention that makes the underlying disease disappear. Do not train every complement to retain the positive study label.

CIMIL supplies a relevant research precedent for counterfactual sub-bag assessment, but its verified author implementation is a multistage whole-slide pipeline involving clustering, instance search, a projector and refinement. It does not validate simple MRI leave-one-out scores as local truth. [AAAI 2024 paper](https://ojs.aaai.org/index.php/AAAI/article/view/28135), [author repository](https://github.com/centurion-crawler/CIMIL)

## Cost, prerequisites and decision

- **Student cost:** six pairs × twelve targets = 72 scalar comparisons per study, with no additional encoder forward relative to the existing four-window bag. This is not free relative to the historical one-window training recipe. A frozen teacher is unnecessary during student inference.
- **Storage:** pair target plus weight in float32 for 2,247 studies is about **1.23 MiB**, excluding identifiers. A proposed 20-member × 10-window × 12-target logit packet is about **20.57 MiB**; its actual complete collection and admission are not established.
- **Clean teacher cost:** two inner folds give each fitting row only one out-of-fold teacher. To obtain two eligible members per row needs, for example, **two seeds per inner fold: four half-data fits**, roughly two full-fitting-data training equivalents at equal epochs, plus extraction. Do not assume the proposed committee's shared-prefix optimization is proven or free.
- **Scope of cleanliness:** each teacher must exclude the student's fitting row/component and all protected outer/reserved images from upstream image training. Current accepted public labels have documented report-derived, gold-informed ancestry; fresh image cross-fitting does not restore untouched-gold independence. Unknown-roster public checkpoints remain unsuitable for a clean held-out claim even if their new inference is fitting-only.
- **One-teacher fallback:** if a permitted out-of-fold teacher is available but a second is unaffordable, run A/R and, conditionally, N with `c=1`. This answers whether teacher ordering adds useful study-specific information. It cannot separate epistemic uncertainty. The current unknown-roster public single-member packet does not qualify merely because it has already been reviewed. If no permitted teacher exists, retain the current supervised bag experiment rather than bypass provenance to obtain a cheap result.
- **Avoid an unnecessary distillation bottleneck:** a compact, fold-safe teacher family should also be assessed as a deployable single-model challenger under the already authorized comparison scope. If it is stronger and meets runtime and eligibility constraints, deploying that family may beat spending the remaining budget transferring it back into the incumbent. The objective is the best MRI-only single checkpoint, not preserving DINOv3-L. Published participant scores support testing recipes; they do not isolate an architecture's causal advantage.

**Bottom line:** the defensible possible edge is preserving reproducible, study-specific relative evidence while allowing absolute study severity to remain anchored to the accepted objective. PR38 already addresses focal pooling; this proposal should compete as a narrow loss-only comparator, not duplicate or bundle that redesign. Its strongest early falsifier is failure to beat an anatomically matched donor-pattern control. No primary paper currently establishes this exact method on RSNA knee MRI.

### Evidence and implementation status

The cited primary papers and author repositories were inspected read-only. The Bayesian active-learning author repository is available [here](https://github.com/Riashat/Active-Learning-Bayesian-Convolutional-Neural-Networks); it supports the same-input uncertainty framework, not this MRI weighting rule. Attention MIL's [paper](https://proceedings.mlr.press/v80/ilse18a.html) and [author implementation](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) support learned bag aggregation; existing repo attention is already the relevant implementation baseline. No external package was run, no clinical rows were accessed, and no training or repository changes were performed for this proposal.
