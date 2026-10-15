# What weak supervision can and cannot fix in a single MRI model

14 October 2026 · Mathematical review and synthetic counterexamples only

## Decision

**Keep the bounded image-pseudo-label experiment. Do not claim that cross-fitting, confidence weighting, or a generic noise correction recovers the competition target.** The overlooked risk is that confidence weighting can change the *optimal ranking*, even when the original probabilities are perfectly calibrated. Matching total gradient mass does not remove that change.

There is no defensible numerical supervision ceiling here. In particular, the publisher's 0.8927 label AUC is a different measurement on the gold-guided official 58 cases, not an upper bound on an image model. No result below predicts whether this pipeline can exceed .95.

## 1. The quantity the student actually learns

For one target, let X be the MRI information available to the student, Y the true binary competition target, Q the report-derived soft target, and W its nonnegative effective weight. Assume an unrestricted predictor, a population BCE objective, and positive expected weight. Its optimum is

    m_W(x) = E[WQ | X=x] / E[W | X=x].

The true AUC-optimal ordering is that of η(x)=P(Y=1 | X=x). Therefore the issue is whether m_W preserves η's ordering, not whether Q looks like a probability or whether the training loss is low. Finite models need not attain this population optimum.

Useful sufficient conditions:

- Unweighted supervision works if E[Q|X] is a strictly increasing function of η
- Class-conditional soft noise works if E[Q|Y=0,X]=a and E[Q|Y=1,X]=b are constants with b>a: E[Q|X]=a+(b−a)η
- A fixed convex blend of sources with individually order-preserving conditional means remains order-preserving
- Weighting by a positive function of X alone leaves the unrestricted conditional optimum unchanged; weighting by Q or report state generally does not

Instance-dependent noise needs stronger structure than “labels are usually correct.” The primary theory likewise places additional conditions on AUC preservation; it does not establish a universal noise-correction guarantee. [Menon, van Rooyen and Natarajan, 2018](https://www.microsoft.com/en-us/research/publication/learning-from-binary-labels-with-instance-dependent-noise/)

## 2. Overlooked counterexample: perfect probabilities, harmful confidence weights

Consider two equally common MRI groups. Let the report probability Q be genuinely calibrated even conditional on MRI: draw Y conditional on X,Q as Bernoulli(Q).

| MRI group | Distribution of Q | True risk, conditional mean of Q | Optimum with W=2 abs(Q−0.5) |
|---|---|---:|---:|
| A | Always 0.7 | 0.7 | 0.7 |
| B | 1 in 20%; 0.5 in 80% | 0.6 | 1.0 |

Unweighted BCE correctly ranks A above B. Confidence-weighted BCE reverses them. With these equal group sizes, true AUC changes from approximately **0.555 to 0.445**. All calculations are synthetic and were checked with exact rational arithmetic.

The mechanism is explicit:

    m_W(x) = E[Q|X=x] + Cov(W,Q|X=x) / E[W|X=x].

Confidence weighting selects a different report distribution within an MRI group. It does not merely reduce noisy gradients. Multiplying all weights for a target by a constant cannot undo this ratio. A positive weight floor limits the size of the effect but gives no universal ordering guarantee.

**Practical consequence:** the publisher's suggested 2|Q−0.5| remains an empirical candidate, not a source-mandated correction. Its benefit must earn a matched test. Do not infer missingness from final Q=0.5: the reviewed v4 is a probability-space blend, so that number can combine different upstream states. [Publisher data card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)

## 3. What same-source self-distillation can improve

Suppose an ideal teacher learns t(X)=E[Q|X]. For a fixed blend α, the student's new target is (1−α)Q+αt(X). Then

    E[(1−α)Q+αt(X) | X] = E[Q|X].

The population target is unchanged at every α. The same fixed-point statement holds with matched effective weights when the teacher learns m_W. Cross-fitting removes self-fitting exposure; it does not change this identity or erase systematic source bias.

Even giving an ideal teacher the full study Z and a student a sampled partial view X does not automatically change the target expectation. If view sampling is conditionally independent of Q given Z, then

    E[E[Q|Z] | X] = E[Q|X].

This condition fails for a label-dependent sampler unless its additional selection information is handled appropriately. It also does not cover arbitrary nonnested teacher/student inputs.

**This is not an argument against the reported image-pseudo-label successes.** A finite teacher, ensemble, different architecture, optimization path, augmentation policy, or regularization can produce a useful different approximation. Teacher averaging can reduce variance; a model that suppresses a report shortcut can reduce bias. A student can outperform a finite teacher. The theorem only rules out the explanation that repeated *population-optimal same-objective* self-distillation, by itself, identifies an unobserved clinical target. Distillation's finite-sample bias–variance mechanism has direct theoretical support. [Menon et al., 2021](https://proceedings.mlr.press/v139/menon21a.html)

A useful interpretation of a successful experiment is therefore “this teacher construction improves the finite MRI learner,” with a matched softening control optional when attributing the mechanism, not required merely to choose a practical winner. It is premature to call it clinically corrected supervision.

## 4. Why generic noise correction cannot supply the missing truth

Here are two observationally identical worlds. X has equally frequent groups A and B. The observed hard report label Q has probability 0.6 in A and 0.4 in B in both worlds.

| World | True risks η(A), η(B) | Report false-positive / false-negative rates |
|---|---|---|
| I | 0.8, 0.2 | Both groups: 1/3, 1/3 |
| II | 0.4, 0.6 | A: 0.4, 0.1; B: 0.1, 0.4 |

Every class-specific error rate is below 0.5. Reports in World II are 72% correct overall, yet their conditional-mean ordering is reversed relative to clinical risk. A model ranking A above B has true AUC 0.8 in World I and 0.4 in World II.

Because the entire observed distribution of X,Q is identical, no procedure using only these observations can know which ordering is correct. Add an identical constant missingness mask and the impossibility remains. OOF folds, agreement between same-report extractors, and extra self-distillation rounds do not resolve it. Identification needs added restrictions or independent evidence, not just a more elaborate estimator.

Conversely, low label AUC is not a ceiling: if Y is perfectly recoverable from X and binary report labels independently flip with probability 0.2, report-label AUC is 0.8 while E[Q|X]=0.2+0.6Y gives perfect image ranking. This separate synthetic example also explains why the publisher's label score cannot bound model performance, even before accounting for its different evaluation scope and gold-guided construction.

## 5. Missingness and severity can change ordering without extraction mistakes

**Masking:** with an observation indicator M and otherwise correct labels, observed-only BCE estimates

    P(Y=1|X,M=1) = η s₁ / [η s₁+(1−η)s₀],

where s_y=P(M=1|X,Y=y). Missing-at-random conditional on X gives s₁=s₀ and preserves η wherever observation probability is positive. A constant positive ratio s₁/s₀ also preserves ordering. An X-dependent ratio can reverse it; estimating P(M=1|X) alone does not identify that ratio.

**Filling with 0.5:** even under conditional missing-at-random, all-cell training on MY+(1−M)0.5 estimates 0.5+s(X)(η−0.5). Risk 0.9 with observation probability 0.1 becomes 0.54; risk 0.7 with full observation remains 0.7. Missingness alone reverses the rank. This is a counterexample, not a claim that the final v4 uses that exact filling rule.

**Severity:** let severity S be 0, 1, or 2 and the true target be Y=1{S≥1}. Group A is mild-positive 80% of the time; group B is severe-positive 60% of the time; otherwise each is normal. If Q=S/2, mean targets are 0.4 and 0.6 while true risks are 0.8 and 0.6. Perfectly extracted monotone severity is therefore insufficient. One must model the qualifying event or its supported severity distribution, not assume expected severity ranks a threshold probability. This does not justify hardening genuine probability targets.

## 6. Cheap falsifier and next decision

**Already completed:** the synthetic construction falsifies the universal “distance-from-0.5 weighting safely denoises calibrated probabilities” explanation. It does not establish that weighting harms these actual labels.

**Candidate diagnostic, not executed:** if an already authorized fitting-only packet contains current Q, candidate weights, original observation status, and existing frozen MRI feature bins, compute per target/bin:

1. Unweighted and weighted means of Q, their difference, and total weight
2. Kish weight-concentration summary, (ΣW)²/ΣW², with zero-support bins explicit; this is not a count of independent patients
3. Rank reversals between sufficiently supported bins, stratified by source and original observation state

No encoder training or new teacher inference is required. Use bins fixed without clinical evaluation outcomes; report sampling uncertainty. Reversals demonstrate that the intervention changes the estimand already at this coarse MRI representation, rather than simply applying global gradient rescaling. They do **not** identify which rank is clinically correct. No reversals is only a weak negative diagnostic because coarse bins can hide finer effects. If no suitable packet exists, do not create a new feature cache just for this check.

**Recommended action:** retain one constant-total-weight, fold-safe image-pseudo-label comparison. Keep confidence weighting separate. Require an explicitly justified source/selection model or newly authorized independent evidence before promoting any method as clinical label correction. No unsupervised diagnostic can certify that missing assumption from the final blend alone.

## Scope

Reviewed the four supplied research documents, with the newer label-provenance memo taking precedence over superseded ancestry descriptions. Consulted primary theory sources. Opened no patient rows, images, labels, weights, or protected evaluations; ran no training, inference, repository changes, or submissions. The formulas and counterexamples are deductions, not measured RSNA outcomes.
