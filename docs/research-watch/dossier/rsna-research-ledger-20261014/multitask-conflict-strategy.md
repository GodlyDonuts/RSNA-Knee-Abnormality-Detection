# Multitask optimization and supervision weighting strategy

Prepared 14 October 2026. Repository evidence is pinned to `8300cfd20c2acaeef9d684a81dbc7fe977282a62`. This is a read-only research proposal. No fitting records, images, protected validation rows, training runs or submissions were accessed or executed for this report.

## Recommendation

The existing four-arm result does not identify gradient conflict. A substantial interaction is possible from per-target normalization alone. First perform a fitting-only, common-checkpoint gradient audit that separates this predictable denominator effect from persistent cross-target interference. If it identifies harmful contributions associated with low source-distance values, the one recommended next training contrast is **bounded source-distance weighting**, against an otherwise identical uniform-weight control. This costs no additional encoder passes or inference work. FAMO is a credible later optimization method, but exact integration into the current eight-microbatch accumulation is appreciably more expensive than its “one backward” description suggests.

Neither method has an established accuracy benefit on this task. All twelve targets and the official macro endpoint remain unchanged.

## What the current loss actually does

For effective batch B, target t, observation weight m, and logit z, define:

`N_t = sum_(i in B) m_it`

`ell_t = sum_(i in B) m_it * BCEWithLogits(z_it, y_it) / N_t`

`L = mean_(t with N_t > 0) ell_t`

Counts are collected over every accumulation microbatch and all DDP workers before backward. Each microbatch contributes its numerator divided by these global counts. There is no second division by accumulation steps. The world-size multiplier compensates for DDP gradient averaging. An entirely unobserved batch skips the optimizer, including weight decay. This is a coherent effective-batch objective, not an obvious denominator bug. [Loss implementation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/loss.py) · [Training loop](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/train.py#L943-L1059)

The four-arm production recipe used microbatch size one, eight accumulation steps, world size one, AdamW, global gradient clipping at 1.0, FP16, rank-32 adapters and adapter dropout 0.05. Thus the usual effective batch contains eight studies. Sparse original masks can leave a target inactive for an entire update; a target can also have only one observed study and still receive the same target-level coefficient as a densely observed target. Equal target coefficients do not imply equal gradient norms, equal positive-class influence, or equally reliable supervision. [Bound run configuration](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/public-run.json)

The full-public overlay supplies all twelve cells per fitting study with binary mask one. There is no class-positive multiplier or source-confidence multiplier in this loss. At one cell, the logit derivative before the target and batch scaling is `sigmoid(z) - q`. A q=.5 target pulls a confident prediction toward .5; it is not ignored. The overlay loader currently accepts only binary masks, even though the generic loss supports nonnegative observation weights. A proposed weighting experiment therefore needs a separately admitted weight artifact and tests; silently replacing its binary-mask contract would be wrong. [Overlay contract](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/label_source.py)

## Why the four-arm interaction is not proof of gradient conflict

The completed macro endpoints are original .87867059654, value-only .85368128089, coverage-only .86930231330, and full-public .89978406481. Their descriptive interaction is `.05547106716`. The full-public medial-meniscus change was negative, approximately −.01468. These are outcome observations, not measurements of the proposed mechanism. The coverage-only confidence interval crosses zero. [Joint-source comparison](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/decision.json) · [Value-only result](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-fixedmask-production-20261010-v1/fixed-final/agent-decision.json) · [Coverage-only result](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-coverage-production-20261011-v1/fixed-final/agent-decision.json)

Here is an exact local calculation, not a claim about the final trained AUCs. Hold the model parameters, studies, images and stochastic draws fixed. For one target with nO originally observed cells and nU added cells, let g00 be its mean gradient on original labels, g10 its mean gradient after replacing those labels with public values, and gU its mean gradient on added public cells. Then:

`g01 = (nO*g00 + nU*gU) / (nO+nU)`

`g11 = (nO*g10 + nU*gU) / (nO+nU)`

Consequently:

`g11 - g10 - g01 + g00 = -nU/(nO+nU) * (g10-g00)`

The identity applies when the original target is observed and the same parameters/logits are used. The macro-gradient expression needs an additional active-target-count adjustment when originally inactive targets become active. It already shows that coverage attenuates the value-replacement gradient through its denominator, without requiring any conflict between diseases. The endpoint interaction also includes changed optimization paths, Adam states, clipping, representation learning and nonlinear AUC ranking. It cannot be decomposed by this identity alone.

Cross-target interference remains plausible because the twelve outputs share adapters, projection and parts of attention. It is one hypothesis alongside label semantics, uncertainty-induced shrinkage, changed sampling noise, and genuinely improved supervision. Target-specific attention does not eliminate sharing. [Model structure](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/model.py)

## The smallest useful diagnostic

This is a proposed audit to run only after its own execution approval, using existing fitting boundaries and aggregate-only exports.

A concrete starting budget is two predeclared saved states and sixteen fixed effective batches per state, followed by at most two of those batches per state for selected-adapter probes if warranted. Cap the diagnostic at one tenth of a reference full fit's measured compute. Freeze this budget before seeing results; insufficient rare-target support means unresolved, not absence of conflict. The same cases may be replayed across recipes because the purpose is a controlled gradient comparison, not a new validation estimate.

1. **Measure supervision first.** On the exact fitting roster, report per-target original/added cell counts, active-effective-batch frequency, public-value histogram, exact and near-midpoint counts, target entropy, and original-mask strata. Do not infer clinical truth from q>.5. This phase needs no new image processing.
2. **Use one common state.** Predeclare a small set of already preserved checkpoints and fitting batches. Apply all four value/mask recipes to identical logits and input draws at each state. Comparing gradients from four differently trained final models would mix the supervision change with its learned consequences.
3. **Start at the shared head.** With available fitting-only embeddings, or one approved frozen forward, measure each target's gradients on the shared projection and the shared slice representation before target-specific pooling. Target-specific output rows have disjoint parameter support, so near-orthogonality there would be a misleading reassurance. Cached frozen features can test head interference; they cannot establish what happens in trainable encoder adapters.
4. **Measure harm, not just angle.** Report per-target norms; the 12-by-12 cosine matrix; each target gradient's dot product with the summed gradient; batch-to-batch variability; and original-versus-added and midpoint-versus-other contributions. Negative pairwise cosines alone do not show that the actual update hurts a task. Check the exact denominator identity above before attributing residual differences to conflict.
5. **Account for the optimizer.** At a saved optimizer state, a virtual AdamW update on a disposable copy can estimate `g_t dot delta_theta`, including clipping and separate head/encoder learning rates. A positive value predicts a local increase in that target loss. Restore all parameters, moments and RNG after each virtual calculation. Raw-gradient geometry is not the same as Adam's actual direction.
6. **Escalate sparingly.** Only if shared-head evidence is consistent should a few predeclared fitting batches receive per-target vector-Jacobian products on selected shared adapter blocks. Twelve full model-gradient traversals on every update would turn a diagnostic into an expensive method. No clinical records or individual predictions need leave the execution environment.

Useful evidence would be repeated under-optimization of the same targets, especially medial meniscus, across batches and more than one saved state; an adverse actual-update projection; and an identifiable original/added-cell contribution. A one-batch negative cosine is insufficient. A result that changes sign across batches or disappears after denominator accounting weakens the case for a conflict optimizer. No direct FAMO, PCGrad or per-target gradient-conflict experiment was found in the canonical loss/training code and searched experiment records; the existing supervision arms perturb this mechanism but do not isolate it.

## What existing aggregates say about source confidence

The pinned source comparison covers 2,603 accepted training-split studies outside fold 0, before the later observed-row eligibility filter. The public blend is finite on all 31,236 cells, ranges from .005 to 1.0, and has only **four exact .5 values among 13,760 originally observed cells**: ACL three, MCL one, every other target zero. Medial meniscus has zero among 1,914 observed cells. This does not provide midpoint counts for the 13,204 newly activated cells in the production overlay, nor continuous confidence histograms. The differing audit and production rosters must remain explicit. [Aggregate comparison](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/comparison.json)

Therefore suppressing only exact .5 cannot be the principal explanation for the value-only arm's behavior on originally observed cells. Continuous soft values and newly activated cells are separate questions.

The source is byte-identical to Steven v6's report-LLM blend. Its original-extraction .5 convention and recommended distance-based weighting do not make the blended number a calibrated probability of label correctness. A midpoint can arise from unmentioned evidence, uncertainty, or disagreement between parent sources; an extreme value can still be confidently wrong. Declared gold-guided source design also prevents using the 58 gold studies as untouched confirmation. It does not establish that an MRI teacher trained on the outer images. [Frozen lineage and semantics](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/sources.json) · [Publisher dataset card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)

## One recommended remedy if the diagnostic supports it

Test **bounded source-distance weighting**, not a new backbone or a hard missingness reconstruction. Keep the current public target q, all fitting studies, all twelve masks, the sampler and the inference graph fixed. Predeclare:

`c_it = 2*abs(q_it-.5)`

`w_it = .5 + .5*c_it`

`L_weighted = (1/12) * sum_t [sum_i w_it*BCE(z_it,q_it) / sum_i w_it]`

The .5 floor is a proposed conservative safeguard, not the publisher's recipe or an established optimum. It makes every cell positive-weight, retains all target activations, and limits relative emphasis rather than deleting ambiguous examples. Do not tune the floor per target. For a full eight-study batch, an individual normalized coefficient is between approximately .533 and 1.778 times its uniform counterpart. Each target's coefficients still sum to 1/12. This preserves target-level coefficient mass, **not** total gradient magnitude or direction.

The smallest predictive test is two matched fits: uniform weights versus this one fixed weighting rule, using identical initialized tensors, RNG, study order, input bags, total gradient-bearing examples, accumulation, optimizer, learning-rate schedule, precision and fixed final update. No checkpoint choice on the outer curve. Existing control reuse needs exact compatibility proof. At the older 18.36-hour fit rate, two full fits would be approximately 36.7 GPU-hours before changed infrastructure costs; this is a planning anchor, not a forecast. There are no additional encoder passes, no new inference parameters and zero added inference cost.

Before those fits, run a **fitting-only shuffled-weight null at the same frozen states**. Permute weights within target, original-mask stratum and descriptive q-side strata; preserve weight histograms and total side-specific weight mass. This tests whether observed alignment is more than generic heterogeneous weighting. Preserving every exact q value would make the permutation ineffective because c is a deterministic function of q. The null still will not exactly match each target's gradient norm. For a norm-matched diagnostic, explicitly rescale its per-target shared gradient on the disposable probe and disclose that this is an analytical control, not the production loss. Log norm changes and clipping; do not claim that equal scalar coefficients match gradient budgets.

A uniform-versus-weighted accuracy gain would establish a useful reweighting package, not prove calibrated confidence or conflict repair. A causal claim about cell-specific reliability would require an additional fully trained shuffled-weight control, beyond this minimal proposal.

## Why FAMO is deferred rather than dismissed

FAMO adaptively combines task gradients using scalar loss histories, avoiding storage of one model-gradient vector per target. Its model coefficient is proportional to `softmax(task_logit)/(task_loss - lower_bound + epsilon)`, normalized across tasks; it updates task logits from before/after log-loss changes. The paper aims to balance relative progress, not maximize macro AUC. [FAMO primary paper](https://papers.neurips.cc/paper_files/paper/2023/file/b2fe1ee8d936ac08dd26f2ff58986c8f-Paper-Conference.pdf)

The author example performs one model backward **and a second same-batch no-gradient forward after the optimizer step**. With our accumulation, exact full-effective-batch coefficients must be known before the first microbatch backward. Retaining all graphs costs activation memory; a replay implementation instead needs a prepass, gradient-bearing pass, and post-update pass, approximately `3F+B`, versus today's `F+B`. This is a derived integration cost, not a measured RSNA runtime. With B approximately 2F, the ratio is about 1.67; checkpointing changes that estimate. Dropout/RNG and mutable buffers must be replayed consistently. [Author example](https://github.com/Cranial-XIX/FAMO/blob/main/famo.py#L25-L41) · [Same-batch post-step pass](https://github.com/Cranial-XIX/FAMO/blob/main/famo.py#L77-L85)

Soft BCE contains an irreducible target-entropy term: `BCE(q,p) = H(q) + KL(Ber(q)||Ber(p))`. Consequently raw relative loss progress can reflect different soft-label entropy floors. Subtracting entropy leaves ordinary BCE gradients unchanged but changes FAMO's weights; it is another design choice. The author implementation also assumes valid losses for every task, rather than supplying the needed missing-task state policy. Feeding zero for an absent task is invalid. [Author implementation](https://github.com/Cranial-XIX/FAMO/blob/main/methods/weight_methods.py#L97-L132)

PCGrad explicitly computes task gradients and projects conflicting directions. Its author implementation retains a task-gradient matrix and performs pairwise projections: roughly O(KP) gradient storage and O(K²P) projection work for K tasks and P shared parameters. Twelve tasks do not imply a measured twelvefold slowdown, but the additional task-gradient work is real. [PCGrad author code](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py#L25-L61) · [Primary paper](https://arxiv.org/abs/2001.06782)

These methods warrant a later trial only if the diagnostic finds persistent shared-gradient harm after accounting for normalization and source-value effects. They are not the recommended simultaneous second intervention.

## Stop and promotion criteria

- **Do not dispatch** if the exact fitting-roster weight/entropy audit is unavailable, provenance is mischaracterized, or the required source/mask/weight admission is missing.
- **Stop the mechanism claim** if conflict appears only in task-specific rows, a single batch, or the predictable denominator interaction; or if it does not correspond to adverse shared-update projections.
- **Do not weight by source distance** merely because the author recommends it. Require an identifiable contribution pattern on the fitting-only probe. If the signal comes from medically ambiguous but useful mild findings, source-distance weighting may remove the information the model needs.
- **Stop implementation admission** if masks or activation counts change, per-target normalization differs across accumulation/DDP layouts, controls do not reproduce, or weight-floor/finite-value checks fail. No protected validation is needed for these tests.
- **Keep the fixed training endpoint.** Do not stop or tune based on favorable outer targets. Promote only using the unchanged full twelve-target macro result, its paired component-level uncertainty, and all per-target changes. Report undefined bootstrap draws and repeated-search limitations.
- **Protect rare outcomes without gaming the metric.** Macro AUC gives every disease equal endpoint weight, while BCE can be dominated by its negatives. Do not remove MCL, Baker's or fracture because support is small, nor declare victory from those few components alone. About 73% of the earlier full-public macro gain came from these three sparse targets; that makes uncertainty important, not those targets dispensable. [Support diagnostic](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/macro-support-diagnostic.json)
- **Kill the proposed recipe** if the fixed comparison shows no supported full-macro improvement or material supported-target deterioration. A null result rejects this bounded rule, not every uncertainty model, image teacher or optimization method.

Direct AUC optimization is a separate research direction, not ruled out simply by microbatch size one. Moment-based objectives can avoid explicit positive-negative pairs in one forward, but current library assumptions about binary labels, class priors and empty-class batches need a distinct design for masked soft labels. It is not a drop-in substitute for this controlled experiment. [LibAUC implementation](https://docs.libauc.org/_modules/libauc/losses/auc.html)
