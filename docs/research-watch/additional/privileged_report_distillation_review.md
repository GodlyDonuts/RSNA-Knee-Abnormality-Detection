# Training only report supervision for knee MRI

Research decision memo | 14 October 2026

## Recommendation

Defer a new report plus image teacher and a full TRACE implementation for the eight day deadline. There is no verified ready concept table, and the current twelve report derived soft targets likely already capture much of the information a diagnostic report teacher would return. The supplied 0.872 versus claimed 0.951–0.953 public scores do not establish that missing report concepts explain the gap; the competitor claims were not independently verified in this review.

The only defensible new branch is a tightly capped test of extra, image visible morphology or location information, conditional on the existing targets. It is an optional research probe, below the existing fold safe image pseudolabel strategy in priority. Without ready labels, a fold safe checkpoint and honest validation, do not start it. A 24–36 hour experiment would exceed the supplied 18.36 hour reference fit and is not a cheap diagnostic.

## What the primary evidence supports

Bergström, Yue and Johansson directly studied training only report distillation. Their multimodal teacher uses frozen image and text features, separate attention pooling and a joint classifier. On diagnostic labels extracted from reports, its stronger accuracy did not translate into a student advantage over image only self distillation; the finding also appears in their pulmonary embolism experiment. Gains depended on task and backbone. This is unusually relevant negative evidence for adding a report teacher to an existing report label pipeline. It is not a knee MRI experiment or an evaluation of DINOv3 LoRA. The inspected paper says some code and splits will be released; this review did not verify a usable author implementation of its full experiment. [Paper, Sections 2.2 and 5.2 and Appendix B.2](https://arxiv.org/html/2510.24385)

TRACE is a recent, relevant proof of the image only deployment pattern. It reports breast ultrasound results using six expert annotated attributes, including 0.916 versus 0.897 AUC for a ResNet50 baseline on BUSC-BUSBRA. These are not weak multilingual knee report labels. Its concept missing perturbations are dataset dependent: the paper does not demonstrate a universal benefit. Its single malignancy risk ordering should not be transplanted to twelve independent knee findings. [TRACE paper](https://arxiv.org/html/2608.20809v1)

The author code provides an important qualification: the privileged teacher simply returns the supplied concept targets. The image editor predicts a residual from coarse concepts; it does not consume the report at inference. Concept refinement uses clamping, and residual distillation detaches its target. Thus TRACE is not evidence that an independently learned report plus image teacher necessarily adds information. [Author model](https://github.com/wentao-2/TRACE/blob/e79e4a7f2fb55f935d40acc38c487d3417c21c87/trace/trace_model.py) and [author loss](https://github.com/wentao-2/TRACE/blob/e79e4a7f2fb55f935d40acc38c487d3417c21c87/trace/trace_loss.py)

The broader privileged information literature supports a different potential use: annotation process information can help isolate label noise. It also warns that privileged features which almost reveal the target can hurt. This supports distinguishing report confidence and missingness from extra anatomical supervision, rather than assuming every report feature is useful. TRAM is an alternative mechanism, not knee evidence or an additional recommended branch here. [ICML 2023 analysis](https://proceedings.mlr.press/v202/ortiz-jimenez23a.html), [TRAM](https://proceedings.mlr.press/v162/collier22a.html)

## When the signal is new

Let Q be the existing vector of twelve soft targets, W their weights and masks, C extra report concepts, and X the MRI. Rephrasing Q as concepts, duplicating it in an auxiliary head or producing a teacher that simply recovers Q does not introduce new case specific supervision. A new head can still change optimization, capacity or regularization; that is why a matched Q only control is necessary.

Training case report facts are legitimate privileged supervision, even when they imply a training label. That is not automatically evaluation leakage. Leakage arises when held out information influences training or selection, directly or through a teacher or label pipeline.

For one output, a weighted sum of binary cross entropies is exactly a scaled BCE against the weighted average target:

    a BCE(y,p) + b BCE(q,p) = (a+b) BCE((a y+b q)/(a+b), p)

This identity assumes the same prediction p and ordinary BCE at the same temperature. Multiple heads or temperature changes can alter optimization, but are not by themselves evidence of additional report information. Preserve independent sigmoid outputs; do not impose a twelve class softmax.

A related algebraic check: without clamping and with translation invariant error loss, matching residual d to c−c0 has the same numerical error as matching c0+d to c. Detaching the residual target changes gradient routes, not the annotation content. A claimed editing gain therefore needs a direct concept prediction baseline with matched capacity, losses and gradient scale. This is a deduction, not an assertion that the implemented TRACE gradients are identical.

Plausible additional concepts include explicitly described meniscal tear morphology or subregion, displaced fragments or extrusion, and compartment specific cartilage depth or surface involvement. Select only attributes not already encoded in Q. These are examples to audit, not a claim that a suitable table exists. Avoid adding another presence score, another severity score already used to create Q, a global abnormality score, language style, clinical history or inferred diagnoses as if they were new visual evidence.

Extra concepts need not improve a label predictor given Q to be useful: they might make an MRI representation easier to learn in finite data. Conversely, nonzero residual variation in C is insufficient; it may be report style, extraction noise or nonvisual information. The relevant practical test is whether aligned concepts improve MRI only predictions beyond label repetition and equally strong unaligned supervision.

## Missingness and severity contract

Every extracted field needs its evidence span and separate status: explicit present or category, explicit absent, uncertain, unmentioned, not applicable, or contradictory. Unmentioned is not absent and is not a probability of 0.5. The latter can be a masked storage placeholder only. Uncertainty is not an ordinal midpoint. A normal meniscus does not automatically provide a negative label for every tear morphology unless the schema explicitly defines a not applicable state.

For the pilot, supervise only explicit, noncontradictory attributes. Keep unknown and inapplicable cells masked. Apply each concept loss only over its observed cells, normalize by observed count, then average concepts so verbose reports do not dominate. Keep the twelve main target losses, source weights and masks exactly unchanged across arms. Do not convert severity into disease probability without independently validated mapping. Prefer categorical morphology and location over a shared scalar risk axis. Random concept masking does not repair source dependent omission; it only removes additional observations.

## One conditional experiment with a hard budget

This is a proposed protocol, not an experiment run in this review. Its hypothesis is narrow: extra report morphology or location contains MRI learnable, case aligned supervision beyond Q and report provenance, useful within a short continuation of the current model.

### Entry gate with at most two hours of preparation

First verify an admissible concept table already exists. If none exists, allow only a local extraction audit within two hours total preparation time. Use an already available approved local parser or model; do not assume a new remote API, model download or manual annotation project. Time twenty reports, estimate total extraction and review cost, and stop if the full preparation cannot fit. No original reports need leave the current authorized environment.

Freeze at most six concept fields, at most twenty four encoded dimensions, from an outer training subset only. Inspect a fixed sample of 100 extracted field instances stratified by source and observed status; require at least 95 correctly supported instances, with no negation reversal or hallucinated evidence. Drop unsupported fields. As a practical sparsity gate, require at least fifty observed examples for every retained category and support in at least two available source groups. These are engineering gates, not confidence guarantees. If the audit cannot meet them inside the cap, defer.

Other entry requirements are an existing fold safe converged checkpoint, a reproducible baseline continuation and an untouched evaluation set with usable positives and negatives. If producing these requires another full fit, the small experiment is no longer small. If the only reference labels are fifty eight studies already used to design extraction or thresholds, they cannot also be treated as untouched validation. Report derived validation can diagnose training, but cannot establish agreement with image reviewed pathology.

### Controls and objective

Within outer training set F, cross fit a small nuisance predictor e(C | Q,W,S,M), where S is report provenance available for analysis and M is the observed concept mask. Use three internal group folds. Fit all preprocessing and standardization within F; each training case receives a nuisance prediction from a model that did not fit that case. No evaluation reports or labels enter fitting. Define observed residual R = C−e, standardize each retained dimension using F only and keep M fixed.

This is association control, not guaranteed removal of label information. Estimated residuals can still encode Q; even perfect conditional mean residualization does not establish independence. Continuous twelve dimensional Q makes exact stratification sparse. Report nuisance model held out errors and residual dependence on Q and source; do not describe the result as causal.

Use four arms, each with the same added auxiliary head shape and the same underlying target attention architecture:

- A: unchanged main losses, auxiliary loss weight zero
- B: auxiliary prediction of standardized nuisance expectation e, which uses Q and provenance but no case specific concept residual
- C: auxiliary prediction of standardized aligned residual R
- D: auxiliary prediction of R permuted between patients within observed concept status, source and coarse parent target probability bins

For D, leave masks attached to their original cases and permute only observed values, separately per field. Require at least twenty donor cases per stratum; drop unsupported strata from auxiliary supervision in every arm. The shuffle preserves approximate, not exact, relationships to continuous Q. Check resulting Q imbalance explicitly. B and D are complementary controls, not an independence proof.

Attach the auxiliary head to the current relevant target pooled MRI representation. Use masked SmoothL1 averaged per concept. The auxiliary prediction must influence a shared trainable attention or LoRA representation; fitting a separate head on completely fixed final features cannot improve the unchanged main predictions. Do not add a second image encoder or report encoder. Use identical checkpoint initialization, batches, augmentations, main loss, optimizer, learning rate, clipping and number of updates in all arms. Calibrate a fixed auxiliary coefficient on the same first 32 training batches so B, C and D each contribute roughly 10% of the main gradient norm at the shared representation. Freeze it thereafter; log gradient norms. This reduces a simple gradient scale explanation without pretending to match every gradient direction.

### Compute and decisions

Cap all four training arms combined at 0.20 of the reference full fit. At 18.36 GPU hours per fit, this is 3.672 GPU hours total, roughly 55 minutes per arm if throughput is unchanged. Include pilot timing and gradient calibration inside this cap. Preparation is at most two hours and evaluation should remain lightweight. Do not acquire a new feature cache or train a multimodal teacher to make the pilot possible. These are budget ceilings, not measured runtime predictions.

Use a fixed endpoint, no per arm early stopping or public leaderboard tuning. Score main MRI only outputs on the same untouched set: paired patient level bootstrap of twelve target mean AUC, all per target deltas and source slices where sample size permits. Predeclare a practical go threshold of +0.005 mean AUC versus A, positive differences versus both B and D, and a paired 95% interval excluding zero for the primary C−A comparison. Treat the numerical threshold as a decision rule, not a predicted gain. Rare targets with inadequate support or unstable bootstrap estimates make the result inconclusive, not a pass.

If C improves concept prediction but not main AUC, stop. If B performs equally well, there is no evidence the new case specific concepts were needed. If D performs equally well, there is no evidence aligned concept content helped. Any unsupported extraction, ancestry violation, inference report dependency or time overrun is a no go. A promising pilot permits only a prebudgeted independent fold or seed confirmation; it is not sufficient to replace the final model. A null result in a short continuation means defer for this deadline, not that concepts can never help.

## Split and teacher ancestry checklist

Group all repeat exams, patients where identifiers permit, and near duplicates before splitting. The outer evaluation set must be excluded from every ancestor: image checkpoint supervision, teacher training, pseudolabel generation that informed training, extractor examples, prompt tuning, concept vocabulary selection, nuisance fitting, confidence calibration and checkpoint selection. Evaluation reports may be parsed with a frozen rule solely for a separately reported diagnostic analysis; their concepts must not enter student optimization.

A teacher can leak evaluation information indirectly through pseudo labels on different training cases. An apparently out of fold student is not clean when its teacher or warm start previously learned from that evaluation fold. Log model version, training groups, extraction prompt or parser version, label sources and calibration groups for every generated table. A preexisting all data fine tuned checkpoint is not a valid outer fold initialization.

For a future report plus image teacher test, the minimum comparisons are matched image teacher KD, Q only teacher KD and image plus report teacher KD, with identical student schedules and teacher output loss. Generate teacher targets through internal cross fitting within each outer training fold; no teacher ancestor may use outer evaluation data. Cross fitting reduces self memorization, but does not make report shortcuts visually transferable. A superior multimodal teacher AUC alone is not the success criterion. The MRI only student must win.

## Deployment and final decision

Export exactly the current MRI encoder, attention and twelve output heads. Remove auxiliary training heads and all report parsing from inference. Verify predictions are unchanged when report files are absent or their contents are randomized, and verify inference accepts only MRI plus already authorized image metadata. No concept table, report lookup, teacher service or per case text embedding may be required.

For this deadline, the entry prerequisites are not currently verified, especially the concept table. Defer privileged concept training and report plus image KD unless those prerequisites appear almost for free. Preserve the idea as a falsifiable, bounded option rather than opening another full training branch. The stronger immediate lesson is to audit whether current weak labels already use the report information that the proposed teacher would supply.

## Scope of this review

Public primary papers and author repository source were inspected. No private repository or clinical data was accessed, no clinical data or weights were downloaded, no model was trained, and no author was contacted. The quantitative experiment budgets and decision thresholds above are proposals. The official RSNA announcement confirms the October 22 deadline and expert annotated assessment data, but the Kaggle data and evaluation pages returned no readable body in this environment; private dataset counts and validation readiness were not independently checked. [RSNA announcement](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri)
