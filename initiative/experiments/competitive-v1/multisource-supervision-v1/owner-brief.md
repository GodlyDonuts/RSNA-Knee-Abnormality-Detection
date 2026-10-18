# Codex brief: learn from multiple annotation sources without confusing disagreement with new MRI data

Prepared October 17, 2026. Repository evidence was read at commit `734c28e843508630834db2ec869b7daed2e4c436`, except the handoff initially read from main. Refresh current ownership and recipes before execution. This is a proposed research/implementation brief, not an instruction to restart existing jobs and not a demonstrated accuracy improvement.

## Executive instruction

The owner proposes using their GPT Astra annotations and other providers' annotations across training epochs. The useful hypothesis is **complementary supervision on the same MRI studies**, not the epoch boundary itself. Investigate and implement that hypothesis without a broad new model search.

Do not dismiss the idea merely because multi-annotator learning is established. Do not call it a frontier breakthrough merely because several providers appear in the source list. Determine whether the owner has additional, definition-aligned, provenance-distinct information that the current training target actually omits. Then train one existing MRI architecture on a precisely defined, immutable multi-source target overlay, with one matched comparison. Keep annotation disagreement and provenance available for a possible second-stage source-observation model.

The default implementation is a deterministic, missingness-aware, ancestry-aware soft-target mixture. It requires one image forward pass, not one forward pass per annotator. Whole-epoch source cycling is not the default. A source-aware observation model is a conditional extension only when its parameters are supportable; do not let an unconstrained label model invent clinical truth from two disagreeing CSVs.

Preserve all healthy spatial/path work, completed models, scoring owners, accepted original releases, evaluation exclusions, and existing compute/storage constraints. Implement directly in a separate proposed experiment namespace; do not create another series of planning documents without a working target constructor and bounded scientific comparison.

## 1. What is verified, what is not, and why it matters

### 1.1 Current supervision is not one uniform “Astra dataset”

The current full-encoder parent-adaptation comparison uses Steven v6 versus S6 v5 source labels, 3,887 fitting studies, uniform IID16 windows, common masks/teacher/initialization, a 25% source and 75% parent objective, and a fixed three-epoch/1,458-update schedule. Its original protocol isolates changing source values. Consult current-state for completed training/scoring, because the original protocol retains historical pending paragraphs. [R1]

The newer spatial reader comparison uses a different supervision contract: 75% original binary report supervision and 25% parent on observed cells, with detached parent supervision on missing cells. Its encoder is frozen N3 E16. Do not copy an anchoring coefficient from one family into the other or assume that the owner's annotation source supplies every current branch. [R2]

The phrase “GPT Astra annotations” is the owner's identification of their annotation work. The sources inspected for this brief did not establish the precise model identifier, prompts, input modality, extraction version, calibration, or whether every current “original” label derives from that run. Trace the actual training tensor back to its manifest. Do not infer provider identity from a friendly filename.

### 1.2 Some apparently different datasets are already related

The repository's provenance audit identifies `Shio-Koube/knee3`, pinned revision `373f4e1daf36984fa48bf3fc8fd9883fe17cb584`, member `llm_labels_v4_blend.csv`, as byte-identical to Steven's Kaggle version 6. Its SHA256 is `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`. Steven describes a probability-space blend of Steven v2 and lixin73's GPT-5.6-Sol report labels. Exact mixture coefficients and parts of the extraction pipeline are not disclosed. [R3]

A historical fitting-only audit also found two other tables identical in 9 of 12 columns over 2,599 shared selected studies. It found out-of-range values in one table, reaching 1.2745. These are dated diagnostic findings, not current cohort counts or permission to clip values. [R4]

Therefore, a Hugging Face mirror, the upstream file, the upstream file's blend, and its constituent extraction cannot be counted as four independent labelers. Nor does a different API provider establish independent errors when all providers interpret the same report.

### 1.3 Numerical values do not have universal semantics

The Steven card, as recorded in the repository, uses 0.5 for not-addressed findings. A separately audited Gemini source uses 0.05 for unmentioned findings and 0.5 for uncertainty, with severity encodings also present. A value of 0.05 in that source is not automatically a calibrated 5% probability of disease. [R3, R4]

Do not globally mask every 0.5, globally trust every 0.05, globally threshold at 0.5, or average arbitrary numeric columns. Source- and version-specific interpretation is required. A missingness marker may become inseparable from a genuine probability after an undocumented blend; mark that limitation rather than reverse-engineering an unsupported semantic rule.

### 1.4 The public blend is not independent of Gold58

Steven's documented target engineering, a Synovitis proxy, and blend comparisons used Gold58. Excluding those 58 rows from a new fit does not make the upstream recipe independent of that evaluation pool. Report-label AUC is not an MRI model's Kaggle AUC. [R3]

Do not use protected gold/confirmation values to choose source weights, prompts, state mappings, providers, thresholds, epochs, or eligibility. Use current accepted boundaries, not cohort counts copied from historical audits.

### 1.5 Claims not established by this brief

No particular Anthropic-generated candidate was provider-verified in this inspection. The owner may have such a source; admit its exact card, revision, process, and rights before attributing it to Anthropic. No new annotation arrays, reports, pixels, protected labels, or private predictions were analyzed here. No improved Kaggle score or clinical accuracy is claimed.

## 2. Mathematical answer to the epoch-rotation proposal

For one study and finding, let z be the student's logit, p = sigmoid(z), and y_s the usable label from source s. Standard binary cross-entropy with logits is

    ell(z, y) = softplus(z) - y*z.

For fixed nonnegative weights w_s summing to one,

    sum_s w_s ell(z, y_s)
      = softplus(z) - z sum_s w_s y_s
      = ell(z, q),

where q = sum_s w_s y_s. This is an exact equality of loss functions at the same logit, not an approximation. The logit derivative is p - q. The network gradient is (p - q) grad_theta z. The same result holds for independent binary outputs when the comparison keeps all source masks, weights, and reduction conventions consistent. [M1, mathematical derivation]

A source sampled with probabilities w_s has the same expected loss and gradient at fixed parameters and fixed image input as the deterministic mixture. Its source-sampling contribution to conditional gradient covariance is

    Cov_s(grad_theta ell) = Var_s(y_s) J J^T,
    J = grad_theta z.

For multiple findings and a shared sampled source, the corresponding term is J^T Cov_s(y_vector) J, using a Jacobian whose rows are the finding-logit gradients. This can correlate gradient noise across diagnoses. Averaging analytically removes this particular source of noise; it does not remove image/batch/augmentation noise or guarantee better generalization.

This is a conditional-expectation, or Rao-Blackwellization, argument. It establishes an objective/gradient relationship, not equality of final network parameters under different optimizer trajectories.

### 2.1 What epoch cycling changes

Whole-epoch cycling follows a time-varying objective. Learning rates, Adam moments, the representation, and which cases have been memorized differ between epochs. There can be curriculum effects, forgetting, and recency effects. More iterations on the same labels are not more independent annotations.

Do not write “epoch cycling is exactly the same training run as averaging.” The objective equivalence assumes the same logit/input and fixed source distribution. Do not write “cycling must improve because the model sees both views.” The temporal ordering supplies no additional annotation information beyond the observed sources.

One epoch per source also does not ensure equal effective supervision when sources have different row counts, masks, class balances, preprocessing, batch denominators, or number of steps. With changing weights/schedules, derive the actual objective instead of invoking the simple equal-weight identity.

### 2.2 Missingness and normalizers are part of the model

Let m_ics mark a meaningful observation and w_ics be a predeclared source weight. For study i and finding c,

    d_ic = sum_s m_ics*w_ics
    q_ic = sum_s m_ics*w_ics*y_ics / d_ic, when d_ic > 0.

When d_ic = 0, there is no multi-source target. Keep its availability mask false. A placeholder used for storage must never acquire a supervised loss merely because it is finite.

Averaging a normalized BCE per source is generally not the same as a per-cell-normalized mixture if source coverage differs. Choose one estimand and preserve the existing effective-batch/per-target denominator in the matched comparison. Report effective weights, not only nominal weights.

Weighting each annotation separately without per-cell normalization can upweight a study simply because it has more copies or annotators. That may be intentional in a specified probabilistic observation model, but it is not the default disease-supervision objective here.

### 2.3 Teacher anchoring does not invalidate the identity

For a fixed source fraction alpha and fixed detached parent probability t,

    alpha*ell(z, q) + (1-alpha)*ell(z, t)
      = ell(z, alpha*q + (1-alpha)*t).

Do not simultaneously change alpha, labels, masks, and spatial mechanisms, then call the result a label-source effect. Preserve the chosen reference's alpha. The previous strong claim that a 75% parent term mathematically prevents useful source corrections was already weakened by the repository's fitting-only diagnostic; do not resurrect it. [R5]

### 2.4 Concrete interpretation

Hypothetical, not actual clinical cases:

- Source A says 1 and source B says 1: an equal mixture is 1.
- A says 1 and B says 0: an equal mixture is 0.5, not “yes because someone said yes.”
- A says 1 and B genuinely abstains: use A's observation without interpreting abstention as 0 or 0.5.
- Both abstain: no direct source-label target is available.

Do not use a logical OR or `1 - product(1-y_s)` across annotators. Those annotations are alternate measurements of the same event, not distinct independent causes of disease. A positive-union policy can systematically manufacture positives and erase the specificity advantage of a better source.

## 3. Why a combined target could still outperform each source

The goal is not disagreement for its own sake. It is reduced error in the supervision that teaches the MRI representation.

Under a deliberately simplified additive model, let each source estimate an ideal target with error epsilon_s. For normalized weights w, the mixed error is w^T epsilon. Its squared-error decomposition is

    MSE = (w^T b)^2 + w^T Sigma w,

where b = E[epsilon] and Sigma is error covariance. For S equal-variance sources, equal weights, and common pairwise error correlation rho, the variance term becomes

    sigma^2 * [rho + (1-rho)/S].

This explains why multiple sources can help and why copied/shared errors give little benefit. It also explains why including a badly biased source can hurt. These are assumptions for reasoning, not measured properties of the present labels and not an AUC guarantee.

Observed label correlations do not identify error correlations without reference truth. Low correlation can reflect complementary errors, incompatible target definitions, differing missingness, or simply a bad source. High agreement can reflect shared report omissions. Never invert the empirical covariance of the raw label columns and call the result optimal reliability weights.

The owner's annotation budget is potentially valuable because it may have bought more accurate extraction, better coverage, or richer evidence. Price and provider identity are not empirical reliability estimates. The candidate should exploit demonstrated differences in available information, not use spending as a weight.

This is also a training-time combination. It need not increase the number of MRI encoders at inference, and it need not tune any public-leaderboard prediction blend. A single model can learn from multiple annotation sources. It can still overfit if weights or corrections are repeatedly selected using the same leaderboard or development pool.

### 3.1 AUC adds another important qualification

Suppose, as a restrictive idealization, that source s has a constant false-positive rate f_s and true-positive rate t_s independent of the image conditional on the true binary finding. If eta(x) is the true image posterior, then

    E[Y_s | X=x] = f_s + (t_s-f_s)*eta(x).

When t_s > f_s, this is a strictly increasing affine transform of eta(x). A fixed weighted mixture with positive overall slope has the same population ordering. An infinitely flexible, perfectly optimized classifier with infinite data would therefore not gain AUC merely from this kind of constant calibration correction. This is a derivation under stated assumptions, not a claim that real report labels satisfy them.

A useful RSNA improvement must come from effects those assumptions exclude: instance-dependent extraction errors, definition mismatches corrected with evidence, differing coverage, finite-sample/optimization effects, or a representation that generalizes better. A model returning less extreme probabilities is not automatically ranking studies better. Diagnose changes in actual within-target ordering rather than presenting softer confidence as an AUC mechanism.

## 4. Source admission and the minimum useful evidence table

Before touching a GPU, reuse the existing source catalog and original private arrays. Do not re-download MRI archives, re-label the entire corpus, or transfer report text to new services as a prerequisite.

For every candidate source record:

    source_id, provider_claim, verified_model_id_or_unknown,
    immutable_revision, filename, content_sha256,
    input_modality, report_version_or_pixel_exposure,
    target_schema_version, extraction_prompt_hash_or_unknown,
    raw_value_type, missingness_rules, uncertainty_rules,
    severity_rules, report_evidence_availability,
    ancestor_ids, copied_columns, blend_parents,
    training_and_evaluation_exposure,
    license_and_competition_admission_status.

For each admitted study-target assertion, retain privately:

    exact_study_key, leakage_component, target,
    raw_value, semantic_state, canonical_value_if_defined,
    observed_mask, source_family, evidence_reference,
    accepted_override_reference, reason_for_abstention_or_exclusion.

Semantic states must distinguish at least explicit positive, explicit negative, unresolved/uncertain, unmentioned, and unsupported/incompatible encoding. More detailed ordinal states require a documented target mapping; do not invent them for a source that released only binary labels.

Do not turn a labeler's self-reported confidence into calibrated probability of clinical correctness. Keep extraction confidence separate from disease probability and from “not mentioned.”

### 4.1 Family handling

Represent source ancestry as a graph. Deduplicate exact mirrors and known copied columns before aggregation. An ancestor and a derivative blend must not become extra independent votes. A conservative default is one canonical representative per overlapping ancestry component. Components may be target-specific when copying occurs only in selected columns.

When original primitive extractions are available and eligible, prefer those over opaque blends for modeling reliability. When only a blend is available, treat it as one composite source, record its ancestors, and exclude overlapping descendants/parents from extra votes. Do not pretend its undisclosed coefficients are known. A family group is a dependence-control convention, not proof that the remaining groups have independent clinical errors.

If no trustworthy reliability estimates exist, use equal weights across the admitted canonical families as a transparent first comparison, not as a claim of statistical optimality. Do not optimize a dozen per-target weights on rare positives. An existing adjudication with accepted provenance can take precedence, but not every LLM-generated hard label should be promoted to immutable truth merely because it was expensive.

### 4.2 Same scans versus genuinely new scans

Join shared studies by exact IDs and existing component mapping, verifying duplicates/versions. Multiple annotation files for the same studies increase supervision diversity, not the number of patients.

A source with additional distinct patients is a different intervention involving population, sampling, target mapping, and overlap. Do not mix it into this first comparison and call the result a benefit of rotating annotation sources. Preserve the same MRI roster and image exposure.

## 5. Default candidate: one deterministic, lineage-aware overlay

### 5.1 Trace the true baseline

Print a short machine-readable trace from the actual candidate's loss target back to source files and accepted overlays. Include mask counts, meaningful-probability counts, sentinel/ambiguous counts, actual anchoring fraction, and cohort. The owner's Astra labels may be unused by some current branches; discovering that is more important than launching an epoch scheduler.

Use the strongest already-completed matched training recipe for which the proposed new source values can be the only change. The licensed-parent uniform16 family is an existing candidate for this, subject to current status and exposure constraints. Do not silently substitute the newer spatial reader or dense-path architecture to test this label idea.

### 5.2 Label-values-only comparison

For the most economical first comparison, keep original MRI inputs, order, views, masks, batch normalization, initialization, total optimizer updates, schedule, precision, source-parent mixture, and inference exactly as in the selected reference.

Build a new immutable target overlay only on currently supervised cells where the baseline annotation and at least one nonoverlapping admitted family supply meaningful, definition-compatible values. Mix at the study-target level. Outside those cells, preserve the baseline target and mask exactly. Log how many cells actually change and why.

This is a deliberately narrow test of complementary label values. It does not test coverage expansion. It also does not silently fix a legacy missingness treatment. If baseline 0.5 placeholders were trained as regularization, unmodified fallback cells remain the legacy baseline behavior; do not relabel that behavior as semantically observed evidence or imply the whole baseline has now been repaired.

If correct source admission requires changing the baseline's supervised-cell mask, do not reuse an unmatched old run. Prepare a semantically normalized matched control and candidate with the same mask, or explicitly limit the first comparison to safe common-observed cells. Preserve and disclose the coverage limitation. No live recipe is mutated.

Genuinely accepted training-only adjudications retain their accepted precedence in both arms. Keep original files immutable; a research overlay is not an undocumented overwrite of the accepted release.

### 5.3 Preserve the full annotation tensor

Store the admitted labels and masks as study x target x source-family arrays, even when the first candidate trains on the mean. Also retain family count and descriptive disagreement. A consensus of [1,0], two genuinely uncertain [0.5,0.5] observations, and no observations are different evidence situations even if naive scalar encodings coincide.

Do not automatically downweight disagreements to zero. They may be the hard, rare, or boundary cases that matter. Do not automatically upweight agreement as truth. Disagreement measures are initially diagnostics, not a calibrated loss multiplier.

### 5.4 No repeated MRI passes per source

The mixture target is precomputed from frozen annotation tables. The trainer performs its original image forward pass and its original BCE/anchoring/reduction with the new target tensor. Target-source identity is not an MRI input. Reports and source-specific explanation text never enter the deployed image-only classifier.

For illustration, a 3,887 x 12 x 3 float32 target tensor is 559,728 bytes before masks/provenance. This is an arithmetic size estimate, not a reservation or a claim that three eligible sources have already been admitted. The dominant cost remains the selected MRI training recipe, not averaging a few target columns.

## 6. Optional source sampling, and why it is not the primary experiment

A source-sampling implementation is useful to verify the owner's temporal proposal and for streaming systems. Sample among available admitted families at the study-target level using the same weights as the deterministic mixture. Save the generator state in the checkpoint. Source order must not be tied to hospital, patient order, or learning-rate phase. Missingness must not change the effective batch denominator relative to the analytical comparator.

Verify expected losses/gradients numerically on fixed synthetic tensors, including missing cells. The accompanying reference implementation provides a small deterministic-mixture constructor and a sampler for this purpose; it is not integrated into the repository trainer.

A model trained with source sampling can differ from a deterministic-mixture model because of gradient noise and optimizer dynamics. That is a separate hypothesis about optimization. Test it only after confirming that genuinely additional source information exists and that the mixture changes useful supervision. Do not consume the first competition entry comparing A/B/A versus B/A/B epoch order.

Do not call a difference in source-sampled fitting loss a disease improvement. Do not use the final source of an epoch cycle as an implicit preferred provider. Do not reset Adam or the learning-rate schedule at a source change unless that is the explicitly separate intervention.

## 7. Advanced extension: model the observation process rather than averaging all mistakes

This is the stronger conceptual version of the owner's idea: distinguish what is in the image from how each source describes it.

Let p_ic be a shared image-based probability for finding c. For a canonical binary observation from source s, a simple source observation layer is

    r_ics = f_sc + (t_sc - f_sc) p_ic,

where f_sc is the source's false-positive tendency and t_sc its true-positive tendency relative to the modeled latent target. Fit the observed annotation using r_ics, while deployment outputs p_ic. Missing annotation cells contribute no observation term. This can represent sources with different sensitivities/specificities instead of forcing the image model to compromise between their raw numbers. Multi-annotator layers and confusion models are established methods. [M2, M3]

There are substantial qualifications:

1. Those parameters are not identifiable from arbitrary disagreement alone. Two unconstrained binary observers plus an unknown latent prevalence have more free parameters than their joint binary table supplies. A flexible image network does not automatically solve that ambiguity. Label-noise identifiability requires additional assumptions/information. [M4]
2. With many sources, inherited blends and shared reports violate naive conditional-independence claims. Three provider names are not automatically three independent observations of disease.
3. A per-source BCE sum is a composite marginal objective. Do not call it the exact joint latent likelihood. Under an explicitly assumed binary latent model, a full joint likelihood would be a sum over the shared latent state of products of conditionally independent emissions. Dependence changes that model.
4. Soft LLM probabilities and severity numbers are not binary draws without an explicit observation model. Do not fit Bernoulli confusion matrices to arbitrary rank scores, sentinels, or blended confidences and claim recovered clinical sensitivity.
5. The same report can omit a finding for all extractors. A source layer cannot manufacture the missing image evidence or cure the underlying report's error.

### 7.1 Admission gate for the advanced model

Use it only when the project already has eligible, fitting-only adjudications or justified constraints that identify and orient the source-observation model. Separate those training anchors from independent final evaluation. Shrink reliability estimates when support is small; report uncertainty. Do not derive reliability from a public model's own exposed fitting predictions or calibrate on protected Gold58.

A known reference source can orient a constrained model, but fixing it to identity defines “truth” relative to that source; it does not establish clinical correctness. Treat such a construction as source-relative adaptation, not recovered latent truth.

When the information is insufficient, retain the transparent family-level mixture and disagreement metadata. An honest non-identifiability conclusion is preferable to a sophisticated but arbitrary confidence model.

### 7.2 Where the owner's annotation effort could create an advantage

Trace whether the Astra workflow retained explicit report spans, negation, temporality, laterality, uncertainty, and accepted correction rationales. If it did, those are potentially more valuable than the final 0/1 columns alone.

Use source disagreement to prioritize a bounded fitting-only review queue stratified across findings and disagreement types. Resolve a source's misreading of an explicit statement using verifiable text evidence and the fixed competition rubric. Separate “clear extraction mistake” from “report ambiguity” and “possible report/image disagreement.” A report-only review can address the first two, not establish the third without appropriate image adjudication.

Provider agreement is not the acceptance rule. No large-scale new API labeling job is authorized by this brief. Existing records should be reused, and any external handling of private reports must follow current permissions. Preserve reviewer provenance and do not feed validation outcomes back into the review instructions.

This queue need not stall the initial mixture construction. It is a parallel way to determine whether disagreements contain actionable information rather than just different uncertainty conventions.

## 8. One bounded scientific comparison, not an experiment explosion

### Primary comparison

Reuse an existing fixed control only if its full numerical recipe, initial state, fitting roster, image schedule, masks, optimizer updates, and inference match. Train at most one new deterministic multi-source candidate under that contract. Score all predeclared endpoints, not only the finding with the largest apparent improvement.

If no suitable control exists because the necessary source-semantic correction changes masks, run one properly matched pair rather than claiming causality against a historical mismatch. Do not also add dense windows, a different encoder, new spatial paths, longer schedules, or a new anchor coefficient.

### Cost and budget

Target construction and lineage review are small-data CPU work. Reuse downloaded annotation artifacts and original image caches. No full image decoding is required for comparing label arrays. Cap new training at the empirically measured budget of the selected existing recipe; one candidate should not require additional encoder evaluations merely because it has several sources.

Before launch, record the exact measured wall time, peak memory, checkpoint/storage liabilities, and remaining inference/submission allowance. Do not invent an H100 runtime or claim GPUs are allocated because the owner authorized their use. Respect the project's physical 1/2-H100-node restriction; physical 4/8-GPU nodes remain excluded. Keep all existing healthy jobs and scorers untouched. [R2]

### Evaluation and independence

Use existing all-target paired, patient/leakage-component-aware analysis with the actual fixed endpoint. Report undefined bootstrap draws and rare-class support. No dropping difficult findings, picking the best epoch, or editing evaluation labels after inspecting disagreement.

The full-parent source comparison is useful for a competition candidate, but internal evaluation on studies seen by the released encoder is not independent whole-model evidence. A clean-lineage N3 comparison can provide development evidence if its complete fitting lineage excludes the evaluation groups, but repeatedly inspected development data are not pristine final confirmation. [R2]

Do not use the same source mixture as both training target and the only definition of success. Better agreement with the mixed weak labels is a sanity check, not proof of improved disease classification. Do not label internal metrics “public-equivalent.” Public results alone do not establish private-final gains.

### Decisions that change the next action

- No second semantically admissible, ancestry-distinct source: do not launch a nominal multi-source fit. Finish the evidence table and identify precisely what is missing.
- New sources are copies or the mixture is numerically identical to the current target: no new classifier information is added; skip duplicate training.
- The candidate changes few cells: report the actual target-by-target changes, not “three datasets used.” Decide whether a bounded overlay still addresses a meaningful error class without inventing an impact estimate.
- The candidate improves only agreement with its own weak labels: no demonstrated clinical/private gain. Retain as an unvalidated candidate, not a promoted mechanism.
- The fixed candidate fails the matched disease-ranking comparison: keep the negative result and do not search public-leaderboard weights until one looks good.
- A promising gain survives the appropriate predeclared endpoint: replicate using a suitable independent/clean lineage check as available, and qualify exact offline inference. Its source-combination benefit is still separate from private-final performance.

Do not require an arbitrary statistically significant rare-target result when there are only a handful of positives, but do not represent that lack of power as evidence of benefit. Distinguish an exploratory promising candidate from an established mechanism.

## 9. Implementation deliverables

Use existing repository mechanisms rather than adding another training framework. Suggested names below are proposals; inspect the tree before creating conflicts.

1. **`multisource_manifest.json`**: immutable source versions, meaning of values, ancestry groups, evaluation exposures, admitted populations, and all unresolved claims.
2. **`multisource_overlay.py`**: deterministic family-normalized target constructor, availability/disagreement arrays, common-observed replacement policy, and exact single-source equivalence.
3. **`multisource_target_audit.json`**: fitting-only per-target coverage, source pairs, missingness, copied columns, conflict counts, target-change magnitudes, retained overrides, and effective weights. No individual labels or reports in public Git.
4. **`multisource_fixed_candidate.json`**: one exact reference, one exact new source objective, shared masks/schedule/images/inference, measured budget, predeclared endpoints, and original owner identities.
5. **`multisource_result.md`**: the actual question answered, observed metrics, uncertainty, exposure limitations, cost, and a promotion/rejection decision. Lead with predictive evidence, not hash counts.

Important implementation checks are narrow and scientific: single-source identity, explicit-BCE-sum versus mixed-BCE loss/gradient equality at fixed logits, no missing-sentinel votes, family deduplication, no evaluation membership in fitting arrays, exact denominator preservation, and deterministic resume if source sampling is enabled. The companion module demonstrates only the algebraic target construction; it does not admit clinical semantics or automatically integrate with an existing anchor/loss normalizer.

## 10. Final message Codex should be able to return

Return a concise statement answering:

- Which label files are actually used by the current candidate, and does the owner's Astra work enter that loss?
- Which additional source families are eligible and genuinely nonduplicative, and what meanings do their values carry?
- How many current training cells change under the proposed mixture, versus how many are merely repeated votes or newly covered cells?
- Is the new candidate a label-values-only comparison, a separately controlled coverage change, or a source-observation model?
- What exact completed control is reused, what new training actually ran, and what actual independent or competition result exists?

Do not reply only with “multi-source training is implemented,” “all tests pass,” or “training has started.” Those do not answer whether the owner's additional annotation information helps the MRI classifier.

## References and exact source pointers

Repository facts below are anchored to the read commit, not presumed current forever. These sources contain historical paragraphs; current-state and original terminal evidence override stale operational statements.

[R1] Matched S6 source-label contrast:
https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/734c28e843508630834db2ec869b7daed2e4c436/initiative/experiments/competitive-v1/s6-source-label-contrast-v1/README.md

[R2] Research-agent handoff:
https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/734c28e843508630834db2ec869b7daed2e4c436/docs/research-agent-handoff.md

[R3] Steven v6 provenance addendum:
https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/734c28e843508630834db2ec869b7daed2e4c436/docs/research-watch/additional/rsna-steven-v6-provenance-addendum-2026-10-14.md

[R4] Historical adapter/public-supervision source audit:
https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/734c28e843508630834db2ec869b7daed2e4c436/initiative/audits/adapter-source-research-v1/README.md

[R5] Fitting-only teacher-anchor diagnostic:
https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/734c28e843508630834db2ec869b7daed2e4c436/initiative/audits/fitting-teacher-anchor-tension-20261017-v1/README.md

[M1] PyTorch, BCEWithLogitsLoss, official documentation:
https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html

[M2] Rodrigues and Pereira, Deep Learning from Crowds, AAAI 2018:
https://ojs.aaai.org/index.php/AAAI/article/view/11506

[M3] Tanno et al., Learning From Noisy Labels By Regularized Estimation Of Annotator Confusion, 2019:
https://arxiv.org/abs/1902.03680

[M4] Liu, Cheng, and Zhang, Identifiability of Label Noise Transition Matrix, ICML 2023:
https://proceedings.mlr.press/v202/liu23g.html

[M5] Ratner et al., Snorkel: Rapid Training Data Creation with Weak Supervision, 2017:
https://arxiv.org/abs/1711.10160

The mathematical derivations, proposed decisions, and candidate implementation in this brief are analysis/proposals. None is a claim of a newly established RSNA AUC improvement.
