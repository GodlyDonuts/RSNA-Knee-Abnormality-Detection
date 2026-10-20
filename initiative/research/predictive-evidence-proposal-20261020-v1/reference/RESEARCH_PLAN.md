# Hypothesis-conditioned MRI prediction with censored spatial evidence

Research proposal | October 20, 2026 | Goal: a legitimate competitive improvement toward 0.97

## Status and decision

This is a new proposed learning formulation for this project, not a trained model or a promise of a winning score. The included PyTorch file implements and tests its probability calculation only. No MRI files, report rows, protected labels, external medical weights, production jobs or competition submissions were accessed or modified by this package.

The recommended bet is to learn which finding hypothesis best predicts withheld, actually observed MRI detail. Do not build a large diffusion generator. Do not synthesize lesions and call them ground truth. Do not train another model primarily to reproduce a public teacher's twelve probabilities.

The distinctive combination to investigate is: (1) diagnosis-conditioned prediction of real held-out local MRI descriptors, (2) positive-versus-negative evidence ratios using the same context and nuisance model, and (3) marginalization over possible lesion supports that preserves the probability mass of supports outside the currently observed region. This is a proposed synthesis, not a claim of worldwide novelty or of a proven clinical causal model.

## Evidence boundary

The retrieved current-state snapshot was updated October 20, 2026, 05:37 UTC. It records the R6 reproduction at 0.958, whole94/influence/paired/higher-learning-rate adaptations at 0.949, and final cross-lineage submission 57422143 at 0.949. It records ranking and source-zero work as active. That snapshot is not a fresh read of live scheduler state.

The historical creative plan already includes patch-witness, plane-memory, geometry, neighboring-slice and supervision experiments. An attention head, a cross-plane graph, or another generic label audit must not be represented as an entirely new direction. A failed frozen-feature probe also does not rule out a representation that must be learned end to end.

Project sources inspected or supplied in this conversation:
- AGENTS.md (retrieved blob becba8cb4f7b6075288d0f19406f0a20945322e7).
- docs/current-state.md (retrieved blob 3899e3d59336f45adac2d68776845ad8848e380e).
- docs/creative-research-plan.md (retrieved blob 45c9bc623f9eae73a745f4e1ce19f7ffa4a8656e).
- rsna/competitive/released_context.py: accepted source/teacher mixing.
- rsna/competitive/licensed_full_bag.py and original whole94 completion receipt.
- rsna/competitive/teacher_pair.py: opposite render/mirror views, not proof of independent acquisition correspondence.

## Research hypothesis

Global exam probabilities give weak information about where disease evidence is and what distinguishes it from normal anatomy. A different objective can train the encoder to model the image evidence supporting a finding, rather than principally preserve another model's decision function.

This may improve sample efficiency and error diversity. It does NOT create new patients, remove report-label errors, guarantee lesion localization, or establish that generative models are superior in knee MRI. Its value must survive a strong discriminative control with the same images, initialization and computation.

## Model

### Observation units

Maintain an acquisition-aware complete roster of non-overlapping target regions and their slice/series provenance. These are physical or renderer-traceable regions, not assumed independent copies of overlapping 2.5D windows. Cross-acquisition geometry is optional and requires quality checks; valid within-series geometry does not prove cross-series registration.

Keep the existing accepted fitting cohort and supervision immutable for the first experiment. Create new, separately versioned masked-image inputs. Do not change inputs or sources of active jobs.

Use two complementary, label-independent block-mask patterns from one fixed family in both training and serving. Produce all original input windows from the masked source volume. If a held-out target appears in another overlapping window/channel/mirror, remove that duplicate too. Apply a halo around the target. Never mask only a late feature vector after a full-image CNN has already encoded the answer.

The encoder sees visible context C. Targets t are fixed descriptors extracted from actual held-out image regions, not public-teacher probabilities. Initial implementation: deterministic multiscale patch descriptors, such as a pinned DCT/intensity-gradient transform with normalization fitted only on the fitting partition. Keep descriptor width at 64 initially; descriptor sufficiency is an explicit risk. No learned target encoder may collapse together with the predictor. No target-dependent image normalizer is available to the visible-context encoder.

Use the existing encoder architecture initially to isolate the learning change. Initialize all matched arms identically. An inherited MRI checkpoint remains an inherited-exposure lineage even without a teacher loss. A strict Kaggle-medical-data-only lineage requires a vetted generic initialization or a competition-only initialization; repeat the admitted mechanism in that lineage before making a provenance claim. Do not import another medical checkpoint for this proposal.

Remove the old teacher-target objective and old parent-logit additive path. Train the new predictive model end to end after a brief decoder warm-up. A new context-only disease prior is permitted; it is not the old parent prediction.

### Predictive hypotheses

For target c and local descriptor region j, predict from visible context:

    f0(t_j | C,c) = Normal(mu0_cj(C), Sigma_c)
    f1(t_j | C,c) = Normal(mu0_cj(C) + delta_cj(C), Sigma_c)

The first is target-c-negative, not an assertion that the entire knee is healthy. Both hypotheses share variance and nuisance context. Fix or tightly control the variance in the first experiment so a class cannot win by inflating its predictive uncertainty. These are descriptor-space densities, not calibrated likelihoods of full clinical MR images.

Local log evidence is:

    ell_cj = log f1(t_j|C,c) - log f0(t_j|C,c)
           = delta_cj^T Sigma_c^-1 (t_j - mu0_cj)
             - 0.5 delta_cj^T Sigma_c^-1 delta_cj

It requires a hypothesis to predict the actual held-out descriptor better, not merely assign a high class score to an observed feature.

### Latent lesion support and censoring

Let Z be a candidate set of regions supporting finding c. Let S_c be a fixed family of possible supports, and pi_c(Z|C) a normalized prior independent of held-out target values and the observation-selection policy. The first version uses fixed geometry-only priors. Include several support scales, so diffuse findings are not forced into a single lesion voxel. Multi-component support is a later declared extension, not an assumed property.

Under the prototype's conditional-independence model, with O the observed target regions:

    BF_c(O|C) = sum_{Z in S_c} pi_c(Z|C) * exp(sum_{j in Z intersect O} ell_cj)

This follows by dividing the support-mixture positive density by the negative reference density. It is exact for that specified model, not proof that real MRI regions satisfy its assumptions.

IMPORTANT: keep all complete-roster supports in the sum. A support with no overlap with O contributes its prior mass times 1. Do not throw it away and renormalize the prior over visible supports. That operation silently conditions on having observed the lesion.

Consequences:
- No observed target evidence means BF=1, or log BF=0.
- Identical positive/negative predictive models mean BF=1.
- BF is bounded below by prior mass on completely unobserved supports.
- Duplicating an alternative support and splitting its prior mass leaves predictions unchanged.
- Duplicate image renderings must not be counted as independent spatial evidence.

A context-only log-odds head a_c(C) is needed:

    p_c^(m) = sigmoid(a_c(C_m) + log BF_c(O_m|C_m))

Conditional likelihood ratios alone are not a complete disease posterior if the context itself is informative. Across overlapping mask contexts, average p_c^(m); do not multiply them as independent confirmations.

### Training

Keep accepted q_ic and observed-label mask M_ic unchanged. They remain weak targets.

    L_exam = masked_BCE(p_final, q)
    L_density = mean_valid [-q log p1(t_O|C) - (1-q) log p0(t_O|C)] / observed_descriptor_count
    L = L_exam + lambda_density * L_density

Pin lambda_density before the serious comparison; 0.1 is an initial engineering value, not an established optimum. Missing label cells receive no class-conditioned supervised loss. Label-free reconstruction may still use eligible fitting images, but no protected images enter it.

All arms can share a short, identical label-free reconstruction warm-up. Its positive result would support representation learning, not specifically the new classifier. Stage the objective in a deterministic, separately versioned module. Predictions at inference require only images and acquisition metadata.

Initial matched training recipe, subject only to the tiny-subset numerical acceptance: effective batch 8, AdamW with encoder learning rate 3e-5, new-head learning rate 3e-4, weight decay 0.02, 5% warm-up and cosine decay to 10% of peak, six epochs (2,916 optimizer updates at 3,887 studies). Freeze BatchNorm running statistics in both arms if retaining the existing small-batch encoder path. Use mixed precision for the encoder and FP32 for density/log-sum-exp/loss calculations. These are proposed engineering settings, not validated optimal values. Any necessary change applies identically to both arms before the serious comparison; no per-arm tuning.

Use all inexpensive target-descriptor/readout evaluations for each chosen training mask. Do not reproduce the old sampled-readout/full-readout mismatch by summing a different set of target likelihoods at serving. Stream local predictions and sparse support sums; do not materialize dense full-image attention or a huge support-by-region tensor in the production implementation. The supplied dense tensor code is a small reference oracle only.

## Three experiments within one research program

### E1: Does diagnostic prediction add more than generic representation learning?

Primary full comparison:
- A: strong end-to-end discriminative model with per-finding spatial/series readout.
- B: A plus diagnosis-independent masked descriptor prediction.
- C: hypothesis-conditioned prediction with censored support evidence and a context-only prior.

All use the same fitting cohort, label masks, initialization, raw image coverage, mask family, encoder-forward budget, optimizer-step budget and checkpoint schedule. Make direct comparator heads sufficiently strong; match head parameters and measured FLOPs within a reported tolerance, rather than claim an exact equivalence that changing architecture makes impossible. The discriminative control must also be able to read the observed target descriptors, not only the masked context.

The fastest decisive pair is B versus C, not C versus the historical 0.949 number alone. Run B/C first on available ordinary one-/two-H100 capacity. A can be a reduced-cost scientific control if resources permit. Keep a direct readout on C for an inference-ablation: is its improvement in the new predictive score or merely its learned representation?

### Required reparameterization control

The shared-variance Gaussian log ratio is linear in the target descriptor when context is fixed. Therefore its algebra alone is not a new expressive classifier. Run an identical-architecture control C0 with the class-conditioned density coefficient set to zero, retaining the same observed descriptors, support calculation, parameters and forward operations. Both C and C0 still train on the exam-level objective. This isolates the effect of learning to predict actual image evidence from a mere reparameterization of discriminative classification. Also retain B, the diagnosis-independent reconstruction control, to distinguish a diagnosis-conditioned benefit from generic masked-image regularization. These ablations are scientific requirements, not evidence that a gain has already occurred.

### E2: Does correct observation handling matter?

Hold predictive decoder, data and support family fixed. Compare retained missing-support prior mass with the explicit wrong-observation control that renormalizes only over visible supports. Evaluate multiple predeclared coverage levels without selecting the best on gold/public scores. Compare full spatial support with equal-size shuffled spatial grouping. The latter preserves individual regions and compute while breaking the claimed anatomical structure.

If correct censoring has no useful effect but C improves, retain the useful model and withdraw the causal claim about censoring. If only a frozen probe fails, give one bounded end-to-end run before rejecting the representation hypothesis.

### E3: Is this useful beyond R6?

Freeze the scientifically selected model. Evaluate the new standalone model and its incremental value with the accepted R6 ensemble on an evaluation set allowed by project policy. Pick any blend policy before final confirmation, not by trying many public-LB weights. A class specialist can be useful even when it does not dominate all 12 targets.

The final target is competitive 0.97, not a self-imposed demand that one original solo model must score 0.97. At equal macro weighting, moving 0.958 to 0.970 requires +0.144 summed per-target AUC, for example +0.048 in three targets with no other changes. These are arithmetic examples, not predicted gains or evidence that these gains are attainable.

## Execution allocations

The official published deadline is October 22, 2026, 23:59 UTC. This plan is not an asynchronous job or a promise of later delivery.

0-6 hours: implement target descriptor and raw-level masking contracts, inspect fitting-only cases, bind splits/provenance, run leakage and source-parity tests. Validate within-series geometry; cross-series geometry is not a prerequisite for version 1.

6-12 hours: real-MRI tiny-subset overfit, reconstruction-versus-identity baseline, gradient check, and an early exact two-T4 inference timing test. Frozen-feature pilots are debugging, not the final scientific gate.

12-36 hours: B/C end-to-end pair, fixed budget. Reserve an initial 12-16 H100-GPU-hours per arm, only if actual available capacity and throughput permit. One physical one-/two-H100 node is allowed; no physical four-/eight-GPU nodes. A six-epoch two-mask/94-window reference involves 3,887*94*2*6 = 4,384,536 window forwards per arm, before backward/recomputation costs. This arithmetic is not a runtime measurement. If needed, lower shared backbone cost rather than silently shrinking only the candidate's evidence.

36-48 hours: fixed candidate inspection, censoring/geometry/inference ablations, predeclared seed repeat if effect remains credible, and final checkpoint preservation. Do not extend all variants.

48-60 hours: locked evaluation if authorized, full offline packaging, representative and tail-case T4 run, exact complete CSV validation and the one controlled competition evaluation. Begin the final full offline/scored run early in this window; do not postpone its nine-hour allowance until the window ends. Set an absolute latest launch target of October 22 at 12:00 UTC, leaving approximately three hours beyond a full nine-hour run before the published deadline. Reserve the accepted baseline and do not depend on a last-minute unqualified fallback.

Actual GPU availability and input throughput were not read by this research package. Existing source-zero/ranking jobs remain unchanged. Reconcile the actual owners, quota and funded storage before dispatch.

## Failure tests

Reject an implementation immediately if held-out target pixels can affect the predictive mean through duplicate windows, unmasked cache features, targets as inputs, or target-derived normalization. A locality graph is not anatomical registration merely because it uses DICOM coordinates.

The predictive mechanism is unsupported if:
- class-conditioned predictions do not improve held-out conditional descriptor fit relative to B;
- image-target shuffling within acquisition strata leaves the purported new evidence equally useful;
- nearly all diagnostic performance comes from a_c(C), while BF contributes no reproducible information;
- improvement disappears against B at matched image/compute budget;
- report/site/scanner shortcuts explain the apparent gain on group-aware checks;
- a permitted independent evaluation does not support clinical benefit;
- offline inference cannot finish with the required complete UID set.

One source-proxy null does not prove that MRI contains no further signal. One positive proxy does not prove clinical accuracy. Do not turn arbitrary +0.001 rounded public AUC into a reliable win, or impose a rule that eight targets must improve when an honest specialist could help an ensemble.

## Validation discipline

Patient/duplicate-disjoint splits precede caches and any self-supervised adaptation. Keep protected gold/confirmation out of fitting and architecture selection. Respect the existing policy for any single final confirmation; this package does not authorize reading it. An inherited MRI encoder may have upstream exposure to an apparent local holdout. Label the limitation, and use a vetted generic initialization in the clean-lineage experiment.

Soft-label loss is meaningful against the given weak targets. Thresholding those probabilities does not create expert ground truth. Teacher agreement, attention maps and successful reconstruction are mechanism metrics, not clinical validation. Correcting arbitrary instance-dependent label noise from the same noisy observations is not generally identifiable without additional assumptions or independent information [5].

## Offline deployment budget

Two T4s, 16 GB each; four CPU cores; about 29 GB host RAM; nine-hour hard wall-time cap; no internet; exact submission.csv.

Design target: one shared encoder, two masked contexts, 12 paired small predictive heads, streamed descriptors and sparse support sums. No diffusion denoising loop, text model at inference, external retrieval, or 24 full CNN runs. Replacing two mirrored 94-window encoder passes with two masked 94-window passes keeps the nominal encoder-window count equal, but is NOT a measured throughput or accuracy equivalence.

Acceptance envelopes: <=12-13 GB VRAM per T4; <=22 GB host RSS; <=7 hours projected end-to-end wall time on the actual hidden-size bound, leaving at least two hours margin. Never claim a memory/runtime pass from the reference mathematical tests.

Let R be actual measured per-GPU encoder-only window throughput, W=188 initial encoder windows per study, N the authorized hidden-study upper bound. Encoder-only planning estimate is N*W/(2R). Add observed decode, descriptor, predictor, CSV, setup and tail costs. At hypothetical R=40 windows/sec, 5,000 studies need 3.26 encoder-only hours, 10,000 need 6.53; neither is a claim about actual test size or actual T4 throughput. Benchmark immediately. Maintain complete-output fallback only through a separately qualified baseline, not a partial or silently constant CSV.

## Prior work and distinctions

[1] Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks (2012). Architectural/optimization example, not evidence that this proposal will replicate its success.
https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

[2] He et al., Masked Autoencoders Are Scalable Vision Learners. Masked image reconstruction is established, not our invention.
https://arxiv.org/abs/2111.06377

[3] Li et al., Your Diffusion Model is Secretly a Zero-Shot Classifier (ICCV 2023). Generative conditional density can serve classification. We do not transfer its natural-image performance or propose its expensive inference loop.
https://arxiv.org/abs/2303.16203

[4] Sudakov et al., Revisiting Autoregressive Models for Generative Image Classification (ECCV 2026), version 2 dated July 1, 2026. Useful precedent for efficient generative classification and conditioning-order effects, not a knee-MRI result. Its official README uses a 0.3B RandAR configuration and multiple trials; it is not a plug-in nine-hour MRI solution.
https://arxiv.org/html/2603.19122v2
https://github.com/yandex-research/ar-classifier

[5] Liu et al., Identifiability of Label Noise Transition Matrix (ICML 2023). Limits on recovering true labels from noisy observations.
https://proceedings.mlr.press/v202/liu23g.html

[6] Ren et al., Likelihood Ratios for Out-of-Distribution Detection (NeurIPS 2019). Motivates cancellation of background statistics; does not prove our conditional Gaussian heads cancel MRI nuisances.
https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html

[7] Zhuang et al., Enhancing Knee Disease Diagnosis via Multi-View Graph Representation With Multi-Task Pre-Training (IEEE TMI 2026). Shows that graph MRI and masked pretraining are already studied. Official code requires segmentation assets in its illustrated data pipeline. Borrow ideas only; do not import MRNet/in-house data, segmentation assets, or restricted derived weights.
https://pubmed.ncbi.nlm.nih.gov/41632682/
https://github.com/zixuzhuang/KGNet

[8] Kaggle official competition requirements and deadline.
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation

## What has actually been done in this response

Inspected repository source/state/history evidence; reviewed primary papers and official implementations; derived the support-mixture likelihood identity; implemented an original small PyTorch reference; ran 18 synthetic CPU tests including exact Gaussian-mixture equality and autograd finite-difference checking. No medical-image training, actual lesion localization, T4 qualification or competition improvement has occurred.
