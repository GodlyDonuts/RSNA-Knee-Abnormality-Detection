# RSNA final week experiment decision

14 October 2026, 19:20 UTC. Decision synthesis from the completed source and experiment audits. This is a recommendation for future authorized work, not a request to execute, alter queues or expand validation access.

This prioritization supersedes the broader experimental schedules in the earlier research notes for the October 22 competition deadline. Their additional controls remain useful for mechanism research; they are not all required before selecting a stronger eligible final model.

## Recommendation

**Allow at most two new performance experiments: one directly deployable compact challenger, then one conditional image target blend on the strongest eligible family.** Leave either slot unused when its prerequisites fail. Recover already-paid evidence first, without allowing an inaccessible result to hold the whole decision indefinitely.

Do not make DINOv3 distillation, a new architecture paper, or a causal explanation prerequisites for selecting a better eligible model. A compact model that meets the single-model definition, improves the admissible performance evidence and fits deployment constraints is itself a potential final model.

The two slots are ordered by option value, not a commitment to run them. A blocked compact recipe loses priority to an executable teacher blend. Bounded source-distance weighting, window-relation transfer and new geometry heads do not automatically inherit a slot because their implementation looks small.

## First collect the evidence already paid for

Reconcile the completed second-seed supervision training with its missing final evaluation, the current all-development checkpoint results, and any completed mean/query, dense-adaptation or patch-focus results. Preserve unavailable outcomes as unavailable. The cancelled dense fit is not a negative accuracy result, and a recovery test is not a full scientific comparison. These findings may remove the need for either new experiment. The all-development model has no internal holdout; its .872 public result and the older .899784 weak-development result are different fitted models and evaluation populations. [Historical status and terminal receipts](experiment-ledger-reconciled.md)

This is a request to use existing evidence when lawfully available, not to restart every interrupted run. Do not require completion of an entire research backlog before making the final-model decision.

## Experiment 1 Compare one compact whole recipe as a final model

**Question:** does one concrete compact pipeline offer better eligible competition performance and deployment reliability than the strongest completed incumbent?

**Why it earns first refusal:** it can improve the final artifact directly and avoids paying for a second model merely to transfer its information back into DINOv3. It also provides a possible teacher for the second experiment. The failed .878287 source-slot package did not reproduce the newer CoAtNet recipe and does not close this option.

**The source gate is currently substantive.** The .953 ConvNeXt-T and .951 DINOv2-S are author reports with incomplete teacher, training and checkpoint details. They are motivation, not ready reproductions. The inspectable nartaa .949 is one SWA inference state with two views, but inherited OAI/prize obligations remain unresolved and its MRI training covered all 4,349 nongold studies, including our outer/reserved cohorts. It cannot provide an uncontaminated local comparison merely because subsequent fine-tuning is fitting-only. Competition eligibility and local evaluation independence are separate questions. [Exact artifact and recipe audit](../rsna-single-model-evidence-2026-10-14.md), [pinned .949 notebook](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397)

There are two permissible forms of this **one comparison**, depending on admission:

- An already trained compact artifact, if separately admitted under the existing work agreement, can be considered directly through an authorized evaluation route. Its inherited exposure must remain visible. No new local holdout, protected-row evaluation or implied permission to download/use weights follows from this recommendation.
- Otherwise, compare one explicitly specified **new own compact recipe** from an admitted ImageNet initialization on the existing fitting population. The available Raptor training/inference code is enough to define a Raptor-like fully fine-tuned CoAtNet bag with current accepted labels, consistent preprocessing and a fixed endpoint, without borrowing all-4,349/OAI weights or copying Gold58 selection. Complete disclosure of somebody else's .95 recipe is not required for this legitimate experiment. Preserve the evaluation roster and full twelve-target endpoint; describe it as a new whole-recipe challenger. [Released Raptor training code](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model?scriptVersionId=347608187)

**Match the decision, not every internal operation.** A whole-recipe comparison may deliberately change backbone, optimizer, crop and training schedule. Give both recipes the same admitted information boundaries and a declared resource envelope; report their actual training and inference costs. Identical initialization or optimizer states across different architectures are neither possible nor needed. This comparison answers which package to choose, not which component caused the difference.

**What changes the choice:** a credible gain on the permitted endpoint plus acceptable provenance and full-scale offline execution makes the compact model a direct finalist. A tie favors the simpler, cheaper or already-qualified artifact. An unbound author score, an ineligible source, or a worse completed comparison does not justify more backbone variants. The real blocker for an own compact candidate is input/GPU/deployment admission and remaining budget, not lack of the winning author's exact recipe. If that engineering work cannot fit, advance the already-prepared bag or known-scope teacher option instead.

## Experiment 2 Test one image target blend only when its supply chain is ready

**Question:** can image-derived predictions improve the strongest eligible recipe at a reasonable remaining cost?

Use one frozen, admitted teacher whose MRI training, initialization and tuning ancestry respect the student's outer/protected boundary. Preserve the current winning label recipe, fitting roster, masks, images, update schedule and final endpoint. Blend a single predeclared fraction of teacher probability into that recipe's target, keeping total target weight unchanged. Do not simultaneously revert to the original sparse mask, expand the population, replace the pooling head or tune target-specific mixture weights. Study predictions supervise the study bag, not every isolated window.

**The cheapest valid version does not require OOF targets.** A teacher trained only on the existing fitting set can predict those same fitting studies without exposing the outer set. This tests in-sample teacher transfer and may merely reproduce its label errors, but its outer comparison can still be valid. It is different from claiming cross-fitted denoising. The current report-label source's gold-guided design and repeated reuse of the weak outer endpoint remain shared limitations. An all-3,403 model is not a fitting-only teacher for the 2,247/1,156 split. An unknown-roster imported checkpoint does not become safe through fitting-only extraction. [Boundary and cost analysis, including provenance correction](../rsna_reports/rsna_teacher_strategy_red_team_20261014.md)

If a deployable teacher itself is stronger, retain it as a finalist before training a student. If useful known-scope teacher outputs do not already exist, two inner-fold teachers are an optional, more expensive way to obtain OOF targets. Half-data teachers may be weak. Do not turn their construction into an obligatory prerequisite for every transfer experiment.

**What changes the choice:** compare the blended candidate with a compatible completed control, reusing the control where exact recipe equivalence is established. A practical full-macro gain can justify consideration without an additional shuffled-teacher fit. If the candidate does not improve, stop after this fixed contrast. If teacher ancestry, target identity or the remaining delivery budget is unresolved, leave the slot unused. Do not rescue the branch with multiple blend strengths, committees or window losses.

## Cost envelope

Let C be the measured cost of one full fitting-set run of the chosen family, H the total cost of two half-data teachers, I teacher inference, and C_all its final all-development run.

- Direct compact candidate: C plus evaluation; a ready admitted checkpoint may avoid C. Promotion ordinarily adds C_all and full deployment verification.
- Existing known-scope teacher blend: C + I when a compatible control is reusable; 2C + I otherwise.
- Two-fold teacher blend: H + C + I with reusable control, or H + 2C + I without it. Equal epochs and comparable per-image cost suggest H approximately C; equal update budgets can make H approximately 2C.
- A successful OOF recipe may additionally need a fresh all-development OOF table and final student. At equal epochs and identical family costs, that is approximately 2C_all plus extraction, not merely one final fit.

The historical **18.36-hour native fit** makes the two-fold comparison roughly 36.7 hours with an exact reusable control, or 55.1 without, under the equal-epoch assumption; full update budgets increase it. These are native-recipe illustrations, not compact-model or window-bag forecasts. Four-window source bags use up to 24 encoder images per study versus six for the old one-window source recipe; native uses up to 96. Encoder size, input resolution, backpropagation, decoding and hardware still determine wall time. [Measured native reference](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/public-result.json)

Before reserving either experiment, subtract existing commitments, final training, target extraction, full-scale offline inference and recovery allowance from the actual available compute window. A three-example notebook runtime does not establish the nine-hour hidden-run requirement. The final deadline is October 22 at 23:59 UTC. [Official timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)

## Evidence sufficient for a competition decision

Keep all twelve targets and report sparse-support uncertainty, including undefined bootstrap draws. Avoid selecting a recipe solely from a few rare-positive components or from the best point on its outer curve. The reused weak-development cohort is informative but not independent clinical confirmation; public feedback is also subject to selection. Use the best evidence the authorized protocol actually supplies and state its limits.

A favorable matched whole-recipe result does not need to prove its mechanism to be useful. Shuffled-teacher, donor-window and norm-matched controls are required for strong attribution claims; they are not automatic deployment gates. Conversely, a tiny uncertain gain is not made convincing by calling it practical. When evidence is tied or unstable, preserve the better-qualified incumbent rather than spending the remaining budget explaining noise.

## Stop researching these branches for this deadline

- New window-relation committees, uncertainty estimators and PR38 extensions. PR38 is already implemented as a standalone proposal; its teacher and trainer prerequisites remain. The relation proposal adds more targets, admission and optional causal controls before it has an accuracy result. [Current status](research-update-1900.md), [relation proposal](../rsna_window_relation_proposal_20261014.md)
- New physical-extent/severity heads, FAMO/PCGrad, direct AUC loss, new knee pretraining and larger backbones. They add integration or supervision uncertainty without a demonstrated near-term advantage. Generic physical slice-distance auxiliary supervision was already diagnostic-tested; it should not be repitched as new. Continuity readout is distinct but does not earn a final-week full fit.
- Automatic source-confidence weighting. Distance from a blended .5 is not calibrated correctness; cheap implementation is not evidence of competitive value. Retain its aggregate diagnostic, but do not displace the two higher-value options without new evidence. [Weighting analysis](multitask-conflict-strategy.md)
- Broad 501-row expansion, fresh anatomy selectors, repeated averaging and more generic literature. Preserve their audit findings for later. Do not expand protected boundaries or treat a bundled negative result as a universal mechanism failure.

**Final choice:** keep the strongest completed admissible single model, preserve a direct compact deployment option, and buy at most one additional teacher-transfer contrast when its inputs and total delivery cost are already credible.

## Selection-policy clarification, 14 October 20:15 UTC

The later [score-gap audit](../rsna-score-gap-selection-challenge-2026-10-14.md) verifies a closely matched .783→.854 public supervision contrast and existing disagreements between weak-development and public rankings. This does not identify the cause of the remaining gap, make the leaderboard independent validation, or elevate another same-source teacher by default. It does require qualifying the earlier promotion language: a weak-only gain is evidence about that report-observed endpoint, not established competition superiority.

Before any future fit, predeclare whether one fixed downstream competition comparison is separately authorized and affordable. An inconclusive or sparse-target-driven weak result need not automatically veto that already-planned comparison; it does not authorize a new submission, endpoint alteration, protected-cohort access or a sweep. Without such a route, retain the qualified incumbent and label the compact comparison exploratory. Clear broad regression or failed deployment qualification remains a practical stop. No existing preregistration is retroactively changed.

The [supervision mathematics review](../rsna_reports/rsna_supervision_ranking_limits_20261014.md) separately shows that confidence weighting can reverse rankings even for calibrated probabilities, and ideal same-source distillation preserves the original population target. Finite-model pseudo-label gains remain plausible. Neither the source-label AUC nor those idealized results establish a numeric MRI performance ceiling.
