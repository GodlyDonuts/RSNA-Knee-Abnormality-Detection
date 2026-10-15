# Historical evidence supplement — October 14, 2026

Pinned snapshot: 8300cfd20c2acaeef9d684a81dbc7fe977282a62. This supplement extends the separately delivered ledger without modifying its delivered version. Read-only aggregate documentation/code searches; no case records, models, training, submissions or GitHub writes.

## Coverage delta

Nine additional full audit READMEs were inspected: ranking-supervision-exposure-v1; public-report-supervision-diagnostic-v1; effusion-focused-reliability-v1; public-label-fixedmask-20261010-v1; public-label-coverage-production-20261011-v1; dinov3-patch-attention-epochs-v1; shared-control-final-20261008-v1; fitting-series-coverage-20261007-v1; public-recipe-gap-review-20261008-v1. Targeted default-branch searches for complementary selection and shared controls returned the same pinned snapshot. This adds one explicitly quantified completed fold to an already-covered scientific family, two consequential diagnostics, exact intervention definitions and completed final-control equivalence. It does not discover a new winning model or establish that every historic audit has been read.

## 1. Missing completed patch-attention fold0 result

Job845218 completed ten epochs /2,810 updates on2,247 fitting studies, one H100/two CPUs, with final macro **0.7451169387** on1,156 development studies versus frozen CLS **0.7783102649**: delta **−0.0331933262**. MCL fell from0.5973 to0.2162. All ten epochs were privately preserved. No patch-attention checkpoint was submitted to Kaggle.

This fills the first ledger's missing explicit fold0 number. Combined with fold1's modest improvement and fold2's regression, it strengthens rejection of that particular frozen replacement head. It does not test the current identity-preserving residual patch-focus branch and does not reject all spatial information.

[Original fold0 receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/dinov3-patch-attention-epochs-v1/README.md)

## 2. Final common-control equivalence is actually complete

Original CPU882567 compared the two privately verified ten-epoch controls. Exactly equal:
- All258 common learned tensors, frozen proofs and encoder provenance
- Named common head and ordered identical-encoder AdamW states
- AMP scaler, all RNG state and progress counters
- All2,810 six-field numerical update records
- Complete final1,156-study prediction CSV bytes

The only graph addition is131,072 all-zero patch-summary parameters; optimizer state is absent or zero. Whole graph/binding serialization still differs. This closes the earlier epoch5/early-log-only evidence gap.

Consequence: for a prospective compatible contrast, establish exact source/default-input initialization, reference/recipe compatibility and actual GPU recovery, then reuse the accepted reference rather than automatically spend another ten-epoch fit. It is not permission to substitute this control for source-slot, a new bag-training exposure, changed labels, or changed runtime without those checks.

[Final proof](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/shared-control-final-20261008-v1/README.md)

## 3. Complementary series selection remains a distinct unanswered question

The original audit replayed the frozen selector on12,476 eligible series from2,247 fitting studies.214 studies omit an available non-fluid category:103 sagittal,131 coronal,20 both. All omitted categories have a tier0 candidate.924 selected bags repeat plane/fluid/fat-suppression categories, but these can be genuinely distinct informative images. All available three fluid-sensitive fat-suppressed plane categories are retained.

A new same-budget selector reserving complementary categories before filling six slots was proposed, preserving16 slices/384pixels and effectivebatch8. The audit explicitly says no selector integrated or training dispatched. Targeted repository search found the proposal and later bundled public-source input work, but no isolated completed selector contrast. Absence from these searches is not proof of absence in every historical branch.

The later source-slot negative changes crop, input resolution, windowing, augmentation, head and sampling together. It therefore does not answer the isolated selector question. Before testing, resolve descriptor semantics: the table has no T1 field and no sagittal fluid/non-fat-suppressed category. Non-fluid is a proxy, not proof of T1 or equivalence to public slots.

[Coverage receipt and proposed control](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/fitting-series-coverage-20261007-v1/README.md)

## 4. Exact failed supervision interventions do not rule out image-teacher blending

Fixed-mask ablation retained exactly2,247 fitting studies,13,760 observed cells and original masks. It **replaced**13,750 observed values with pinned Shio-Koube/knee3 probabilities; masked numerical values normalized to0 while masks stayed0. It was not an interpolation of accepted labels and image-teacher OOF predictions.

Coverage-only retained the original observed values and used pinned public soft targets only in formerly masked fitting cells. It was not replacement of observed labels and was not a tested learned confidence mixture.

Full-public changes both values and coverage; its observed internal improvement and the negative/inconclusive ablations leave a possible interaction unproven. None of these tests constitutes a failure of fold-safe image-teacher soft blending, ranking transfer or same-window-aligned teacher supervision. Those still require teacher ancestry, nested fold exclusion, exact input/loss and compute-matched controls. Existing ordinary/confidence-weighted teacher and committee protocols must be reviewed rather than re-proposed as new.

[Fixed-mask definition](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-fixedmask-20261010-v1/README.md) · [Coverage-only definition](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-coverage-production-20261011-v1/README.md)

## 5. Three public label sources must not be conflated

The earlier training-only diagnostic used Pilkwang's rsna-knee-llm-labels dataset ID11529209/version1, markedCC0, with YES/NO/UNK verdicts. Its author described scores as rank-usable and uncalibrated. The provided script imported an absent llm_labeler.to_scores implementation, so the exact numerical mapping was not independently reproduced. This is distinct from the current Shio-Koube/knee3 overlay.

Metadata for Steven's and JEV-Knee labels described blending/target choices tuned on58 or74 gold-like reports; those were not admitted in this diagnostic. Neither same filenames nor source-page claims establish identical ancestry or clean protected-cohort use.

Measured fitting-only disagreement examples are not population error estimates:739 observed Effusion negatives received publicYES/0.68, plausibly reflecting mild rather than moderate/large effusion;197 MCL negatives receivedYES/0.68. In the exact fitting cohort272 original-positive Effusion annotations were masked and280 cells carried incomplete-reliability holds; these overlap and must not be summed. All-development count422 masked-positive Effusion annotations is a different scope.

Independent report rereading agreed on8/9 proposed diagnostic states, with one fracture acuity disagreement retained. A narrow follow-up retained one ambiguous Effusion hold rather than treating multiple historical agreements as clinical truth. The subsequent31-cell intervention already has completed model results in the main ledger. These diagnostic samples support severity/missingness questions without proving erroneous accepted labels.

[Diagnostic and source distinctions](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-report-supervision-diagnostic-v1/README.md) · [Focused reliability process](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/effusion-focused-reliability-v1/README.md)

## 6. Ranking exposure was measured before implementing a ranking loss

The actual trainer forwards one study at a time and accumulates eight microbatches. A pairwise same-target term inside each ordinary forward has **zero** positive/negative study pairs for every target. Accumulation alone does not build a coupled ranking graph.

Even if comparisons are assembled across the effective eight-study update, the original strict labels supply cross-component positive/negative pairs in:
- MCL15/843 updates (1.78%)
- Fracture5–7/843 (0.59–0.83%)
- Baker17–18/843 (2.02–2.14%)
- Synovitis16–22/843 (1.90–2.61%), limited by negatives
- Medial meniscus820–826/843

Counts were replayed at seeds20260922/20261004, with exact sampler/masks/rosters and no protected targets, and checked against843 actual GPU observed-cell counters. This is an execution/support diagnostic, not a failed ranking-training run. A new teacher-order objective may supply different pair support, but it still needs fresh differentiable paired outputs or correctly replayed gradient caching, explicit compute accounting and stable teacher-order selection. Do not fix zero-comparison support by arbitrarily increasing the loss coefficient or inventing rare negatives.

[Original measured exposure](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/ranking-supervision-exposure-v1/README.md)

## 7. Recipe differences predate the latest experiments

The October8 source review already documented original DINOv2-small last-six-block training, backbone/headLR8e-6/1e-3,30epochs/cycle10,130mm crop/robust contrast/central neighboring channels, versus current all-block adapters1e-4/3e-4 and ten fixed epochs. The source text also described image-annotated-study weighting. These were observational differences, not a controlled demonstration of a capacity bottleneck. The failed native mean-patch branch did not reproduce the public full input recipe.

[Original recipe-gap review](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-recipe-gap-review-20261008-v1/README.md)

## Practical conclusion

The meaningful coverage changes are: complete fold0 patch-head failure; genuine final-state shared-control reuse; still-untested same-budget complementary selection; and a sharp distinction between failed public-value replacement and not-yet-tested fold-safe image-teacher blending. These change what should be considered new. They do not justify fresh GPU work or label admission by themselves.

