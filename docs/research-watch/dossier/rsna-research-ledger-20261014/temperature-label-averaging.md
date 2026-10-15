# Temperature sampling and repeated report-label extraction

Research-only recommendation, 2026-10-14. No report rows, labels, images, weights, or secrets were opened; no labels generated, training run, repository edit, external data transmission, or submission performed. Source context: the existing `label-provenance.md` audit in this directory, read in full.

## Recommendation

**Worth a controlled pilot if authorized: modest stochastic sampling plus aggregation can reduce unstable extraction errors. Increasing temperature alone is not an improvement, and two agreeing generations are not independent clinical evidence.** Prefer explicit structured report states and evidence spans; use repeat-pass disagreement to identify uncertain extractions. Test several samples from one good extractor against similarly budgeted genuinely distinct strong extractors rather than assuming one must win. Two draws can reveal disagreement; four is a reasonable small pilot size, not a literature-established optimum.

No suitable unexposed clinical reference has been established as available. Current weak-validation targets are themselves v4; better agreement with v4 is not independent clinical validation. Independent adjudication of what a report actually states can validate extraction faithfulness, but report omission and report–MRI mismatch remain, so that adjudication is not an MRI clinical-truth endpoint.

This is not a novel recommendation to “blend two labelers”: the active v4 table already blends Steven v2 and lixin73 GPT-5.6-Sol report-label probabilities. The audited publisher description does not disclose exact mixing weights, reproducible parent-generation settings, or all missing-cell rules. Both use the same reports; distinct authors/models do not establish independent errors or ground truth. The official 58 cases guided/evaluated upstream choices and must not become the tuning or independent-confirmation set for this proposal. [Pinned source audit](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/README.md) and [publisher card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)

## What the evidence supports

- Wang et al. show that sampling diverse reasoning paths and aggregating their final categorical answers improves arithmetic/common-sense benchmarks. Main experiments used 40 samples; temperatures varied by model (including 0.5 and 0.7). This supports an aggregation experiment, not a guarantee for clinical report extraction, arbitrary temperature increases, or two-sample gains. [Self-consistency paper](https://arxiv.org/html/2203.11171v4)
- Farquhar et al. estimate uncertainty over meanings rather than surface wording. Their method detects inconsistent confabulations but explicitly does not fix systematically wrong outputs. Most QA experiments used ten generations, with source passages deliberately withheld; direct transfer to source-grounded report extraction is unproven. A structured positive/negative/uncertain/not-addressed schema already gives useful semantic categories without a separate open-text clustering model. [Semantic entropy paper](https://www.nature.com/articles/s41586-024-07421-0)
- Soft-label training need not round the final aggregate: for a common binary target definition, identical masks, and fixed example weights, the arithmetic average of binary cross-entropies equals cross-entropy against the arithmetic mean target. This algebra says what objective is optimized; it does not validate the target as truth or make its numbers calibrated.

## Why correlation matters

For exchangeable errors with common variance sigma² and pairwise correlation rho:

Var(mean error) = sigma² × [rho + (1 − rho)/n]

With rho = 0.8, two runs retain 90% of the single-run error variance; four retain 85%. Even infinitely many retain 80%. A shared systematic bias remains. These are illustrative assumptions, not measured values for our labels. Independent random seeds can produce conditionally independent token draws while preserving strongly correlated case-level errors because the same model, report omissions, prompt, and target misunderstandings are shared. New models/prompts can reduce some shared errors, but poorer individual accuracy can erase that benefit.

The meaningful diversity is different correctable mistakes, not simply more different text or numbers. Estimate error complementarity against an authorized reference; disagreement alone cannot identify which extractor is right. Temperature may raise diversity by degrading an otherwise clear extraction.

## Three different quantities

1. **Vote fraction:** fraction of sampled outputs declaring a report-state positive. Under independent draws it estimates that decoder's positive-output frequency at the chosen prompt/temperature, not the probability the MRI finding is clinically present. At n = 2 it has only 0, 0.5, and 1 as values; at a true decoder frequency of 0.5 its independent-sampling standard error is about 0.354.
2. **Mean supplied soft number:** average of elicited scores/probabilities. This preserves score strength but inherits each model's interpretation, calibration errors, and missingness conventions. For example, scores 0.6 and 0.8 average to 0.7, while thresholding both at 0.5 gives a positive vote fraction of 1. These are different estimands, not interchangeable aggregations.
3. **Clinical probability:** requires an appropriate clinical endpoint and validation/calibration against relevant reference outcomes. Neither numerical averaging nor unanimous extraction establishes it.

Keep not-addressed separate from explicitly uncertain, negative, and positive. Preserve per-pass state, evidence span, and any explicit probability separately. Do not translate silence into a negative or automatically weight by distance from 0.5. In the existing blended CSV, 0.5 cannot reliably reconstruct the original not-addressed mask; the provenance audit documents that ambiguity.

## Predeclared experiment, only after authorization

- Freeze cohort eligibility, source versions, target definitions, prompt/schema, budget, aggregation/masking rules, and primary metrics before looking at outcomes. Use only an approved unexposed development reference. Do not access protected confirmation cohorts or reuse the official 58 for model/prompt/temperature/blend selection. If no suitable reference exists, the work can measure extraction stability, not establish correctness gains.
- Keep the current v4 blend as the operational baseline and a fresh low-temperature single extraction as a methodological baseline. Compare repeat low-temperature extraction, modest-temperature same-model extraction, and a distinct-strong-extractor ensemble at the same total generation budget. Match total tokens/cost or measured compute, not merely request count; disclose differences in model capabilities/costs. For a small illustrative four-call setup, compare 4 draws from one model with 2 draws each from 2 models. Temperature values must be supported by the selected model and fixed before evaluation; 0.4–0.7 is only a plausible pilot range where supported, not a universal optimum.
- Retain all per-pass outputs privately. Measure explicit-state extraction errors, negation/uncertainty/not-addressed mistakes, unsupported assertions, coverage and abstentions. Compare fixed-state vote aggregation with fixed-semantics soft-score averaging. Report per-target results and paired, leakage-component/group-aware uncertainty; do not count repeated samples as extra independent studies.
- Only if both budget and authorization permit, propose one predeclared downstream comparison; otherwise defer. Keep image data, masks/weighting policy, split, architecture, training budget and seeds matched so label changes are the only intended difference. Never tune by repeated protected/outer peeks. A meaningful extraction gain requires improved independently adjudicated report-state accuracy. A downstream gain needs a predeclared held-out endpoint with its reference limitations stated: scoring against v4 can show agreement with the existing weak target but cannot establish better clinical truth. Smoother labels or agreement with the same ancestor table are not independent evidence.

## If “temperature” means the MRI classifier

For a fixed binary classifier output, p_T = sigmoid(z/T) with a fixed positive T changes confidence, not the underlying prediction randomness. Running the same deterministic network again gives the same output. It is strictly monotonic per target, so a single fixed positive temperature preserves that target's AUC (apart from numerical ties), and leaves the 0.5 decision threshold unchanged. Select calibration temperature using a separate allowed calibration partition, not a protected cohort. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)

Actual MRI prediction diversity requires independently trained/checkpoint models, justified test-time augmentations, or explicitly stochastic inference such as MC dropout. Those are different interventions with different costs and assumptions; turning dropout on arbitrarily is not equivalent to a validated uncertainty method. [Deep ensembles](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) and [MC-dropout framework](https://proceedings.mlr.press/v48/gal16.html)

Bottom line: test repeat extraction, but spend the extra calls on identifiable evidence-backed error reduction. Same-model sampling offers a cheap instability signal; different strong labelers may offer complementary errors. Neither substitutes for an unexposed reference, and the already-blended v4 must remain the comparison to beat.
