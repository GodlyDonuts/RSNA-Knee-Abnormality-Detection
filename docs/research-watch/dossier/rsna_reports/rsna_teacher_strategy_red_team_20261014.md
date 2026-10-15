# RSNA image teacher strategy and validation audit

14 October 2026

## Recommendation

Test whether image-derived soft targets improve a single student, but keep that question separate from backbone replacement and expansion of the training population. The smallest complete cross-fitted experiment on the current fold needs two image teachers and two matched students. A cheaper existing-teacher experiment can still have a valid outer evaluation, but it answers an in-sample distillation question rather than an out-of-fold label-denoising question.

Do not describe an experiment as fully leakage-clean merely because its newly trained teachers exclude the outer fold. The ancestry of their initial weights and training targets also matters. The current public soft-label table has verified bytes but incompletely established generation history.

The current verified public best is 0.872, reached by two checkpoints of the same 3,403-study native384/public-supervision model. The older 0.899784 weak-development result belongs to a different 2,247-study fit that scored 0.854 publicly. Those numbers are different estimands and cannot identify one bottleneck by subtraction. [Verified checkpoint curve](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/daily-kaggle-comparison-queue-20261014-v1/all-five-result-analysis.md)

Repository evidence below is pinned to commit `8300cfd20c2acaeef9d684a81dbc7fe977282a62`. This report proposes experiments; it does not report new training, inference or submission results.

## What the new competitor claims do and do not establish

The discussion being reviewed contains creator reports of approximately 0.953 from a ConvNeXt-T and 0.951 from a DINOv2-S using image pseudo-labels without external data. A CoAtNet author reports benefit from soft out-of-fold blending and negative results from hard replacement or masking. These are author-reported pipeline outcomes, not reproduced controlled comparisons. Exact checkpoints, training memberships, target-generation histories and scored submission bindings remain necessary to audit them. [Original discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)

The plausible lesson is broader than adopting CoAtNet: several smaller backbones may become competitive when given a suitable image pipeline, adaptation recipe and supervision. The claims do not prove that DINOv3 is inadequate, that a particular architecture is superior under our conditions, or that the entire gain comes from pseudo-labeling. They also do not establish that external MRI data are necessary.

Our own completed ablations already warn against hard conclusions about labels. Full public supervision improved weak-development macro AUC from 0.878671 to 0.899784, but public values under the original masks scored 0.853681, while filling only previously masked cells scored 0.869302. Neither isolated follow-up reproduced the joint intervention. Coverage-only uncertainty spans zero. [Completed source experiments](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/docs/research-handoff-20261012-corrected.md)

Moreover, 72.89% of the full-public macro gain came from MCL, Baker's and fracture, supported by only two, three and two positive outer studies. This is a fragility diagnostic, not permission to remove these targets. [Exact aggregate decomposition](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/macro-support-diagnostic.json)

## What the current public labels actually are

The current fitting overlay uses `llm_labels_v4_blend.csv` from Shio-Koube/knee3, revision `373f4e1daf36984fa48bf3fc8fd9883fe17cb584`, CSV SHA256 `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`. The pinned project record says that revision has no dataset card or license metadata. The filename does not establish identity with another author's same-named table. It also does not establish whether the values originate only from reports, include image-model predictions, use out-of-fold models, or underwent gold-guided selection. Those questions remain unresolved. [Pinned source admission](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/experiments/competitive-v1/public-label-source-v1/README.md)

The implementation replaces all twelve fitting values and masks on the existing fitting roster. Every present public source row receives twelve observed masks; it is not an additional image-teacher objective and does not retain original masks inside that arm. The original accepted release and outer evaluation masks remain unchanged. [Overlay builder](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/scripts/research/build_label_source_overlay.py) · [Bound overlay loader](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/label_source.py)

A separate Pilkwang image-teacher evidence utility exists, but current canonical training does not use it. Its teacher training roster is explicitly unknown. Its confidence heuristic subtracts across-window probability spread from decisiveness; different anatomical windows are not repeated measurements of the same evidence. That heuristic should not be used to infer clinical confidence or prove pseudo-label reliability. [Teacher evidence implementation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/teacher_evidence.py)

## Smallest complete nested teacher and student experiment

### Freeze the question

Use the existing fold0 fitting population T of 2,247 studies and its 1,156-study outer development pool V. Preserve leakage components, accepted source identities and all protected boundaries. Choose one teacher recipe, one student recipe, one fixed training endpoint and one soft-target weight before looking at the experiment's outer results. Do not combine this contrast with a new backbone, cohort expansion, new input crop or patch-focus module.

The teacher's initial supervision must have established ancestry. Verified report-only supervision is the clean starting point. If the current public table is retained despite unresolved ancestry, describe the experiment as excluding V at the new-teacher stage, with upstream independence unresolved. New cross-fitting cannot erase contamination already present in teacher targets or initialization.

### Generate fitting targets with two teachers

1. Partition T into two approximately equal inner groups by complete leakage component, retaining the frozen outer boundary. Check target support; do not silently break components or invent labels to balance them.
2. Train teacher A on inner group A only and teacher B on inner group B only. Use fixed endpoints. Neither teacher may use its opposite group for early stopping, target selection, calibration or hyperparameter choice if the goal is strictly out-of-training target generation.
3. Teacher A predicts group B; teacher B predicts group A. Each fitting study receives one cross-fitted, twelve-target, whole-study soft prediction from a teacher that did not train on that study's component.
4. Store raw pre-rank outputs, exact teacher and renderer hashes, training-component digests and missing-view/fallback counters. Do not convert cohort percentile ranks into BCE probabilities.

This is the minimum complete OOF table for T: two teacher fits. Each teacher sees only about 1,123 or 1,124 studies before component imbalance. Their strength may be substantially below a teacher trained on all 2,247. A negative student result could therefore mean weak half-data teachers, not that image pseudo-labels are useless.

### Compare two students

Train a baseline and a soft-target candidate from the same initialized state with identical images, augmentation, study order, update schedule, optimizer, precision and final endpoint. Keep the base supervision unchanged. Add one predeclared bounded soft-teacher contribution in the candidate; no per-target tuning or hard replacement selected on V.

Evaluate both once on the same V masks, report all twelve targets, paired whole-component uncertainty, undefined resamples and rare-component sensitivity. An existing baseline may be reused only after exact compatibility of initialization, input exposure, schedule, normalization and final artifact is established. A superficially similar older run is not automatically a control.

Use teacher targets at the study-bag level. A teacher that saw the complete study cannot legitimately teach every randomly selected window as if each contained the lesion. The existing mean/query bag implementation addresses this loss-unit problem, although its four training windows versus ten inference windows remain a separate input-distribution question. [Whole-bag implementation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/window_bag.py)

### Do not confuse loss names with new information

For the same student logit, two weighted BCE terms are mathematically equivalent to one BCE against their appropriately weighted blended target. With different masks and per-target normalizers, the effective blending weight varies by cell and batch. Calling one term an auxiliary loss does not make the supervision independent or create a novel mechanism. The real intervention is target ancestry, target values, observation weights and the image evidence available to the teacher and student. [Actual normalization](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/loss.py)

## Cost and scheduling

Let F be the number of outer folds, K the inner teacher folds, n_f the fitting population for outer fold f, E_t and E_s the teacher and student epochs, c_t and c_s their measured compute per study exposure, and I_t(n) the cost of teacher inference over n studies.

With two student arms, total fits are F K teachers plus 2 F students. Ignoring unequal component sizes and fixed overhead, training and target-generation cost is approximately:

`sum_f [ (K - 1) n_f E_t c_t + 2 n_f E_s c_s + I_t(n_f) ]`

Each study is predicted once for its OOF target, so inference is n_f studies per outer experiment, not K n_f, unless additional views or an ensemble are explicitly requested. Repeated decoding, startup, validation and checkpoint preservation add costs.

For one outer fold and two teachers, equal epochs and equal per-image costs imply roughly three full-2,247-study training equivalents: one equivalent across the two half-data teachers and two students. The prior full-public run took about 18.36 hours including its recorded run work, so 55 hours of GPU time is a rough planning anchor, not a measured forecast. Parallel execution does not remove that GPU demand, memory/storage requirements or recovery work. [Measured prior run](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/public-result.json)

If each half-data teacher instead receives the full original update count, teacher work is approximately two equivalents and total work approximately four, or roughly 73 hours at that same reference rate. That is a different training-budget design. Warmup and cosine schedules must be specified accordingly; fixed epochs and fixed updates cannot be conflated.

For three outer folds and two inner folds, there are six teachers and six students. Across an N-study three-fold development set, total fitting membership is 2N; equal-epoch training exposure is approximately 6N study-epochs. With three inner folds it becomes nine teachers, six students and approximately 8N study-epochs. These totals exclude any teacher tuning, repetitions and final all-development retraining.

If the final recipe succeeds, a fresh final OOF target table on all 3,403 development studies requires K final teachers plus a final student, unless a compatible, correctly excluded existing table is available. Do not average outer-experiment pseudo-label tables casually: their training scopes and teacher strengths differ.

## Cheaper alternatives and their limits

### Reuse the completed four arm experiment first

The four existing supervision endpoints form a useful recipe-level two-by-two design. Define factor A as replacing public values on cells that were originally observed, and factor B as adding public supervision on cells that were originally masked. The original arm has neither factor, value-only has A, coverage-only has B, and full-public has both. There were 13,760 originally observed fitting cells and 13,204 newly observed cells in the complete 26,964-cell table.

The fixed-mask and coverage admission records have the same configuration as the full-public run after removing the fitting-supervision artifact field: seed 20260922, ten epochs, 2,810 updates, warmup 281, original initialization/RNG, input roster and scientific source a9902f3. The two follow-up source-file hash maps also agree. [Fixed-mask admission](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-fixedmask-production-20261010-v1/CPU-accepted.json) · [Coverage admission](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-coverage-production-20261011-v1/CPU-accepted.json)

The descriptive macro-AUC interaction is:

`full_public - values_only - coverage_only + original = 0.055471067159061915`

This is post-hoc endpoint arithmetic, not a significance result or a clean biological effect of missingness. Factor B also changes target activation and per-target loss denominators; the resulting interaction includes optimization and multi-task effects. AUC is a nonlinear ranking endpoint.

A cheap next analysis could apply the same whole-component bootstrap multiplicities jointly to all four existing prediction vectors, compute the interaction within every draw, retain undefined draws, and decompose it by target. Do not add the separate pairwise confidence interval bounds. Such an analysis needs no new training and no protected rows, but would remain conditional on this one seed, development set and already-selected set of experiments. No archived numerical interaction analysis was found in the sources searched for this report.

| Alternative | New fits for one outer experiment | What it can answer | Main limitation |
|---|---:|---|---|
| Aggregate source and gradient audit | 0 | Whether targets, masks and gradients conflict under unchanged inputs | Does not establish clinical truth or an accuracy gain |
| Existing known-scope teacher on all T plus two students | 2 students | Whether in-sample distillation improves outer performance | Teacher targets are in-sample; not a test of OOF denoising |
| One-way split: teacher on A, two students on B | 1 teacher plus 2 half-data students | Cheap out-of-training transfer signal | Student sees only half T; asymmetric and not a full-cohort result |
| Two-fold complete OOF table plus two students | 2 teachers plus 2 students | Minimal complete cross-fitted target experiment | Half-data teacher strength may be limiting |
| Unknown-membership public teacher plus two students | 2 students | Exploratory transfer and eventual hidden-test performance | Outer independence cannot be claimed until teacher ancestry is established |

For the existing-teacher route, a known teacher trained only on T can produce in-sample pseudo-labels without contaminating V. OOF generation is not logically required for V to remain unseen; it is required for a claim that fitting pseudo-labels come from teachers that did not train on those fitting cases. These are different guarantees.

## Does a strong public teacher make independent validation impossible

Not in principle. If its exact training and tuning ancestry proves exclusion of V and every protected component, V can remain independent of that teacher's fitting. If membership is unknown or includes V, filtering the student to T does not repair the problem: knowledge from V can pass through the teacher's outputs on T.

An external public teacher may still yield a useful competitive student. A hidden competition test can assess that deployed pipeline, subject to eligibility and the normal limits of repeated leaderboard selection. It does not retrospectively make a contaminated local validation clean. Under the present prohibition on opening new protected validation, no new independent clinical truth can be manufactured by teacher consensus.

Global OOF is also insufficient for nested evaluation. A teacher that predicts one training row out of fold may nevertheless have trained on the student's outer fold. Generate targets inside each outer training population, or provide exact membership evidence showing the stronger exclusion. [Existing teacher-boundary protocol](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-teacher-notebooks-v1/README.md)

## Challenge the 501 row expansion hypothesis

The 4,349-versus-3,403 gap is exactly 441 reserved confirmation studies, four quarantined studies and 501 accepted training-split rows with all original masks zero. Current selection removes those 501 before the public overlay. They are not all documented correspondence failures, and their exact reason partition has not been established from the available aggregates. [Release population](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/docs/data-readiness-report.md) · [Selector](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/train.py#L471-L493)

Even admitting all 501 would add only 14.7% relative to 3,403 and would reach 3,904, still short of 4,349. More importantly, selection may be correlated with report completeness, pathology, language, scanner protocol or unresolved examination identity. Their value is not established by count. Some may add useful image diversity; some may be unsuitable single-knee units. Safe image-only admission is not equivalent to permission to trust their report labels or public pseudo-labels.

The first useful deliverable is an aggregate reason and technical-eligibility partition, not training. A later admitted-subset experiment must keep protected boundaries and compare against a compute-matched control. Avoid using the large .854-to-.872 population change as a precise learning curve: checkpoint age and selection differ.

## Challenge backbone replacement as the explanation

Strong reports from ConvNeXt-T, DINOv2-S and CoAtNet weaken a claim that a uniquely powerful backbone is required. They do not prove interchangeability. Inputs, crop, depth coverage, adaptation scope, optimizer, supervision and data population may all differ.

Our source-slot result of 0.878287 versus native-public 0.899784 is a bundled negative result, not a pure backbone or pooling test. The dense-adaptation run was cancelled and has no completed accuracy outcome. A new-backbone package additionally needs its own input/weight mapping and actual runtime admission; implementation success would not resolve its scientific value. [Source-slot outcome](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/source-slot-fixed-final-20261014-v1/README.md) · [Dense cancellation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/training-preservation-cancellation-20261014-v1/README.md)

Prioritize a matched supervision or adaptation test before changing backbone, cohort and pooling together. If a smaller backbone is chosen for speed, report that as a practical package comparison with measured quality and cost, not evidence that its architecture alone solved the problem.

## Decision gates

1. Recover the completed second-seed supervision result before commissioning another label-source full fit. Preserve an unavailable result as unavailable.
2. Require explicit teacher weight and label ancestry, student-outer exclusion, and source eligibility before calling a result clean. Unknown ancestry can support an exploratory track only with that limitation visible.
3. Use a cheap diagnostic or one-way split to establish that the proposed teacher has useful signal before paying for complete nested teacher training. A weak half-data teacher is a stopping reason for that teacher recipe, not for all distillation.
4. Run one fixed soft-target contrast. Keep the inference model, inputs and population unchanged. Do not choose target-wise mixtures or weights on the same outer results.
5. Promote only after repeatable supported-target behavior and the unchanged full twelve-target endpoint. Report rare-target fragility and model-search uncertainty; conditional bootstrap intervals are not independent confirmation.
6. Retain time for final all-development training and actual offline deployment. The official final submission deadline is 22 October 2026 at 23:59 UTC, with a nine-hour inference limit. A nested protocol that cannot finish and deploy before then is a research plan rather than a competition delivery plan. [Official timeline and code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/)

## Provenance correction added 14 October 2026 at 1844 UTC

This addendum supersedes the earlier blanket description of the current public-label source as having unresolved image-versus-report ancestry. Direct inspection of the pinned September 29 audit establishes that the 630,153-byte Shio-Koube `llm_labels_v4_blend.csv` is byte-identical to `stevenleehans/rsna-knee-llm-report-labels` version 6, SHA256 `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`. The audit records declared CC0 licensing, parent sources described as the author's v2 report extraction and lixin73's GPT-5.6-Sol extraction, gold-guided changes, and an effusion-derived proxy for missing synovitis. These are frozen provenance receipts and recorded publisher declarations; this correction does not claim a fresh reproduction of the original publisher metadata. [Exact source receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/sources.json) · [Audit interpretation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/README.md)

The resulting validation limitation is specific: gold-guided source design means the 58 gold studies cannot be described as untouched confirmation for models using this blend, even when their rows are excluded from local fitting. This is distinct from an image teacher having trained on the student's outer images. The available evidence points to report-LLM label generation and does not establish that image-training contamination pathway. Full parent-level reproducibility remains incomplete, but an unknown image-teacher roster should not be presented as an established defect of this table.

The proposed nested image-teacher protocol still prevents new teacher fitting from crossing the student's outer boundary. With the public report blend as supervision, its outer comparison can assess transfer against the existing weak report-derived endpoint, subject to historical model-selection and label-semantic limitations. It cannot restore untouched-gold independence or turn report-label agreement into clinical truth. The source's synovitis proxy also gives a concrete supervision-alignment hypothesis worth a fitting-only diagnostic; it does not itself prove that the proxy harms hidden-test performance. The earlier generic warnings about unknown public image teachers remain applicable to genuinely unknown model weights, but should not be attributed to this report-label blend without additional evidence.
