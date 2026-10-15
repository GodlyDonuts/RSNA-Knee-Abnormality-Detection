# RSNA knee MRI strategy and experiment decision tree

Research draft dated 14 October 2026. This is a proposal for a single deployed MRI backbone and checkpoint. It does not authorize training, data acquisition, spending, submissions, or changes to existing experiments.

## Recommendation

Prioritize an auditable test of image-derived soft supervision and sufficient model adaptation. Keep one smaller DINO or convolutional model as a matched challenger, rather than treating a family switch as the explanation for the score gap. Finish interpreting the dense adaptation, whole-bag, and patch-focus work already underway before starting another version of those ideas.

The strongest new question is: **Does independently generated, fold-safe image evidence improve the exact current supervision pipeline, beyond generic target softening, gradient rescaling, and seeing more MRI?** A recovered source audit and live source card identify the current overlay as a Steven Lee report-label blend, including gold-guided adjustments and an effusion-derived fill for undecided synovitis. The named parents are report extractions; no image-teacher parent is documented. Exact reconstruction of every blend/proxy coefficient remains incomplete. The proposed experiment adds traceable image evidence to this known report-derived supervision pipeline.

My earlier preference for a CoAtNet challenger as the leading next move was too strong. Fresh creator reports show competitive single DINO and convolutional models using similar image-pseudo-label pipelines. This raises the priority of supervision and adaptation while leaving architecture as a practical control. None of these reports guarantees that the method will transfer to this project.

## Evidence that changes the diagnosis

### Relevant creator reports

These are first-party comments inspected in the live competition discussion, not independent reproductions or checkpoint-level verification.

- Colum2131 reports single ConvNeXt-T 2.5D at .953 and DINOv2-S 2.5D at .951, both trained with image pseudo-labels and no external data. A follow-up describes common preprocessing, head, augmentation, and pseudo-labels, with backbone-specific learning rate and epochs. [Results](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3535687), [matched recipe description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3535841)
- Archit reports a .950 single-fold, 224-pixel CoAtNet. He describes OOF image predictions, soft blending with report labels, and gains from increasing image-label influence; hard replacement and hardening performed worse. His discussion includes silent/report-negative synovitis. [Model](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3525138), [OOF description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3525193), [soft targets](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3525701), [synovitis](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3526342)
- Nicolai reports that a 50/50 blend improved gold and original weak-label validation but did not improve the public leaderboard. [Null result](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3529062)

The comments do not establish exact teacher counts, complete training ancestry, final blending weights, all inference-time operations, or an uncontaminated selection protocol. Archit's use of the 58 gold cases for diagnosis/calibration is incompatible with treating those cases as this project's protected model-selection set. Copy the testable mechanism, not that selection procedure.

### Correct the input parity claim

The inspectable nartaa .949 V2 recipe selects 96 source slices in total, allocated as 26 sagittal-fluid, 22 sagittal-other, 18 coronal-fluid, 12 coronal-other, and 18 axial slices. The current native model also nominally selects up to 96, through six sequences times 16 slices. Thus a generic claim that the competitor simply sees more unique anatomy is unsupported. Missing series and repeated selected images can reduce actual unique coverage in either pipeline.

The concrete differences are allocation, stack span, physical crop, neighbor-channel encoding, training views, adaptation, supervision, and pooling. Nartaa's actual source samples 2–98% of the stack, trains on 12 random triplets at 384, and evaluates 94 overlapping triplets at a native 320 center crop with two anatomical views. That is 188 encoder images, not 188 unique slices. Its triplet centers run through the concatenated slots, so some windows cross series boundaries. The .949 model is one SWA inference state with mirror inference, not one raw epoch and one pass. [Pinned scored notebook](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397)

### What our existing experiments actually establish

The current verified solo public score is .872, superseding the earlier .854 reference. These came from different fitting scope and checkpoint age, so the difference is not a clean one-factor gain. The .87829 source-faithful internal result concerns the adapted Pilkwang DINO input/head recipe; it is not a failed CoAtNet reproduction. [Current state](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/docs/current-state.md), [source-slot implementation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/rsna/competitive/source_slot.py)

The public overlay is pinned to Shio-Koube/knee3 revision `373f4e1daf36984fa48bf3fc8fd9883fe17cb584`, file `llm_labels_v4_blend.csv`, SHA256 `eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`. It covers all 2,247 fitting studies and replaces fitting values and masks with 12 observed soft targets per study. The mirror itself has no dataset-card/license metadata. [Current overlay protocol](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/initiative/experiments/competitive-v1/public-label-source-v1/README.md)

A more specific, previously overlooked frozen audit records byte-for-byte identity with `stevenleehans/rsna-knee-llm-report-labels` version 6, a declared CC0 license, author-v2 and lixin73 GPT-5.6-Sol parent extractions, gold-guided changes, and synovitis proxy filling from effusion. The live source card confirms report probabilities blended in probability space, with the v2 proxy filling undecided synovitis without overriding explicit synovitis. Blend coefficients and the exact numeric proxy rule are not established here. This is stronger evidence than inferring origin from the filename. Gold-guided source construction means the 58 gold studies cannot be treated as untouched validation; it does not establish exposure of the 1,156 outer-development images to an image teacher. [Pinned source audit](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/sources.json), [publisher dataset card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)

- The fixed-mask comparison retained the original 13,760 observed fitting cells and replaced 13,750 values with public probabilities. It did not interpolate original and public targets. [Fixed-mask audit](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/initiative/audits/public-label-fixedmask-20261010-v1/README.md)
- Coverage-only preserved original observed values and added public targets on previously masked fitting cells. It did not test independently cross-fitted image supervision. [Coverage audit](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/initiative/audits/public-label-coverage-production-20261011-v1/README.md)
- Epoch averaging, contrast changes, patch means, and neighboring-slice interventions have negative or inconclusive prior results. They are not fresh starting points. A confidence interval crossing zero does not establish equivalence or disprove every version of the underlying mechanism. [Research history](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/docs/creative-research-plan.md)
- The fitting-only Pilkwang teacher collector and 20-member committee design already exist. They are not yet a demonstrated student improvement; the canonical trainer does not currently use this teacher objective. The teacher training roster is unknown. [Single-teacher code](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/rsna/competitive/teacher_evidence.py), [single design](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/tools/research/public_teacher_evidence/design.json), [committee design](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/tools/research/public_teacher_evidence/committee_design.json)

## The mathematical distinction that matters

For a predicted probability p, binary cross entropy is affine in its target. For nonnegative effective weights a and b:

`a BCE(y,p) + b BCE(q,p) = (a+b) BCE((a y + b q)/(a+b), p)`

Here y can already be a soft public target, and q is the teacher output. The identity holds when a+b is positive; both zero means the cell contributes nothing. Include masks, confidence weights, per-target denominators, and the auxiliary multiplier inside a and b.

Consequently, writing a separate teacher BCE does not create fundamentally new supervision mathematics. What can change the result is the teacher's information and provenance, which cells receive it, its confidence weighting, the input unit it describes, and the resulting gradient scale. Separately normalized losses can change all of these effective weights.

The clean first comparison should keep total cell weight fixed and use `y_mix = (1-alpha) y_current + alpha q`. Choose one alpha before the outer evaluation, for example .25 as a conservative experimental setting, not an evidence-based optimum. If using the existing additive implementation, explicitly match total effective weight or include a gradient-scale control. Do not let a doubled loss magnitude impersonate better labels.

For 12 independent findings, use 12 Bernoulli targets. A 12-way softmax would incorrectly make the findings mutually exclusive. Multilabel distillation research supports label-wise treatment, but its image-classification benchmarks are not knee-MRI evidence. [L2D paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Multi-Label_Knowledge_Distillation_ICCV_2023_paper.html), [author implementation](https://github.com/penghui-yang/L2D)

## Why image pseudo-labels might help and why they might fail

An image model trained across many studies can average inconsistent report phrasing and learn repeatable anatomy. Cross-fitted predictions may therefore be a lower-variance target than each study's individual report-derived annotation. A committee trained from genuinely different label sources can add information even when all members use similar backbones. This is a plausible mechanism, not proof that confidence is clinical correctness. Distillation theory identifies a bias-variance trade-off: a smoother but systematically wrong teacher can still harm the student. [Statistical perspective on distillation](https://proceedings.mlr.press/v139/menon21a.html)

Four failure modes must stay separate:

1. **Systematic label bias.** Teachers trained on the wrong severity definition may confidently reproduce it. Cross-fitting does not recover missing truth by itself.
2. **Input insufficiency.** A study-level teacher sees a lesion in a window that the training student does not receive. The student is asked to infer unavailable evidence.
3. **Adaptation or optimization limits.** The student receives adequate evidence but cannot fit teacher relations with its current trainable capacity, objective, or update budget.
4. **Validation mismatch.** A real image-level improvement may reduce agreement with report-derived evaluation labels. Conversely, better weak-label AUC can fail to improve competition AUC, as the creator null report illustrates.

No single weak-label AUC comparison distinguishes all four. The decision tree below uses inexpensive diagnostics before allocating another full fit.

## Provenance and leakage gates

### Training only on fitting rows is necessary but insufficient

For outer development set E and fitting set F, every learned ancestor of a purportedly clean teacher must exclude E and its leakage components. That includes teacher labels, teacher pretraining/adaptation, checkpoint selection, learned normalization, calibration, and ensemble weights. Exclude protected confirmation and official evaluation inputs as required by the existing protocol.

A globally OOF prediction table is not automatically safe for a nested student experiment: a teacher predicting one fitting row might have trained on E. If an overlay descends from an image teacher with unknown outer exposure, making new nested teachers does not repair that uncertainty. For the current overlay, the recovered evidence points to report-LLM extraction and established gold-guided adjustments; outer-image teacher exposure has not been established. Preserve that distinction rather than turning incomplete ancestry into an allegation of leakage.

Two defensible tracks are possible:

- **Clean-provenance track:** teachers and control/student supervision use a verified report-derived source whose ancestry excludes protected image labels and image-based selection. Nest all teachers within F. This permits a genuinely fold-safe pipeline claim, conditional on the documented split and label limitations.
- **Practical incremental track:** keep the current overlay exactly fixed in both student arms and add a traceable, newly cross-fitted teacher. This can answer whether the added information improves this engineering pipeline. If overlay ancestry remains unresolved, it cannot restore or claim fully independent validation. Describe the unresolved upstream exposure explicitly.

The clean track may have a substantially weaker teacher because the older accepted labels are sparse. Failure of that teacher does not disprove the value of a stronger image-label pipeline. Conversely, an unknown-roster public teacher cannot become clean merely because we only collect its predictions on F. [Existing teacher audit rules](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/initiative/audits/public-teacher-notebooks-v1/README.md)

### Keep scientific scope distinct from competition eligibility

Public availability, an open-source code license, permission to adapt weights, permitted competition use, and clean training ancestry are separate checks. A high-scoring creator claim is not a license receipt. MRI pretraining that used OAI, fastMRI, private cohorts, or undisclosed data needs its own eligibility audit. No new clinical data, reports, or images need to leave an authorized environment for the proposed diagnostics.

The inspectable nartaa .949 checkpoint has an additional boundary problem independent of licensing: its reported training scope is all 4,349 nongold studies, encompassing this project's development and reserved-confirmation studies, with repeated Gold58 selection. Fine-tuning it only on F cannot remove that upstream exposure. Use its architecture and disclosed recipe with a fresh allowed initialization for a clean-boundary experiment; do not automatically admit its trained weights. A later hidden competition test would answer a different performance question and does not grant permission to change the present contract. [Author reconstruction](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792), [scored release](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397)

## Broad mechanism inventory

These are alternatives and conditional branches, not a plan to run every experiment. “One fit” below means one complete matched student run at its specified frame and update budget; the actual GPU-hours must come from a pilot. Controls remain on the same roster, target semantics, inputs, initialization, and fixed-final evaluation unless a row explicitly changes one of them.

### A Supervision and denoising

1. **Cross-fitted soft image targets.** Highest-priority unresolved mechanism. Compare the unchanged overlay with a constant-weight mixture containing independently generated teacher scores. Add a conditional-mean or permuted-teacher control if the first result is promising. Cost: teacher construction plus one candidate fit; a fresh matched control if the existing run is not reusable. Kill the claimed image-specific explanation if its benefit is matched by non-image softening.

2. **Different report sources feeding different teachers.** Diversity of label errors may matter more than diversity of backbone names. Use only independently verified, admitted label versions; otherwise this risks reintroducing held claims. Start with two sources only if one-source distillation works. Cost: approximately double cross-fitting. Stop if prediction diversity is mostly calibration and there is no held-out complementary information. Twenty correlated checkpoints are not twenty independent opinions.

3. **Confidence without penalizing focal anatomy.** Compare unweighted teacher soft targets against a bounded between-model disagreement weight, with total effective target weight controlled. The existing single-teacher score subtracts across-window range from decisiveness; the committee design subtracts between-model standard deviation instead. Window variation may encode genuine anatomy. Cost: no extra encoder passes after packet collection. Stop if weighting only removes rare positives or if unweighted targets work as well.

4. **Source then teacher curriculum.** Introduce the same fixed teacher mixture after a short source-supervised warm-up rather than from the first update. This tests optimization conflict, not better labels. Match total teacher exposure or report the difference. Cost: one conditional fit. Stop if an equivalent learning-rate/gradient-norm schedule explains the gain.

5. **Target-balanced exposure.** Report actual per-target gradient mass, observed positives, and teacher confidence mass before adding focal loss, oversampling, or positive weighting. A macro-AUC task can be dominated during optimization by frequent/easy negatives. Cost: inexpensive instrumentation followed by at most one weighting arm. Stop if the intervention only increases duplicated rare-case exposure or worsens most targets. Do not tune 12 independent weights on tiny validation counts.

6. **Admit omitted studies separately.** The current selector retains the old any-observed roster; public coverage does not automatically add the omitted training studies. A larger fitting roster might improve generalization, but it is a data-volume experiment requiring source and split admission. Cost scales with added studies. It must not be folded into the first label-source comparison.

### B Matching teacher information to the student

7. **Same-window teacher targets.** Before forcing a single-window student to match a whole-study teacher, compare the teacher's score on the student's actual window with its full-study score. Then test aligned-window versus study-level targets at identical student frame budget. Cost: existing per-window packet can make the diagnostic cheap. Stop if substantial disagreement is absent or matching windows does not improve fit. This diagnoses functional mismatch; it does not prove which window contains pathology.

8. **Whole-bag student training.** Give the student the information needed to learn a study-level target. This is already active in the mean/query work, not a new proposal. The source-slot setting has four training windows versus one, roughly four times encoder-frame exposure. Count that cost and use a matched-frame control if claiming better aggregation rather than more images.

9. **Within-target rank transfer.** Preserve teacher order between studies for the same finding instead of forcing exact probabilities. Suitable if ordering transfers but probability matching struggles. Use soft pair targets or a tie margin fixed from fitting data. Cost: fresh cross-study graphs or exact recomputation, not ordinary accumulation. Stop if the Bernoulli baseline transfers the same order at lower cost. Do not rank unrelated target classes against each other. [DIST](https://arxiv.org/abs/2205.10536), [RankDistil](https://proceedings.mlr.press/v130/reddi21a.html)

10. **Label-wise feature or relation transfer.** A small projection can align target-specific embeddings or patch relations when scalar predictions discard useful information. Compare with scalar KD under the same teacher access and overhead. Cost: teacher feature generation/storage and another fit. Stop if shuffled feature identities or a generic regularizer perform equally well. Cross-family dimensions require a learned projection, not arbitrary channel matching. [L2D implementation](https://github.com/penghui-yang/L2D)

11. **Leave-one-view teacher diagnostics.** Measure how removing each plane/sequence changes teacher outputs; compare with removing a random equally sized view set. This can reveal whether a student omits a crucial evidence source. Cost: extra teacher inference, not immediately a new student. A score drop is dependence, not proof of anatomical causality. Stop if dependence is dominated by slot-count or scanner signatures.

12. **Target-specialist teachers with one student.** A legitimate teacher may be strong on cartilage but weak on effusion. A shared student can receive per-target evidence from different eligible teachers while remaining one model at inference. Cost: committee inference and selection risk. Use a predeclared small rule or fitting-only selection, not 12 gold-optimized blend weights. Stop if apparent specialization disappears under component resampling.

### C Severity and missingness

13. **Presence and qualifying severity as separate heads.** Train an auxiliary presence concept only when the source explicitly supports it; retain the competition head for qualifying severity. This could prevent mild findings from contaminating a severe-positive target while still learning anatomy. Cost: a small head, but trustworthy concept extraction is the real expense. Stop if concepts merely duplicate the final label or inherit the same ambiguity.

14. **Ordinal or interval supervision.** Use explicit known severity ordering or permissible intervals instead of fabricating exact grades. Distinguish no disease, below threshold, qualifying disease, and unknown only where evidence permits. Cost: modest head/loss work, potentially substantial annotation review. Stop if there are too few reliable ordered pairs or gains arise only from unapproved label expansion. Ordinal MIL evidence comes from ulcerative-colitis severity, not these knee targets. [Ordinal MIL paper](https://arxiv.org/abs/2411.14750), [author code](https://github.com/Shiku-Kaito/Ordinal-Multiple-instance-Learning-for-Ulcerative-Colitis-Severity-Estimation)

15. **Separate depth, extent, and acuity.** Cartilage depth is not automatically lesion size; ligament abnormality is not automatically acute high-grade disruption. Auxiliary concepts should preserve these distinctions and connect to the final target only through supported rules. Cost: small model change, difficult labels. Stop if the required dimensions are unobserved. [Project rubric](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/protocols/report_weak_v2.md), [cartilage evidence](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/main/docs/cartilage-grade-evidence.md)

16. **Model reporting propensity as a diagnostic.** Test whether observability depends on report style, anatomy, scanner, or co-findings. Missingness is plausibly not random; its mask alone does not identify hidden labels. Randomly masking observed labels tests an MCAR scenario, not the actual missingness mechanism. Cost: low for diagnostics. Do not apply inverse-propensity correction without support and assumptions. [Selected-at-random positive-unlabeled learning](https://proceedings.mlr.press/v94/bekker18a.html)

17. **Blinded fitting-only image review.** A small qualified review of probability-sampled cases plus separately tracked teacher/report disagreements can distinguish severity errors from omission. Include agreements and normal controls to avoid a biased disagreement-only picture. Cost: human expertise and separate authorization, not assumed free. Stop if access, reviewer reliability, or adjudication cannot be established before the deadline.

### D Local pathology with anatomical context

18. **Finish target-query and patch-focus tests.** These are already implemented or active. Do not rename them as a new attention breakthrough. Their value is a matched gain beyond the global classifier with adequate gradient flow and comparable exposure. Stop or retain according to their original fixed-final protocol, not an appealing attention visualization.

19. **Distribution of local evidence rather than one maximum or mean.** Give a tiny head a few predeclared statistics such as peak evidence, upper-tail mass, and contiguous extent, together with global context. Focal tears and distributed effusion need different summaries. Cost: low head overhead on existing features. Compare with equal-parameter mean/max heads. Stop if gains vanish when slice count or padding is matched. Distribution-pooling evidence is from pathology, not knee MRI. [Paper](https://www.sciencedirect.com/science/article/pii/S1361841523000749), [author code](https://github.com/onermustafaumit/mil_pooling_filters)

20. **Anatomical adjacency rather than arbitrary repetition.** Reward corroboration across physically adjacent slices where the target definition supports it, without requiring every positive to appear in two planes. Compare true positions with shuffled positions and distant slices at equal frame count. Previous neighboring-channel results were inconclusive/negative, so a new test needs a genuinely different aggregation mechanism. Stop if shuffled anatomy performs equally well.

21. **Duplicate-invariant extent.** Overlapping windows can make one lesion look like repeated evidence. Pool over physical coverage or unique slice positions so duplicating an identical window does not increase predicted extent. Cost: low if geometry is already reliable. First test exact duplicate invariance without training. Stop if the current model is already insensitive or if geometry cannot be trusted.

22. **Instance-label disambiguation.** Avoid assigning every window the positive study label; alternate or jointly learn local evidence and bag prediction. CIB-MIL supplies a recent mechanism precedent in whole-slide pathology, not validation for MRI. Cost: another training formulation and potentially unstable latent labels. Compare against the already active whole-bag baseline. Stop if instance pseudo-labels collapse or random-window controls reproduce the gain. [CIB-MIL](https://pubmed.ncbi.nlm.nih.gov/42470794/), [author code](https://github.com/TencentAILabHealthcare/CIB-MIL)

23. **Counterfactual local evidence test.** Remove or replace a proposed witness region and compare the score change with equal-area random or anatomy-matched removal. This challenges whether attention identifies causal evidence. Cost: bounded inference first; model changes only if informative. Avoid interpreting synthetic masking artifacts as lesions. It is a falsification tool before it is a loss function. [Counterfactual MIL precedent](https://ojs.aaai.org/index.php/AAAI/article/view/28135)

### E Adaptation and model family

24. **Complete dense adaptation before escalating LoRA complexity.** Current adapters demonstrably update, but that does not prove enough task adaptation. Dense adaptation is already active and has passed bounded memory checks; operational interruption is not a scientific failure. Compare at a matched learning-rate search budget and input policy. Cost: measured actual run cost. Stop if better fitting consistently fails to generalize and regularization does not repair it.

25. **One smaller full-fine-tuned challenger.** DINOv2-S, ConvNeXt-T, or CoAtNet at 224/appropriate native resolution is practical. Choose one with an auditable implementation and compatible cached inputs. A small model can receive more complete bags and updates for the same wall time. Cost: one matched fit, then a second supervision arm only if justified. Report both equal-exposure and fixed-wall-time comparisons; they answer different questions. Creator claims support feasibility, not a guaranteed family ranking.

26. **Layer-wise learning rates or selective unfreezing.** Test whether the backbone needs more adaptation in later versus earlier layers. Use a single predeclared schedule, not a large search. Cost: one conditional fit. Stop if it only changes total update magnitude and a simple learning-rate control matches it.

27. **LoRA gradient correction or low-rank optimizer states.** LoRA-Pro and GaLore address optimization and memory respectively. Neither is a first choice when a dense path already fits memory and remains untested. LoRA-Pro's implementation changes optimizer behavior; it is not a harmless flag. Cost includes integration and recovery testing. Stop if the actual bottleneck is data/input or wall time rather than optimizer memory. [LoRA-Pro](https://arxiv.org/abs/2407.18242), [implementation](https://github.com/mrflogs/LoRA-Pro), [GaLore](https://arxiv.org/abs/2403.03507)

28. **Preserve local features during task fine-tuning.** If dense fine-tuning improves fitting but destroys useful local structure, sample patch-pair Gram relations from a frozen initialization as an anchor. Compare with simple weight anchoring at matched overhead. Cost: additional frozen forward work; no giant dense cache required. Stop if local degradation is absent or ordinary regularization works equally well. DINOv3's Gram anchoring is an SSL result, not proof of a knee fine-tuning benefit. [DINOv3](https://arxiv.org/abs/2508.10104), [author explanation](https://github.com/facebookresearch/dinov3/issues/65)

29. **Eligible knee-pretrained initialization.** A relevant MRI model might help more than a newer generic architecture. Establish exact checkpoint availability, ancestry, overlap, terms, and permitted adaptation first. KneePreM does not distribute pretrained checkpoints and reports a four-day A100 pretraining run, making from-scratch replication unattractive with eight days left. OrthoFoundation requires a separate ancestry/eligibility audit. [KneePreM](https://arxiv.org/abs/2609.31461), [author code](https://github.com/BiomedSciAI/fuse-med-ml/tree/master/fuse_examples/imaging/oai_example), [OrthoFoundation release](https://github.com/ytrsk/OrthoFoundation-MSK-5M)

### F Inputs and generalization

30. **Series quality before more slices.** Check whether the deterministic selected series preserves the required anatomy, fluid sensitivity, orientation, and coverage. Compare a quality-aware selection rule with the existing rule at identical series/slice budget. Cost: metadata audit and possibly cache work. Stop if differences only reflect scanner shortcuts or the selected series already covers the evidence.

31. **Physical crop and laterality consistency.** Audit whether anatomy occupies comparable physical field of view and whether orientation transforms preserve medial/lateral semantics. Do not assume arbitrary pixel flips are anatomically valid. Prior geometry and neighbor work already exists; focus on a specific verified inconsistency rather than generic augmentation. Cost: low for audit, potentially large if a new cache is needed. Stop if no material mismatch is found.

32. **Protocol robustness with legitimate sequence dropout.** Missing-sequence training can reduce dependence on a scanner-specific slot pattern. Compare against intensity-only regularization, keeping complete-study evaluation unchanged. Cost: one conditional fit without extra frames. Stop if complete-study performance worsens or metadata-only controls explain the gain.

33. **Nuisance-stratified stress tests.** Break down errors by scanner/protocol proxies, slice count, original label observability, and report style when both classes exist. Add metadata-only and feature-shuffled controls. Cost: mostly inference analysis. These are diagnostics; report style is not a verified hospital identity and should not become an inference input unavailable at test time.

### G Deployment and selection

34. **Single-checkpoint inference coverage.** A single backbone can process multiple windows without being a multi-model ensemble. Select a fixed coverage policy using development data and measure full DICOM-to-output runtime. Cost: proportional to windows. Stop adding windows if they repeat the same anatomy or violate runtime margin. Keep a strict distinction between one checkpoint, one pass, TTA, SWA, and checkpoint ensembles when comparing creator scores.

35. **Weight averaging only with a new reason.** Epochs 6–10 averaging has already failed here. Do not spend another run on generic SWA unless an independently observed flatter-basin or checkpoint-instability mechanism changes the hypothesis. A competitor's single-SWA model is one deployed checkpoint but not evidence that our prior averaging choice should work.

36. **Calibration is not a standalone AUC strategy.** A strictly increasing per-target transformation preserves AUC apart from numerical ties. It can change a probability blend or training targets, but temperature scaling alone cannot repair ranking. Rank-transfer experiments must therefore be evaluated as learning interventions, not post-hoc calibration tricks.

## Minimal experiment decision tree

### The first experiment is finite

The immediate proposal is one fixed test of the practical incremental track, using the existing stable native DINOv3 recipe. It does not wait indefinitely for a perfect label source. A bounded provenance review determines the claim we can make; it does not automatically expand into another research campaign.

1. Keep the existing 2,247 fitting and 1,156 development studies, leakage components, native 384 inputs, 16 selected slices per sequence, maximum six sequences, current rank-32 adaptation graph, and current public-label overlay. Do not add the omitted studies or change masks.
2. Partition F into two approximately balanced component-aware inner folds, frozen before fitting. Train two teachers with the same existing recipe and a fixed ten-epoch schedule, each excluding the other inner fold and all of E. Scale teacher warmup/cosine endpoints to their declared shorter ten-epoch update count; do not silently give each half-data teacher a full-data update budget. These are newly generated teachers, not the unknown-roster public committee. They still inherit the current overlay's gold-guided source construction and incompletely reproduced parent extractions, so the entire pipeline is not thereby certified clean.
3. Generate one OOF probability vector per fitting study using the corresponding excluded-fold teacher. Use the same native sequence/slice selection as the student and deterministic evaluation preprocessing, without extra windows, TTA, or gold-selected calibration. Store the 12 probabilities and provenance; no new dense feature cache is needed.
4. Compare S0 with unchanged public targets against S1 with `0.75 y_current + 0.25 q_OOF`, using identical official initialization, student seed, training draw schedule, total cell weights, and the existing ten-epoch/2,810-update fixed-final endpoint. Alpha .25 is a predeclared test choice, not an asserted optimum. Do not add a confidence heuristic, rank loss, class weighting, or architectural change to this first contrast. Any source-support exclusion must be fixed before fitting and applied as a declared scope rule, not selected from the endpoint.
5. Primary endpoint: paired change in the existing all-12 masked development macro AUC, accompanied by every target, component uncertainty, undefined draws, and measured cost. This measures the incremental pipeline against its existing weak evaluation labels; it does not establish clinical truth or clean upstream independence. If the completed S0 state does not match these exact conditions, include a fresh S0 rather than borrowing an unmatched score.
6. If S1 is worse or shows no useful signal at the fixed endpoint, stop this specific self-distillation recipe. Do not search alpha, extend epochs, or add committee members on the same outer result. A promising result earns the predeclared information-destroying control and a matched replication. If provenance prevents even this bounded use under the project's existing policy, take the one-family challenger branch rather than silently bypassing the restriction.

This costs approximately one fitting-pool-equivalent of teacher training plus one or two student fits, depending on control reuse, and teacher inference. It directly tests whether image-model denoising adds value beyond the existing overlay. It does not test the maximum attainable benefit from stronger, independently labeled teachers; a negative result should be scoped accordingly.

**Cheaper alternative if cross-fitting cannot fit the available budget:** the live publisher card recommends target confidence `2 abs(y_current - 0.5)`, while the current overlay uses all-one observation masks. One fixed loss-weight contrast could test that disclosed recipe difference without training teachers. Preserve label values and observation masks, use a distinct weight field, predeclare the normalization and zero-weight-target behavior, and match target/gradient budgets in the interpretation. This is a new experiment rather than a correction compelled by ground truth: in a blended table, distance from 0.5 is not a proven clinical confidence. Do not combine this change with the image-target mixture in the first test. [Publisher card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)

### Step 0 Freeze the comparison and audit the actual bottleneck

Record the current .872 checkpoint identity, the relevant internal comparator, code and data hashes, fitting/evaluation membership, exact supervision values/masks, frame counts, optimizer steps, inference policy, and wall time. Keep the already active jobs and their scientific protocols intact. Do not bypass blocked compute access or launch duplicates.

Resolve what can be learned about the current overlay's ancestry. If unresolved, choose and label the practical incremental track rather than claiming a repaired clean pipeline. Review any new source, masks, teacher access, or concept policy before use.

Run only cheap, fitting-only diagnostics first:

- Teacher versus current-target agreement, probability spread, and confidence mass by target and original observability stratum.
- Whole-study versus aligned-window teacher disagreement at the student's actual input unit.
- Effective gradient mass and fresh cross-study pair counts, rather than nominal batch size.
- Input coverage and compute throughput on the exact proposed student.

The existing eight-study teacher probe found no cases satisfying its predeclared sparse-high-window pattern across 96 target cells. That small result does not establish population focal suppression or justify asserting it. The single-teacher window-range penalty remains a structural concern to test; the committee design already avoids that specific penalty.

### Step 1 Obtain the smallest useful teacher packet

If an eligible teacher with verified ancestry is already available, reuse it only within the admitted scope. Otherwise, construct two inner folds inside F, split by leakage component. Train each teacher on one half and predict the other half. Every fitting study receives a prediction from a teacher that did not train on it; neither teacher sees E.

Start with one affordable teacher family and a verified source. Freeze the component-aware split and its per-target support before training. A teacher fold with no meaningful class support cannot supply credible evidence merely because its outputs are confident; retain the control supervision for that target under a predeclared admission rule. A two-fold teacher is trained on less data than the eventual student, so a weak result may reflect teacher quality. Do not call it an upper bound on what image pseudo-labeling can achieve. Increase teacher strength only after diagnosing this limitation, not reflexively adding twenty models.

For a teacher benchmark on E, the two inner teachers can predict E without exposure, but their half-data training makes that a different comparator from a full-F teacher. A separate full-F teacher is a conditional extra cost, not a free reference. If there is no independent image-label validation, teacher weak-label AUC measures report agreement, not true clinical accuracy.

### Step 2 Run one supervision contrast on the best stable existing student

**Control S0:** exact unchanged current supervision and training recipe.

**Candidate S1:** same student, initialization, input draws, frames, updates, masks, optimizer, and total effective cell weight; mix one fixed amount of cross-fitted image target into the current target. Retain originals in a versioned experiment overlay. Where the teacher does not meet the declared admission policy, leave the control target unchanged rather than inventing certainty.

This first test should not simultaneously switch backbone, resolution, number of views, class weighting, and teacher confidence rule. Reuse S0 only if its actual scientific state and fixed-final endpoint match; a convenient historical score is not automatically a control.

**If S1 is promising:** add one nuisance control S2 with the same per-target teacher distribution but no matched study-specific teacher information: use a predeclared permutation within original-label/mask/coverage strata. Avoid tiny strata and declare bin-merging before evaluation. Keep alpha and gradient scale identical. A deterministic conditional-mean teacher target is an alternative softening baseline, but it does not preserve the original teacher distribution and must not be described as doing so.

**If S1 beats S0 but not S2:** favor the simpler regularization explanation. Do not claim that the teacher discovered pathology.

**If S1 beats both:** repeat the fixed contrast on a second matched seed or eligible fold. Promote only after inspecting per-target failures, component sensitivity, and runtime. An inconclusive small comparison remains inconclusive.

### Step 3 If distillation fails diagnose one cause

- **Large aligned-window mismatch and poor teacher fitting:** test same-window targets first, or use the already active whole-bag student. Keep extra-frame cost explicit. Do not immediately increase model size.
- **Inputs aligned but student cannot fit stable teacher ordering:** use the dense adaptation result or a smaller fully trainable student. If scalar probability matching is the specific failure, test within-target rank transfer next.
- **Student fits teacher well but validation fails:** investigate teacher bias, severity mismatch, nuisance correlation, or incompatible validation labels. More capacity is unlikely to fix a faithfully learned bad target.
- **Teacher itself is weak or nearly identical to the existing overlay:** improve the admitted source or teacher, or stop this branch. A student cannot be expected to extract new information from a redundant target merely because it is called distillation.
- **Teacher/label disagreements concentrate at severity boundaries:** consider the presence/severity or qualified-review branch. Do not relabel disagreements automatically.

### Step 4 Use one family challenger to test the remaining explanation

Compare one compact full-fine-tuned DINOv2-S or ConvNeXt/CoAtNet model under the same supervision and input budget. Pick a family based on implementation readiness and measured throughput, not claimed leaderboard ordering alone.

If family replacement wins under unchanged supervision, adaptation/inductive bias becomes a stronger explanation. If teacher supervision helps both families similarly, supervision becomes a stronger explanation. Only run the complete two-by-two architecture-by-supervision comparison if the first contrasts justify its cost. Keep backbone-specific optimization tuning budgets comparable; an intentionally untuned challenger is not an informative falsifier.

## Ranking and relation transfer require real batch accounting

The historical one-study forward has zero cross-study pairs. Gradient accumulation to eight studies does not create the missing graph. The repository's audit also shows very few observed-label pairs for several rare targets even at the effective-batch level. Soft teacher scores can provide more pairs, but potentially many uninformative near-ties. [Ranking exposure audit](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/tree/main/initiative/audits/ranking-supervision-exposure-v1)

Two honest implementations are available:

1. Retain two fresh study graphs for each pair, or a genuine eight-study graph if memory permits. Peak activation memory increases; measure it.
2. Use exact gradient caching: collect outputs without retaining encoder graphs, compute the coupled loss gradient, then replay the same inputs and random state at the same weights to backpropagate each study. This adds approximately one forward pass. A rough arithmetic estimate is 4/3 of ordinary compute if backward costs two forwards, but actual wall time must be measured. [Gradient Cache paper](https://aclanthology.org/2021.repl4nlp-1.31/), [author implementation](https://github.com/luyug/GradCache)

Do not start with stale prediction/feature queues unless their staleness is explicitly controlled. No optimizer step may occur between the cached forward and exact replay. Treat same-input and same-RNG equality as an acceptance gate, not an assumption.

## Compute and storage budget

Let H be the measured GPU-hours for one full teacher/student training run over F at the declared epoch and frame budget. Different backbones and whole-bag policies have different H values; do not reuse a throughput estimate blindly.

- K-fold cross-fitting trains K models on approximately (K-1)/K of F. At fixed epochs, this costs about `(K-1) H_teacher`, plus inference and startup overhead: two folds about 1H, three folds 2H, five folds 4H.
- A minimal two-fold teacher plus S0/S1 screen is approximately `H_teacher + 2 H_student + inference`. If an exactly matching control is already complete, the incremental student cost is only one fit.
- The recorded full-public baseline took 66,103 seconds, about 18.36 hours, including its recorded run work. At that reference throughput, three full-fitting equivalents would be about 55 GPU-hours before additional overhead; this is a planning anchor, not a forecast for a different model or a guarantee of allocation. [Actual result receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/public-result.json)
- A conditional softening control plus a second-seed S0/S1 pair adds about 3H_student. A family challenger adds its own H_challenger. This is why the branches must be gated rather than launched together.
- Four-window training uses roughly four times the source-slot encoder frames of one-window training; aggregation and optimizer costs prevent an exact wall-time equivalence.
- A fitting packet of 2,247 studies × 20 members × 10 windows × 12 targets in float32 is 21,571,200 bytes, about 20.57 MiB, before probabilities and metadata. The proposed 64 MiB committee cap is plausible for logits, not dense feature maps.
- Shared-prefix committee inference reduces ideal block work from 20×12 blocks to 6+20×6 = 126 blocks, compared with 240. That arithmetic is not a measured 1.9× end-to-end speedup. Exact source-graph equality and real-input equivalence must pass before reuse.
- Dense Gram or feature distillation should compute selected relations on demand. A full patch-token cache can be hundreds of times larger than a pooled-feature cache.

With eight days left, keep the first day for lineage/input diagnostics and throughput, days 2–4 for completed active results plus one decisive supervision contrast, days 4–6 for confirmation and the chosen final fit, and the last two days for offline inference, exact checkpoint binding, failure cases, and runtime margin. This is a planning envelope, not a promise of cluster availability. Drop speculative pretraining and new optimizer frameworks first if access or runtime slips.

## Evaluation and stop criteria

Use the unchanged evaluation frame and masks for the primary matched comparison. Report all 12 target AUCs and their positive/negative study and leakage-component counts. Report undefined bootstrap draws rather than replacing them with 0.5 or silently dropping targets. Component bootstrap intervals quantify sampling uncertainty conditional on the trained models; multiple seeds address a different uncertainty.

Predeclare a practically meaningful gain relative to the measured cost and uncertainty, rather than choosing a threshold after seeing the result. For promotion, require a reproducible directional gain with no consistent material target collapse and a feasible offline runtime. A confidence interval crossing zero is neither proof of improvement nor proof of equivalence. Rare-target gains supported by one or two components are fragile; show leave-one-positive-component-out sensitivity.

The strongest falsifiers are:

- The image-teacher arm matches a conditional-mean/permuted soft-target control: the proposed image-specific mechanism is unsupported.
- Teacher order is already reproduced, but performance remains poor: scalar KD capacity is not the bottleneck.
- Same-window and whole-study teacher targets agree closely: input-target mismatch is not a compelling main explanation.
- Shuffled positions or duplicate windows preserve a claimed anatomical aggregation gain: the mechanism may be count, regularization, or nuisance correlation.
- Dense adaptation or the compact challenger improves fitting only: adaptation alone has not solved generalization.
- Apparent gains disappear when label scope, exposure, or gradient magnitude is matched: the original causal interpretation fails.
- A claimed clean source cannot establish ancestry: stop the clean-validation claim even if the engineering result looks good.

Short pilots can reject numerical instability, broken losses, gross input mismatch, or infeasible runtime. They cannot reliably rank mature model families from early learning curves. Do not extend a losing arm, choose the best epoch, or change target-specific weights after looking at the outer endpoint and still call it the original fixed-final test.

## High upside hypotheses beyond reproducing the leaders

Reaching parity is a useful control, not the final ambition. The defensible route to an advantage is to exploit a measured weakness that a stronger model still inherits. The following bets have more upside than adding another similar checkpoint, but they require sharper evidence. None is established as absent from every leading private solution.

### First edge Separate real synovitis evidence from a label construction proxy

**Concrete opportunity.** The recovered source audit explicitly records synovitis filling from effusion. That creates an artificial training correlation in addition to any real biological correlation. An ensemble of image teachers trained on the same proxy can become very confident while preserving the same mistake. This is a specific weakness in our target construction, not a speculative claim that all missing labels are wrong.

The live publisher card narrows the claim: the proxy fills undecided synovitis in the v2 parent and does not override explicit synovitis observations. Therefore a correction must preserve that distinction; it would be inaccurate to claim the source equates every synovitis label with effusion.

**Primary medical evidence and limit.** MOAKS distinguishes effusion-synovitis and Hoffa-synovitis as noncontrast MRI assessment constructs; neither name automatically matches RSNA's separate thickened/inflamed-lining target. In a 111-knee OA study using contrast-enhanced assessment, definite synovitis occurred both with and without effusion. A separate 104-knee end-stage-OA study found effusion-synovitis a better noncontrast surrogate than Hoffa-synovitis, but every included knee had some effusion-synovitis and contrast-defined synovitis. That cohort cannot establish specificity in the competition's broader population. The lesson is to retain useful fluid correlation while learning additional supported morphology, not force independence or claim noncontrast MRI can always separate the two. [Original MOAKS](https://pubmed.ncbi.nlm.nih.gov/21645627/), [Roemer 2010 primary study](https://pubmed.ncbi.nlm.nih.gov/20691796/), [Crema primary comparison](https://www.oarsijournal.com/article/S1063-4584%2816%2930297-7/fulltext)

**Proposed intervention.** If cell-level provenance can be reconstructed from admitted source logic, distinguish direct synovitis evidence, explicit absence, and proxy-only targets. Keep direct evidence intact. Test reducing only proxy-derived supervision and learning the remaining image signal through a separate, admitted teacher policy. Missing or proxy-only cells must not become negatives. A small auxiliary head can share the existing encoder; another backbone is unnecessary.

A sharper alternative is a zero-initialized morphology residual on a known-scope synovitis predictor: retain its existing fluid-associated baseline and let shared image features add or subtract evidence, supervised only by admitted direct synovitis observations. Compare with an equally sized generic residual and with feature identities shuffled within effusion strata. Do not claim that a residual becomes identifiable without enough reliable discordant examples. If an external OAI-pretrained model is considered, audit its actual label mapping: an OAI effusion-synovitis label must not silently become RSNA synovitis. The exact mapping used by nartaa is not established here.

**Control and strongest falsifier.** Match the reduction in supervised cells with random removal within the same target and source strata, and preserve effective loss normalization. On supported development cases, report performance separately for concordant effusion/synovitis and discordant combinations. If random removal works equally well, the proposed proxy-specific mechanism is unsupported. If there are too few reliably labeled discordant cases, report the limitation; the current near-ceiling weak synovitis AUC cannot establish clinical superiority.

**What competitors lack.** The identified proxy is verified in our source audit. Leading creators' complete target-construction code is mostly undisclosed, so we cannot say they all have this defect. Some explicitly recognized the synovitis-report problem. This branch is most valuable if their remedy merely smooths the same proxy rather than supplying independent image evidence.

**Cost and gate.** One target-scoped loss contrast with the existing encoder, plus the source-lineage reconstruction. No new labels should be inferred from the proxy itself. If direct versus proxy cells cannot be identified, do not pretend a broad synovitis reweighting is this experiment. Independent image review would greatly strengthen the test but needs separate access and authorization.

### Second edge Learn the competition boundary rather than abnormality alone

**Concrete opportunity.** Mild abnormalities can be visually obvious yet correctly negative under the competition rubric. A highly accurate abnormality teacher can therefore be a biased severity teacher. Known below-threshold negatives carry different learning information from normal anatomy, even though both have final label zero.

**Proposed intervention.** Use the already distinguished explicit-absence and below-threshold states where source evidence supports them. A shared encoder predicts presence and qualifying disease through separate heads. For cartilage, preserve depth and extent as separate concepts; for ligament injury, preserve disruption and acuity. Use only actual supported concepts or intervals, with missing concept masks. Do not calculate a calibrated qualifying probability by multiplying uncertain concept probabilities or treating attention span as a measured lesion diameter.

The ambitious version is severity-aware distillation: learn from an image teacher's useful presence representation while preventing its broad abnormality score from automatically overriding a documented below-threshold final negative. This is a new, explicit experimental supervision policy, not a silent change to accepted labels. A monotonic relation between genuinely nested concepts may be imposed, but a universal maximum-severity operator is inappropriate for targets with extent requirements.

**Control and strongest falsifier.** Compare boundary-focused exposure with equal-count random-negative exposure and a same-size auxiliary head whose concept assignments are permuted within final-label strata. Keep images, total updates, and final target scope fixed. The mechanism fails if gains persist after destroying the severity concepts, or if discrimination improves only against normal knees and not supported below-threshold cases.

**Evidence and novelty boundary.** The project rubric establishes the threshold mismatch. Ordinal MIL and noisy-ordinal progression papers show ways to use severity or comparative information in other diseases; they do not show that unannotated knee severity can be recovered for free. Current competitor comments do not disclose enough to establish whether their teachers handle these boundaries. [Ordinal MIL](https://arxiv.org/abs/2411.14750), [noisy ordinal progression](https://arxiv.org/abs/2503.10440)

There is unusually direct recent support for the cartilage construct problem: a 2025 primary sMOAKS study adds a separate >50%-thickness-loss dimension because traditional MOAKS distinguishes any cartilage loss and full-thickness loss without isolating deep partial-thickness loss. Two radiologists evaluated 40 knees for the proposed scoring supplement. This supports auditing an OAI/MOAKS-to-RSNA label mapping rather than assuming its numeric grades already encode the competition boundary. It does not prove a competitor used an incorrect mapping, supply the missing labels in this project, or resolve the separate physical-size criterion. [sMOAKS primary study](https://pubmed.ncbi.nlm.nih.gov/40857520/)

CoNNS provides a newer, narrower design precedent: represent finding presence, attributes, and evidence separately; ignore cross-patient pairs containing unknown states; use contradictions in explicit attributes rather than automatically treating every different patient as a negative. Its experiments are chest radiography with large-scale vision-language training, not knee MRI. Its author repository contains training and dataset code, but importing that full framework is not a sensible final-week default. The transferable idea is concept-aware supervision/masking on the existing MRI encoder. [CoNNS paper](https://arxiv.org/html/2605.19374v1), [author implementation](https://github.com/DopamineLcy/conns)

**Cost and gate.** Heads and losses are cheap; reliable concept coverage is the bottleneck. This is a strong upside branch only if fitting-only evidence contains enough genuine boundary examples. If not, prioritize the simpler image-supervision test instead of fabricating ordinal targets.

An auxiliary head can also hurt through shared-encoder gradient conflict. A shuffled-concept control is insufficient if it changes valid-cell exposure or gradient magnitude. Match a bounded auxiliary gradient budget on shared parameters, retain the primary objective, and log any normalization, clipping, and Adam effects. A zero-initialized head may initially have zero encoder gradient, so do not divide by that norm or infer the mature optimization regime from initialization alone.

### Third edge Measure distinct physical support rather than repeated views

**Concrete opportunity.** Ninety-four overlapping triplets are not ninety-four independent observations. A model can mistake repeated coverage for extensive disease or dilute a small lesion among many irrelevant windows. The same generic pooling operation need not be appropriate for focal ligament injury and distributed fluid or cartilage abnormalities.

**Proposed intervention.** Extend the existing target-query/local-evidence work with a small per-target evidence summary that retains peak strength, unique physical support, and within-series continuity. Normalize repeated windows by the physical slices they cover. Keep global anatomical context alongside local evidence. Do not merge coordinates across series without verified registration, and do not interpret a learned extent score as a clinical centimeter measurement without validation.

**Control and strongest falsifier.** Before a fit, duplicate selected windows while keeping their pixels identical, permute physical positions, and replace neighboring views with equally sized distant views. Compare a physical-support head with an equal-parameter distribution head without geometry. If shuffled positions work equally well, the gain is not attributable to anatomy. If duplicate-invariant pooling offers no benefit over the current head, stop this version rather than adding more geometry machinery.

**Verified competitor limitation.** The inspected nartaa implementation uses a concatenated stack whose triplets can cross slot boundaries; it does not guarantee physically adjacent MRI channels. With five populated slots, two triplets around each of four boundaries can cross slots, up to eight of 94 windows. That arithmetic describes the layout, not the measured contribution of those windows. A boundary-safe, frame-matched alternative is a concrete experiment; it may improve or worsen performance because the original model could exploit cross-sequence context. Other leaders' private implementations are unknown.

**Cost and gate.** The duplicate/position diagnostics can reuse existing features or predictions where available. The head should reuse the same encoder frames. A new physical-extent fit is worthwhile only if current outputs demonstrate the relevant sensitivity; ordinary neighboring-channel and patch-mean experiments have already failed to establish an advantage.

### Fourth edge Separate anatomical variation from teacher uncertainty

**Concrete opportunity.** A focal finding can produce strong disagreement across windows while all competent teachers agree about the informative window. Penalizing window spread then removes exactly the difficult local evidence a student needs. Conversely, agreement among twenty closely related teachers does not protect against shared label bias.

**Proposed intervention.** Factor the existing member-by-window output packet into distinct questions: do members agree on the same input, do outputs change when the input's anatomy changes, and is the pattern stable under safe transformations? Retain this distinction in weighting or local-versus-study transfer. Begin with unweighted soft targets as the control. Only introduce a reliability gate after its behavior is measured across the full fitting packet.

**Control and strongest falsifier.** At equal effective teacher mass, compare a between-member rule with the current single-teacher window-spread rule and with no weighting. Report mass by target and original observation state, not only average confidence. If the gain disappears after matching confidence mass, the result may be sample selection or gradient weighting rather than better uncertainty. If focal-style patterns remain absent in a sufficiently broad fitting diagnostic, this is unlikely to explain the main gap.

**What is already done.** The committee design already rejects across-window spread as epistemic uncertainty. That correction is not a new invention. The untested edge is whether the full factorized packet supplies useful localized evidence or a better student objective. Scalar teacher logits cannot provide patch localization that the source model does not expose.

**Cost and gate.** Mostly packet analysis and a bounded loss contrast. The small eight-study probe is insufficient either to establish or refute a population-level focal effect. Teacher eligibility and shared-prefix equivalence remain separate requirements.

### Fifth edge Allocate a fixed image budget to the most useful anatomy

**Concrete opportunity.** Equal total slice counts can hide different coverage. Six equally sampled series may spend views on redundant regions while a narrow lesion lies between selected slices. A compact fully trainable backbone may be useful because it makes adaptive coverage affordable, not because convolutions inherently dominate transformers.

**Proposed intervention.** Reuse one backbone in a coarse-to-fine policy: a coarse global pass supplies context and candidate locations; the same learned backbone processes a fixed number of high-resolution or adjacent-slice views. Retain a fixed global-coverage quota so attention cannot discard the anatomy needed to interpret a suspected lesion. Learn the selector only inside fitting folds, with a deterministic policy at inference. This remains one backbone/checkpoint but uses more than one pass.

**Control and strongest falsifier.** Compare learned selection with uniform selection and random selection, all given the same coarse pass, the same high-resolution budget, and the same input information. Compare against simply spending that budget on more uniform views. If the selector's apparent benefit vanishes after this accounting, it is extra compute, not better coverage. If selections track scanner or padding signatures, stop.

**What competitors lack.** The inspected nartaa recipe uses fixed slot allocations, random training windows, and dense fixed evaluation windows. An adaptive selector is absent from that disclosed path. Its presence in higher-scoring private methods is unknown. The existing patch-focus work may already cover part of this mechanism; assess its actual selector before implementing a duplicate.

**Cost and gate.** This has higher engineering risk than a head or label contrast and potentially significant DICOM/cache cost. It earns a pilot only after a cheap coverage diagnostic shows missed informative regions and the current patch-focus result does not already solve the problem.

### Sixth edge Keep evidence confidence separate from disease strength

The source audit records that an original extraction mapped unaddressed findings to 0.5. A blend can also produce 0.5 by averaging conflicting source values. These are different evidence states, and neither establishes a calibrated 50% clinical probability. Severity grades encoded as fractional targets are yet another quantity. Recover source provenance where possible and keep presence/severity targets and evidence weights separate. Do not reconstruct masks by checking the final target for equality to 0.5.

The live source card's suggested `2 abs(p - 0.5)` weighting is an especially cheap first hypothesis within this branch, not a validated uncertainty estimator. Test it against the current all-one policy with identical target values and an explicitly matched weighting budget; report how it changes rare-positive and ambiguous-cell exposure. Removing the pull toward 0.5 may help, or it may discard the precise boundary cases needed for generalization. [Live publisher card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)

A minimal test would preserve final target values while changing weights only on source-identified uncertainty states, with an equal-weight-mass random-cell control. A more ambitious concept teacher would retain each admitted source's explicit evidence states rather than compressing them prematurely to a scalar. This needs source-level records or reproducible generation logic; the final CSV alone is insufficient. A gain that vanishes under matched gradient mass is a weighting effect, not evidence that the model resolved uncertainty semantics.

### A cheap gradient diagnostic before adding more tasks

At one frozen, known-scope checkpoint, use identical fitting-only study bags and RNG to evaluate the four existing label/mask policies on the same logits. This requires no new supervision targets. Decompose originally observed and newly activated cells, while treating final 0.5 values as a numeric stratum rather than a recovered missingness label.

Measure per-target normalized residual energy and gradient norms on the actual shared projection. Pairwise cosine is useful, but also report each target's first-order directional loss change under the combined update. For an optimizer parameter displacement `Delta theta`, a negative `gradient_j dot Delta theta` predicts a local decrease in task j's loss. Include the optimizer's preconditioning and clipping when interpreting that direction. Target-specific output rows are disjoint and can look orthogonal even when their shared-encoder updates interfere.

Repeat on a fixed collection of fitting batches to distinguish systematic conflict from minibatch noise. If only feature-space gradients are affordable, label them as a proxy: the encoder Jacobian can change angles when mapping them to parameter gradients. This analysis can motivate one bounded weighting intervention; it cannot prove that balancing noisy task losses improves macro AUC. A new multitask optimizer should not be introduced merely because some gradient cosines are negative.

There is also a simple normalization explanation to exclude before invoking task conflict. At common fixed logits, for a target with the same active-task accounting and `n_original > 0`, the per-target four-arm gradient interaction is `g_full - g_value_only - g_coverage_only + g_original = -(n_added / n_total) (g_value_only - g_original)` under per-target mean normalization. The new cells cancel and the old-cell value change is diluted by the larger denominator. This identity does not describe the AUC interaction after four separately trained models, but it demonstrates why the endpoint interaction alone cannot identify a biological missingness effect or cross-task gradient conflict.

### Order the high upside bets by what the next evidence shows

The new lineage finding makes source-induced target coupling and severity-boundary errors the most concrete ways a successful teacher pipeline could still have headroom. Physical-support aggregation is the best architecture-level bet if outputs are sensitive to duplicate coverage or lose focal evidence. Adaptive selection is the larger engineering bet if fixed sampling demonstrably misses useful anatomy. None should displace a completed matched experiment merely because it sounds more original.

An ambitious result would be one student that learns complementary image evidence while avoiding shared source-label errors and spending its image budget more effectively. That combination could exceed a teacher or a reproduced baseline; it is not mathematically impossible for a student to outperform its teachers. It is also not guaranteed by distillation theory or creator comments. Each added component must earn its place through its own falsifier before being combined.

## What I would do first if execution were separately approved

First, finish the current result ledger and supervision-lineage audit. Next, inspect fitting-only teacher/window disagreement and exact gradient exposure. Then run one constant-total-weight, cross-fitted image-supervision contrast on the best stable existing student, with provenance limits stated honestly. A promising result earns a softening control and replication. A failure selects one diagnosis branch, not a larger ensemble by default.

Keep one compact full-fine-tuned family challenger ready as the practical alternative. The best expected next gain currently appears more likely to come from better traceable supervision plus enough adaptation than from an elaborate new pooling module. The broader inventory remains useful because the diagnostics can identify the specific situation in which one of those less conventional ideas becomes the right experiment.
