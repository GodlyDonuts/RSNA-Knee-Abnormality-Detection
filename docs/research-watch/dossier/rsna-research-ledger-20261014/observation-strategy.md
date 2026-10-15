# A single model that spends its observations better

Research snapshot: October 14, 2026. Repository pinned to `8300cfd20c2acaeef9d684a81dbc7fe977282a62`.

This is a research proposal, not an executed experiment. No MRI cases, clinical reports, private prediction rows, training, model downloads, repository changes, or external publication were used for this supplement. It combines aggregate project evidence, targeted source inspection, primary papers, and the coordinated public-notebook audit.

## Recommendation

The useful alternative to a larger encoder is **one shared encoder with an explicit observation budget**: preserve complementary acquisition categories, inspect broad context cheaply, and spend expensive views where they add information. The strongest low-risk experiment is a minimal complementary-series exchange. The more ambitious experiment is a fixed-budget low-resolution survey plus high-resolution detail, with an adaptive selector tested against a uniform selector.

Neither is evidence that a 0.95+ result is attainable. The existing record actively argues against “more views will fix it”: whole-series coverage, doubled unique training-slice exposure, neighboring channels, and plane memory have already failed to establish a macro improvement. A credible advantage must preserve something those experiments discarded, or eliminate a specific source of redundancy. Supervision provenance remains at least as important as this observation strategy.

For a leader-specific challenge, the most interesting cheap questions are different: Are cross-slot triplets useful or accidental? Does the second mirrored view add less than a complementary full-field view? Does the model confuse repeated correlated windows with greater anatomical extent? These deserve falsifiable tests, not assumptions that the published method is broken.

## 1. What is already implemented, tested, or genuinely missing?

Direct source checks:

- [The native selector](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/data.py) reserves a series from each available plane, then fills by ordering/tier. It does not explicitly reserve every complementary plane/contrast category
- [The competitive loader](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/data.py) already supplies signed physical position relative to each series midpoint, plus a geometry-known flag. It deliberately does not invent positions
- [The model](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/model.py) already has target-specific slice and series attention, plane/contrast features, optional patch readouts, and masking
- [NeighborSliceContext](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/context.py) already mixes previous/current/next features within a series, using physical gaps where known. “Add a cheap neighbor mixer” is therefore not new
- [Plane memory](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/plane_memory.py) pools patches into target-specific sequence summaries and exchanges evidence across sequences. It does not claim registration or retain explicit physical support length
- [The completed neighbor-input family](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/neighbor_input.py) uses bounded neighbor-delta channels, with original-volume donors and matched distant donors. That is not identical to packing three raw sampled slices into RGB

The [completed-experiment ledger](../../additional/rsna-research-ledger-20261014/experiment-ledger.md) establishes these no-repeat boundaries:

| Existing comparison | What it actually says |
|---|---|
| Whole-series, 224 input / 256 cache | 0.86034 versus 0.86209; no demonstrated gain from adding all coverage under that recipe |
| Stratified versus uniform | Unique source exposure rose from 185,929 to 364,647 at identical presentations; internal delta +0.00091 with interval spanning zero |
| Physical/distant neighbor channels, three epochs | No established primary or macro advantage for physical neighbors; not a full-schedule test of raw RGB triplets |
| Plane memory, three epochs | Macro +0.00428, interval spanning zero; secondary meniscus improvement does not replace its primary endpoint |
| Frozen patch-attention replacement | Aggregate decline; do not pitch the same replacement again |
| Native mean-patch residual | Fixed-final macro decline of about 0.01864; generic patch averaging is not the answer |
| Whole-window mean/query bags | Implemented and recovery-tested, without a final accuracy result; four-group training already spends about four times the one-group encoder work |
| Patch-focus residual | Already implemented with exact zero-query identity and target-specific pooling; no final accuracy result at the pinned snapshot |

**Concrete opening:** the fitting-only selector audit found 214/2,247 studies with an available sagittal/coronal non-fluid category omitted, and 924 studies with repeated categories. The 214 is an upper bound on a conservative exchange experiment's affected population, not proof of missing diagnostic evidence. Category duplicates can be genuinely complementary. [Audit](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/tree/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/fitting-series-coverage-20261007-v1)

## 2. The public leader is an observation-allocation clue, not a causal experiment

The [exact nartaa 0.949 source](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397) uses 96 selected slices across five slots, allocated 26/22/18/12/18. Native also nominally uses 96 slices. Its 94 overlapping triplets and two views mean 188 encoder presentations, not 188 unique anatomical observations. Its source uses a 140 mm field rendered to 384, then a native 320 center crop. See the [full coordinated source audit](../rsna-single-model-evidence-2026-10-14.md) for hashes, training differences, score evidence, and unresolved provenance questions.

Three structural deductions follow. They are hypotheses about opportunity, not measured defects:

1. **Slot boundaries are not physical neighbors.** With all five slots present, two triplets cross each of four boundaries: eight of 94, or 8.51%. Missing internal slots add another complication. This creates a controlled test of accidental cross-acquisition mixing
2. **A central crop excludes peripheral field.** The idealized retained width is 140 × 320/384 = 116.7 mm before any additional source-specific geometry effects. That arithmetic does not establish that a finding lies outside it
3. **A mirrored view is mainly an invariance observation.** A complementary field may supply new evidence for the same computational price. The source audit reports only a rounded 0.001 author-observed mirror gain, which is insufficient to predict the outcome of replacing it

The model can still be excellent despite all three. Cross-contrast channels might help; peripheral content might be noise; mirroring might regularize precisely the relevant failure. Its higher score also comes with materially different supervision, fitting population, adaptation, and selection history.

## 3. Twelve concrete hypotheses

### H1. Minimum-disruption complementary-series exchange

**Mechanism:** within the same six-series budget, replace a redundant selected category with an omitted tier-0 sagittal/coronal non-fluid category. Preserve existing fluid-sensitive fat-suppressed plane coverage. Start with the smallest exchange set rather than rewriting selection globally.

**Distinct from prior work:** changes information type, not total depth or number of series. The full-series experiment did not isolate this exchange. It is an existing diagnostic idea, not a newly discovered omission.

**Kill test:** if the new category is mostly a descriptor alias, unsuitable quality, or a redundant acquisition after source-semantic review, stop before fitting. In a future matched fit, no improvement in the prespecified affected subset and no overall gain means stop; do not escalate to many selector variants.

**Cost / bet:** approximately unchanged encoder work. Best final-week engineering risk, probably modest accuracy ceiling because most studies are unchanged.

### H2. Replace redundant TTA with complementary spatial views

**Mechanism:** for a crop-based model, compare central crop + mirror with central crop + full-field resized view, all at the same encoder resolution and presentation count. A training-compatible variant uses one global low-detail view and one true-pixel detail crop with the same shared weights.

**Distinct:** this is field-of-view complementarity, not the failed isolated p1–p99 contrast change, not synthetic super-resolution, and not a second model. Native already sees the full field, so it is primarily a leader/source-slot challenge rather than a diagnosis of the native baseline.

**Kill test:** compare the complementary view to an equal-cost second central view and to mirror. If benefits vanish after matching crop scale or are confined to acquisition strata, reject the anatomical explanation. A test-time-only change can fail from distribution mismatch and should not be called a general refutation of paired-view training.

**Cost / bet:** exactly equal encoder calls when output sizes match; extra decode/resize time still counts. Attractive cheap diagnostic, conditional on lawful access and a relevant checkpoint.

### H3. Broad low-resolution survey plus a fixed high-resolution budget

**Mechanism:** per series, encode up to 32 slices at 192 and eight at 384, using one shared DINOv3-L. Retain broad contextual coverage while spending fine-detail compute on fewer slices. Pool by source identity so a high-resolution revisit is not counted as independent corroboration.

**Distinct:** whole-series coverage previously increased observations at one resolution; this exchanges resolution for coverage under a fixed token budget. It does not repeat mean-patch pooling or window-bag training.

**Kill test:** uniform detail selection must first remain competitive with native 16×384. If low-resolution survey views erase focal evidence and eight detailed views cannot recover it, stop. More observations at lower fidelity can be worse.

**Cost / bet:** 32×144 + 8×576 = 16×576 patch tokens per series. Attention-pair terms are 62.5% of the reference, ignoring special tokens; this is not a wall-time claim. Higher upside, moderate-to-high implementation risk. See Experiment B.

### H4. Progressive detail acquisition with a coverage floor

**Mechanism:** extend H3 only after its uniform control: reserve half the detailed views for a physical coverage grid and choose the rest by scout-feature novelty or disagreement. Use a fixed maximum budget, not unconstrained uncertainty escalation. Start with a deterministic selector, not reinforcement learning.

**Distinct:** attention chooses how to read already encoded images; this decides which additional images deserve expensive encoding. Existing patch-focus does not solve that problem.

**Kill test:** equal-compute uniform and diversity-only policies must be beaten. High attention is not lesion localization, and low confidence is not necessarily useful information. Require stable behavior under benign intensity or sampling perturbations; artifacts winning all refinement slots is a failure.

**Cost / bet:** same high-resolution budget as H3; serial routing and batching overhead may erase savings. AdaFocus establishes the general dynamic-compute mechanism on videos, not knees. [Primary paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_AdaFocus_V2_End-to-End_Training_of_Spatial_Dynamic_Networks_for_Video_CVPR_2022_paper.html), [author code](https://github.com/LeapLabTHU/AdaFocusV2)

### H5. Boundary-safe triplets with deliberate cross-contrast fusion

**Mechanism:** construct raw RGB triplets entirely inside one actual series. Spend the freed boundary-window budget on real in-series observations. If cross-contrast mixing is useful, represent it explicitly as typed acquisition evidence after the shared encoder rather than pretending it is a spatial neighbor.

**Distinct:** own source-slot processing is already boundary-safe; this challenges the public concatenated-window recipe, not an undiscovered native bug. The existing bounded-delta neighbor test is not raw-triplet equivalence.

**Kill test:** compare original boundary windows, matched boundary-safe replacements, and boundary-only perturbations at equal count. If original cross-slot windows consistently improve a matched model, physical purity is the wrong optimization criterion. A source-faithful reproduction and a corrected alternative must retain separate identities.

**Cost / bet:** constant encoder count; small affected fraction limits expected gain. Missing-slot handling and mirror behavior make this a valuable correctness audit even if it does not improve AUC.

### H6. Duplicate-neutral physical-support summaries

**Mechanism:** preserve a target-specific summary of evidence spread in physical space alongside the normalized pooled feature: unique covered intervals, contiguous support, and separate strongest-evidence statistics. Weight overlapping observations by their unique physical support rather than presentation multiplicity. Do not multiply probabilities as if adjacent windows were independent.

**Distinct:** target queries and plane memory already exist, but neither explicitly represents absolute union support. The proposed addition must retain order/measure, not merely add another query.

**Kill test:** duplicate the same represented interval and require unchanged support; extend identical evidence into a new adjacent interval and require a changed support summary. The [synthetic counterexample](../../additional/rsna-research-ledger-20261014/synthetic-extent-counterexample.json) proves this operator-level distinction only. Current position-bearing features can already carry some extent implicitly, so it is not a proof that the full native model cannot represent extent.

**Cost / bet:** small head cost after existing embeddings. Potentially useful for severity/extent, but an unlocalized attention score is not a measurement of pathology, and through-plane length does not measure cartilage depth. Good medium-term mechanism, not a guaranteed leaderboard fix.

### H7. Cross-plane corroboration gated by actual coordinate validity

**Mechanism:** add sparse messages between patches whose transformed physical coordinates are compatible, using acquisition type as context. Do this only where orientation, position, spacing, crop transforms, and frame compatibility are validated. Else retain independent-series pooling.

**Distinct:** existing series-relative scalar positions and plane-memory exchange do not establish cross-plane correspondence. A same-index sagittal/coronal match is invalid.

**Kill test:** a coordinate shuffle that preserves image/plane statistics should destroy the proposed benefit. If shuffled geometry performs equally well, the branch is learning co-occurrence rather than correspondence. Known patient motion or invalid frames must disable matching rather than create confident false agreement.

**Cost / bet:** sparse head work is cheap; recovering correct transforms and verifying registration assumptions is not. Poor final-week bet unless the geometry already exists. [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html) defines position, orientation and spacing; it does not guarantee absence of inter-series motion.

### H8. Train for missing acquisitions without equating absence to normality

**Mechanism:** structured series dropout with explicit availability masks, preserving at least one usable acquisition and comparing full-input and naturally missing-input strata. Reallocate unused view budget only under a fixed rule, not according to guessed diagnosis. Use whole-study supervision on the remaining bag; do not assign the study label to every surviving slice.

**Distinct:** masking padded inputs prevents numerical contamination; it does not train robustness to missing informative acquisitions. Native already has correct masks, so this is a training-distribution hypothesis.

**Kill test:** simulated dropout must improve naturally missing-series cases rather than just its own artificial test distribution. A material full-input regression defeats the final-week argument. Missingness can be informative about protocol or selection, so forcing invariance is not automatically correct.

**Cost / bet:** usually fewer encoder presentations; matching total learning exposure remains necessary. ModDrop supplies the general precedent, in gesture recognition rather than MRI. [Primary paper](https://arxiv.org/abs/1501.00102)

### H9. Separate uncertainty about unobserved anatomy from low class probability

**Mechanism:** when considering early stopping or reallocating views, track coverage and prediction stability separately. A stable negative from an incomplete survey is not proof that unobserved anatomy is negative. Use a mandatory per-plane/contrast observation floor and a fixed worst-case cap.

**Distinct:** this is a stopping-policy safety condition, not probability calibration to improve AUC. It could free budget for hard studies, but it must preserve ranking.

**Kill test:** evaluate error/ranking loss conditional on rare targets and missing acquisitions, not only average confidence. If the calibration subset has too few positives to validate exits, retain the full budget. Do not promise distribution-free guarantees from a reused small development set.

**Cost / bet:** possible average savings, unchanged worst-case budget. Poor final-week bet given sparse positives. [Fast yet Safe: Early-Exiting with Risk Control](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea5a63f7ddb82e58623693fd1f4933f7-Abstract-Conference.html) motivates explicit risk control rather than a confidence threshold alone.

### H10. Audit anatomical mirror semantics instead of applying universal flips

**Mechanism:** verify the transformation in patient coordinates, including slot order and channel order. On boundary-crossing triplets, a mirror rule based only on the central slice can apply one plane's convention to other-plane content. Consider excluding such triplets from mirror TTA or using typed series-local transforms.

**Distinct:** this is not generic flip augmentation. It concerns exact orientation, handedness, and cross-slot semantics in the inspected public source.

**Kill test:** the transformation should be an involution with preserved identity, masks, and target meanings. But passing the double-flip test is necessary, not sufficient for anatomical validity. If geometry cannot verify the rule, describe it as an image transform rather than anatomical mirroring. Any trained/evaluated comparison needs the original mirror route as its equal-budget control.

**Cost / bet:** negligible extra arithmetic; potentially expensive geometry QA. Useful source audit, unknown accuracy impact. Avoid claiming the current public transform is wrong without original-orientation evidence.

### H11. Protocol-shortcut audit before metadata-guided adaptation

**Mechanism:** test whether acquisition counts, field of view, spacing, plane/contrast availability, or border/background cues explain apparent ranking. Preserve useful contrast information while challenging spurious protocol associations. A diagnostic metadata-only model is not a proposed submission and would require separate execution approval.

**Distinct:** existing metadata features are legitimate interpretive context; simply deleting them can hurt. The issue is robustness to acquisition shifts, not moralizing about all shortcuts.

**Kill test:** if an image change helps only a protocol-defined subset and a metadata-only diagnostic captures the same effect, the claimed visual mechanism is weakened. Use group-disjoint splits and protocol-stratified reporting; do not mine site identifiers for submission gains.

**Cost / bet:** low-cost diagnostics; full metadata-guided domain adaptation is a separate research project. [FINO](https://arxiv.org/html/2606.05107v1), linked from the [official DINOv3 repository](https://github.com/facebookresearch/dinov3), offers a 2026 method for retaining informative factors while suppressing spurious ones. Its medical evidence is chest radiography, not knee MRI. [Zech et al.](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683) demonstrates the general danger of hospital-associated shortcuts in a different task.

### H12. Token compression with a protected detail reserve

**Mechanism:** merge or drop redundant/background tokens early enough to reduce encoder work, then spend the saved budget on more original views. Preserve a uniform token floor and selected detail regions. Compression after the expensive encoder saves little encoder compute.

**Distinct:** patch-focus changes readout; token compression changes internal compute. [ToMe](https://arxiv.org/abs/2210.09461), [TokenLearner](https://arxiv.org/abs/2106.11297), and [PatchDropout](https://arxiv.org/abs/2208.07220) establish different forms of token efficiency in other settings.

**Kill test:** measure end-to-end speed, not attention FLOPs alone. A rare lesion can resemble a low-salience background token; if the detail reserve does not protect it, reject. DINOv3 rotary positions, dense feature assumptions, mixed precision, and checkpoint recovery all need fresh verification.

**Cost / bet:** theoretically substantial, practically uncertain. Worst final-week bet here: changes the most mature numerical path and may discard exactly the weak evidence the task needs.

## 4. Best experiment A: complementary categories without buying more compute

**Question:** does preserving an omitted acquisition type help more than retaining a second acquisition of an already represented type?

**Control:** the exact native-public six-series / 16-slice / 384 recipe, official DINOv3-L rank32, accepted fitting targets/masks, original fold boundaries, fixed final epoch and unchanged optimizer schedule. Use the verified original compatible control only if scientific and numerical equivalence is established; similar names are insufficient.

**Candidate selection rule, frozen before outcomes:**

1. Begin with the exact original six-series selection
2. Identify an omitted tier-0 sagittal/coronal non-fluid category using validated existing descriptor semantics; do not relabel it “T1” without evidence
3. Replace only a redundant selected category while preserving all originally retained fluid-sensitive fat-suppressed plane categories and at least one series per plane
4. Choose at most two exchanges with deterministic existing quality/tie-break rules
5. For the strict equal-presentation primary comparison, allow an exchange only when `min(original_depth,16)` matches `min(replacement_depth,16)`. Otherwise retain the control choice and report the exclusion count. This may reduce the affected population below 214
6. Preserve source order, real geometry, masks, image scale, and exactly the same unique-slice limit. No fabricated or repeated slices

**No-label engineering gate:** produce only aggregate category counts, eligible-study count, exact encoder-presentation equality, and unchanged-case identity checks. Validate that the descriptors identify truly different acquisitions. This task has not performed that gate.

**Matched controls / falsifiers:** the primary two-arm comparison isolates the category exchange. If selection changes also alter quality tier, slice count, crop, or supervision, it is no longer that experiment. An optional subsequent sham exchange within the same category can test whether any new-series perturbation helps; do not silently add this after a convenient outcome. Record the overall result and the prospectively defined affected subset. The latter is a mechanism diagnostic, not a replacement primary endpoint.

**Budget:** same maximum 96 encoder images/study. Target measured training/inference time no more than 1.05× the exact control, including I/O. Planned scientific horizon: ten epochs / 2,810 updates on the existing 2,247/1,156 split, not a three-epoch verdict. A conservative planning ceiling is 48 H100 GPU-hours for a fresh two-arm comparison plus finite recovery/evaluation; admit it only after measured throughput demonstrates it fits. This is a proposed ceiling, not a reservation or predicted runtime. Reusing a genuinely equivalent completed control can cut the new fit count to one.

**Promotion rule:** predeclare the minimum useful improvement before running. A reasonable research screen is +0.005 macro AUC point estimate, no clear broad-target regression, and an affected-subset result consistent with the proposed mechanism; that screen is not independent confirmation. Use the same leakage-component paired analysis and report undefined bootstrap draws. A rare-target-only jump or an interval compatible with meaningful harm is insufficient for a strong claim. If the subset is too small for a clear answer, say inconclusive rather than growing the scope after seeing results.

**Why it could fail:** category omission may be harmless, duplicate series may have better quality, labels may not reward the extra information, or changes may affect too few studies to matter. A clean negative should close this narrow question.

## 5. Best experiment B: shared-encoder survey and refinement

**Question:** can the model retain broader anatomical coverage at the same encoder-token budget without sacrificing focal detail, and can a selector spend that detail budget better than uniform allocation?

### Stage B0: establish whether the resolution trade is viable

Compare native 16×384 per series against a mixed-resolution arm with up to 32×192 survey slices and eight×384 detail slices. Use uniform physical coverage for detail selection initially, falling back to validated ordinal coverage where geometry is unknown. Keep the same six-series selector to avoid confounding with Experiment A.

One encoder, one learned inference state, one study head. Add a view-scale/type indicator if needed, but use source identity and hierarchical pooling so the detailed revisit replaces/refines the matching coarse evidence rather than counting as a second independent location. Retain the original grayscale/input normalization contract; do not silently introduce the public crop/contrast bundle.

For six fully populated series and patch size 16:

- Native: 96×576 = **55,296 patch tokens**
- Mixed: 192×144 + 48×576 = **55,296 patch tokens**
- Sum of per-image squared patch counts: **31,850,496 → 19,906,560**, or **0.625×**
- Encoder image presentations: **96 → 240**

The arithmetic excludes special tokens and does not model projection/MLP costs, kernel launches, decode overhead, memory traffic, backward graphs or batching. The mixed arm may be slower despite equal patch-token count. Short series use only actual unique source slices; do not fill to 32 by duplication. Report realized, not nominal, budgets.

### Stage B1: isolate the adaptive selection mechanism

Only if the uniform mixed-resolution arm is viable, compare two otherwise identical mixed-resolution fits:

- **U:** eight uniformly allocated detail slices per series
- **A:** four uniform anchor detail slices plus four novelty-selected survey locations, with a fixed physical-separation rule and deterministic tie-breaks

Use stop-gradient normalized survey features for a predeclared diversity selector, rather than adding a separate scout backbone or an RL policy. If task-attention selection is later considered, it needs a separate comparison; it is not interchangeable with diversity. Uniform anchors prevent complete policy collapse but do not guarantee lesion coverage.

Both arms have the same survey images, high-resolution count, learned graph, losses, optimizer updates, precision, crop, and series selector. Only detailed locations change. Changing routing during training changes the learning trajectory; it cannot be evaluated as an inference-only speedup claim.

**Falsifiers:**

- B0 falls materially below native, showing that reducing detailed observations loses more than broad context contributes
- A does not beat U at equal measured cost, or selects mostly artifacts/background
- Selection is unstable to benign intensity or survey-grid changes
- Gains require extra teacher labels, more training steps, a larger head, or hidden extra high-resolution views
- Runtime or host memory exceeds the deployment envelope even though FLOPs look favorable

**Budget:** do not launch all three arms together. First require actual end-to-end throughput at ≤1.10× native for the mixed arm and fit the original offline two-T4 inference envelope with margin, using an eight-hour internal target under the nine-hour hard limit. A later U/A pair gets a separate proposed 48-H100-hour ceiling and fixed ten-epoch horizon, contingent on measurement. B0 and B1 together can cost several full fits; if current pending experiments leave insufficient time, do not call this a final-week quick win. Preserve the two-concurrent-node and storage-reserve constraints already documented by the project.

**Why it could dominate rather than merely imitate:** it can see additional source locations while reserving native-resolution evidence, without adding parameters or relying on correlated mirrored windows. The mechanism is genuinely different from the leader's dense triplet aggregation. **Why it probably will not dominate immediately:** the scout may fail precisely on subtle lesions, the selector lacks localized supervision, mixed-resolution training can destabilize the pretrained representation, and noisy study labels may reward shortcuts instead of better inspection.

## 6. Three cheap leader-specific falsification tests to design before any new fit

These are proposed tests only; no predictions were generated here.

1. **Boundary sensitivity:** preserve the 86 within-slot triplets and manipulate only the eight cross-slot triplets in a fully populated 94-window bag. Compare replacement, removal with weight renormalization, and a count-matched within-slot perturbation. Effects identify dependency on boundary content, not whether the resulting diagnosis is more correct. Missing-slot cases need a separately defined roster
2. **View complementarity:** under one unchanged checkpoint, compare central + mirror with central + full-field at equal image sizes/counts. Assess future held-out ranking and target-specific deltas only under authorized evaluation. Test-time distribution mismatch remains a limitation
3. **Measure versus multiplicity:** on synthetic embeddings, distinguish one interval, duplicates of that interval, and distinct adjacent intervals with identical evidence. Then, in a future matched MRI study, test whether a physical-support residual helps beyond existing positions and plane memory. Synthetic success is only an admission test

## 7. Literature: what transfers and what does not

- **Adaptive observation:** AdaFocusV2/V3 demonstrate dynamic spatial/temporal compute allocation, including the optimization difficulties of learning where to look. They are video recognition evidence, not proof of MRI lesion detection. [V2](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_AdaFocus_V2_End-to-End_Training_of_Spatial_Dynamic_Networks_for_Video_CVPR_2022_paper.html), [V3](https://arxiv.org/abs/2209.13465)
- **Minority evidence:** Attention2Minority selects salient instances relative to normal-slide representations. This motivates a selection failure hypothesis, but a normal MRI reference can encode protocol differences and weak-negative errors. Its author code says its selected instances work better with non-contextual models. [Paper](https://arxiv.org/abs/2301.07700), [author code](https://github.com/JoeSu666/SiiMIL)
- **Dynamic MIL:** PAMIL couples sampling, representation and decision-making in whole-slide pathology. It supports a progressive-observation research direction, not the claim that adding its RL machinery is cheap or appropriate here. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Dynamic_Policy-Driven_Adaptive_Multi-Instance_Learning_for_Whole_Slide_Image_Classification_CVPR_2024_paper.html)
- **Order context:** OAS-MIL uses a shared ResNet18 with set attention and a bidirectional sequence branch on tumor-centered liver MRI. Its 183-patient, single-center, expert-localized setting differs fundamentally from unlocalized twelve-target knee screening. Three-slice slabs outperform five in its ablation; more context is not monotonically beneficial. Its result motivates a control for order, not a repeat of existing plane memory or neighbor context. [Primary preprint](https://arxiv.org/html/2609.11271v1)
- **Resolution flexibility:** NaViT and FlexiViT support the broad idea of trading resolution/tokenization against efficiency in one model. They do not establish that a DINOv3 knee checkpoint can switch resolution or patch size without adaptation. The proposed first experiment changes image resolution, not patch-embedding weights. [NaViT](https://arxiv.org/abs/2307.06304), [FlexiViT](https://openaccess.thecvf.com/content/CVPR2023/papers/Beyer_FlexiViT_One_Model_for_All_Patch_Sizes_CVPR_2023_paper.pdf)
- **Dense features:** DINOv3's dense-feature and resolution-transfer results make shared coarse/detail encoding technically plausible. The project's failed patch-readout experiments show why generic foundation-model claims cannot replace an RSNA control. [Primary report](https://arxiv.org/abs/2508.10104)

MRI k-space undersampling papers were deliberately not used as evidence for these classification proposals. Choosing which measurements to acquire for reconstruction is a different problem from selecting already reconstructed slices for study-level diagnosis.

## 8. Adversarial final-week assessment

1. **Information quantity is not the demonstrated bottleneck.** Existing coverage experiments are negative/inconclusive, while the full-public-supervision recipe produced the clearest practical improvement
2. **The largest possible gain may require a better target, not a better view.** Report silence, severity thresholds, image pseudo-label ancestry, and fitting population remain confounded with high public scores
3. **Most new heads optimize an unverified proxy.** Attention, novelty, uncertainty, contiguous support, and cross-plane similarity are not ground-truth lesion localization
4. **The budget includes integration.** Mixed shapes, unique-source bookkeeping, source transforms, CUDA graphs, recovery, and hidden-test memory can cost more time than the nominal experiment
5. **Short runs are not decisive.** The native public learning curve improved substantially after early epochs. Another three-epoch result can become an expensive inconclusive result
6. **Sparse development positives make “domination” easy to hallucinate.** Preserve undefined bootstrap draws, component grouping, prespecified endpoints, and all negative outcomes. Do not replace macro with a convenient subset after seeing scores
7. **Do not fork every idea.** Reconcile existing patch-focus/window-bag/dense work first. A result from an already-running lineage is usually worth more than a new untrained abstraction
8. **No first-principles guarantee closes 0.872 to 0.95+.** The right ambition is to expose a measurable information bottleneck that competitors leave on the table, then kill weak explanations quickly

**Practical order:** finish/reconcile existing work; audit selector semantics and public boundary/mirror contracts; choose one fixed-compute experiment; keep the ambitious shared survey/refinement design ready only if measured runtime and remaining research budget justify it. Avoid a new large backbone, generic attention module, token-surgery rewrite, or RL policy as the default last-week move.
