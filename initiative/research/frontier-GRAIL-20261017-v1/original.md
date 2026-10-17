# Frontier Research Memo: An Original Path From 0.950 Toward a Winning RSNA Knee Model

## Executive conclusion

The project has now reached the point where **another backbone swap, another crop, another sampling heuristic, another longer fit, or another public-leaderboard blend is unlikely to create a defensible lead**.

The current evidence is unusually informative. Your independently trained N3 is a real, fully fine-tuned 73.1M-parameter MRI model and scores **0.894**. The project’s best measured **0.950** is a reproduction of released S6 weights/inference, not an internally discovered mechanism. The released CoAtNet is 0.949. Whole94 epoch 1 also scored 0.949, while context/regional/full-field additions scored 0.948. fileciteturn10file0L2-L23 fileciteturn10file3L79-L87

Meanwhile, the public frontier has already absorbed the obvious ideas. The 0.949 CoAtNet uses 96 selected slices, 94 three-slice windows, finding-specific MIL, 384-pixel training, a native 320-pixel inference crop, anatomical mirroring, soft-label blending, external knee MRI, and weight averaging. citeturn18search2turn24search8 The 0.950 S6 ConvNeXt uses finding-specific **spatial attention plus window-level MIL**, full-FOV and anatomy-localized 100-mm crops, 3.5-mm physical spacing, 64 sampled windows, 16 epochs, and 4,263 report-labelled studies. citeturn18search5turn24search10 A publicly shared multi-model recipe has reached **0.956**. citeturn18search3turn18search4 Your repository additionally contains an exact external **0.958 reference** consisting of CoAtNet branches, a four-fold physical/state-aware branch with stackers, and a three-checkpoint 2.5D reader, although your own committed reproduction had not yet produced a measured score at the last verified state. fileciteturn7file2L47-L70

So the objective should not be “improve MIL.” It should be:

> **Build a reader that represents the radiologist’s decision rule more faithfully than current bag-of-windows models, and train that reader so weak report supervision is accepted only when its gradient points in a direction consistent with image-derived expert labels.**

That leads to one coherent proposed system, rather than a collection of experiments:

**GRAIL-Knee: Geometry-Relational, Adjudication-Informed Learning.**

It has two scientific innovations:

1. **Criterion-relational MRI reasoning:** stop treating the examination primarily as an unordered collection of window features. Preserve physical patient coordinates and explicitly model relationships such as “the same abnormal signal appears on two neighboring images,” “this structure loses continuity across a short physical trajectory,” “evidence occupies at least a certain anatomical extent,” and “two planes intersect at the same anatomical location.”

2. **Clean-gradient supervision:** do not decide globally whether Steven/S6/JEV/report labels are “best.” For each target and training example, estimate whether the weak-label gradient is aligned with the gradient induced by held-out expert image labels, and down-weight weak examples whose update direction points the model away from the expert task.

The validation work is not a third model idea. It is the mechanism that stops us from fooling ourselves while developing the other two.

This is important because the competition public leaderboard is calculated on only about **30% of the test set**, while the final private leaderboard uses the other **70%**. citeturn19search3 With that split, a large search over public blends is exactly the environment in which leaderboard selection can choose a model that captured public-specific noise. The final deadline is **October 22, 2026 at 23:59 UTC**, so there is no room for a broad architecture search. citeturn18search0turn22search3

I also would **not** state that the current first-place score is 0.97 as an established fact. Kaggle’s indexed leaderboard page does not expose the live rows to search. I can independently verify public shared 0.956 models and your project’s externally sourced 0.958 recipe; discussion participants are predicting or aspiring to roughly 0.97–0.975, but that is not the same as a verified current leaderboard result. citeturn18search3turn22search1 **0.97 is nevertheless the correct engineering target** if the objective is a plausible winning submission rather than merely catching the public notebooks.

## What the evidence actually says

The most important distinction now is between an **established observation** and a **causal explanation**.

| Question | Established evidence | What is *not* established |
|---|---|---|
| Is your own model simply too small? | N3 already trains all ~73.1M parameters and reaches 0.894. fileciteturn10file3L79-L87 | That more parameters will close the gap. |
| Is basic window coverage the missing breakthrough? | Strong public CoAtNet evaluates 94 windows; S6 trains with 64 windows; your Whole94 did not improve at E1. citeturn24search8turn24search10 | That another 16-vs-32-vs-94 sampling adjustment will create +0.02 AUC. |
| Is “use adjacent slices” the answer? | Adjacent geometry has helped in some public controlled systems, but N3 already deliberately selects slices and stacks triples; your own neighbor experiment did not establish a gain. citeturn24search0 fileciteturn11file0L2-L15 | That successful models universally require consecutive original slices, or that changing adjacency together with preprocessing would isolate the effect. |
| Are weak labels materially different? | Yes. S6/JEV supervision and your current supervision differ across all 3,887 fitting studies; fracture rank correlation is only **0.51446**. fileciteturn9file0L2-L17 | Which source is closer to image-derived competition truth overall. |
| Are reports themselves ground truth? | No. The host explicitly says the expert labels were assigned from the images independently and are authoritative when the report disagrees. citeturn19search0 | That optimizing report-label agreement is equivalent to optimizing the competition task. |
| Is scanner/source shift real? | The competition spans heterogeneous sites/scanners/protocols; a competition metadata probe fell from 0.6516 random-fold AUC to 0.5981 under scanner grouping. citeturn18search1turn19search2 Independent medical-imaging research also finds scanner shifts usually reduce performance and that MRI is particularly sensitive. citeturn21search0 | The exact amount by which your current validation is optimistic. |
| Does current MIL already localize? | Yes. S6 has finding-specific spatial attention and window attention; CoAtNet has finding-specific MIL. citeturn18search2turn18search5 | That generic “add attention,” “add an ROI,” or “use a transformer” is novel enough to beat them. |
| Does the public 0.956–0.958 region show complementary model errors? | Yes: the shared 0.956 solution combines different model families, while your 0.958 reference is also heterogeneous. citeturn18search4 fileciteturn7file2L47-L70 | That a large ensemble tuned to public scores will generalize to the 70% private set. |

One fact in the validation lane deserves immediate attention. Your repository already performs meaningful report/pixel leakage protection: the training code checks that report groups and leakage groups do not cross the train/validation boundary, and the full image audit found a cross-study nonconstant pixel edge that is retained in leakage grouping. fileciteturn4file7L114-L123 fileciteturn4file13L207-L216

But a repository pilot examining actual DICOM headers found **PatientID present and non-empty in all ten sampled studies**, with ten distinct values. That pilot explicitly says it is only a presence test and does not establish patient identity or patient disjointness. fileciteturn15file0L1-L12 Another code audit explicitly warns that pseudonymous PatientIDs do not by themselves establish true patient disjointness. fileciteturn17file0L2-L16

That does **not** prove patient leakage. It means there is enough information present to do the full audit rather than infer patient isolation from StudyInstanceUID.

The second deep fact is even more important. The competition labels are not ordinary disease-presence labels. They encode **decision criteria**. The host defines a meniscal tear as signal reaching the surface on at least two images or a qualifying morphologic abnormality; MCL requires a high-grade acute tear with disrupted fibers and adjacent edema; ACL requires high-grade disruption; compartment OA requires a moderate/large region of high-grade cartilage loss; effusion and Baker’s cyst require moderate/large extent. Ambiguous borderline findings are deliberately graded negative. Two MSK radiologists label each examination, with a third adjudicating disagreements. citeturn19search1turn24search5

That changes what an ideal model should compute.

Current high-performing models already learn **where to look**.

The next frontier is teaching the model **what relation among observations constitutes a positive finding**.

## Validation lane: determine whether your metric is lying

The validation agent should not spend its time inventing another cross-validation scheme. It should answer one falsifiable question:

> **Does a split that connects likely same-patient studies and holds out scanner/protocol domains materially change our estimate of generalization or the ranking of candidate mechanisms?**

### Exact experiment

Construct a connected-component group graph over all eligible training studies.

Begin with every edge the repository already recognizes: `leakage_group`, duplicate/near-duplicate report links, exact/cross-study pixel-equivalence links, and other existing verified equivalence edges. The repository has already built explicit leakage-group machinery and report-similarity controls, so these should remain unchanged rather than be rediscovered. fileciteturn4file1L18-L27 fileciteturn4file2L34-L43

Add a **PatientID equality edge only after auditing it**. Read one header from each series, verify PatientID consistency within a study, retain the identifiers only as private hashed values, and count cross-study repetitions. A repeated pseudonymous PatientID plus compatible basic acquisition/anatomical information is strong evidence that the studies should not straddle validation folds; a unique PatientID does *not* prove a unique real person. The ten-study pilot establishes that the field is available, not that its semantics have been proven. fileciteturn15file0L1-L12

Then construct scanner domains from stable acquisition information such as Manufacturer, ManufacturerModelName, software/protocol signatures, field strength, pixel spacing, slice spacing and sequence composition. The competition itself warns that intensities, orientations and resolutions vary across studies, while a public metadata probe identified 265 fine-grained scanner fingerprints and measured a substantial random-vs-scanner-grouped gap. citeturn23search7turn24search11

Do **not** evaluate this with released S6/CoAtNet features as the sole representation, because your repository correctly notes that released encoders have already seen your internal cohort/Gold58; head CV on those representations is not independent whole-model OOF. fileciteturn8file0L2-L25

For a fast diagnostic, use a frozen ImageNet encoder with no RSNA exposure to extract study features once, then train exactly the same small classifier under:

\[
\mathcal S_\text{current}
\qquad\text{versus}\qquad
\mathcal S_\text{patient+scanner}.
\]

Nothing except assignment to folds changes.

This is **not** intended to select your final architecture. It measures whether the validation geometry itself rewards scanner/patient/source reuse.

### The failure criterion

The hypothesis is falsified if all of the following hold:

\[
|\mathrm{AUC}_{grouped}-\mathrm{AUC}_{current}| < 0.005,
\]

no meaningful cross-fold repeated-PatientID component exists, and the candidate ordering remains stable under scanner grouping.

If the macro gap exceeds roughly **0.01**, or if candidate ranking reverses, I would no longer trust conventional weak-label validation as the primary promotion endpoint. If it exceeds **0.015**, treat domain/group leakage as a first-order problem.

Those are operational thresholds, not claims of statistical significance.

### Compute budget

The header graph is essentially CPU/I/O work. Feature extraction needs a single pass over the images. The scientific budget should be **zero new end-to-end training** and at most one frozen feature extraction plus CPU head fits.

The important outcome is not another score. It is a trustworthy ordering criterion for everything that follows.

### Why this matters for winning

MRI is particularly vulnerable to scanner-domain shift in independent multi-domain research. citeturn21search0 More importantly, the challenge data were deliberately collected from a diverse international mixture of scanners, protocols and populations, and prevalence is not guaranteed to match among training, public, and final evaluation sets. citeturn18search1

That combination makes **source-invariant ranking** more valuable than squeezing another 0.001 from the public board.

It also provides a rigorous answer to the ensemble concern: a candidate that only looks complementary under the 30% public leaderboard but loses its advantage under patient/scanner-grouped validation should not be blended into the final submission.

## Supervision lane: use the expert labels to control gradient direction

This is where I think the project has its clearest opportunity for a genuinely original contribution.

You currently treat label source as a global recipe choice: Steven labels, S6/JEV labels, teacher targets, mixtures and masks. But the evidence says this abstraction is too coarse.

The two weak-label systems differ in every one of the **46,644 target cells** across the 3,887 fitting studies, and fracture rank correlation between sources is only 0.51446. fileciteturn9file2L39-L54 The host simultaneously tells us the report and image-derived labels answer slightly different questions: image-derived consensus labels use stricter thresholds and take precedence when they disagree. citeturn19search0

The right question therefore is not:

> Which label table is best?

It is:

> Which weak-label updates move the image model in the same direction as the expert-image task?

### The linear algebra

For target \(t\), let weak training example \(i\) produce loss gradient

\[
g_{it}=\nabla_\theta \ell(f_\theta(x_i),\tilde y_{it}).
\]

Let a held-out subset of the 58 image-adjudicated cases produce a clean gradient

\[
g_t^{G}=\nabla_\theta L_t^{G}(\theta).
\]

If the next weak-data update is

\[
\theta'=\theta-\eta\sum_i w_{it}g_{it},
\]

then a first-order Taylor expansion of the clean loss gives

\[
L_t^{G}(\theta')
\approx
L_t^{G}(\theta)
-\eta
\sum_i
w_{it}
\left(g_t^{G}\right)^\top g_{it}.
\]

That dot product is the key.

When

\[
\left(g_t^{G}\right)^\top g_{it}>0,
\]

the weak example points approximately in a direction that reduces the expert-image loss.

When it is negative, the weak supervision is asking the network to move in a direction that locally *increases* the expert-image loss.

This is not an exotic mathematical trick. Learning to reweight noisy examples according to their effect on a clean validation objective was established by Ren et al.; their method explicitly derives example weights through a meta-gradient against a clean unbiased set. citeturn20search0turn20academia48

The RSNA setting gives that idea an unusually strong interpretation: **the clean set and the weak set correspond to two genuinely different annotation procedures.**

### Why ordinary gradient alignment will overfit Gold58

Fifty-eight examinations are too few to trust a full 73M-dimensional gradient direction.

So do not use the full gradient.

For each target, form gradients only with respect to the final target head and optionally the last encoder block:

\[
G_t=
\begin{bmatrix}
g_{1t}^{G\top}\\
g_{2t}^{G\top}\\
\vdots
\end{bmatrix}.
\]

Compute

\[
G_t=U_t\Sigma_tV_t^\top
\]

and retain only the leading \(r\) right-singular directions:

\[
V_{t,r}=[v_1,\ldots,v_r],
\qquad r\in\{2,3,4\}.
\]

Project every weak gradient into this clean subspace:

\[
\hat g_{it}
=
V_{t,r}V_{t,r}^\top g_{it}.
\]

Then define a robust alignment weight such as

\[
a_{it}
=
\frac{
(g_t^{G})^\top\hat g_{it}
}{
\|g_t^{G}\|\|\hat g_{it}\|+\epsilon
},
\]

\[
w_{it}
=
\operatorname{clip}
\left(
w_{\min}+
(1-w_{\min})\sigma(a_{it}/\tau),
\,w_{\min},1
\right).
\]

This low-rank projection is doing something important: it throws away weak-gradient directions that the tiny gold set has no evidence to identify reliably.

I would also compute separate clean directions for gold positives and gold negatives when support permits. Otherwise those gradients can partially cancel while representing distinct error modes.

### Cross-fitting is mandatory

Partition Gold58 into four leakage-safe groups.

For fold \(k\), learn the gradient subspace and weak-example weights from only

\[
G\setminus G_k.
\]

Make predictions on \(G_k\).

After all four folds, concatenate predictions and compute the twelve AUCs once over the entire set of 58 cross-fitted predictions. This avoids the frequent undefined-AUC problem caused by trying to calculate a rare-target AUC independently inside every tiny fold.

No Gold58 example may participate in producing the weights or checkpoint used to predict itself.

And do not use a released encoder for the scientific proof if that encoder previously saw Gold58. Your repository already warns that this invalidates whole-model independence. fileciteturn8file0L2-L25

### The actual falsifiable supervision experiment

**Hypothesis:** weak examples whose gradients oppose the cross-fitted expert-label gradient account for a measurable subset of the model’s gold ranking errors; suppressing those gradients improves expert-image ranking without changing image preprocessing, sampling, architecture, optimizer schedule or label source.

**Matched baseline:** same frozen clean encoder, same features, same weak target values, same mask, same batches, same optimizer, same initialization.

**Treatment:** change only \(w_{it}\) using low-rank clean-gradient alignment.

This is more informative than yet another Steven-vs-S6 label-table run because it tests the mechanism we actually care about.

Promotion requires:

\[
\Delta \mathrm{AUC}_{Gold58,\ crossfit} \ge 0.008
\]

together with improvement on a majority of well-supported targets, no severe regression on a well-supported target, and no deterioration exceeding roughly 0.003 on the scanner-grouped weak-validation endpoint.

Because Gold58 is small, I would use a group bootstrap over the final cross-fitted predictions and treat the probability mass above zero as evidence strength rather than demanding a conventional 95% interval from every target.

**Failure:** if opposed-gradient examples are not enriched among actual gold misrankings, or reweighting does not improve cross-fitted gold ranking, stop. That would tell us that weak-label direction is *not* the main remaining bottleneck.

The first implementation can be a head/last-block experiment. It does not deserve a full 73M-parameter fit until it passes.

### Why this is stronger than “better labels”

This mechanism can discover that the same label source is good for one subset of ACL studies and harmful for another subset, or useful for OA but misleading for fracture.

That is consistent with your observed fracture source correlation of only 0.51446 without assuming either table is globally correct. fileciteturn9file0L2-L17

It also directly targets the actual competition semantics. The host labels borderline findings negative, whereas clinical reports can mention milder or uncertain abnormalities. citeturn19search0turn19search1 A weak report statement can therefore be linguistically correct and still produce the wrong training direction for the leaderboard task.

That is a much deeper supervision problem than ordinary label noise.

## MRI modeling lane: replace bag pooling with criterion-relational reasoning

This is the part where I would push furthest beyond the current competition consensus.

Strong current systems already solve

> “Which windows matter for target \(t\)?”

They do **not explicitly solve**

> “What relationship among those windows satisfies the target’s clinical criterion?”

Those are different problems.

### The limitation of present MIL

Abstract a current target-specific MIL head as

\[
h_i=f_\theta(W_i)
\]

for window \(i\), with target-specific attention

\[
a_{ti}
=
\frac{\exp(q_t^\top K h_i)}
{\sum_j\exp(q_t^\top K h_j)},
\]

and pooled representation

\[
z_t=\sum_i a_{ti}Vh_i.
\]

A classifier produces

\[
s_t=w_t^\top z_t.
\]

This is a powerful set function. I am **not** claiming it is mathematically incapable of learning every pairwise phenomenon.

But in the common implementation, the target attention score for each window is generated mainly from that window’s own representation. The model receives explicit within-window 2.5D context, but relationships between two *different* windows, two different series, or two intersecting planes must be inferred indirectly through independent embeddings and scalar aggregation.

That is an unattractive inductive bias when the target definition itself is relational.

For example, the competition says a meniscal tear can require qualifying signal on **at least two images**. citeturn19search1 A weighted mean of “tear-likeness” is not the natural statistic for that rule.

### Preserve physical geometry instead of just sequence position

DICOM already supplies the geometry needed to place pixels and slices in a common patient coordinate system. The standard states that Image Position (Patient), Image Orientation (Patient), and Pixel Spacing specify slice position/orientation relative to the patient coordinate system. citeturn20search6 DICOM also specifies that inter-slice spacing can be derived by projecting differences in Image Position onto the normal derived from Image Orientation. citeturn20search28

For slice \(i\), let its row and column unit vectors be \(r_i,c_i\), position \(p_i\), and normal

\[
n_i=r_i\times c_i.
\]

Its plane is

\[
n_i^\top x=d_i,
\qquad
d_i=n_i^\top p_i.
\]

For two nonparallel planes, their intersection direction is

\[
u_{ij}=n_i\times n_j.
\]

This allows the model to know that a sagittal token and a coronal token correspond approximately to the **same physical part of the knee**, rather than merely knowing “one is sagittal and one is coronal.”

There is independent MRI research supporting this principle. Anatomy-oriented self-supervision using relative imaging-plane orientation and relative slice location has improved transfer on knee MRI tasks, specifically exploiting the fact that anatomically prescribed planes contain meaningful spatial relationships. citeturn21search1 Knee MRI transformer research likewise reports benefits from explicitly modeling three-dimensional spatial correlations rather than relying on simple slice aggregation. citeturn21search2

### GRAIL’s graph

Keep the strong 2.5D encoder. Do **not** replace the whole image model with a giant 3D transformer.

Each encoded window becomes a node

\[
v_i=
W_hh_i
+
E_{\mathrm{plane}(i)}
+
E_{\mathrm{sequence}(i)}
+
PE(x_i)
+
E_{\mathrm{anatomy}(i)}.
\]

The anatomy term can reuse an existing knee localization/segmentation system such as the SKM-TEA anatomy network already used by the public S6 pipeline, instead of inventing a new localization network. S6 already demonstrates that anatomy-conditioned crops are technically viable in this competition. citeturn18search5

Construct only sparse, clinically meaningful edges:

\[
E=E_{\text{neighbor}}
\cup E_{\text{cross-plane}}
\cup E_{\text{cross-sequence}}.
\]

`neighbor` links physically nearby windows in the same series.

`cross-plane` links windows whose image planes intersect near the same anatomy.

`cross-sequence` links different acquisitions that observe the same anatomical region with different contrast properties.

This is a few hundred meaningful relations, not \(O(N^2)\) unrestricted attention.

### The real innovation: differentiable clinical criterion operators

Do not merely run a graph transformer and hope it discovers something.

For each target, give the network operators corresponding to the form of evidence used by the challenge rubric.

For local target evidence

\[
e_{ti}=u_t^\top v_i,
\]

define an **existence operator**

\[
\phi_t^{\text{exist}}
=
\operatorname{LSE}_i(e_{ti}),
\]

which works naturally for a focal abnormality that may be compelling in one region.

For targets requiring repeated evidence, define

\[
\phi_t^{\text{repeat}}
=
\operatorname{LSE}_{(i,j)\in E_{\mathrm{neighbor}}}
\left[
e_{ti}+e_{tj}
-\lambda_d d_{ij}
+\rho_t(v_i,v_j)
\right].
\]

This explicitly asks for mutually supporting evidence in two physically related observations.

For continuity-sensitive anatomy,

\[
\phi_t^{\text{break}}
=
\operatorname{LSE}_{(i,j)\in E_{\mathrm{neighbor}}}
\left[
m_{tij}
\,
\|P_tv_i-P_tv_j\|_2
\right],
\]

where \(m_{tij}\) learns when a feature discontinuity occurs along the relevant ligament/bone trajectory.

For extent-based findings,

\[
\phi_t^{\text{extent}}
=
\sum_i
\sigma(e_{ti})\,\Delta s_i\,A_i,
\]

where \(\Delta s_i\) is physical slice support and \(A_i\) is the physical area represented by the relevant localized patch.

For cross-plane corroboration,

\[
\phi_t^{\text{cross}}
=
\operatorname{LSE}_{(i,j)\in E_{\text{cross-plane}}}
\left[
e_{ti}+e_{tj}
+\psi_t(v_i,v_j,x_i,x_j)
\right].
\]

Then the new branch predicts

\[
r_t
=
\beta_{t,1}\phi_t^{\text{exist}}
+
\beta_{t,2}\phi_t^{\text{repeat}}
+
\beta_{t,3}\phi_t^{\text{break}}
+
\beta_{t,4}\phi_t^{\text{extent}}
+
\beta_{t,5}\phi_t^{\text{cross}}.
\]

This is not hand-coded diagnosis. The local features, compatibility functions and operator weights are learned. What is hand-designed is the **space of relationships the network is encouraged to test**.

That is exactly what an inductive bias should do.

MRI work such as RadioFormer has independently found value in structuring models around radiologist-like hierarchy—slice extraction, cross-sequence aggregation, then inter-slice aggregation—particularly when training data are limited. citeturn21search3 Recent large-scale MRI pretraining work, MARS, similarly separates anatomy-invariant structure from sequence-specific information rather than treating sequence heterogeneity as generic nuisance variation. citeturn20search2 These results do not establish that GRAIL will improve RSNA; they support the underlying design principle.

### Target-specific mapping

The target/operator mapping I would initialize—not freeze—is:

| Finding | Most relevant initial operators |
|---|---|
| Medial/lateral meniscus | repeated adjacent evidence + cross-plane corroboration |
| ACL | continuity/break + repeated evidence |
| MCL | continuity/break + edema-sensitive cross-sequence evidence |
| Medial/lateral/PF OA | extent + cross-plane anatomical agreement |
| Effusion | extent across fluid-sensitive acquisitions |
| Synovitis | distributed extent + sequence-conditioned evidence |
| Baker’s cyst | localized existence + physical extent |
| Contusion | cross-sequence/cross-plane marrow evidence |
| Fracture | focal existence + line/continuity break + cross-plane corroboration |

This is grounded in the host’s target definitions, not a claim that these exact operators will be optimal. citeturn19search1turn24search5

### The matched experiment

Your Codex agent’s warning about adjacency is correct.

The first GRAIL test must **not** simultaneously modify adjacency, rewindowing, preprocessing, augmentation, label source and loss.

For N3, preserve its actual selected-slice/triple construction exactly.

For S6, preserve its 3.5-mm physical windowing, 320 cache, 288 crop and current image encoder exactly. citeturn18search5

Change only this:

\[
\text{finding-specific MIL aggregation}
\rightarrow
\text{criterion-relational residual aggregation}.
\]

Cached rewindowing must not be advertised as restoring image information; once spatial detail has been discarded, no later rewindowing can reconstruct it. Therefore the first test deliberately operates on the **same representations** as its control.

**Hypothesis:** explicit physical pair/extent relations improve findings whose reference definitions require multi-image, continuity or extent evidence.

The predeclared primary target group should be:

- MCL,
- medial meniscus,
- lateral meniscus,
- PF OA,

with ACL as a useful secondary target.

A public controlled ablation from another participant independently observed poorer performance on small/thin/graded findings and found adjacent slice geometry more consequential than a simple resolution increase in that system. That is supportive competition evidence, not proof for your pipeline. citeturn24search0

Promotion should require approximately:

\[
\Delta AUC_{\text{primary targets}}\ge +0.015
\]

and

\[
\Delta AUC_{\text{macro}}\ge +0.005
\]

on the clean validation endpoint, with no material degradation on the diffuse/fluid findings.

If it cannot deliver that as a cheap head-level mechanism, **do not spend a long full-encoder run on it**.

### Compute budget

The first experiment should keep the image encoder frozen and consume cached embeddings/feature maps. A relational head with sparse edges is tiny relative to the 73M encoder.

I would cap the discovery experiment at roughly **one feature extraction pass plus a few GPU-hours of head training**. The full encoder is unlocked only after the relation mechanism survives the matched test.

That is frontier research without turning the remaining days into architecture roulette.

## The integrated model: GRAIL-Knee

If both scientific mechanisms survive their falsification gates, they fit together cleanly.

Start from the strongest legally eligible image model available for final fitting. At this point your measured project reference is S6 exact at 0.950; do not pretend the external 0.958 recipe is a project score until the actual submission finishes. fileciteturn10file0L2-L23 fileciteturn7file2L47-L70

Let its existing target logit be

\[
b_t(X).
\]

The criterion-relational branch outputs

\[
r_t(X).
\]

The final model is

\[
s_t(X)=b_t(X)+\alpha_t r_t(X).
\]

That formulation is deliberate.

**Do not train a completely independent new classifier and then search Kaggle for its blend weight.**

Train a **residual reader** whose job is specifically to change rankings the strong base gets wrong.

### Make the residual learn new directions

For a batch of base logits \(b_t\) and residual outputs \(r_t\), add a mild redundancy penalty

\[
\mathcal L_{\text{redundancy}}
=
\sum_t
\operatorname{Corr}(b_t,r_t)^2.
\]

A harder linear-algebra version is to project relation features \(R_t\) away from the span of the base prediction:

\[
P_b
=
b_t(b_t^\top b_t+\lambda I)^{-1}b_t^\top,
\]

\[
R_t^\perp=(I-P_b)R_t.
\]

Then the residual head is fit on \(R_t^\perp\).

I would begin with the soft covariance penalty rather than hard projection, because some overlap can genuinely be useful.

The rationale is simple: AUC only improves if ordering changes. A residual branch that merely reconstructs the base ranking consumes compute without creating a new source of discrimination.

### Add gradient-filtered weak supervision

The final fitting loss becomes schematically

\[
\mathcal L
=
\sum_{i,t}
w_{it}^{\text{CGR}}
\,m_{it}\,
\ell(s_t(x_i),\tilde y_{it})
+
\lambda_R \mathcal L_{\text{redundancy}}.
\]

Only after the representation and supervision mechanisms are established would I consider a short ranking-aware fine-tune.

The competition metric is macro AUC. citeturn18search0 Deep AUC maximization methods have been specifically developed and tested for medical image classification. citeturn20search1 But I would **not** make direct AUC optimization the main breakthrough: pairwise AUC losses cannot recover anatomical information the model never represented, nor fix a systematically wrong weak target. They are a final metric-alignment layer after the hard problems are solved.

### Domain robustness only if the validation agent proves it is necessary

Do not blindly add adversarial domain training.

If the scanner-grouped experiment demonstrates a substantial drop, then factor the relation representation into

\[
h_i=h_i^{A}+h_i^{S},
\]

where \(h^A\) is encouraged to preserve anatomy while being less predictive of scanner identity and \(h^S\) retains sequence/contrast information.

The relational geometry operates mainly on \(h^A\), while fluid/sequence-sensitive target gates can consume both.

That design is inspired by the same broad principle used by MARS—separating anatomy-invariant and sequence-dependent MRI representation—but a positive MARS result does **not** establish that this modification will help this competition. citeturn20search2 It should be conditional on an actual shift finding.

## What I would authorize now

The three agents should no longer be told “find ideas.” They should receive exact scientific contracts.

| Lane | Falsifiable hypothesis | Exact change | Budget | Kill condition |
|---|---|---|---|---|
| **Validation** | Current validation is materially optimistic because patient/source/scanner grouping is incomplete | Change split grouping only; audit PatientID and scanner domains; fixed unexposed feature learner | CPU audit + one frozen extraction pass | Grouped gap <0.005, no meaningful patient crossings, candidate ordering stable |
| **Supervision** | Some weak-label gradients systematically oppose the expert-image objective | Change only per-example/per-target weights using cross-fit low-rank gold-gradient alignment | Head/last-block scale, few GPU-hours | Cross-fit Gold58 <+0.008 macro, no error enrichment, or scanner-grouped validation worsens |
| **MRI modeling** | Set-style MIL is missing useful explicit multi-image/physical relations | Replace only aggregation with sparse criterion-relational residual head; preserve N3/S6 preprocessing exactly | Frozen encoder/feature-map experiment, few GPU-hours | Primary relational targets <+0.015 mean or macro <+0.005 |

These lanes can execute **in parallel** because they do not depend on one another.

They are not three hyperparameter searches.

They answer three different scientific questions:

\[
\text{Can we trust the metric?}
\]

\[
\text{Are weak labels pushing in the wrong direction?}
\]

\[
\text{Is the reader representing the right examination-level relationships?}
\]

After those results, there should be at most **one major new end-to-end training family**: GRAIL with whichever mechanisms survived.

That is a fundamentally different research cadence from serially waiting for Whole94, then uniform, then labels, then another sampler, then another backbone.

### What should stop immediately

I would not spend the remaining scientific budget on:

- another generic backbone enlargement;
- another crop-size search;
- another “more windows” run without a new evidence-integration mechanism;
- another context/regional serving variation;
- changing adjacency, preprocessing, augmentation, loss and population in one run;
- choosing label sources from fitting loss;
- choosing a model because it wins only on released-encoder head CV;
- tuning ensemble coefficients against the public leaderboard.

The project has already accumulated enough negative evidence that those are low-information directions. Whole94, uniform/influence sampling, expanded fitting, neighbor variants and context additions have either failed to establish a gain or remain merely unscored—not demonstrated breakthroughs. fileciteturn11file0L2-L15 fileciteturn12file0L2-L28

### How I would handle the final ensemble problem

Your concern about ensembles is correct in one sense and too broad in another.

**Ensembling is not inherently what causes private-leaderboard failure. Public-selection is.**

A blend of genuinely complementary models with a coefficient fixed from leakage-safe cross-fitted data can reduce variance and generalize well. A blend assembled by repeatedly asking “did 0.21/0.79 beat 0.25/0.75 on public?” is a model-selection procedure on 30% of the test set. Since final ranking comes from the unseen 70%, that is exactly the behavior to avoid. citeturn19search3

The cleanest final construction is therefore not a large ensemble at all:

\[
\boxed{
\text{strong base}
+
\text{one original geometry/supervision residual}
}
\]

with the residual coefficient learned **once** from independent grouped/cross-fitted evidence and frozen before submission.

A secondary heterogeneous public model should be added only if its residual errors are demonstrably complementary under the same protected validation, not merely because the public board rises.

### What reaching 0.97 actually requires

Going from 0.950 to 0.970 is not a rounding improvement.

Because the metric is the unweighted average of twelve AUCs, it requires a total increase of

\[
12(0.970-0.950)=0.24
\]

AUC-points distributed across targets. citeturn19search5

That could mean, for example, six difficult targets improving by an average of roughly **0.04** while the rest remain constant.

The public 0.956 family shows that complementary models can recover about six thousandths of macro AUC over the 0.950 single-model tier on the public split. citeturn18search3turn18search5 That is valuable, but it does not by itself solve a 0.020 gap.

A winning original method therefore needs to attack **the targets where current readers are structurally wrong**, not make every target 0.001 better.

That is why the clinical criterion matters so much.

Meniscus is not simply “find a bright pixel.”

MCL is not simply “attend to the medial side.”

OA is not simply “detect abnormal cartilage.”

The adjudication rules encode **repetition, continuity, severity and extent**. citeturn19search1 Current models are extremely good image feature extractors, but their examination-level operators remain much more generic than the task itself.

That mismatch is where I would place the bet.

The most credible frontier move I can derive from the evidence is therefore:

\[
\boxed{
\textbf{expert-aligned weak gradients}
+
\textbf{patient-coordinate relational evidence}
+
\textbf{criterion-specific differentiable aggregation}
+
\textbf{residual learning against the strongest existing reader}
}
\]

rather than “train another better classifier.”

That is an original mechanism with a reason to improve the exact errors that still matter, a mathematical explanation for why it adds representational capacity, a way to exploit Gold58 without simply overfitting it, a validation design intended to survive the 70% private set, and three cheap falsification gates that can kill it before the remaining compute is wasted.