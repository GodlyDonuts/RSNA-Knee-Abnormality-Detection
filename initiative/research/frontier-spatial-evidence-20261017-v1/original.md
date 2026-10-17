**Codex’s distinction is correct. The next original bet I would back is not another backbone, another label blend, or another attention head over the same summaries. It is a model that preserves and verifies spatial relationships between MRI acquisitions before your current pipeline discards them.**

I found a concrete limitation in the implementation, derived a way to test whether it matters, and built runnable mathematical diagnostics. That gives us a more specific research direction than “try anatomy-aware attention.”

Frontier_RSNA_Breakthrough_Plan_2026-10-17.md[Read the full research paper](sandbox:/mnt/data/rsna_frontier_plan/Frontier_RSNA_Breakthrough_Plan_2026-10-17.md) · RSNA_Frontier_Research_2026-10-17.zip[Download the paper and runnable diagnostic bundle](sandbox:/mnt/data/RSNA_Frontier_Research_2026-10-17.zip)

## 1. The opening I found in your actual implementation

Your current evidence supports the distinction between independently training N3 to **0.894** and reproducing S6 at **0.950**. It also records completed negative context/regional experiments and completed but not yet demonstrated improvements from the newer training comparisons. Nothing in that record establishes a breakthrough yet. 

But those negative experiments do **not** test every way of integrating MRI evidence. Here is the specific distinction.

### Your regional model retains summaries, not detailed spatial correspondence

In `regional_features.py`, the released encoder produces a **1,024-channel, 10×10 spatial feature map** for each window. The regional branch reduces it to five regional mean contrasts, a spatial standard deviation, and a maximum contrast. Those summaries are concatenated with the global feature vector before entering the context model.  

That reduction has an exact mathematical limitation.

Let \(H\) be the spatial feature map and \(S(H)\) everything retained by this summary operation. Certain rearrangements \(P\) of the spatial features satisfy

\[
S(PH)=S(H),
\]

even though the features occupy different locations.

For example, moving a complete feature vector between two positions belonging to the same quadrant and the same center-region membership preserves all seven summaries. The means stay the same. The standard deviations stay the same. The maxima stay the same.

**Consequently, no downstream head receiving only those summaries and unchanged metadata can distinguish the two arrangements.** Adding more transformer layers after that compression cannot recover the missing distinction.

I executed a synthetic check of the summary arithmetic:

| Constructed comparison | Result |
|---|---:|
| Difference between the underlying spatial feature maps | Nonzero |
| Maximum difference between retained global means | **0.0** |
| Maximum difference between all seven regional summaries | **0.0** |

A separate correspondence statistic distinguished aligned from misaligned evidence despite identical per-view evidence distributions.

**This proves information loss in the representation. It does not prove that clinically important information is being lost on your actual error cases.** The encoder can still encode substantial anatomy inside its channels. What we have established is a specific blind spot of the retained representation, not the cause of the entire leaderboard gap.

That is nevertheless a stronger starting point than assuming that another generic head will discover something new.

### There is also a separate restriction in the existing correction head

`ContextResidual` applies a correction of the form

\[
\Delta z_c=0.5\tanh(g_c(H))
\]

at its default limit. Its constructor does not permit a larger limit. Thus, that context/regional family can move each view’s logit by at most \(0.5\) in either direction. 

For a single view, a wrongly ordered positive/negative pair separated by more than **1 logit** cannot be reversed by that residual alone.

For the actual mean-of-probabilities test-time augmentation, the corresponding bounds are

\[
L_i=\frac1V\sum_v\sigma(z_{iv}-0.5),
\qquad
U_i=\frac1V\sum_v\sigma(z_{iv}+0.5).
\]

A positive case with \(U_+<L_-\) cannot outrank that negative case within this correction envelope.

The bundle includes a diagnostic that measures this optimistic ordering ceiling from authorized development predictions.

**This is not a revival of the disproven teacher-anchor argument.** Your fitting-data audit found that the 25% source component already changes the preferred ordering of most strongly source-opposed pairs. That result is a reason not to remove anchoring based on the earlier story. The bounded residual is a different, directly visible restriction, and its practical importance still needs actual pair-margin measurements. 

## 2. The frontier proposal: preserve joint evidence, then verify it

The model should be able to ask:

> **Do these observations provide compatible evidence about the same anatomical finding, or are we combining unrelated suspicious appearances?**

That differs from asking which windows deserve higher attention.

Attention-based MIL and correlation-aware instance modeling are established methods. TransMIL already explores morphological and spatial relationships, so simply putting a transformer over instances is not a new contribution. The opportunity here is the particular information your current representation removes and the way to establish its incremental value. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v80/ilse18a.html)

### Preserve spatial tokens before pooling

The new reader should receive spatial feature tokens before the seven-summary reduction, together with their source membership and preprocessing geometry.

For the released-parent branch, the existing `RegionalCapture` hook identifies the relevant pre-pooling location. For an independent N3 experiment, Codex must use **N3’s actual native map dimensions, windows, and preprocessing**, not transplant the released model’s 94-window contract and call the comparison matched.  

The existing regional summaries cannot be expanded back into those maps. A bounded forward pass over the already available decoded pixels would be needed. That is different from downloading or rebuilding the original dataset.

### Treat geometry as uncertain, not automatically registered

Your current context code explicitly says its position features describe selected-slot position, **not patient millimeters or anatomical registration**. That is an important limitation, not a bug. 

DICOM position, orientation, and pixel spacing define a mapping into patient coordinates, but using those fields correctly does not establish perfect alignment between separate acquisitions. Crop transforms, resizing, padding, mirroring, slice support, and possible movement must remain part of the geometry. [DICOM](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)

I would represent a token by a nominal location \(\mu_u\) and an uncertainty/support matrix \(\Sigma_u\). Correspondence between tokens from different acquisitions would be a **soft compatibility**, not an asserted exact match.

When geometry is missing or inconsistent, the cross-plane branch should abstain and retain the original prediction. It should not manufacture a registration or interpret missing evidence as a negative finding.

### Measure a relationship that marginal pooling does not retain

Suppose \(e_c^{(s)}\) and \(e_c^{(t)}\) are disease-specific evidence vectors over spatial tokens in two acquisitions.

Let \(\Pi\) be a normalized, nonnegative correspondence matrix derived from qualified geometry. Its row and column marginals are

\[
a=\Pi\mathbf 1,\qquad b=\Pi^\top\mathbf 1.
\]

A useful candidate statistic is

\[
\boxed{
C_c^{st}
=
(e_c^{(s)})^\top
\left(\Pi-ab^\top\right)
e_c^{(t)}
}
\]

This compares two quantities:

- Evidence meeting at geometrically compatible locations.
- What would be expected from the same marginal evidence without that spatial dependence.

In the synthetic example I ran, the marginal evidence was unchanged, but this statistic changed from **+0.25 to −0.25** when the correspondence changed.

That is the central modeling distinction: **the input now contains a joint relationship that the previous summary representation could not distinguish.**

The statistic is not a disease probability, and positive co-support is not automatically a lesion. A compact learned reader would need to interpret it alongside sequence identity, uncertainty, single-view evidence, and the original logits.

### Do not make “agreement across planes” a compulsory rule

The design must preserve a strong single-acquisition path. It should learn when another acquisition is informative, rather than require every finding to appear in every view.

Likewise, mirrored test-time augmentation is not an independent acquisition, and overlapping windows are not independent confirmations. Counting them that way would create artificial confidence.

Recent multi-sequence MRI research provides a useful warning here. VISTA explicitly studies failures involving interactions between sequences, rather than treating every shift as an isolated intensity problem. However, its experiments concern approximately registered brain-MRI segmentation. They do not establish a knee-classification gain or justify copying its test-time adaptation into this competition. [arXiv](https://arxiv.org/html/2605.17433v1)

**The proposed first system is one encoder plus a compact spatial evidence reader. Not a new foundation-model training campaign, a registration network, and a larger ensemble all at once.**

## 3. Supervision needs a different mathematical framing, not another confidence heuristic

The supervision work should support that spatial model by distinguishing four objects:

**The underlying finding → what the report says → what an extractor outputs → what the competition target means.**

A label disagreement can arise at any of those transitions. Two disagreeing label files alone do not tell us which transition failed.

A principled model could introduce a latent state \(Z\), with source observations \(O_m\), and learn

\[
P(O_m\mid Z,\text{source characteristics})P_\theta(Z\mid X).
\]

But there is a serious identifiability problem.

Even a simple binary latent label with two binary observers has more unknown prevalence/error parameters than the observed joint distribution can generally determine. Correlated extractors make the independence assumptions less defensible. The label-noise identifiability literature explains why additional observations or structural assumptions are necessary. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v202/liu23g.html)

**A more complicated latent-label model does not solve this merely by being more complicated.**

My proposed starting point is therefore **provenance-supported sets of possible states**, not invented certainty.

For a case with admissible states \(\mathcal S_i\),

\[
\mathcal L_{\text{partial}}
=
-\log\sum_{z\in\mathcal S_i}P_\theta(z\mid X_i).
\]

An accepted, explicit statement can restrict the state set. An ambiguous statement leaves a larger set. When the evidence imposes no restriction, the set contains every state and contributes no forced label.

This differs from simply giving an uncertain case a smaller BCE weight: it changes **which distinctions the observation actually supports**.

However, the state definitions must come from the authorized target rubric and accepted evidence. Postoperative status, severity, uncertainty, and absence should not be casually collapsed onto one invented ordinal scale.

This approach also cannot correct a wrong original report without additional trustworthy evidence. Neither agreement among teachers nor a low correlation between two label sources supplies that evidence automatically.

## 4. Validation must distinguish genuine learning from an easier source-identification problem

Your agent is right that released encoders’ exposure to internal studies prevents treating a new head’s cross-validation score as independent whole-model evidence. A new fold assignment cannot undo earlier encoder exposure. 

I would maintain two separate conclusions:

**Mechanism evidence:** Does the spatial reader help on the cleanest available N3-lineage, patient-disjoint development comparison?

**Transfer evidence:** Does adapting the stronger released model improve genuinely unexposed evaluation data?

The first can justify further development without establishing the second. Also, the existing N3 development endpoint has been repeatedly inspected, so it should not be described as pristine final confirmation.

### Decompose where the AUC improvement comes from

For each finding, define \(A_{gh}\) as the probability that a positive from acquisition domain \(g\) outranks a negative from domain \(h\), with half credit for ties.

Then

\[
\mathrm{AUC}
=
\sum_{g,h}\pi_g^+\pi_h^-A_{gh}.
\]

This is an exact decomposition.

It separates improvement within comparable domains from improvement between domains. A model can improve pooled ranking partly because it distinguishes sources with different disease prevalence or reporting conventions, without improving discrimination within those sources.

That between-domain improvement is not automatically illegitimate. But it does not, by itself, establish robust anatomical learning.

The provided `pair_audit.py` calculates this decomposition from fixed predictions, without training another model. Its equal-domain summaries are explicitly labeled as sensitivity analyses, not substitutes for the competition metric.

The sparse targets remain a hard limitation: your handoff records very few positive observations for several findings. Thousands of positive-negative pairs generated from two positive cases do not constitute thousands of independent positive examples. 

**A sustained lead requires evidence that survives this scrutiny. It cannot be inferred from a more impressive internal decimal.**

## 5. The three assignments I would give Codex

These are three bounded tests serving one research program, not three open-ended architecture searches. The resource figures below are **proposed caps**, not measured runtimes or claims that the hardware is available.

| Assignment | One decisive comparison | Proposed resource cap | What would stop promotion |
|---|---|---|---|
| **Validation** | Use identical fixed predictions to audit exposure and decompose within-domain versus between-domain ranking gains. | **2 CPU-hours**, using existing artifacts. | Gains exist only on exposed data or unsupported comparisons; independent evidence remains unavailable. |
| **Supervision** | Same clean-lineage features, latent output dimensions, roster, masks, initialization, and fitting budget; compare collapsed binary observation handling with provenance-supported admissible-state likelihoods. | **4 CPU-hours** for small heads and analysis, conditional on the required evidence already existing. | State semantics cannot be justified, or improvement merely tracks the source that supplied the labels rather than independent/adjudicated evidence. |
| **MRI modeling** | Same spatial tokens and model, but genuine qualified correspondence versus a marginal-preserving correspondence null. Retain the existing baseline as a reference. | **8 H100 GPU-hours total**, only if separately funded and admitted by a measured scout. | No gain over both the appropriate baseline and null, unreliable geometry, fitting-only improvement, or unacceptable deployment cost. |

For the MRI comparison, **only correspondence should change**. Keep the encoder, image content, labels, masks, initialization, optimizer updates, output bound, and inference budget matched.

The null also needs care. An arbitrary shuffle does not automatically preserve weighted degrees, missingness, or coverage. Those invariants must be checked. And beating a deliberately damaged null is insufficient: the candidate must also improve the appropriate baseline.

All three lanes can prepare their contracts and implementation while the existing scorers continue. There is no reason to interrupt completed campaigns or wait for every public result before doing the source-level and mathematical work.

There is also no reason to admit an unbounded GPU campaign before a fixed fitting-only scout has measured geometry validity and end-to-end cost.

### Keep the proposed reader small enough to deploy

For the released-parent example, 94 windows with 100 spatial positions produce 9,400 tokens per view. Dense all-pairs attention is unnecessary; bounded neighborhoods or streamed correspondence reductions are more appropriate.

An illustrative two-view FP16 cache with 64 projected channels across 3,887 studies would occupy approximately **8.71 GiB before metadata and temporary storage**. Keeping all 1,024 channels would require approximately **139.38 GiB**.

Those are arithmetic estimates, not storage reservations. The original N3 geometry must be priced separately from its actual implementation. Nor is a 64-channel projection guaranteed to retain the useful signal.

The deployment target remains the existing offline inference environment. A small-case success cannot establish full hidden-runtime compliance, as your own deployment receipts explicitly acknowledge. 

## 6. What this means for the 0.97 goal

Moving from AUC **0.950 to 0.970** means reducing pairwise error mass, with half credit for ties, from **0.050 to 0.030**.

That is a **40% net reduction in ranking error**, not a small cosmetic improvement.

For a 12-target equally weighted macro-AUC, improving one target by 0.10 contributes only about **0.00833** overall. Closing a 0.02 gap requires cumulative per-target improvement of 0.24 after accounting for regressions.

That is why I would favor a mechanism capable of improving several finding families over a leaderboard-sensitive correction for one rare target. But the evidence must establish that the mechanism actually transfers across those families.

Your concern about ensembles should also remain focused on **selection behavior**, not just model count. Repeatedly selecting a single architecture or epoch from public scores can overfit too. A new single model does not acquire private-generalization protection merely because it is original.

## My recommendation

**Build the spatial evidence reader as the main new research bet. Use the validation and supervision lanes to determine whether it is learning a useful anatomical relationship rather than another source-dependent correlation.**

The crucial advance is not “more attention.” It is:

> **Retain information the existing model discards, demonstrate that the retained relationship changes meaningful predictions, and show that the improvement survives an evaluation the model has not already seen.**

I completed the repository/literature audit, the mathematical analysis, and the runnable synthetic diagnostics in the attached bundle. I have **not** launched separate background agents, inspected the private MRI/prediction arrays, trained this reader, or demonstrated a new competition score. The synthetic result establishes a representational distinction. The matched MRI experiment must still establish that it is worth learning.