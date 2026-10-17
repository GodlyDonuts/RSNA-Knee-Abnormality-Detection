# Spatial joint evidence: owner proposal and concrete decision

The [supplied text](original.md) is preserved with [exact provenance](provenance.json).
The separately named Downloads Markdown and sandbox ZIP are not accessible in
this workspace. No claim is made to have executed that supplied bundle. Its
three assignments are research proposals; execution remains one coordinated
program without spawning new agents or interrupting existing scorers.

## What changes

Preserve spatial tokens **before** their global/regional reduction. This is a
different representation from the [implemented longitudinal GRAIL branch](../frontier-GRAIL-20261017-v1/README.md),
which consumes pooled window embeddings. That earlier branch cannot test the
new proposal's spatial blind spot and will not be renamed as having done so.

The prepared, unsubmitted 32-study N3 scout now retains the actual original
pre-pooling FP32 spatial maps and window embeddings in the **same original
forward**, along with selected source identities, original matrix/letterbox
geometry and base logits. No fixed 10x10 parent grid or 94-window contract is
transplanted to N3. Actual map shape, number of windows, bytes and end-to-end
cost must be measured. The prior proposed full-cache mode has been removed;
a full spatial cache needs its own measured representation/storage admission.
The scout's separate 3GiB node-local envelope includes at most 2GiB of features
and a retained 512MiB head/publication allowance. No full fit is yet admitted.

## Established source facts and mathematical checks

- Actual `regional_features.regional_summary` maps 1024x10x10 parent features
  to seven channel summaries. An exact source test swaps two feature vectors
  within one quadrant/outside the center region: feature maps differ while the
  global mean and **all seven summaries are bit identical**. This establishes
  a representation collision, not a clinical error or a cause of the score gap.
- Actual `released_context.ContextResidual` limits each view correction to
  +/-0.5 logit. A positive/negative single-view logit gap above 1 cannot be
  reversed by that branch. Mean-probability TTA needs the per-case sigmoid
  interval calculation, not the same single-logit argument. Its practical
  ceiling has not been measured here; no limit is relaxed based on speculation.
- `spatial_correspondence.centered_cosupport` implements
  `diag(E1.T @ (Pi - a b.T) @ E2)` without materializing the centered matrix.
  Fixed evidence and fixed marginals can produce +0.25 versus -0.25 as the
  coupling changes. This is a weighted covariance, not a lesion probability.
- Its four-cycle null preserves **both weighted marginals and every positive
  edge/structural zero**, not just feature histograms. It changes geometry
  without reading evidence or labels. Sparse couplings without usable cycles
  explicitly abstain. A matched comparison needs improvements over both the
  unchanged baseline and this null; beating only a damaged control is inadequate.

Nine focused source tests cover the collision, covariance, independent/constant
null, gradient propagation, exact-support marginal preservation, determinism,
sparse abstention and invalid couplings. These establish engineering only.

## Geometry is the gating scientific uncertainty

Actual full-roster CPU949445 found all 17,671 selected series have usable
within-series metadata, but **none retains FrameOfReferenceUID or thickness**.
Disjoint original N3 windows are separated by a median about20mm. That supports
coarse longitudinal experiments, not registered cross-plane clinical evidence.

The native pipeline retains original matrix shape, in-plane spacing, positions,
orientation, resized dimensions and square letterbox policy. Those fields allow
nominal coordinates once the actual representation/preprocessing is bound.
They do not prove inter-acquisition registration, lesion location, scanner motion
or the effective receptive-field support of a learned feature. The spatial scout
will not silently declare any cross-acquisition coupling qualified.

A defensible first geometry operator uses qualified within-acquisition relations
and a retained single-acquisition path. Cross-acquisition links require a bounded
original-header/consistency review and an uncertainty/abstention policy. Mirrors
and overlapping windows never count as independent acquisitions. A small shared
projection may make storage practical, but its representation loss must be tested;
64 channels are not accepted merely because the paper's arithmetic fits a budget.

## What we accept and push back on

**Validation:** AUC's within/between-domain pair decomposition is exact and useful.
Use genuine acquisition-domain metadata, frozen predictions and support counts;
do not guess scanner/site from filenames or call sparse pairs independent cases.
An original N3 endpoint avoids released-encoder image fitting exposure but is
repeatedly inspected development, not pristine confirmation. A new head split
cannot erase released-encoder exposure. Accepted folds/protected cohorts remain.

**Supervision:** Partial-state likelihood is appropriate only when the target
states and admissible sets are justified by accepted source evidence. For a
binary target, the unrestricted set {0,1} has exactly zero loss, equivalent to
masking that cell. Thus rebranding our existing missing-label mask as a new
partial-label mechanism would be empty. A nontrivial experiment needs supported
additional states or observation structure; current source disagreements alone
cannot identify latent truth. No silent label changes or Gold meta-training.

**Novelty:** Attention MIL is established; the potential contribution is the
preserved relation, its qualified geometry and a controlled MRI benefit. The
proposal's link labeled TransMIL actually points to Ilse et al.'s 2018 attention
MIL paper, not TransMIL. The relevant primary works were checked independently:
[Ilse2018](https://proceedings.mlr.press/v80/ilse18a.html),
[TransMIL2021](https://arxiv.org/abs/2106.00908),
[label-noise identifiability2023](https://proceedings.mlr.press/v202/liu23g.html),
[DICOM image-plane geometry](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html),
[VISTA2026](https://arxiv.org/abs/2605.17433).
VISTA's adult/African/pediatric brain-tumor segmentation results do not demonstrate
knee classification or justify test-time self-training here.

## Promotion contract

Freeze encoder, original MRI inputs, masks/labels, initialization, update budget,
output bound, stopping point and inference cost across aligned/null arms. Retain
unchanged base as reference. Use fixed-final all12 targets, source/exposure
qualifications and whole-component paired uncertainty with undefined support.
No gold/confirmation tuning, best-epoch fishing or new score claimed from tests.
A promising result needs another matched seed/fold and offline runtime acceptance.

Current best remains externally reproduced0.950; own N3E16 is0.894. The new
spatial reader has **no trained result or competition score**. A 0.950→0.970
move requires40% net reduction in macro ranking-error mass; this is the target,
not an expected gain established by the memo.

## Actual progress after the original source publication

The original H100 scout was scheduler-preempted after26complete MRIstudies,
not rejected for input/numerical failure. Actual maps are1024x12x12, unlike the
released-parent10x10 example. Owning CPU949734fully verified all26original
files/1,912windows. A separate normalQOS continuation949735is PENDING and
reuses those completed forwards, with only six new encodings authorized.
The full spatial fit and an accuracy gain remain unobserved.

The [actual source-domain audit](../../audits/N3-fixed-source-domain-20261017-v1/README.md)
also completed, reconstructing existing0.9209253491internal AUC exactly across
12manufacturers/42manufacturer-model domains. Common-target within-domain
discrimination did not broadly collapse; sparse-target differences remain
uncertain. This is one fixed model, not an independent comparison or a gain.

## Actual acceptance and next design

NormalQOS H100recovery949735completed0:0 in51seconds. Owning CPU949785
independently collected the exact original terminal: **32studies/2,345windows**,
26completed forwards reused and six new, maps1024x12x12FP32, **zero maximum
base-logit repooling error**,1,311,275,207retained feature bytes and2.614GBpeak
GPUallocation. The measured recovery worker elapsed21.46seconds includes reuse,
so it is not an all-new-study throughput estimate. No training or score gain.

[Actual accepted scout](../frontier-GRAIL-20261017-v1/actual-spatial-scout-accepted.json)
and [next experiment decision](next-experiment-decision.json). The whole cohort's
raw spatial maps would require135.67GiB; a fixed64-channelFP16sketch would be
4.24GiB, with unproved information loss. Neither full cache is admitted. The
selected next design learns a projection on full maps online, using the frozen
original encoder; it exchanges H100compute for bounded storage. The actual
real-engine recovery/online-provider acceptance and full matched fit are still
required. This is a concrete design decision, not a trained spatial model.
