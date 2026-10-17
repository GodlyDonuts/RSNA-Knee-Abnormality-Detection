# Frontier memo: decision and execution

The [original owner-supplied memo](original.md) is preserved verbatim with its
[byte provenance](provenance.json). Its embedded citation IDs are not verified
links. This is a research hypothesis, not evidence of a 0.97 model.

## Decision

Prioritize **criterion-relational residual learning**, with a matched aligned
versus sham geometry control. Preserve the strong base and its original inputs;
learn evidence relationships rather than another crop, backbone or blend sweep.
Finish existing scoring owners unchanged. There is no new accuracy claim.

The new [actual achieved-response diagnostic](../../audits/fitting-achieved-response-20261017-v1/README.md)
strengthens this direction: on all 3,887 fitting studies, substantial probability
movement often yielded little new ranking discrimination. Across four completed
head runs, none of 5,832 updates clipped. In the S6-label/full94 arm, median
fracture probability movement was 0.0750, but only 77 of 660 pointwise-correctable
teacher-opposed strong-source pairs actually reordered, while 55 previously
ordered pairs inverted. These are weak-label fitting pairs, not clinical errors
or independent validation. They do not prove geometry will help.

The first experiment changes only the residual evidence integration. Retain
existing MRI preprocessing, windows, encoder, source targets, masks and base
predictions. Aligned and sham branches share parameter count, initialization,
evidence histogram, metadata, updates and stopping point. The sham permutes
evidence-to-position assignments within a series; it must not alter the base
reader or silently remove acquisition covariates from its control.

Preserve the memo's full design as the intended research direction. Admit its
operators in stages according to actual source support: within-series distinct
evidence and continuity first; cross-plane correspondence and physical extent
require the additional evidence described below. A geometry-only census is an
input check, not a trained GRAIL model or a smaller definition of success.

## Corrections that matter to implementation

1. **Overlapping windows are not independent images.** Original N3 triplets
   share two selected slices between neighboring windows. The first graph
   retains original SOP membership and permits corroboration only between
   disjoint-support windows. Continuity edges are separate. This conservative
   window corroboration is not a literal implementation of the clinical
   two-image rule; coarse sampling can miss nearby supporting evidence.
2. **Planes intersect along a line, not a known lesion location.** Image
   Position/Orientation and Pixel Spacing define patient coordinates. Cross-series
   links additionally need a common admitted frame, motion/registration checks
   and anatomy/ROI correspondence. Never identify equal slice indices across
   planes as a shared anatomical point. [DICOM image planes](https://dicom.nema.org/medical/dicom/2026c/output/chtml/part03/sect_C.7.6.2.html)
   and [frame of reference](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.4.html).
3. **Center spacing is not observed tissue volume.** Current native metadata
   has centers and spacing, but not a validated lesion mask/area or slab union.
   An attention-weighted span is a proxy, not millilitres of effusion or cartilage
   surface loss. Missing geometry stays missing. Normalize log-sum-exp/counts
   and deduplicate support so repeated inputs cannot manufacture evidence.
4. **The patient audit is not limited to ten studies.** The existing
   [full one-header audit](../../audits/patient-header-audit/patient_header_audit.json)
   covers all 4,410 released train/public-example studies: no missing PatientID
   and no repeated identifiers; issuer is missing throughout. It does not prove
   real-patient independence or within-study consistency. Reuse it; a per-series
   consistency/scanner extension is a distinct remaining question, not a reason
   to rerun the completed audit or change accepted folds by assumption.
5. **Gold gradient alignment consumes labels as training.** The proposed method
   is related to [Ren et al., ICML2018](https://proceedings.mlr.press/v80/ren18a.html),
   which learns weights against a clean objective. Gold58 has historical
   calibration exposure and released encoders saw this cohort. Four-way head
   cross-fitting cannot undo upstream exposure or make it pristine. Pooled AUC
   across fold-specific models also adds cross-fold score comparisons; report
   within-fold pair-weighted endpoints and undefined support, not just pooled
   AUC. Do not silently turn the current protected evaluation into meta-training.
   This lane needs an explicit data-role decision and independent evaluation;
   it does not block the geometry branch.
6. **Supporting papers are precedents, not RSNA proof.** RadioFormer uses
   registered sequences and annotated tumor crops, with brain/breast/liver
   evaluations. Those inputs are stronger than our unregistered knee windows.
   Its [actual paper/author response](https://papers.miccai.org/miccai-2025/0744-Paper1494.html)
   supports hierarchical reasoning as a hypothesis, not a transferable gain.

## Immediate executable work

`rsna/competitive/criterion_geometry.py` now constructs unchanged N3-window
source-support graphs. It separates overlapping continuity from disjoint
corroboration, preserves actual millimetre gaps, breaks links across large native
acquisition gaps and refuses fabricated geometry. Synthetic cases explicitly
show that one source slice cannot provide both sides of a repeat edge. This is
engineering, not an accuracy result.

The completed owning-node CPU census bound original native cache metadata, exact
2,247 fitting / 1,156 development rosters and original series selection, reading
no images or clinical target values. It measures graph coverage, physical gaps
and missing frame/slab fields before selecting a compatible feature extraction.
It retains source identities privately and exports only aggregate/hash receipts.
No GPU, new cache copy or model fitting is authorized by that census.

Then admit one frozen-encoder extraction on the unchanged roster and one
aligned/sham residual comparison with the unmodified base as a third reference.
Use fixed-final all12-target results and whole-component uncertainty; predeclare
MCL/menisci/PF-OA as a secondary mechanism group without replacing the macro
endpoint after outcomes. Released-encoder feature validation is descriptive,
not independent whole-model OOF. Use an exposure-safe own-model endpoint for
the scientific claim, and separately qualify any stronger-base deployment.

The old slice-distance auxiliary task, plane-memory residual, source-ranking
head and focal-window prototype remain separate. None already establishes this
distinct-source criterion graph, and none should be renamed GRAIL retrospectively.
Do not bundle gold weighting, new crops or encoder changes into this first test.

## Actual census and implemented branch

CPU949445completed the full metadata census in7.82seconds. Fitting/outer
selected-series counts are11639/6032; all have admitted within-series geometry.
It bound162651/84333unchanged N3windows and127734/66237disjoint-support repeat
edges. Every study has such edges. Median repeat separation is20mm in both
roles, with extrema about5.4–48mm. This conservative graph cannot be advertised
as the clinical two-adjacent-image criterion. All17671selected series lack cached
frame-of-reference and slice-thickness fields; cross-plane and physical-volume
operators require additional source/header/registration evidence.

The actual source is preserved as actual-census-geometry-v1.py.txt. The current
module adds explicit Python float casts in its orthonormal-basis validation to
satisfy typing; relation construction arithmetic is unchanged.

The new CriterionResidual branch is now implemented separately from all live
models. It uses source-disjoint corroboration, distance-weighted ordered evidence
and latent variation, plus image moments and acquisition nuisance statistics.
All branches start at exact zero residual. The sham changes only within-series
feature/geometry correspondence; global image moments and nuisance inputs stay
bit equal. This is local engineering, not a trained GRAIL result.

A published-source32-study H100frozen-feature pilot is prepared, now expanded
under the [spatial proposal](../frontier-spatial-evidence-20261017-v1/README.md)
to retain full original spatial maps as well as window embeddings. It uses the
original own N3E16checkpoint and native source cache, with no clinical target
reads and one original forward per study. Encoder hooks retain exact original
FP32window tokens; repooling must reproduce the actual base logits within1e-6.
A proposed3GiBnode-private envelope covers at most2GiBof scout features and
512MiBhead/publication reserve, after all original liabilities. Actual fresh
admission and a one-time dispatch remain required. Full extraction and residual
training are not yet dispatched; the scout has no full-cache mode. Measured
spatial dimensions/cost must precede a separate full representation admission. This gate measures fidelity, throughput and
actual representation availability before spending a full-run budget.
