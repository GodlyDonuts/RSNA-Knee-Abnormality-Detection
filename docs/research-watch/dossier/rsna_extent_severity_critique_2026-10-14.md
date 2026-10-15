# Physical extent and severity heads need a narrower claim

Read-only critique, 14 October 2026. Reviewed the final strategy and observation supplement against implementation commit `8300cfd20c2acaeef9d684a81dbc7fe977282a62`, with a targeted PR38 update at `e4f288d6d5cf605aa79935cd456359bd7eb498e1`. No experiment was run.

## Verdict

Physical-support pooling is a plausible **inductive bias**, not a demonstrated missing capability of the current model. Separating presence from qualifying severity is clinically meaningful, but its named concepts require independent supervision. Combining both changes now would make a positive result difficult to interpret. The smallest worthwhile experiment should isolate **within-series continuity of image evidence**, keep the existing representation and labels fixed, and explicitly challenge acquisition shortcuts.

**Deadline recommendation: do not make this a new full-training priority before October 22.** A capped frozen-feature probe is worth considering only if a compatible feature cache and geometry bindings already exist. If it requires new cache construction, substantial geometry engineering or fresh concept labeling, defer it. Interpreting and completing existing supervised adaptation and focal/window experiments has a stronger near-term case.

## What the current attention can already represent

The native loader supplies signed physical distance from each series midpoint, scaled by 100 mm, plus a geometry-known flag. The model adds its position embedding to the visual feature **before both attention scoring and value pooling**. It already has target-specific slice/series attention. Optional neighbor context also uses physical gaps, and plane memory exchanges sequence evidence. None is equivalent to an explicit support integral, but “the model has no geometry” or “softmax cannot represent extent” would be incorrect. [Loader](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/data.py#L22-L45) · [Actual pooling](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/model.py#L250-L303) · [Existing neighbor context](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/context.py)

A counterexample to the broad impossibility claim: identical visual features at positions −d and +d, position-valued pooling, and scores proportional to position yield pooled position `d*tanh(beta*d)`. This distinguishes spans. The actual gated score can approximate such behavior over a bounded range; the example is an expressivity argument, not a claim about learned weights. Relative lesion occupancy can also be represented when coverage and slice count are fixed.

Normalized pooling is invariant when **every complete token**, including its coordinates, is replicated equally. That is desirable for duplicate observations of the same anatomy. Selectively duplicating one token generally changes its relative attention mass. Adding similar pixels at a different coordinate is not exact duplication. The existing synthetic example drops this distinction by holding pooled features identical without representing the native positional pathway.

Native uniform/stratified sampling already avoids repeated source slices. Overlapping triplet multiplicity is chiefly a concern for the public/window-bag pipelines, not an established native sampling defect. A duplicate test can expose sensitivity; it cannot establish that ordinary native inputs contain the error.

The newer **PR38 focal-window module already exists**: disease-specific bounded pooling, separate per-window teacher evidence, and no penalty for anatomical window disagreement. Its review records 127 passing focused tests, but no trainer wiring, GPU admission or accuracy result. It does not implement physical interval support, yet proposing another focal-query pool would duplicate this work. Its zero-teacher-loss branch is exact; the focal head itself is not initially identical to ordinary pooling, so its separate pooling-only control remains necessary. [Pinned root review](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/e4f288d6d5cf605aa79935cd456359bd7eb498e1/initiative/audits/focal-window-PR38-root-review-20261014-v1/README.md) · [Module](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/e4f288d6d5cf605aa79935cd456359bd7eb498e1/rsna/competitive/focal_window.py)

## The head can learn the scanner rather than the lesion

For local evidence e and integration weights w, an integral `E=sum(w*e)` equals coverage `C=sum(w)` times normalized burden `B=sum(w*e)/C`. Giving a new head E while withholding B or C from its control would create an information or arithmetic advantage. A gain may reflect easier multiplication rather than recovered anatomy.

Millimeters also expose scan span, spacing, selected-series composition and geometry availability. These can correlate with protocol, acquisition quality, reporting practice or disease mix. Centering coordinates at the scan midpoint does not remove those associations, and the midpoint is not a fixed anatomical landmark.

More importantly, **slice-center gaps are not measured lesion intervals**. Integrating across widely sampled slices assumes something about the unseen gaps. DICOM distinguishes nominal slice thickness from center-to-center spacing; the native head receives center positions, not a validated slab-coverage measure. A true union-of-observed-slabs proposal needs explicit interval widths and gap handling. Do not join series without verified correspondence. [DICOM image-plane definitions](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)

Through-plane support also differs from cartilage surface extent and percentage thickness loss. The project's rubric requires separate constructs. An attention map or CLS-derived score is not a validated cartilage segmentation, lesion diameter or severity measurement. [Project target rubric](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/protocols/report_weak_v2.md)

## Severity factorization does not create its own labels

A final negative may mean normal anatomy or a visible finding below threshold. The final label alone cannot teach two heads which case occurred. For illustration, if a qualifying probability were factored as `q=p*s`, every `p` between q and one with `s=q/p` gives the same observed prediction. Calling the heads “presence” and “severity” does not identify them. This illustrates the problem; the reviewed strategy correctly warns against multiplying uncalibrated concept probabilities.

Explicit absence, supported below-threshold findings and supported qualifying findings can provide genuine auxiliary labels. Unspecified severity and report silence cannot. Before a severity-head experiment, require an aggregate fitting-only inventory of these states and their provenance. Otherwise the branch is only an architectural regularizer. It must not be presented as clinically interpretable severity learning.

## One smallest falsifiable experiment

**Test continuity alone with a frozen representation and two matched residual heads.** This is a proposed diagnostic experiment requiring separate execution approval, not a new production training plan.

Use one already admitted fitting-only checkpoint. Freeze its encoder, positional attention and original twelve outputs. Reuse exactly the same selected slice embeddings and positions, generating them once only if a compatible cache is unavailable. Keep the original fitting/evaluation boundaries and all twelve supervision targets unchanged. Do not add severity concepts in this experiment.

Both residual heads receive identical image-only local evidence scores, normalized evidence moments, peak evidence, total sampled span, gap statistics, series descriptors and geometry-known flags. Predeclare and freeze the local scoring function; its scores are model evidence, not verified lesion probabilities. Give both the same small learned graph, parameter count, initialized tensors, updates and optimizer schedule. The original positional pathway remains intact in both.

- **Aligned arm:** add one predeclared, smooth within-series adjacent-evidence statistic, for example the gap-weighted average of `e_i*e_(i+1)`, calculated on distinct, ordered sample locations. Call it a continuity proxy, not a lesion-length measurement
- **Sham arm:** compute that same statistic after a fixed, label-independent within-series permutation of the local evidence-to-position assignment. Retain the original evidence histogram and all global burden/coverage inputs. Use identical arithmetic and head dimensions

This control preserves the extra representation capacity, global acquisition information and compute while breaking the local alignment the proposed mechanism requires. Preserve unknown-geometry handling identically and disable physical weighting there rather than invent millimeters. Fix the permutation and all settings before outcomes. Both arms should begin with a zero residual and use the same predetermined endpoint.

Before fitting, synthetic admission cases should check exact duplicate neutrality after source-identity deduplication, padding neutrality, dependence on genuinely distinct neighboring locations, and approximate stability when the **same underlying synthetic evidence field** is sampled at different densities. Real-image resampling can discard lesions, so prediction invariance on arbitrary removed MRI slices is not a valid requirement.

**Decision:** the aligned arm must beat both its sham and the unchanged frozen baseline on the full twelve-target endpoint. Report component-paired uncertainty and acquisition-stratified results where support permits. If the sham matches it, the anatomical-alignment explanation fails. If the gain is confined to scan-spacing strata, or a constant-evidence input reproduces the effect, an acquisition shortcut remains a strong explanation. Too little supported data means inconclusive, not success.

No encoder retraining or extra inference frames are needed; the new residual adds small but nonzero head inference work, which must be measured. Admit the probe only if measured extraction and fitting together fit within one tenth of the reference full-fit compute; do not change the sample roster or stopping rule after seeing outcomes to force that cap. A successful frozen-feature result would justify considering one end-to-end matched follow-up after reconciling existing work, not automatically dispatching it before the deadline. It would establish useful image-geometry interaction, **not** accurate physical lesion extent, clinically identified severity, or an expected .95+ score. A negative result closes this narrow readout proposal without claiming that better localization or genuine concept supervision cannot help.
