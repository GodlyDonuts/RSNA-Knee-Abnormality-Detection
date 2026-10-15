# Native anatomical input contract: a negative bug audit and one bounded hypothesis

Audit date: 2026-10-14. Current repository pin: `c83ffa8b58a3bb28dc63e7f10f4805faab0a8893` in GodlyDonuts/RSNA-Knee-Abnormality-Detection. The earlier research pin was `8300cfd20c2acaeef9d684a81dbc7fe977282a62`; the GitHub comparison shows nine subsequent commits, with none changing the native preprocessing, data, backbone, or model files examined here.

## Bottom line

**No demonstrated native transform/ordering mismatch was found.** The native cache already orders by projected patient geometry, preserves physical in-plane aspect, and binds sampled images to the corresponding positions. Competitive training and inference use the same input code. The apparent horizontal-flip concern belongs to the older base dataset, not the competitive dataset used in the current native recipe.

There is a real, narrower representational omission: **the signed position supplied to the model is distance along an acquisition's own slice normal, not a universal patient-axis or medial-to-lateral coordinate.** Image orientation and laterality are not model inputs. An equivalent lossless re-encoding can therefore change the image tensor and reverse the scalar position attached to the same anatomical slice. That is not a broken data contract: the code promises acquisition-relative distances, and an image model can learn orientation and compartment cues from appearance. It supports a conditional robustness hypothesis, not an accuracy claim or a mandatory correction.

**Recommendation:** retain the native input contract for the compact challenger. Add the geometry/channel-order falsifiers below to its synthetic adapter gate. Do not spend a fit or rebuild a cache merely because canonicalization is absent.

## 1. What the current code actually does

All repository links below are pinned to the current commit.

### Decode and cache order

- [`rsna/dicom.py:38–69`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/dicom.py#L38-L69) extracts pixels, ImagePositionPatient, ImageOrientationPatient (IOP), PixelSpacing, and InstanceNumber. It preserves the transmitted in-plane pixel orientation. It does not extract Laterality/ImageLaterality into the `Slice` object
- [`order_slices:72–114`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/dicom.py#L72-L114) validates finite, approximately orthonormal geometry and uses the first IOP vector crossed with the second as the normal. Its primary sorting key is `IPP dot normal`. InstanceNumber and SOP ID are only tie-breakers on that branch
- Missing/mixed geometry explicitly falls back to unique InstanceNumbers, records a warning, and marks ordering as `instance_number`. Duplicate InstanceNumbers without usable geometry raise. Duplicate physical positions are warned at cache creation and rejected later by competitive loading; this is not a silent duplicate-to-neighbor claim
- [`make_volume:117–181`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/dicom.py#L117-L181) stacks pixels and stores SOP identities, positions and hashes in exactly the resulting order. The normal is acquisition-defined; its sign is not forced to a patient-axis convention

**Result:** the claim that native training is accidentally sorted by filename or InstanceNumber despite available geometry is false for this path.

### Resize and intensity

`make_volume` uses physical height = rows × row spacing, physical width = columns × column spacing; scales both by one common factor; then centrally pads to a square. It does not stretch the original pixel matrix blindly to square or swap row/column spacing. Volume-wide min–max intensity scaling and MONOCHROME1 inversion are unrelated to anatomical mirroring.

[`sample_neighbor_images`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/competitive/neighbor_input.py#L36-L58) then divides native uint16 by 65,535 and uses bilinear, antialiased square-to-square interpolation, clamping to [0,1]. This second scaling preserves the established aspect ratio. [`_GrayEncoder._normalized_rgb`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/competitive/backbones.py#L241-L253) repeats the raw grayscale channel three times and applies the bound ImageNet mean/std. It introduces no spatial or stack transform.

The existing [`cache_patch_geometry.py`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/scripts/research/cache_patch_geometry.py) also uses the correct DICOM column/row increment convention and half-pixel resize mapping. It explicitly does not claim cross-series registration. This research helper is not evidence that patient coordinates already enter the production encoder.

### Series, slice and plane semantics

[`select_series`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/data.py#L15-L39) preserves available sagittal/coronal/axial coverage and then fills the budget by the documented ranking. Plane features come from the accepted series table; this is not an IOP-derived canonicalizer. The repository already has a provenance-controlled [`series_metadata.py`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/series_metadata.py#L88-L205) path for reviewed plane corrections. No clinical table or existing correction rows were opened here, and no new incorrect plane assignment is established.

[`CompetitiveStudyDataset._indices/_positions`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/competitive/data.py#L23-L65) uses unique selected source indices and strictly increasing projected physical positions. Unknown geometry produces zero position and a zero known flag, not fabricated millimeters. Position is `(projection − series midpoint)/100` plus the known flag. Selected images and positions use the same indices.

The older `StudyDataset.__getitem__` contains a width flip. **The competitive subclass overrides that entire method and does not call the base method.** Its `augment=True` affects sampling/selection, not pixel flipping. There is an explicit existing regression test, [`test_augmentation_sampling_is_reproducible_and_does_not_flip`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/tests/test_competitive_data.py#L215-L234). Treating that legacy flip as a current native defect would be an audit error.

[`train.py`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/competitive/train.py#L673-L717) and [`infer.py`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/competitive/infer.py#L138-L170) bind the same native dataset, cache resolution and model input choices. No train–inference anatomical transform divergence was found.

## 2. The actual missing assumption

The [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html) defines the patient coordinate system and the relationship between pixel coordinates, IOP, IPP and spacing. For humans, positive patient x points left, y posterior and z superior. An acquisition's normal can point either way along an anatomical axis; a plane label alone does not specify its sign or the in-plane display axes.

Consequently:

- A positive native sagittal position is not guaranteed to mean medial, lateral, patient-left, or patient-right across series
- Patient-left/right is itself not knee medial/lateral. A trustworthy knee-side convention would be required to canonicalize medial/lateral coordinates; the current cache does not preserve a laterality field for that purpose. The sign of an arbitrary patient-coordinate origin should not be substituted for it
- Native [`CompetitiveStudyNet`](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/competitive/model.py#L75-L76) embeds only the scalar distance/known pair and five plane/contrast fields, not IOP or side. Compartment recognition can still come from image morphology and cross-series evidence
- A coherent whole-knee left/right reflection preserves the labels “medial meniscus” and “lateral meniscus” as anatomical identities. Automatically exchanging these target columns would be a new label error, not a necessary part of a mirror

This could make compartment-specific learning less sample-efficient if acquisition conventions vary materially in the admitted cohort. **Neither that prevalence nor any actual failure rate has been measured here.** It is equally possible that protocols are consistent enough, or image cues strong enough, that explicit canonicalization brings no benefit.

## 3. Reversal is three different operations

1. **Permute complete slice tuples:** move image, position and mask together. The plain per-slice encoder plus masked attention model is permutation-invariant to this operation, up to numeric reduction differences. Reordering tuples alone cannot repair its accuracy. This statement excludes the optional neighbor-context branch, whose ordinal adjacency has a separate contract
2. **Reverse images while leaving slot positions fixed:** this reassigns physical position features to different anatomy. It can change predictions and is not a harmless order test. A future geometric transform must preserve the pixel–position identity or intentionally define and document the new coordinate system
3. **Reverse raw channels in a triplet:** `[previous, center, next]` becomes `[next, center, previous]`. A CoAtNet with channel-dependent filters need not be invariant. This is relevant to the proposed compact triplet challenger and public models, but the native baseline uses repeated grayscale, so raw channel reversal there is an identity. Any triplet reversal must occur before channel-specific ImageNet normalization; reversing already standardized channels also changes the normalization convention

The proposed native CoAtNet adapter should therefore call its windows “consecutive selected source indices within a series.” With 16 evenly distributed samples, they are not generally immediately adjacent original slices or fixed-millimeter neighborhoods. That is an input definition, not a bug. The existing native neighbor-delta experiment is different: it uses original-volume donors and geometric admissibility checks.

## 4. Public Raptor/nartaa comparison, with an explicit source limit

Public Kaggle pages/API fetches were unavailable in this pass; no browser/computer fallback was used. The original notebook files were not retained after the environment restart. The following therefore uses the **previously verified source audit**, `rsna-single-model-evidence-2026-10-14.md`, rather than pretending to have reread current notebook cells.

That audit pins [Raptor V1 / 344173275](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model?scriptVersionId=344173275) and [nartaa V2 / 356175397](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397). Raptor's scored version has no TTA. Nartaa's audited mirror dispatch uses the center slice's slot: sagittal triplets reverse channels; coronal/axial triplets reverse width. Its triplets are drawn over the concatenated selected slots.

The useful refinement is that **not all slot-boundary windows have the same problem**. Conditional on the audited full five-slot allocation 26/22/18/12/18:

- 94 total windows
- 8 cross a slot/series boundary
- 4 cross a plane boundary
- **Only 2 span different mirror-operation conventions**, at the sagittal-to-coronal boundary

The coronal-to-axial boundary crosses planes but both sides receive a width reversal under that rule. Same-plane boundaries still mix acquisitions rather than genuine spatial neighbors. These counts are synthetic consequences of the audited allocation, not an estimate for missing-slot cases or proof of poor performance.

A plane-dispatched transform is an anatomical left/right reflection only if its preprocessing/IOP conventions make the chosen image axes correspond to that patient reflection. The available prior audit does not justify claiming those assumptions fail in the original checkpoint's training domain. An involution check alone cannot prove anatomical correctness. A public reproduction must preserve the exact original route; a boundary-safe or geometry-aware alternative is a new hypothesis with its own identity.

## 5. Synthetic falsifiers executed in this audit

Artifact: `rsna-anatomical-audit-current/synthetic_geometry_checks.py`; results: `rsna-anatomical-audit-current/synthetic_geometry_results.json`.

Seven checks passed:

1. Executed the unchanged AST-extracted native `order_slices` on synthetic slice objects with contradictory InstanceNumbers; projected geometry won
2. Removed a synthetic position; the documented InstanceNumber fallback and warning appeared
3. Re-encoded the same artificial physical voxel grid by reversing image columns, negating the first IOP vector and shifting IPP to the opposite image edge. Native ordering reversed the stack. Every voxel's physical coordinate remained identical; the same slice's scalar relative position changed sign. This falsifies “native tensors are canonical under lossless orientation re-encoding,” not “native ordering is correct”
4. Executed native uniform `_indices` for synthetic depths 1–1,000 with the current 16-slice request; selected grids were reversal-symmetric in this bounded check. This is not a proof for all possible depths, requests, or floating-point rounding cases
5. Checked the resize arithmetic for a 16×32 image with 2×1-mm spacing; it becomes square physical content rather than a stretched physical rectangle
6. Reconstructed the prior audited public full-slot boundary counts: 8/4/2 as above
7. A small NumPy attention calculation preserved output under joint feature/position permutation but changed when only features were reversed, illustrating the model-level contract

No PyTorch or pydicom runtime was installed, so this was not a full preprocessing/model test suite. Actual `order_slices` and deterministic `_indices` source functions were executed; rendering and pooling checks were NumPy algebra. No package installation, data loading, training, model download or repository mutation occurred.

## 6. The smallest matched no-new-data test, if separately authorized

**First, a diagnostic, not a fit.** On a fixed, already-admitted fitting-only fixture, test exact sample identity/masks and correct coordinate updates under lossless re-encodings. Hold the selected SOP set, intensity values, field of view, padding and encoder-call count fixed. Do not inspect protected endpoints or add data. Report aggregate maximum prediction deltas under:

- Joint tuple permutation: expected to preserve the plain model's predictions; a material failure is a genuine implementation/adapter issue
- Same physical image represented with coherent in-plane index reversal and corresponding IOP/IPP/stack updates: no invariance guarantee in the current model, so a change is a measured sensitivity, not automatically a defect
- Raw triplet channel reversal before normalization in a triplet model: quantifies directional dependence; never conflate it with swapping channels after normalization

**Only if a material sensitivity exists and convention variation is verified**, consider one matched orientation hypothesis. Keep the existing fitting roster/targets, held-out endpoint, cache bytes, selected slices, backbone, initialization, seed, optimizer, schedule, resolution and presentation budget fixed. Compare the original contract against one deterministic, lossless, explicitly defined orientation-convention transform applied consistently in both training and evaluation. Restrict transformations to orientations that can be validated without interpolation; leave unsupported/oblique cases unchanged and report that subset. Use no guessed laterality and do not exchange targets. This tests encoding-convention robustness, not universal medial/lateral canonicalization.

Predeclare overall twelve-target macro as the primary endpoint, the medial/lateral meniscus and OA results as secondary, and the affected fitting-only orientation stratum before fitting. Report paired uncertainty; do not select transformations on the outer endpoint. A null effect, overall regression, unverified prevalence, or sensitivity explained only by altered selection/normalization ends this branch. Existing orientation variation may already be learned, and canonicalizing a distribution that the pretrained image trunk handles well can hurt.

**Correction versus new hypothesis:** fixing a tuple-identity failure, inconsistent train/evaluation transform, or demonstrably wrong row/column mapping would be a correction. None was found. Replacing scalar positions, adding IOP/laterality features, normalizing orientation conventions, reversing triplets, or changing the public mirror route are new hypotheses. They should not be bundled into the compact challenger's baseline as an alleged bug fix.

## Scope and confidence

Read-only authorized repository source, synthetic fixtures, prior research reports and the public DICOM standard were used. No real DICOMs, clinical metadata/rows, caches, protected data, model weights, training, publication, submission or computer use. New local research artifacts only.

High confidence: native ordering, resizing, position attachment, absence of competitive pixel flip, and lack of explicit orientation/laterality model features. Conditional confidence: prior-public-code boundary deductions, because exact notebook cells were not reread. Unknown: real-cohort prevalence, compartment-specific accuracy impact, and benefit of any new orientation recipe.
