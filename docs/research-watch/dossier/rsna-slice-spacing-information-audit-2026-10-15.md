# Selected-slice triplets versus original neighbors: information support, not a demonstrated accuracy defect

15 October 2026. Read-only source research and synthetic sampling arithmetic. The declared Native-CoAtNet-Bag12 N3 proposal remains unchanged.

## Decision

**Keep N3.** No newly found evidence justifies replacing its selected-slice triplets, delaying it for another gate, or buying a second fit. Original-neighbor triplets are a plausible, narrowly testable alternative because the existing cache retains all acquired slices. But closer channels exchange broader within-window coverage for local evidence; they do not dominate the declared contract. The previous original-donor neighbor experiment was inconclusive, and the strongest completed native pipeline remains the relevant incumbent.

If useful later, the smallest next diagnostic is **one paired, fixed-center support audit** using admitted fitting-only aggregate geometry. It needs no MRI pixels, weights, GPU, training, or new cache. Its role is to establish whether the geometric premise is present and quantify the tradeoff. It cannot establish whether omitted pixels contain essential label information.

## 1. New primary-source mechanisms and their limits

- **LambdaUNet (MICCAI 2021): anisotropic context should be treated differently from in-plane context.** Its explicit inter-slice term uses corresponding feature-map locations in neighboring slices, while separate terms retain dense in-plane context. The author code reshapes the slice dimension separately and applies a temporal kernel, rather than interpreting through-plane and in-plane neighborhoods identically. Its reported DWI ablation favored this restricted 2.5D design over removal of inter-slice context and over a 3D-context variant. This is stroke segmentation on anisotropic acquisitions, not knee study classification, and does not compare our two sampling contracts. Crucially, it warns that even consecutive acquired MRI slices can be discontinuous. [Paper](https://faculty.ist.psu.edu/suh972/LambdaUNet_2021.pdf), [pinned author implementation](https://github.com/YanglanOu/LambdaUNet/blob/d90f1bd6d69d0309fe67232b2b5b4606cfa94dc6/unet/lambda_networks.py)

- **Vuskov et al. (European Radiology, 2025): local mixing plus global aggregation are distinct operations.** The knee model inserts zero-initialized residual 3×3×3 blocks into ResNet18, then aggregates slice tokens using a transformer. The author implementation reshapes by study and series before each local convolution, avoiding cross-series convolution. However, its preprocessing trilinearly resizes every stack to a fixed depth; its neighboring tensors are not necessarily three original acquisitions. There is no inspected matched slice-spacing ablation establishing that original-neighbor channels outperform selected-neighbor channels. This supports the mechanism, not adoption of its architecture or a numerical RSNA gain. [Paper](https://publications.rwth-aachen.de/record/1020710/files/1020710.pdf), [pinned model](https://github.com/spacecheck/Res3DKneeModel/blob/f75ec4f892b0f8d049fdf22d155b35fdc31193e8/src/res3dish.py), [pinned interpolation](https://github.com/spacecheck/Res3DKneeModel/blob/f75ec4f892b0f8d049fdf22d155b35fdc31193e8/src/t_dataset.py)

- **KneeXNet-2.5D (2026): raw three-channel local context is implementable with a 2D encoder.** The released dataset code looks up slice-number offsets −1/0/+1 and stacks them for a ResNet34 U-Net. Missing neighbors fall back to the center. This is a concrete implementation precedent, but file-number adjacency is not itself physical-geometry validation. The task is segmentation, the full method is an ensemble, and its 3D comparison is described as a computational reference rather than a fully optimized competitive baseline. Its results should not be translated into a knee-classification benefit. Do not import its duplicate-center fallback into N3’s distinct-original contract. [Paper](https://www.nature.com/articles/s44401-026-00072-5), [pinned loader](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D/blob/d7aa1fe4e64b5fa915a171126b42a425294a7151/Code/2.5D_U-Net_segmentation_noise_model.py)

- **ELNet (MIDL 2020) supplies useful counterweight:** it encodes slices with 2D operations, then max-pools slice features, without an explicit local ordered-slice interaction. Its normalization and BlurPool ablations concern other mechanisms. This is evidence that useful knee diagnosis need not require explicit neighboring-channel continuity, not evidence that subsampling is harmless. [Primary paper](https://arxiv.org/html/2005.02706v3)

The already reviewed two-slice-touch study supports corroborating abnormal morphology across images; it does not require our exact three-channel encoding and does not turn overlapping windows, interpolated slices, or repeated pixels into independent observations. MRPyrNet, MeniMV and KneeDINO are not being recycled as new evidence here.

## 2. Four quantities must remain separate

For original projected positions z[i] and uniformly selected source indices q[j]:

- Native acquired center spacing is z[i+1] − z[i]
- Selected channel spacing is z[q[j+1]] − z[q[j]]
- Selected triplet span is z[q[j+1]] − z[q[j−1]]
- Original-neighbor triplet span is z[q[j]+1] − z[q[j]−1]

For regularly spaced originals with spacing Δ, selected spacing is approximately (N−1)/15 × Δ when N≥16. A selected triplet spans approximately 2(N−1)/15 × Δ, whereas an original-neighbor triplet spans 2Δ. Rounding makes exact source-index gaps differ.

**Center spacing is not slice thickness or true through-plane resolution.** DICOM defines slice thickness as nominal thickness and spacing as center-to-center distance. Acquisition profile and partial-volume effects are not recovered from source index count. Neither original adjacency nor interpolation creates isotropic resolution. An interpolated slice is derived from observations already available; it can smooth or reparameterize context but supplies no additional acquired measurement. [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)

The native loader verifies increasing projected positions, but that alone does not establish normal local spacing across every edge. Its existing neighbor helper admits edges using positive spacing no greater than 1.5× the series median and in-plane origin displacement no greater than 0.001 mm. Those are repository protocol choices, not universal MRI validity thresholds. Unknown geometry must remain unknown; do not invent millimeters or infer thickness from spacing. [Pinned neighbor geometry](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c83ffa8b58a3bb28dc63e7f10f4805faab0a8893/rsna/competitive/neighbor_geometry.py)

## 3. One smallest matched diagnostic

**Paired fixed-center support audit, optional and nonblocking.** Preserve the exact selected series, original count, q-grid, N3 sampler draws and every selected center. For each chosen interior q[j], compare only:

- Control: [q[j−1], q[j], q[j+1]]
- Local alternative: [q[j]−1, q[j], q[j]+1]

Both use twelve three-channel encodes for a complete six-series study, two per series. The global center locations and center-selection probabilities are exactly identical. Local windows must come from the same verified original series, with unchanged cache intensity, field of view, resize, normalization, center identity, coordinate/known flags, masks and series metadata. Record each channel’s original index and actual projected offset; do not add metadata embeddings or another model change.

A metadata-only audit can enumerate all 91 possible two-center draws per full series. Incomplete studies retain N3’s exact shortage/fill allocation and use paired common draws; do not claim that the full-series arithmetic directly describes their allocation. For unavailable/invalid physical edges, report the affected proportion and leave that case outside any claimed physical-continuity conclusion. Do not silently alter centers, duplicate slices, interpolate, or select a different geometry-qualified roster in one arm.

Report: index and millimeter spans; fraction of truly adjacent triplets; original-slice support gained and lost; maximum unobserved run; unique originals per bag and over all candidate windows; and geometry-unknown proportion. This is sufficient to falsify a broad claim of ubiquitous missing local observations if most actual inputs are unchanged or geometry cannot establish the premise.

For a hypothetical block B=[a,a+k−1] of acquired indices, let m be the number of eligible windows intersecting B, M the total, and K the number selected. Exact conditional miss probability is C(M−m,K)/C(M,K), with zero numerator when too few nonintersecting windows remain. Sweep **all** k and all legal a, not an assumed lesion size. Separately count whether a window contains two original adjacent indices inside B. That second statistic measures availability of a local pair; it is not a clinical detection rule. Uniform-start averages are explicitly synthetic assumptions, not lesion distributions.

### Executed synthetic result

The unchanged native `_indices` function was AST-extracted from the source-only audit copy. Identity, bounds and center preservation passed for N=3…1,000. Complete interval curves for eleven illustrative depths produced 924 rows, independently cross-checked against the exact combinatorial formula.

For N=31, q={0,2,…,30}. Control windows span four original index steps; local windows span two. With two windows from the same 14 centers:

- A uniformly placed one-index witness is touched with probability 18.01% control versus 18.89% local; reduced overlap slightly increases distinct samples
- A uniformly placed two-index block is touched with probability **36.26% control versus 25.71% local**
- An original adjacent pair inside that two-index block occurs in one window with probability 0% control versus 13.33% local

These counterexamples establish a coverage/local-pair tradeoff only. They are not a prior estimate of MRI accuracy or evidence that local sampling is worse. For N≤16 the two contracts are identical. For N=46, all 14 local candidate windows expose 42 originals versus 16 in the control, but omit indices 0,1,44,45. Thus retaining global centers does **not** preserve every peripheral slice or the full observation envelope. Neither contract’s observed support contains the other’s in general.

## 4. Cost, interpretation and stopping rule

The diagnostic itself uses zero encoder calls and zero GPU memory. An eventual image comparison, if separately justified, would keep 12 encodes per study in both arms, at the same resolution and architecture. That matches encoder FLOPs and activation shapes, not wall-clock time.

The native loader already reads, hashes and decompresses the full NPZ volume before sampling. Original neighbors therefore need no DICOM intake or new cache. But a downstream adapter receiving only the existing sixteen sampled images cannot recover discarded pixels; sampling must access the already-verified original volume before that reduction. At training time twelve windows require at most 36 unique original images. At all-window inference, up to 84 triplets can require 252 unique originals instead of at most 96 selected originals, increasing possible preprocessing work by up to 2.625× while retaining 84 encoder calls. Overlap reduces this bound. No runtime or hardware fit has been measured.

The support test can falsify the **geometric premise** or the claim of uniform superiority. It cannot falsify essential clinical information loss: identical metadata can accompany two image populations in which skipped slices either contain decisive information or add nothing. Swapping inputs of a model trained on the other contract would also confound sampling with distribution shift. A frozen ImageNet feature delta is not a diagnostic outcome.

The previously completed neighbor-channel control reported physical−gray medial-meniscus delta −0.000948 and physical−distant −0.000785, with relevant intervals spanning zero. It used a different bounded neighbor-delta representation and three-epoch protocol, so it does not settle raw-triplet full fine-tuning. It does weaken any claim that original neighbors are an obvious untried remedy. [Existing receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/tree/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/neighbor-channels-full-fit-v1)

**Stop/go:** retain N3 now. Skip the optional audit if it competes with completing the single comparison. If performed later, stop this branch when the support change is negligible, geometry is unverifiable, or added handling/runtime is not worthwhile. Material, geometry-verified support differences justify retaining the hypothesis for future consideration, not starting a fit automatically. Clinical relevance remains the missing evidence. No broad 3D model, cache rebuild, interpolation scheme or additional mandatory gate is recommended.

## Scope and artifacts

Public papers/standards, author code, existing aggregate research summaries, and synthetic index arithmetic only. No clinical images, individual clinical rows, caches, labels, model weights, MRI inference, training, competition submission, or repository modification.

- `rsna-slice-spacing-audit-20261015/synthetic_sampling.py`
- `rsna-slice-spacing-audit-20261015/synthetic_sampling_results.json`
- `rsna-slice-spacing-audit-20261015/synthetic_interval_curves.csv`

Primary source URLs were read on 15 October 2026. Author files were read at the pinned commits above. PMC browser-check pages were not solved; corresponding author/publisher material and GitHub source were used instead.
