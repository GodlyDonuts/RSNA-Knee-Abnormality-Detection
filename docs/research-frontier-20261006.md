# Research frontier survey — RSNA Knee Abnormality Detection (2026-10-06)

External literature + competitor write-up sweep, mapped to concrete opportunities
for our single-model-first track. Every claim below is cited; measured effects
are quoted with their source's own numbers, not ours.

## What the field now agrees on

Three independent sources converge on the same bottleneck — **label quality,
not model capacity**:

1. Rank-18 public write-up (0.903, top 2.3%): the report label extractor's own
   ceiling is ~0.83 mean per-class agreement against expert image re-reads, and
   is explicitly named as "the real limiting factor on this whole approach."
   https://huggingface.co/blog/bishnoiyash/rsna-competetion
2. Pilkwang (author of the LLM labels powering the copied 0.936/0.942
   ensembles): "the ceiling of this problem sits ... in the supervision rather
   than of the network." Their design already treats label confidence as a
   first-class output (graded mention strength, per-language coverage
   measurement, no-opinion-when-unmatched).
   https://pilkwangkim.github.io/posts/RSNA-Knee-Twelve-Findings-From-One-MRI/
3. Kaggle discussion ablation: DINOv2-small vs -base at 224px converge to the
   same score — "capacity doesn't seem to raise the ceiling at 224."
   https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738096

Our v2 review (4,407 reports, 52,884 evidence-bound decisions, per-cell state +
evidence + uncertainty) is plausibly the most rigorous label artifact in the
field. The research below says that is exactly where the competition is won.

## Frontier opportunities, ranked

### 1. Label-source ablation — v2 vs public LLM labels (unchanged, still #1)

Same architecture, same folds, same epochs — one fit on `llm_labels_v4_blend`
(pin: `Shio-Koube/knee3@373f4e1d`, sha256
`eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`) vs one on
the v2 reviewed release. This is the experiment that decides whether our 52,884
decisions bought anything the crowd's 35B-parameter extractor didn't. Verified
label CSVs are at `/tmp/public-labels/` (this session) and need a release-format
overlay per `rsna/competitive/supervision.py` precedent.

### 2. Knee laterality normalization — measured +3.2pt elsewhere

Pilkwang measured mean attention gradient alignment improving **0.870 → 0.902**
once left/right knees were mirrored to a common side (coronal in-plane mirror).
Five of our twelve targets are side-dependent (MCL, medial/lateral meniscus,
medial/lateral OA); without normalization, the model must learn medial-vs-
lateral from an axis it cannot observe consistently. Check whether our cache
applies it; if not, this is the cheapest large single-model gain available.
Determinable per-study from DICOM Laterality / ImageLaterality tags plus
orientation. **Census done** (`series-headers-census-v1`): only 43% of studies
carry an L/R tag on any series, so tag-only normalization leaves 57%
unnormalized — but every series carries `ImagePositionPatient` and ~95% are
FFS, so a geometric fallback (sign of median lateral IPP offset) can cover the
rest; 21 studies (1.1% of tagged) carry contradictory tags to flag. Our
pipeline currently applies random 50% flips as augmentation (`rsna/data.py`),
teaching the model to ignore side entirely rather than normalize it.

### 3. Resolution ladder vs the Nyquist bound — now MEASURED corpus-wide

Pilkwang's argument: a meniscal tear is 1–3 mm; at the typical ~160 mm FOV a
224 px resize gives ~0.71 mm pitch — **below** the ~0.5 mm Nyquist requirement
at d=1 mm. Our series-header census (20,874 series, audit
`series-headers-census-v1` on the ablation branch) confirms it at scale:
native pitch is median 0.312 mm/px and median FOV 160 mm, so **99.6% of series
fail the Nyquist bound at the current 224 px input** (Sagittal 100%, Coronal
100%, Axial 98%). Failure rate drops to 38% at 336 px, 14% at 384 px, and
plateaus at ~10% for ≥448 px — the residual is acquisition-limited (native
pitch >0.5 mm, unrecoverable). Resolution ceiling ≈ 448 px. Consistent with
our own ladder: uniform224 fold0 0.770 → native384 fold0 0.785. Proposed:
fixed recipe, resolution ladder 336 → 448 on one fold; memory cost is
quadratic in pixels but slice coverage is linear, so trade slices before
pixels. Through-plane pitch (~4 mm slice thickness) can't be fixed by 3D —
in-plane resolution is the binding constraint.

### 4. Graded/ordinal soft targets from v2 states

Public extractors emit graded mention strength because "only the order of
scores is read" (AUC). Our release already carries `__state` (positive /
uncertain / not-mentioned) + evidence — a natural ordinal target. Two cheap
variants of the ablation arm: (a) binary as now; (b) soft/ordinal targets where
uncertain cells get intermediate values rather than masking or flooring.
Literature support: soft-label learning beats hard labels specifically under
small-N and class imbalance (Machine Learning 2025,
doi:10.1007/s10994-025-06860-8).

### 5. Disagreement mining across label sources

Fuse `llm_labels_v4_blend` with v2 per-cell states and extract every cell where
the sources conflict beyond a tolerance (e.g. |public − v2| > 0.5 on an
unmasked cell). Those cells are the highest-value review targets in the corpus
— a confident-learning-style refinement loop (MICCAI 2020 CL tradition; LiNC,
openreview 0ik0xKYQd1) that spends human/LLM adjudication only where it can
move the label table. Output feeds a v2.1 delta review, not a full re-review.

### 6. Masked-cell losses from the SPML/PU literature

Our masked cells are a missing-label problem, not negatives. The SPML line of
work (Cole et al. 2021; MPC/IPC-FPC-BTC, MICCAI W 2024; ESA, arXiv:2412.02240;
NeurIPS 2025 PU survey) offers concrete losses: entropy-maximization on
unlabeled cells, generalized assume-negative, consistency pseudo-labeling
(weak-aug teacher → strong-aug student). A controlled variant: identical fit,
three masked-cell treatments — ignore (current), EM regularizer, consistency
pseudo-label. Most relevant for the sparse targets (MCL 7, fracture 5,
Baker's 9 observed positives).

### 7. Knee-domain self-supervised pretraining on our own corpus

The strongest genuine frontier swing available offline: OAI-DINO (ISMRM 2025,
self-supervised DINO on knee MRI, beat supervised pretraining on downstream
classification) and MRI-CORE (6M-slice DINOv2 MRI foundation model, +6.97%
Dice in low-label regimes; arXiv:2506.12186). MedDINOv3 (arXiv:2509.02379)
shows domain-adaptive DINOv3-style distillation works on medical volumes. We
hold ~820k unlabeled knee slices in cache form — a DINOv2-small self-distill
on THIS corpus (scanner/protocol distribution matched to test) is a pretrained
encoder nobody else in the competition has published. CPU-infeasible locally;
needs the H100 slot or a Kaggle GPU notebook, but is a one-time ~day-scale
pretrain, not a per-experiment cost.

### 8. Per-diagnosis query attention over plane slots (validated externally)

Pilkwang's head: six slot embeddings (3 planes × 2 sequences) with presence
mask, one learned query per diagnosis attending over slots — deliberately kept
small because label supervision bounds richer aggregation. Rank-18's negative
result is worth noting honestly: gated-attention slice pooling *lost* to plain
max-pool at their scale — so keep both pooling options in the ablation grid
rather than assuming attention wins.

**VERIFIED against our code (2026-10-06):** `CompetitiveStudyNet` already
implements this mechanism — `slice_score` emits a per-target attention score
over slices, `series_score` is a learned [12, hidden] query vector attended
over series slots by `_masked_softmax`, and `output_weight` is a per-target
output projection. `max` pooling is the implemented control arm. The
ablation-grid choice is attention-vs-max at the pooling level, not adding
query attention — the head structure already matches the externally
validated design.

### 9. Inference-budget hardening (engineering, protects score)

From rank-18, all transferable to our submission kernel: adaptive degradation
under a projected 9h budget (eval slices 32→16, then ensemble members 5→3),
checkpoint submission.csv every 25 studies, decode fallback chain
(pydicom→pylibjpeg→gdcm), neutral-0.5 on undecodable studies. None of this is
frontier research, but each protects the score our models already earned.

**VERIFIED against our code (2026-10-06):** `scripts/kaggle/chunked_infer.py`
already implements most of this — chunked/resumable inference with per-chunk
submission writes, neutral rows for unusable studies
(`test_unusable_mri_studies_get_neutral_rows`), resilient healthy-series
alternates, and decoder-plugin fallback for recognized DICOM failures. The
one genuinely missing piece was a **wall-clock budget governor** — BUILT
(2026-10-06, PR #4): `--budget-seconds` projects per-chunk overrun from the
observed per-study rate and `--budget-neutralize` fills the remaining tail
with neutral 0.5 rows so a near-overrun still emits a complete submission
instead of dying at hour 9. Model-input degrade (fewer slices) is not
possible mid-run: cache geometry is checkpoint-bound. Margin remains
comfortable (5.1h measured vs 9h), so this stays contingency engineering.

## Second sweep (2026-10-06): MIL-family and auxiliary findings

### 10. Slice-level MIL losses and local-dependency pooling

SmMIL (arXiv:2410.03276) models a simple prior: **neighboring instances tend
to share the same label** — a local-dependency module between adjacent slices
that improves both classification and localization in medical MIL. BE-WISE
(MICCAI-adjacent 2025 breast MRI) shows slice-level weak supervision lifts
study AUC ~7 points over a slice-transformer baseline. For us, our slices are
spatially ordered (patient-geometry positions are bound and verified), so
adjacent-slice consistency is directly meaningful: a meniscal tear occupies
a contiguous run of sagittal slices, not scattered singletons. Our current
pooling treats slices independently within the gated attention; a
local-smoothness term on per-slice scores (e.g. penalize |score[i+1] −
score[i]| weighted by physical proximity, or explicit MIL losses where the
study label supervises the max per-slice score) is an implementable
pooling/loss variant — pairs naturally with the `attention` vs `max` ablation
grid as a third arm.

### 11. Plane-slot design audit — axial is load-bearing, already covered

TripleMRNet (QIMS 2024) ran all 7 plane subsets on MRNet: for **meniscal
tears the axial plane beat sagittal and coronal** (ACC 0.783, close behind
the 3-plane 0.783 vs sagittal's 0.633); for ACL, sagittal leads with axial
second. VERIFIED against our code: `select_series` already picks one series
per plane before filling remaining slots, so every study's plane coverage is
preserved and the per-target series attention can weight planes freely. No
action — recorded because axial-plane series must never be deprioritized in
future series-selection changes.

### 12. CoPAS co-plane attention — partially implemented, marginal gain

CoPAS (Nat Commun 2024, s41467-024-51888-4) — co-plane attention across MRI
sequences, the largest multi-sequence knee abnormality dataset (1,748
subjects, 12 types) — reached overall AUC 0.812. Our series features already
encode plane + fluid/fat flags and the per-target series attention provides
cross-sequence mixing; a dedicated co-plane attention block is a marginal
architectural variant at best.

## Validation doctrine adopted from the field

- Dual-track arbitration (pilkwang): a large derived-label holdout arbitrates
  recipe choices; gold58 is reported but never arbitrates (SE wider than the
  deltas). Matches our finding that the 1,156-study weak set can't resolve
  ±0.001–0.04 deltas — pair it with bootstrap intervals as arbiter only.
- Gold must never leak into fold selection (rank-18's published bug:
  0.9357 claimed vs 0.8568 honest cross-fitted). We already hold gold out.
- Report-hash leakage grouping (identical template reports → identical
  targets): we already have `leakage_group` — keep it load-bearing in splits.

## Explicit non-opportunities (researched, deprioritized)

- Larger backbones at 224px — converged to same score (public ablation).
- More members in rank fusion — deferred by owner direction (single model first).
- Threshold/calibration tuning — AUC is rank-invariant; never moves the metric.
- Text-branch multimodal models — reports don't exist at inference; ruled out
  by the competition contract itself.
- Default TTA stacking — a 2026 systematic study (arXiv:2604.09697) measured
  consistent accuracy *degradation* from standard TTA on medical benchmarks
  via train/test augmentation distribution shift (up to −31.6pt), with
  intensity-only augs degrading least. Not a default win; any TTA must be
  validated per-model on our holdout before use.
- CORN/CORAL ordinal heads for severity — needs ordinal labels (KL-style
  grades) that our binary targets don't carry; revisit only if severity
  labels materialize.
