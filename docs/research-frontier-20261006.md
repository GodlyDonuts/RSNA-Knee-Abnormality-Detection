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
orientation; needs an audit of tag completeness with a fallback (e.g. fibula
position classifier) for missing tags.

### 3. Resolution ladder vs the Nyquist bound

Pilkwang's argument: a meniscal tear is 1–3 mm; at the typical ~160 mm FOV a
224 px resize gives ~0.71 mm pitch — **below** the ~0.5 mm Nyquist requirement
at d=1 mm. No capacity downstream recovers signal destroyed at resize. This
explains both the 224px capacity plateau in the public ablation and why
0.95-level scores plausibly use higher input resolution. Proposed: fixed recipe,
resolution ladder 224 → 336 → 448 on one fold; memory cost is quadratic in
pixels but slice coverage is linear, so trade slices before pixels.

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

### 9. Inference-budget hardening (engineering, protects score)

From rank-18, all transferable to our submission kernel: adaptive degradation
under a projected 9h budget (eval slices 32→16, then ensemble members 5→3),
checkpoint submission.csv every 25 studies, decode fallback chain
(pydicom→pylibjpeg→gdcm), neutral-0.5 on undecodable studies. None of this is
frontier research, but each protects the score our models already earned.

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
