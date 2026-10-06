# Knee-domain self-supervised pretraining — design and cost plan

**Status**: plan + pinned input manifest. `ssl_series_manifest.parquet`
(built 2026-10-06, `scripts/research/build_ssl_series_manifest.py`):
24,386 series / 819,635 slices / 4,410 studies — the full inventory
(train_series 24,371 series / 819,078 slices / 4,407 studies + 3 sample
test studies). Columns: split, study/series uid, n_slices, plane,
fluid_sensitive, laterality tag, pixel spacing, matrix, slice thickness,
protocol string, header_available (3,512 series lack headers — census
download_failed; usable for SSL but outside header-bound policies).
No labels or reports are read by the builder. A GPU slot is still required
for the actual pretrain. Frontier item
7 (`docs/research-frontier-20261006.md`) — the largest genuine frontier swing
available to us.

## Why

Three external results say domain-matched self-supervision beats supervised
pretraining on medical volumes in low-label regimes:

- **OAI-DINO** (ISMRM 2025) — self-supervised DINO on knee MRI; beat
  supervised pretraining on downstream knee classification.
- **MRI-CORE** (arXiv:2506.12186) — 6M-slice DINOv2 MRI foundation model;
  +6.97% Dice in low-label downstream regimes.
- **MedDINOv3** (arXiv:2509.02379) — DINOv3-style domain-adaptive distillation
  works on medical volumes.

Every public solution in this competition initializes from generic
RadImageNet/ImageNet or DINOv2 weights. An encoder self-distilled on **this
corpus's** ~820k unlabeled knee slices is scanner/protocol-matched to the test
set — an initialization nobody else in the competition has.

## What we have

- ~643,501 slices / 18,906 series / 3,403 development studies already cached
  (CPU875406-verified cache stats) plus the wider training corpus — all
  unlabeled for SSL purposes.
- A native-resolution ladder already measured (median pitch 0.312 mm/px; the
  Nyquist census pins 384–448 px as the sweet spot).
- Multi-plane structure: SSL crops should respect plane and anatomy; per-plane
  aug policies (sagittal vs coronal vs axial have different spatial statistics).

## Design sketch

- **Method**: DINOv2-small student–teacher self-distillation (iBOT-style
  masked-image modeling optional). Small, not base — the public
  small-vs-base ablation at 224px converged; parameter budget is not the
  lever here.
- **Views**: standard DINO multi-crop (2 global ≥224px + 8 local), plus one
  domain augmentation nobody else has — **laterality canonicalization** (the
  `canonical-side` census mechanism) applied before cropping so the encoder
  never burns capacity on left-vs-right anatomy. Grayscale, window
  normalization per series as in the existing pipeline.
- **Depth**: 2D slices (matches the extractor); optionally 3-slice stack input
  later, but pretrain 2D — the extraction head already handles slice context.
- **Budget**: DINOv2-small on ~800k images ≈ 1–2 H100-days for a useful
  checkpoint (100–300 epochs equivalent is overkill; the OAI-DINO regime is
  reachable in a day-scale run). One-time cost, not per-experiment.

### Alternative recipe: KGNet multi-task pretraining (2026-10-06 sweep)

Zhuang et al., "Enhancing Knee Disease Diagnosis via Multi-View Graph
Representation With Multi-Task Pre-Training" (IEEE TMI 45(5):2493-2504,
2026; public code github.com/zixuzhuang/KGNet). Two transferable ideas on
top of vanilla DINO:

1. **Multi-task pretrain**: reconstruct masked knee local patches AND
   segment unmasked ones, i.e. masked-image modeling + a dense anatomical
   task — richer than pure self-distillation for anatomy-biased targets
   (cartilage, ACL — exactly the classes KGNet reports gains on).
2. **Multi-view graph**: radiologists localize in one plane and confirm in
   others; KGNet's gains come partly from structuring axial/coronal/
   sagittal series as a per-study graph. Our per-study manifest already
   has the plane grouping to feed such a structure — item 12 of the
   frontier doc (CoPAS-style cross-sequence attention) is the inference-time
   sibling and was judged marginal; the pretrain-time formulation is the
   stronger claim.

Decision stays the same: whichever recipe wins a short paired probe on the
same manifest becomes `encoder.init`; the arbiter is unchanged (paired
outer-fold AUC vs ImageNet init). No implementation until a funded GPU
slot exists — recorded here so the recipe choice is made deliberately.

## Then what

Same bound harness as every other experiment: SSL-pretrained encoder as
`encoder.init` in the identical recipe/folds — paired against the current
ImageNet/RadImageNet init under the same outer fold and bootstrap arbiter. If
it wins, every subsequent experiment (label source, graded targets, masked-cell
losses, canonical sides) stacks on the stronger init.

## Prerequisites before launch

1. A funded GPU slot: the H100 path on Newton (existing
   `publication_coordinator` pattern) or a Kaggle GPU notebook — the doc is
   ready before the slot exists.
2. A slice-level manifest bound by sha (uid × series × instance × plane ×
   canonical-side flag) — build from the existing cache index; no new data.
3. Decide augmentations per plane; lock them in config before launch
   (post-hoc aug tuning is selection noise).
4. Acceptance criteria written down now: pretraining loss trajectory is
   engineering evidence only; the arbiter is the paired outer-fold AUC
   against identical-recipe ImageNet init.

## Explicit non-goals

- No 3D/volumetric pretraining v1 — the extractor is 2D + pooling.
- No multi-modal report fusion — supervision comes from the labels work, not
  the encoder.
- No foundation-model-size escalation — the small-vs-base convergence result
  says spend compute on domain match, not parameters.
