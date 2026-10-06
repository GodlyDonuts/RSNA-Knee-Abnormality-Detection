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
2026; public code github.com/zixuzhuang/KGNet — inspected 2026-10-06).
Two transferable ideas on top of vanilla DINO, **with a hard prerequisite**:

1. **Multi-task pretrain**: `pretrain` vertex-masks knee-graph vertices
   (fully-masked → reconstruct, kept → predict class, mixed); the seg head
   predicts **8-class × 3-view bone segmentations** (`KneeLoss.seg_` is
   cross-entropy on `data.seg`) — richer than pure self-distillation for
   anatomy-biased targets (cartilage, ACL — exactly the classes KGNet
   reports gains on).
2. **Multi-view graph**: radiologists localize in one plane and confirm in
   others; KGNet structures axial/coronal/sagittal series as a per-study
   graph. Our manifest already has the plane grouping — but the graph's
   vertices come from `construct_graph.py`, which extracts bone vertices
   FROM segmentation masks (`bone_index`, `seg/` folders, adjustFOV).

**Prerequisite cost (code-verified)**: the KGNet recipe needs per-study
bone segmentation maps that our corpus does not have. Options: train a
pseudo-segmenter on OAI-ZIB (public knee MRI segmentations) or an
existing knee segmentation model, then run KGNet-style masking on its
outputs. That makes it a *second*-recipe candidate behind the cheap DINO
self-distill probe — record the dependency now so the GPU session prices
it honestly instead of discovering it mid-run.

Decision stays the same: whichever recipe wins a short paired probe on the
same manifest becomes `encoder.init`; the arbiter is unchanged (paired
outer-fold AUC vs ImageNet init). DINO self-distill needs nothing beyond
the manifest; KGNet-style needs a segmenter first. No implementation until
a funded GPU slot exists.

### Before ANY self-pretrain: evaluate released knee foundation weights

2026-10-06 sweep found a stronger starting point than self-pretraining may
provide: **OrthoDiffusion** (Xu et al., arXiv:2602.20752, 2026) — three
orientation-specific 3D diffusion branches (sag/cor/axi) self-supervised
on **15,948 unlabeled knee MRI examinations** (~1.44M images from a
30,653-study multi-joint corpus), linear-probe and finetune recipes for
8 knee abnormalities, 11-structure segmentation, cross-center/cross-field-
strength robustness, label-efficient (high precision at 10% labels).
Public code `github.com/lt-0123/OrthoDiffusion`; public weights
`huggingface.co/lanstat0123/orthodiffusion` (med-ddpm backbone).

Fit assessment for us: OD is **3D volumetric** — its branches consume
per-view volumes, while our extractor is 2D-slice + attention pooling.
Two integration paths: (a) swap the encoder path to OD-style per-view
volume features feeding our per-target study head (bigger surgery, most
knee-domain signal), or (b) self-pretrain on our manifest but borrow OD's
recipe (view-specific branches + denoising-feature reuse).

**Availability checks (2026-10-06)**: MRI-CORE is Apache-2.0 with ViT-B
weights on Google Drive (github.com/mazurowski-lab/mri_foundation) —
2D-slice, 6M slices/18 body regions, **direct drop-in for our 2D
extractor** and the cheapest probe to run (no architecture surgery).
Triad is MIT with 4 checkpoints on GDrive (SwinB + PlainConvUNet, MAE +
SimMIM variants) but 3D. PRISM and Decipher-MR appear paper-only — no
public weights found. Ranked probe list: (1) MRI-CORE ViT-B init —
drop-in, Apache-2.0 — caveat: knee coverage in Duke-110K's 18 body
locations is unverified (the paper's own similarity-transfer analysis
is the check to run before committing); (2) OrthoDiffusion path — most
knee-domain signal, 3D surgery; (3) own-corpus DINO self-distill —
strongest domain match, day-scale cost; (4) Triad Swin-B — generic 3D,
weaker evidence; (5) KGNet-style multi-task — needs segmenter first.
The single best move if a GPU slot opens: short paired probe of
MRI-CORE init vs current ImageNet init, same head, same fold —
cheapest possible read of whether domain-matched initialization is
the lever.

**MRI-CORE checkpoint verified (2026-10-06, downloaded + decoded)**:
`MRI_CORE_vitb.pth`, 436 MB, sha256 `0ca90aeba09568f2…` — GDrive file
`1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW` (linked from the mazurowski-lab
README). Contents: `{"teacher": {...}}` — a DINOv2-teacher checkpoint,
`backbone.*` ViT-B/16 (768-dim, `patch_embed 3×16×16`, `pos_embed
(1,197,768)` = native 224 px — feed 384 px with standard pos-embed
interpolation) plus `dino_head.*` (drop it — feature extraction uses
the backbone only). Wiring spec for the probe: strip `teacher.`/
`backbone.` prefixes, load into our ViT-B encoder path (same shape
contract as the current init), interpolate pos_embed for the cache
side, leave `dino_head`/masked-token keys unloaded.

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
