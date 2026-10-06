# Laterality normalization v1 — canonical-side mirroring experiment design

Date: 2026-10-06 · Basis: audit `series-headers-census-v1` (measured, this repo)
External motivation: pilkwang measured mean attention gradient alignment
**0.870 → 0.902** after mirroring left/right knees to a canonical side
(`RSNA-Knee-Twelve-Findings-From-One-MRI`). Five of our twelve targets are
side-dependent (MCL, medial/lateral meniscus, medial/lateral OA).

## Measured basis (from the census)

- Only **50%** of studies carry a DICOM L/R tag on any series — tag-only
  normalization leaves half unnormalized.
- The geometric fallback is **validated**: median lateral
  `ImagePositionPatient` offset of a study's sagittal series agrees with the
  tag in **97.5%** of consistently-tagged studies (98.7% at |x|≥20 mm,
  99.1% at |x|≥30 mm). Sagittal series are prescribed on a single knee, so
  they are the reliable probe; coronal/axial x-offsets agree only 78–83%.
- `study-sides.parquet` (4,407 studies): L 2,006 · R 2,257 ·
  ambiguous 144 (|median x|<20 mm) · bilateral 443 (series on both sides) ·
  tag-vs-geometry conflicts 135 (prefer geometry — probable tag errors).
  Bound artifacts for all three outer folds — the sides table is the same
  3,403-study set (a study's side is fold-independent); only the roster
  bindings differ: `canonical-sides.fold0.json` (sha256
  `4cf642c9988f265d88b133fe9b973d3758b4f38721cf3ed84032dac4618184fe`),
  `canonical-sides.fold1.json`, `canonical-sides.fold2.json`.
  Census stats: **1,251 mirrored** (L→R on the fold-0 fitting roster),
  417 unresolved (ambiguous/bilateral/no-geometry — never flipped).

## Current code state

- `CompetitiveStudyDataset` (the path the competitive fits use) performs
  **no horizontal flipping at all** — left and right knees reach the model in
  both orientations, so the network must spend capacity learning medial-vs-
  lateral from an axis it cannot consistently observe.
- The random 50% flip in `rsna/data.py` applies only to the base
  `StudyDataset` used by `rsna/infer.py` — not the competitive path.
- Normalization is therefore purely additive: a deterministic mirror, no
  existing augmentation to remove or conflict with.

## Proposed mechanism

1. **Side table** (`study-sides.parquet`, produced by
   `scripts/research/audit_series_laterality.py`): study → side estimate +
   confidence + flags.
2. **Canonical side = R** (corpus majority, 2,257 vs 2,006).
3. **Deterministic flip**: in `CompetitiveStudyDataset.__getitem__`, after
   `sample_neighbor_images`, `images[slot, :valid] = ...flip(-1)` when the
   study's derived side is L. Applied identically in fitting, dev evaluation,
   and Kaggle inference — this is canonicalization, not augmentation.
4. **All planes** receive the in-plane horizontal mirror (pilkwang's recipe).
   Alternative arm (coronal+axial only, sparing sagittal AP orientation) is
   noted as an ablation variant if the first result is ambiguous.
5. **Exclusions**: ambiguous (144), bilateral (443), and any study without
   geometry are left unmirrored and flagged — a wrong mirror is worse than
   none. Tag-geometry conflicts (135) follow geometry.
6. **Labels unchanged**: medial/lateral stay anatomical; after mirroring,
   medial anatomy lands on a consistent image side, which is the point.
7. **Inference**: the same derivation runs on test DICOM headers inside the
   kernel (IPP is present in every file; the census method needs only a
   64 KiB header read) — no external dependency.

## Binding surface (implemented)

- `CompetitiveStudyDataset` accepts `sides: pd.DataFrame | None` (columns:
  `study_uid`, `flip`). The table must cover the study roster (a superset is
  fine — one artifact serves train and outer); missing coverage fails loudly.
  Channel-agnostic: the flip composes with `neighbor_channels` unchanged.
- `rsna/competitive/canonical_side.py` validates the bound artifact (sha256,
  schema, experiment-only status, release/folds/fold/fitting/outer roster
  hashes) and emits a receipt into the run binding.
- `TrainConfig.canonical_sides` policy `{"kind": "canonical_side_v1",
  "sha256": ..., "enabled": true|false}` plus `--canonical-sides <artifact>`;
  `enabled: false` runs the control arm with the same bound artifact. The
  first contrast is isolated (no fitting-supervision / relation / witness /
  neighbor-channel mixing), matching the repo's contrast conventions.
- Build: `python -m scripts.research.build_canonical_side_table --studies
  <studies.parquet> --folds <fold.csv> --fold 0 --census <series-headers.jsonl>
  --canonical R --output <artifact.json>`

## Experiment protocol (paired, mirrors label-source-ablation-v1)

- Same release/folds/seed (`report-supervision-v1`, fold 0), fixed
  3 epochs / 843 updates, identical recipe.
- Arm A (control): current pipeline. Arm B: + canonical mirroring.
- Primary endpoint: 1,156-study disjoint dev macro AUC; **report the
  side-dependent subset** (MCL, MM, LM, Medial OA, Lateral OA) separately —
  that is where the effect should concentrate; the other 7 targets are the
  internal negative control and should not move.
- Decision rule: adopt if dev macro improves beyond paired-bootstrap noise;
  flag for a public-score check either way since dev AUC on rare targets is
  coarse.

## Caveats

- The residual tag/geo disagreements at confident offsets mean a small
  minority of studies may be mirrored the wrong way; at 0.870→0.902-type gains the wrong
  minority is tolerable, and the ambiguous/bilateral exclusions bound it.
- Sagittal in-plane mirror also reverses the AP axis. If a mirrored sagittal
  hurts (measurable as a dev regression on sagittal-dependent targets), fall
  back to coronal+axial-only mirroring.
