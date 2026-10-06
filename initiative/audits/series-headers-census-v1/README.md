# Series-headers census v1 — laterality coverage and pixel-pitch vs Nyquist

Date: 2026-10-06 · Instrument: `scripts/research/audit_series_laterality.py`
Data: `series-headers.jsonl.gz` (24,386 series; one row per series, DICOM header
fields read via 64 KiB HTTP Range requests against `GodlyDonuts/RSNA-Data` —
no pixel data downloaded).

This census answers two frontier questions measured instead of assumed:

1. Can every study's left/right side be recovered (laterality normalization,
   measured +3.2 pt in the rank-18-style recipe write-up)?
2. Does the current 224 px input satisfy the Nyquist bound for the smallest
   diagnostic target (~1 mm meniscal tear)?

## Coverage

| Metric | Value |
| --- | --- |
| Series attempted | 24,386 |
| Parsed (header within 64 KiB prefix + HTTP success) | 24,386 (100% after retry passes) |
| Series with `ImageOrientationPatient` + `ImagePositionPatient` | 100% of parsed |
| `PatientPosition` | FFS 19,775 · blank 1,093 · FFP 6 |

All 24,386 series parsed after two retry passes on transient download
failures; coverage spans all three planes
(Sagittal 9,864 · Coronal 8,609 · Axial 5,898).

## Finding 1 — Laterality is only partially tagged; geometry fallback is universal

| | Series | Studies |
| --- | --- | --- |
| L/R tag present (Laterality ∈ {L, R, LEFT, RIGHT}) | 12,005 / 24,386 (49%) | 2,204 / 4,410 (50%) |
| Tagged studies with internally consistent side | 2,179 (98.9% of tagged) | — |
| Tagged studies with contradictory tags | 25 | — |
| Side balance (study mode) | R 962 · L 932 | — |

- Tag coverage is too sparse to rely on alone: 50% of studies have **no**
  laterality tag on any series.
- Geometry fallback is viable corpus-wide: every parsed series carries
  `ImagePositionPatient`, and `PatientPosition` is FFS in ~95% (rest mostly
  blank). For FFS, the sign of the lateral component of `ImagePositionPatient`
  relative to the patient midline determines side. 17.6% of series sit within
  ±20 mm of the nominal midline, so the fallback should use the **series-median**
  position, not per-slice sign (validated below).
- 25 studies carry contradictory L/R tags — a measurable noise floor (~1.1%)
  for any tag-based side derivation; the geometric estimate can adjudicate them.

**Geometric fallback VALIDATED against tagged studies** (`study-sides.parquet`):
the median lateral IPP offset of a study's sagittal series agrees with its
DICOM tag in **94%** of tagged studies at any offset, rising to ~97% at
|x|≥20 mm and ~99% at |x|≥30 mm. Sagittal series are the reliable probe —
their slab is prescribed on a single knee (per-series agreement: sagittal
96.9%, coronal 78.5%, axial 83.4%). Every study
has ≥1 sagittal series, so the fallback covers the full corpus.

Derived table `study-sides.parquet` (4,407 studies): L 2,006 · R 2,257 ·
ambiguous 144 (|median x|<20 mm) · high-confidence 4,107 (|x|≥30 mm) ·
bilateral scans 443 (series on both sides — need per-series handling, not a
study-level flip) · tag-vs-geometry conflicts 135 (prefer geometry; probable
tag errors).

- Recommendation (feeds frontier opportunity #2): assign each study's
  canonical side from `sign(median sagittal IPP x)` (validated ~94–99%),
  using DICOM tags only as corroboration and flagging the 114 conflicts +
  144 ambiguous + 443 bilateral studies. Then implement canonical-side
  flipping in the data pipeline (mirror the non-canonical side so medial
  anatomy always appears on the same image side; labels stay anatomical —
  "medial" is still medial — but *random* flip augmentation in `rsna/data.py`
  must be disabled for normalized studies, since it would re-randomize the
  medial/lateral image position the normalization just fixed).

## Finding 2 — 224 px input violates Nyquist for ~1 mm structures on 99.6% of series

Nyquist bound: resolving a ~1 mm meniscal tear needs sampling pitch ≤ 0.5 mm/px.
Native acquisition pitch is median **0.312 mm/px** (p90 = 0.500, p99 = 0.703) —
the scanners already capture sub-mm detail. Median FOV is 160 mm.

Effective pitch after resize to model side *s* ≈ `max(FOV/s, native pitch)`.
Fraction of series failing the 0.5 mm bound:

| Input side | Sagittal | Coronal | Axial | Overall |
| --- | --- | --- | --- | --- |
| 224 (current) | 1.00 | 1.00 | 0.98 | **0.996** |
| 256 | 1.00 | 1.00 | 0.98 | 0.996 |
| 336 | 0.44 | 0.31 | 0.36 | 0.377 |
| 384 | 0.14 | 0.12 | 0.18 | 0.140 |
| 448 | 0.09 | 0.09 | 0.16 | 0.110 |
| 512 | 0.09 | 0.09 | 0.16 | 0.108 |

- At the current 224 px input, **99.6% of all series are sampled below Nyquist**
  for 1 mm structures — the model cannot see the detail a meniscal-tear label
  depends on, regardless of architecture.
- The residual ~10% failing at ≥448 px are series whose *native* pitch already
  exceeds 0.5 mm (Axial 11.6%, Sagittal 9.4%, Coronal 8.7%) — acquisition-limited,
  unrecoverable by resampling. The practical resolution ceiling is ~448 px.
- This is consistent with the measured score ladder: uniform224 fold0 = 0.770,
  native384 fold0 = 0.785 (+0.015) — the 384 run recovered ~86% of series to
  Nyquist and gained; pushing to ~448 recovers most of the remaining headroom
  that resampling can deliver.
- Through-plane sampling (slice thickness ~4 mm) is far coarser than the bound
  and cannot be fixed by 3D modeling — reinforcing that **in-plane** resolution
  is the binding constraint, and 2.5D windows are the right formulation.

## Implications for the frontier plan

- Opportunity #3 (resolution ladder) is now a measured defect, not a
  hypothesis: the default input discards diagnostic-frequency content on
  essentially every series. Expected ordering for the next fits:
  384 ≈ 448 > 336 ≫ 224/256, with 448 near the resampling ceiling.
- Opportunity #2 (laterality) needs the geometric fallback to cover the
  untagged 57% — tag-only normalization would leave most studies unnormalized.

## Reproduce

```bash
HF_TOKEN=... python -m scripts.research.audit_series_laterality \
  --inventory initiative/source/inventory \
  --series initiative/source/train_series.csv \
  --output series-headers.jsonl
# analysis: pandas over the JSONL; pitch = max(PixelSpacing), FOV = spacing × dims
```

Coverage is now complete (24,386/24,386 after retries).
