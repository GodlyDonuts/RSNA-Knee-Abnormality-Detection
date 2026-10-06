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
| Parsed (header within 64 KiB prefix + HTTP success) | 20,874 (85.6%) |
| `download_failed` after retries | 3,512 (14.4%, rate-limit transient; distributions below use the parsed 85.6%) |
| Series with `ImageOrientationPatient` + `ImagePositionPatient` | 100% of parsed |
| `PatientPosition` | FFS 19,775 · blank 1,093 · FFP 6 |

Parse failures are download transients, not header defects (spot-retries
succeed); the parsed set spans all three planes roughly evenly
(Sagittal 9,864 · Coronal 8,609 · Axial 5,898 of attempted), so distributions
are representative.

## Finding 1 — Laterality is only partially tagged; geometry fallback is universal

| | Series | Studies |
| --- | --- | --- |
| L/R tag present (Laterality ∈ {L, R, LEFT, RIGHT}) | 5,912 / 20,874 (28%) | 1,894 / 4,410 (43%) |
| Tagged studies with internally consistent side | 1,873 (98.9% of tagged) | — |
| Tagged studies with contradictory tags | 21 | — |
| Side balance (study mode) | R 962 · L 932 | — |

- Tag coverage is too sparse to rely on alone: 57% of studies have **no**
  laterality tag on any series.
- Geometry fallback is viable corpus-wide: every parsed series carries
  `ImagePositionPatient`, and `PatientPosition` is FFS in ~95% (rest mostly
  blank). For FFS, the sign of the lateral component of `ImagePositionPatient`
  relative to the patient midline determines side; the midline offset can be
  estimated per-study from the position spread. 17.6% of slices sit within
  ±20 mm of the nominal midline, so the fallback should use the **series-median**
  position, not per-slice sign.
- 21 studies carry contradictory L/R tags — a measurable noise floor (~1.1%)
  for any tag-based side derivation; the geometric estimate can adjudicate them.
- Recommendation (feeds frontier opportunity #2): derive study side as
  `mode(series L/R tags)` when available, else `sign(median IPP lateral offset)`
  under the FFS assumption; flag the 21 contradictory studies and the
  side-ambiguous cases (median |offset| small) for review. Then implement
  canonical-side flipping in the data pipeline (right↔left mirror) — currently
  `rsna/data.py` only applies *random* 50% flips as augmentation, which trains
  the model to ignore side information entirely rather than normalize it.

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
| 336 | 0.45 | 0.32 | 0.37 | 0.382 |
| 384 | 0.14 | 0.12 | 0.18 | 0.143 |
| 448 | 0.10 | 0.09 | 0.16 | 0.111 |
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

Limitations: 14.4% of series missed this pass (download transients); the
midline-based geometric laterality fallback is a proposed heuristic measured
for coverage, not yet validated against tagged studies.
