# Image-verified label pass (image-review-v1)

Goal: verify all 4,407 train studies' labels against the **actual MRI images** —
every prior label generation was report-text-derived; this is the first pass that
looks at pixels.

## Substrate

- Public HF pack `kevlar-ethereal-SOTA/rsna-knee-336-12slices` (private
  `GodlyDonuts/RSNA-Data` is gated): per study, 6 series slots × central 12
  slices × 336×336 px, 130 mm crop. Slots:
  `SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1`.
- `tools/make_montages.py` renders six labeled 3×4 montage JPEGs per study.

## Method (v2 informed verification)

- 147 batches × ~30 studies, seeded shuffle (`tools/build_manifest.py`,
  seed 20261007) — one Devin child session per batch.
- Each agent receives a bundle zip: all montage dirs + `studies.json`
  (study_uid, side, slots, report text, prior label per target; `prior=null`
  for the 58 gold studies so reviewer accuracy is measurable blind).
- Protocol in `REVIEW-PROMPT.md`: per study, view all six montages, read the
  report, and confirm/correct each of the 12 targets
  (positive / negative / uncertain / insufficient_evidence) with cited
  slice-level evidence, plus `needs_full_res` flag.
- Official scoring semantics: **mild counts** (small effusion, osteophyte,
  tiny Baker's cyst = positive).

## Findings so far (fleet-shared notes + first 30 batches)

- Gold-15 scored so far: macro sensitivity 0.983, macro specificity 0.792.
  Effusion over-calls (spec 0.12) — adjudication pass will tighten.
- Data-quality traps discovered and handled by agents:
  duplicated series slots (SAG_FLUID_NOFS==SAG_T1 etc.), per-plane display
  flips, ~15% laterality-field errors (fibula is ground truth), a few
  scrambled reports (e.g. a shoulder MRI report attached to a knee study).
- Report-vs-image conflicts are common: unreported Baker's cysts, report-
  denied effusions visible on image, report covering both knees.

## Files here

- `REVIEW-PROMPT.md` — the exact v2 reviewer prompt.
- `batches.json` — 147 batches × study UIDs (deterministic, seed 20261007).
- `image_review.csv` — per-study 12-target image states + evidence (partial;
  updated as batches land).
- `flips_vs_current.csv` — cells where image review disagrees with the current
  `study_labels.csv` values.
- `results_jsonl_partial/` — raw per-agent results.jsonl uploads.
- `*.py` — montage/manifest/bundle/aggregation tooling.
