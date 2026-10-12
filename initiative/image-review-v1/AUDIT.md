# Image Review v1 �� Audit Report (2026-10-07)

Every one of the **4,407 train studies was visually verified** against the actual MRI
images (kevlar-ethereal 336px montage packs: 6 series × central 12 slices). This is the
first label pass in the initiative that looked at pixels at all — every prior generation
(v1 public pseudo-labels, v2 report review, v2.1 adjudication) was report-text-derived.

## Method
- 147 batches of ~30 studies each, one Devin review session per batch (dynamic-workflow
  fleet; ~400 session attempts over ~20 h due to org concurrency churn).
- Reviewer saw: all six series montages, the radiology report, and the current label
  candidate per target (gold-58 priors hidden).
- Instruction: **the image is ground truth; report silence is not evidence of absence.**
  Every cell carries a visual evidence citation (series+slice).
- Per-cell output: positive / negative / uncertain / insufficient_evidence + evidence
  string + needs_full_res flag.

## Coverage and QA
- 52,884 cells: 15,283 positive / 32,979 negative / 3,924 uncertain / 2 insufficient
  (non-gold), gold-58 untouched (696 cells gold_official).
- **Gold-58 reviewer accuracy** (image review vs official labels, priors hidden):
  macro sensitivity 0.887, macro specificity 0.744.
  - Sens 1.00: ACL, MCL, Medial Meniscus, Effusion, Baker's.
  - Weakest sens: Synovitis 0.48, Lateral Meniscus 0.81, Lateral OA 0.82.
  - Effusion spec 0.04 — reviewers call borderline fluid positive far more often than
    official semantics. Pos-flips on Effusion are the lowest-confidence flip class.
- **Inter-reviewer agreement**: 29 batches (867 studies, 10,404 cells) were independently
  reviewed twice (main + tail runs). Exact agreement 86.0%. Hardest targets:
  Baker's 0.586 (locule-vs-vessel at 336px), Contusion 0.772, Synovitis 0.850.
  Strongest: Lateral Meniscus 0.961, Medial Meniscus 0.923, Fracture 0.922.

## What changed — 11,368 flips vs current labels
| target | flips | dominant pattern |
|---|---|---|
| Effusion | 3,577 | fluid visible vs "no effusion"/silent reports |
| PF OA | 1,697 | unreported cartilage defects/osteophytes |
| Baker's | 1,649 | unreported SM-MG interval locules, incl. report-denied giants |
| Medial OA | 1,480 | unreported osteophytes/subchondral change |
| Lateral OA | 945 | same |
| MCL | 714 | periligamentous edema / gr-I sprains reports call "normal" |
| Contusion | 421 | geographic bruises reports omit |
| Fracture | 236 | insufficiency/impaction lines, avulsions |
| ACL | 210 | mostly partial→tear upgrades and mucoid→uncertain |
| Medial Meniscus | 206 | displaced fragments, extrusions |
| Synovitis | 153 | frondy/nodular recess tissue |
| Lateral Meniscus | 80 | — |

This is exactly the failure mode flagged by the project owner: radiologists omit
findings they consider obvious or overlook them entirely. Report-derived labels
systematically miss Baker's cysts, effusions, mild OA, and low-grade sprains.

## Data-integrity findings (affect the *dataset itself*, not just labels)
- **Duplicated/mislabeled series slots**: ~60%+ of studies have ≥1 pixel-identical
  montage slot pair. Dominant patterns: `SAG_FLUID_NOFS==SAG_T1` (one non-FS PD or T1
  serves both slots) and `SAG_FLUID_FS==SAG_FLUID_NOFS`; `COR_FLUID_FS==COR_T1` on ~20%.
  Some FS-labeled slots carry non-FS/T1 contrast — **read contrast, not the label.**
  A few studies have NO true FS sagittal or NO true T1 at all.
- **Directory-name/uid scrambles**: in batches img-017, -038, -075, -077, -098 the
  directory names are valid uids *belonging to different studies* (full permutations).
  All calls were verified content-vs-report before trusting uids.
- **Scrambled/mismatched reports**: one study's report was a shoulder MRI; several
  studies carry two reports (R+L knees) under one study; several reports are truncated;
  a few reports clearly describe the wrong study.
- **Laterality**: ~15+ studies have `side` fields disagreeing with images/report;
  several studies show per-plane display flips (COR radiological / AX flipped).
  The fibular head is the only reliable lateral anchor — never trust the side field.
- **Montage limits**: needs_full_res flagged on ~3,029 studies — mostly borderline
  Baker's (locule-vs-vessel), tiny meniscal signal, and T1-dependent marrow calls.

## Label policy in `image_verified_labels.csv`
- `positive`→1, `negative`→0, `__mask=1`, `__provenance=image_verified`.
- `uncertain`/`insufficient_evidence` → binary kept from prior label but `__mask=0`
  (excluded by masked-cell losses; 3,926 cells).
- Gold-58 rows: binary + provenance `gold_official`, untouched.
- `__flip` marks pos_flip/neg_flip vs the previous label; `__prev` keeps the old value.
- `needs_full_res` is a study-level flag.

## Recommended downstream use
- Use `__mask` for masked-cell training (already supported by the packaged loaders).
- Consider down-weighting Effusion pos-flips or treating them as soft labels —
  gold says our threshold is lower than official on borderline fluid.
- The dup-review files in `results_jsonl_dup/` enable noise modeling: cells where two
  reviewers disagree are the natural mask/uncertainty set (~14% of cells).
- needs_full_res studies are candidates for a full-resolution DICOM pass if HF access
  (`GodlyDonuts/RSNA-Data`) is restored.

## Files
- `image_verified_labels.csv` — corrected labels (schema mirrors study_labels.csv).
- `flips_vs_current.csv` — every flipped cell with evidence + notes.
- `results_jsonl/` — 147 per-batch JSONL evidence files (one row per study).
- `results_jsonl_dup/` — 29 independent second reads for agreement analysis.
- `REVIEW-PROMPT.md` — exact reviewer prompt/semantics.
