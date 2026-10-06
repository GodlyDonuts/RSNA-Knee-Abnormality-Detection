# Tier-0 adjudication: 35 extreme-disagreement cells

Manual second-pass adjudication of every tier-0 cell in `review_queue.parquet`
(27 `public_under_calls_positive` where v2 is positive and the public value is
below 0.5, and 8 `public_positive_v2_silent` where the public value is above
0.9 and v2 is `not_mentioned`). Each cell was re-read against the full source
report in `initiative/releases/report-supervision-v1/studies.parquet` and judged
against `protocols/report_weak_v2.md` — notably the meniscus rule ("signal
reaching a surface on ≥2 images, **or abnormal morphology**; isolated internal
degeneration is negative") and rule 3 (a bare OA diagnosis lacks
compartment-specific image-threshold evidence).

This is a label-semantics analysis, not a diagnostic rereading of MRI images.
Proposed state changes are v2.1 review inputs only — nothing here mutates the
label table.

## Headline findings

1. **The public labels use tear-only meniscal semantics.** All 27 under-call
   cells are meniscus targets where v2 recorded `positive` on degenerative
   morphology — extrusion (10), grade-III meniscopathy (8), fraying/fibrillation/
   maceration (5), volume loss alone (4). Under the v2 protocol these are
   defensible positives: extrusion, surface fraying, and substance loss are
   abnormal morphology, and grade-III degeneration is surface-reaching signal.
   The public extractor scores these same reports 0.015–0.49, meaning it learned
   a narrower tear-only boundary. If the competition's official criterion
   resembles the v2 protocol (signal-to-surface or morphology), the public
   labels carry systematic false negatives on degenerative meniscal disease —
   the largest known structured bias in the public label set.

2. **v2 is systematically conservative on compartment-attributable OA.** Of the
   8 silent cells, only 1 is cleanly correct ([8]: patellar subchondral edema
   without cartilage loss). The rest describe compartment-level degenerative
   evidence — prominent medial femoral osteophyte ([23]), lateral femoral
   condyle cartilage defects ([25]), osteophyte formation explicitly in all
   three compartments ([30]) — that v2 left `not_mentioned` while the public
   extractor scored ≥0.9. V2's silence applies the compartment rule in one
   direction only: it credits meniscal morphology aggressively but withholds
   OA credit when the report names osteophytes or cartilage loss in a named
   compartment.

## Per-cell dispositions

Index refers to the row in the tier-0 queue (study order is stable in the
parquet file). `v2 ✓` = keep v2 state; `v2~` = positive is defensible but thin —
retain unless v2.1 tightens the morphology clause; `→uncertain` / `→positive`
= proposed state change for v2.1 review.

### public_under_calls_positive (v2 positive, public <0.5)

| # | Target | Evidence (translated) | Disposition |
|---|--------|-----------------------|-------------|
| 0 | Lateral Meniscus | marginal amputation of body | v2 ✓ |
| 1 | Medial Meniscus | volume loss + grade-III degeneration + extrusion | v2 ✓ |
| 3 | Medial Meniscus | volume loss only, no grade/surface | v2~ (morphology clause) |
| 4 | Medial Meniscus | maceration, "no discrete tear" | v2 ✓ |
| 5 | Medial Meniscus | attenuated size, "no definite tear" | v2~ |
| 6 | Medial Meniscus | undersurface fraying of body | v2 ✓ |
| 7 | Medial Meniscus | grade-III + volume loss + extrusion | v2 ✓ |
| 9 | Lateral Meniscus | marginal fibrillation of body | v2 ✓ |
| 10 | Medial Meniscus | mucoid degeneration + free-edge fraying, "no tear formation" | v2~ — equivocal tearing; weakest positive in the set, `→uncertain` is defensible |
| 11 | Lateral Meniscus | free-edge fibrillation "that does not meet tear criteria" | `→uncertain` (explicit equivocation) |
| 12 | Lateral Meniscus | partial extrusion, "no signs of rupture" | v2~ |
| 13 | Medial Meniscus | degeneration + extrusion | v2 ✓ |
| 14 | Medial Meniscus | degenerated + extruded, "no evident rupture" | v2 ✓ |
| 15 | Medial Meniscus | grade-III + extrusion | v2 ✓ |
| 16 | Medial Meniscus | extreme degeneration, substance loss, extrusion | v2 ✓ |
| 17 | Medial Meniscus | pathological signal + extrusion | v2 ✓ |
| 18 | Medial Meniscus | grade-III + volume loss + extrusion | v2 ✓ |
| 19 | Medial Meniscus | grade-III + volume loss + extrusion | v2 ✓ |
| 20 | Medial Meniscus | grade-II + volume loss + extrusion | v2~ (extrusion carries it) |
| 21 | Medial Meniscus | degenerated posterior horn + reduced anterior-horn volume | v2~ |
| 22 | Lateral Meniscus | extruded + frayed | v2 ✓ |
| 24 | Lateral Meniscus | reduced body volume (pre-existing) | v2~ |
| 26 | Medial Meniscus | grade-II + volume loss + extrusion | v2~ (extrusion carries it) |
| 27 | Medial Meniscus | reduced body mass consistent with degeneration | v2~ |
| 29 | Medial Meniscus | grade-II + extrusion | v2~ (extrusion carries it) |
| 31 | Medial Meniscus | grade-III + volume loss + extrusion | v2 ✓ |

Score: **v2 clearly correct 16, defensible-but-thin 10, proposed →uncertain 1**;
public extractor wrong-or-incomplete on all 27 under tear-only semantics.

### public_positive_v2_silent (v2 not_mentioned, public >0.9)

| # | Target | Report evidence | Disposition |
|---|--------|-----------------|-------------|
| 2 | Medial OA | impression: "Osteoarthritis of left knee"; cartilage wearing | `→uncertain` (OA present, compartment unattributed) |
| 2 | Lateral OA | same | `→uncertain` |
| 2 | PF OA | same | `→uncertain` |
| 8 | PF OA | patellar marginal subchondral edema only, no cartilage loss | v2 ✓ |
| 23 | Medial OA | prominent osteophyte on medial femoral condyle + medial patellar cartilage thinning | `→positive` (named-compartment osteophyte is image-threshold evidence) |
| 25 | Lateral OA | partial-thickness cartilage defects on lateral trochlea + anterior lateral femoral condyle | `→positive` (named lateral-compartment cartilage loss) |
| 30 | Lateral OA | "central-marginal osteophyte formation in all three compartments" + cartilage destruction | `→positive` |
| 30 | PF OA | same — "all three compartments" includes patellofemoral | `→positive` (or `→uncertain` if the v2.1 reviewer reads the clause as femorotibial-only) |

Score: **v2 correct 1, proposed →positive 4, proposed →uncertain 3**.

## Implications for v2.1

- **No mass fix needed on the under-call side.** Tier 0 confirmed v2's
  morphology-inclusive meniscal semantics are internally consistent; the
  disagreement is a public-label blind spot, not a v2 error class. The one
  equivocal exception ([11]) is the correct boundary case for the
  "equivocal tearing stays uncertain" rule.
- **OA silence needs a targeted recheck, not a broad one.** The error mode is
  narrow: reports that name compartment-level osteophytes or cartilage defects
  but were left `not_mentioned`. A v2.1 pass restricted to OA targets where
  public ≥0.9 and v2 is `not_mentioned`/`negative` and the report contains an
  osteophyte or compartment-named cartilage term would catch this class —
  the tier-1/2 queue rows already cover most of it.
- **These 35 cells should be re-queued at the top of v2.1** with the
  dispositions above as starting proposals, since tier-0 was already the
  highest-priority tier.
