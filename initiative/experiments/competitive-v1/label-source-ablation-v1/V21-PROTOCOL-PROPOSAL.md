# v2.1 review proposal: severity-bar recalibration on severity-qualified targets

Consolidated recommendation from `GOLD-ARBITRATION.md`,
`OA-CONFLICT-TRIAGE.md`, `TIER0-ADJUDICATION.md` and the corpus supervision
coverage table. This is a proposal for the next review pass — it does not
edit `protocols/report_weak_v2.md` or any label.

## The measured problem

On the 58 gold studies, v2's recall collapses on severity-qualified targets:

| Target | v2 recall | public recall | v2 fitting supervised% | v2-neg & public≥0.9 |
|--------|----------:|--------------:|-----------------------:|--------------------:|
| Effusion | 0.41 | 1.00 | 65.8% | **712** |
| Medial OA | 0.50 | 0.93 | 42.0% | 115 |
| Lateral OA | 0.67 | 0.82 | 43.7% | 97 |
| Baker's | 0.60 | 0.92 | 34.7% | 234 |
| PF OA | 1.00 | 0.81 | 37.2% | 139 |

Reading the gold-disagreeing cases: gold counts *mild* effusion ("leve
derrame articular"), mild osteophyte/cartilage findings, and small cysts as
positive; v2's severity bar classes them below-threshold-negative. The bar
is set above official semantics — ~1,300 fitting cells are v2-negative while
the public extractor is ≥0.9 confident positive, and gold suggests most are
real positives. Contusion/Fracture show NO such pool (0 and 3 cells ≥0.9) —
the severity bar is correctly placed for traumatic targets.

## Proposed v2.1 protocol changes

1. **Effusion: drop the severity qualifier.** Gold semantics appear to be
   "any report-stated effusion is positive" (gold prevalence 35/58 = 60%).
   Proposed rule: any explicitly stated effusion → `positive`;
   trace/synonym hedges ("possible minimal fluid") → `uncertain`; explicit
   absence → `negative`. This single change re-classes the largest error
   pool in the release.
2. **OA targets (Medial/Lateral/PF): lower the positive threshold to
   compartment-named degenerative evidence.** Named-compartment osteophytes,
   cartilage loss/thinning/defect of any depth, or joint-space narrowing in
   the named compartment → `positive`. Diffuse/superficial chondromalacia
   without compartment attribution → `uncertain`. Explicit intact cartilage
   in that compartment → `negative`. This converts both the v2-negative
   misses AND the uncertain pool (821 Medial OA, 1,099 PF OA cells at
   public≥0.9) into decidable cells.
3. **Baker's: any named popliteal/Baker's cyst is positive** regardless of
   size qualifier — gold counted small cysts. Only non-Baker's cysts
   (ganglion, meniscal) stay negative.
4. **Contusion/Fracture: unchanged** — v2-negative and public agree (0–3
   conflict cells); the traumatic-severity bar matches gold.
5. **Meniscus/ligament/synovitis: unchanged** — v2's morphology-inclusive
   semantics beat public on both recall and precision in arbitration.

## What this does NOT propose

- No gold-derived labels — the proposal changes the *protocol*, then the
  review re-reads reports under it. Gold stays evaluation-only.
- No corpus-wide relabel: the re-review pool is bounded — the ~1,300
  high-priority v2-negatives (public≥0.9) plus the ~3,400 uncertain cells on
  the four severity targets. Tier-1/2 queue rows already cover most of it;
  the queue should be re-tiered so severity-bar cells rank above meniscus
  conflicts (the meniscus tier-0 adjudication showed those are mostly
  v2-correct already).
- The `blended-source-v1` overlay is a stopgap bound on the same policy —
  if the re-review lands, it supersedes the blend's public-sourced cells
  with adjudicated ones.

## Re-review effort estimate

- 1,297 cells: v2-negative on the 4 targets with public ≥0.9 (top priority).
- ~3,400 cells: v2-uncertain on the same targets with public ≥0.5.
- ~400 cells: v2-uncertain with public <0.5 (likely stays uncertain).
- Total ≈ 5,000 cells over ~2,000 unique reports — roughly the size of the
  original v2 pass at ~35% of its cell count, fully dedupable via
  `report_group` (3.7× already measured).
