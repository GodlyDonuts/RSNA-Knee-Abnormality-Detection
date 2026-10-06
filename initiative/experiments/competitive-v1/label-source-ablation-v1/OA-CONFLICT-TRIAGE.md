# OA decisive-conflict triage: 501 cells, v2 `negative` vs public ≥0.9

Follow-up to `TIER0-ADJUDICATION.md`. The queue's largest OA conflict class is
`public_over_calls_negative`: v2 recorded an explicit `negative` on a
Medial/Lateral/PF OA cell while the public extractor scored it ≥0.9. 501 cells
over 417 studies (Medial 159, Lateral 127, PF 215).

## Method

For each cell, the source report was split into sentences and scanned for
OA-evidence terms (osteophyte, cartilage loss/thinning/defect/fissure/ulcer,
joint-space narrowing, chondromalacia/chondropathy/condropatía/kraakbeenlijden/
kondromalazi, subchondral sclerosis/cyst, gonarthrosis/degenerative arthritis —
matching English, Spanish, French, German, Dutch, Turkish, Greek, Croatian,
Bulgarian surface forms) co-occurring with the cell's compartment term. A
sentence with both is a "compartment hit". Buckets:

| Bucket | Cells | Meaning |
|--------|-------|---------|
| compartment hit | 302 | report names compartment-level OA evidence |
| generic OA only | 22 | whole-joint OA claim, no compartment |
| no OA signal | 177 | no OA-evidence sentence for that compartment |

## Sample verification (14 cells read end-to-end)

The keyword buckets mislead without reading the evidence quotes. In every
sampled compartment-hit cell, **v2's negative is a below-threshold call backed
by a quoted mild finding**, not a miss:

- "Condropatía rotuliana superficial difusa" (superficial diffuse patellar
  chondromalacia) → v2 negative, public 0.91
- "mild heterogeneous signal of the cartilage without evidence of a discrete
  defect" → v2 negative, public 0.91
- "Minor fraying of the patellar articular cartilage... mild patellar
  chondromalacia" → v2 negative, public 0.91
- "minor fraying of the medial compartment articular cartilage **but no
  cartilage loss**" + mild-moderate osteophytes → v2 negative, public 0.92
- "partiële dikte fissuur inferolateraal patellair" (partial-thickness) → v2
  negative, public 0.91
- "Chondropatie fémoropatellaire légère sans fissuration profonde" (mild, no
  deep fissuring) → v2 negative, public 0.91
- "léger amincissement et fibrillation superficielle" (mild superficial
  thinning/fraying) → v2 negative, public 0.91
- "Hondromalacija medijalne tibijalne hrskavice I. stupnja" (grade I
  chondromalacia) → v2 negative, public 0.91

Per `protocols/report_weak_v2.md`, OA positives are severity-qualified — mild
or superficial chondromalacia, partial-thickness fissures, and fraying without
cartilage loss are explicitly below the positive criterion. These are
below-threshold negatives, exactly the protocol's intended behavior.

## Finding: a second, mirrored public-label bias

| Conflict class | v2 behavior | Public behavior | Verdict |
|----------------|-------------|-----------------|---------|
| Meniscus morphology (27 cells) | positive on extrusion/grade-III/fraying | 0.015–0.49 (tear-only) | public **under-calls** degenerative meniscal disease |
| OA below-threshold (501 cells) | negative on mild chondromalacia | ≥0.9 | public **over-calls** mild chondromalacia as OA |

Both biases come from the same root: **the public extractor's severity
boundaries are mismatched to the v2 protocol in opposite directions per target
family** — too strict on meniscal morphology, too loose on cartilage severity.
Since the v2 protocol was written to track the organizer's published criteria
(signal-to-surface/morphology for menisci; severity-qualified findings for OA),
v2 is the better-calibrated source *under that semantics assumption*. The
label-source ablation (public vs v2, same model/folds) arbitrates which
semantics the official scoring actually rewards — and it now has a concrete
mechanistic hypothesis to test rather than a generic "label quality" question.

## Caveats — read `GOLD-ARBITRATION.md` first

**Correction after gold-58 arbitration (added same day):** the 58-study expert
labels show gold counts *mild* effusion, mild osteophyte/cartilage findings
and small Baker's cysts as **positive** (v2 recall collapses to 0.41–0.67 on
Effusion/Medial OA/Baker's while public recall is 0.82–1.0). These 501 cells
are therefore probably mostly genuine positives that v2 wrongly calls
below-threshold-negative — the "public over-calls" framing above describes
agreement with the v2 protocol's severity bar, and gold indicates that bar is
set too high on severity-qualified targets. Read the per-target arbitration
table in `GOLD-ARBITRATION.md` before reusing these buckets; the per-cell
bucket assignments remain valid as a map of which sentences each source
read.

- Keyword triage counts are indicative, not exact — negated sentences and
  clause boundaries can produce false compartment hits; the 302/177 split is a
  prior for the v2.1 reviewer, not a measured error rate.
- The 4 tier-0 `→positive` proposals in `TIER0-ADJUDICATION.md` (cells 23, 25,
  30) involved *named-compartment* osteophytes/cartilage defects — a different,
  stronger evidence class than the mild diffuse findings dominating this
  bucket; the two documents are consistent, and gold arbitration supports
  the `→positive` direction.
