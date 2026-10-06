# v2.1 severity re-review — pass 1 adjudication record

The v2.1 proposal (`V21-PROTOCOL-PROPOSAL.md`) relaxes the severity bar on the
four severity-qualified targets: any named effusion, any named Baker's cyst,
and compartment-named degenerative evidence count positive. Gold-58
arbitration (`GOLD-ARBITRATION.md`) showed v2's severity bar collapses recall
on exactly these targets (0.41–0.67 vs gold).

**Pool**: every cell with `v2_state = negative` and public value ≥ 0.9 on the
five severity targets across the corpus — 1,842 cells. This is the
highest-confidence conflict subset (public extractor near-certain, v2 rejected
on severity). Lower-confidence public cells and `uncertain`-state cells are a
later pass.

## Method

Every cell was re-adjudicated by reading the v2 evidence quote recorded for
the negative call (the sentence the v2 reviewer saw and severity-rejected) and
checking whether it names the target finding at any severity. Cells whose
evidence did not name the finding (normal/preserved cartilage, a finding in a
different compartment, or no compartment named) stayed negative.

## Result — `v21_adjudication.csv`

| target | pool | positive | kept negative |
|--------|------|----------|---------------|
| Effusion | 998 | 998 | 0 |
| Baker's | 343 | 343 | 0 |
| Medial OA | 159 | 153 | 6 |
| Lateral OA | 127 | 120 | 7 |
| PF OA | 215 | 212 | 3 |
| **total** | **1,842** | **1,827** | **15** |

Every effusion and Baker's cell carried an explicit naming quote ("Small joint
effusion", "Малка Бекерова киста", "ύδραρθρο", "Lichte hydrops", "mayii",
"épanchement") — the v2 severity bar rejected every one. Kept-negative cells:
"Cartilages normal", "Unauffälliger Knorpelbesatz femoro-tibial", lateral-only
findings under Medial OA, medial-only findings under Lateral OA, a 3 mm chondral
lesion naming no compartment, and a Bulgarian "б.о." (no abnormality) report.
Notably no public-extractor hallucinations surfaced in this pool — the public
labels were corroborated cell-for-cell by v2's own recorded evidence.

## Artifact

`v21_severity_rereview.fold{0,1,2}.overlay.json` — bound overlays (release
sha, folds sha, per-fold roster sha, fold-0 verified through
`load_fitting_label_source`). Positive flips per fitting roster: fold0 995,
fold1 1,015, fold2 982; fitting supervised coverage 14,588 / 15,015 / 14,981.
Built by `scripts/research/build_v21_overlay.py`.

Run configs: `run-configs/label-source-v21.fold{0,1,2}.train.json` (emitted by
`scripts/research/emit_arm_configs.py`).

## Interpretation

This pool is where gold-58 says v2's bar is wrong — Effusion recall 0.41,
Baker's 0.60, Medial OA 0.50. If the severity interpretation holds on a gold
fit, the v2.1 overlay should recover most of the per-target gap while keeping
v2's verified semantics on the other 8 targets. It remains experiment-only;
evaluation masks are untouched.

## Next passes

- `uncertain`-state cells on the severity targets (~3,400 cells corpus-wide).
- public < 0.9 conflict cells and the v2-`not_mentioned` bucket — these need
  full-report context, not just the recorded evidence quote.
