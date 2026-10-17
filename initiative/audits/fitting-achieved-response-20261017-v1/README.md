# Achieved fitting response of completed readouts

This diagnostic asks whether the four already completed, fixed-final readouts
actually realize the source-label ranking corrections permitted by their loss.
The previous analytical pseudo-target screen did not measure learned response.
No training recipe or active scoring owner changes.

Before reading results, fix both existing weak label sources, all four completed
epoch-3 heads, all 3,887 fitting cases, both original 94-window views and their
mean probabilities, all 12 findings, strong-source thresholds 0.1/0.9 and the
original source weight 0.25. Count teacher-opposed strict pairs corrected,
correctable pairs actually corrected, newly introduced inversions and ties.
Also report probability movement, full-bag pseudo-target error/cross-entropy and
the actual per-epoch gradient clipping history. Evaluate every head against both
sources, rather than selecting its preferred labels. No significance interval
is calculated from dependent pairs.

Use one CPU and no GPU on the existing owning node. Reuse the original feature
reader, immutable model/state and full-byte/private publication proofs. No CNN
inference, new optimizer, cache copy, checkpoint, Dataset or competition entry.
Only aggregate output leaves the node; predictions remain in RAM. The separate
2 MiB diagnostic cap fits inside the existing 256 MiB envelope, preserving the
16 MiB future control reserve and all other obligations. Fresh quota and exact
source, roster, final-state and private-publication admission precede execution.

These are fitting responses to weak source labels on a representation that has
seen this cohort. Neither agreement nor a lower objective demonstrates clinical
truth, independent generalization or a better Kaggle model. A pointwise optimum
does not guarantee representability by a shared model. The full-94 objective is
a diagnostic, not a reconstruction of sampled training batches. Findings may
motivate a separately matched experiment; they do not select a new recipe here.

## Actual result

CPU **949320** completed in **41.24 seconds**. Every original completed head and
both source tables were evaluated on all 3,887 fitting studies. The 288
target/source/role/view summaries are in [actual.json](actual.json); the compact
role summary is [summary.json](summary.json). No predictions or case records
were exported. All original model/private-byte proofs and numerical sources
were retained.

| Head / its training source | Mean12target strong-source pair-agreement delta | Pointwise-correctable pairs actually corrected, target range | Clipped updates |
|---|---:|---:|---:|
| Original16 | -0.0002655 |1.83–9.40%|0/1458|
| Original94 | +0.0005822 |2.23–9.18%|0/1458|
| S6-label16 | -0.0002345 |1.59–11.36%|0/1458|
| S6-label94 | +0.0001640 |2.19–11.67%|0/1458|

These are mean-probability fitting summaries against each head's training weak
source, not model accuracy. All cross-source and individual-view results remain
in the full receipt. Maximum actual gradient norm was0.135618; clipping at1
never activated in any of5,832updates. Removing clipping is therefore not a
justified correction for these runs. Large probability changes often coexist
with little net source ranking change. This does not prove classwise calibration
alone explains the updates, that weak source labels are correct, or that the
representation cannot learn better discrimination.

For S6-label94 fracture, median absolute probability change was0.075012, yet
77/660pointwise-correctable opposed pairs reordered and55previously ordered
pairs newly inverted. Net strong-source pair agreement changed+0.0003164.
No statistical significance is claimed from the dependent pairs. Do not promote
a head or choose a new label/loss/LR from these fitting counts.

Original CPU949292failed at import in the management Python before any input or
intent. Distinct numerical-runtime949305stopped before analysis because the local
observer had added elapsed_seconds to the otherwise identical original receipt.
Read-only949308confirmed all source/model/private fields match and no output
directory existed. Only then did distinct949320run; no diagnostic was duplicated.
The exact actual analysis source is preserved in analysis-v1.py.txt.
