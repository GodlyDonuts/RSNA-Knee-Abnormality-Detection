# Measure the report label and teacher mixture signal

This new CPU-only audit uses all3887 already-admitted fitting target rows and their
corresponding two-view parent logits. It reads no MRI, clinical report, held-out
cohort, gold, checkpoint or learned features. No labels, live losses or model
parameters change. There is no new GPU allocation or model transfer.

The production objective mixes25% observed report label with75% teacher
probability; masked labels use the teacher. Count how often teacher and binary
weak label disagree, including both-view and uncalibrated0.9-probability cases,
and how often the mixture target stays on the teacher's side of0.5. Four fixed
source weights are an algebraic sensitivity report, not a recipe sweep, accuracy
metric or a claim that weak labels are right. The mixture's pointwise optimum
is NOT a hard bound on the trained network's predictions or ranking performance.
The exact loss derivative at parent logits is tested against anchored_loss; it
is not an actual head/encoder gradient measurement.

The result can motivate a later matched source-weight experiment; it cannot
choose the winning weight, diagnose label errors or explain the Kaggle gap by
itself. Existing paired and higher-LR comparisons finish unchanged first. A new
64KiB control child fits within the original paired8MiB envelope only after
funding all still-unwritten epoch sources and the1MiB scoring child; fresh node
and shared reserves are checked before any output. All output is aggregate.

## Actual scope and decision

CPU960426 completed the new count in0.023seconds. All46644 fitting study-target
cells are observed in this baseline bundle, but46622 are soft probabilities:
only22 are exact binary values, all positive. Therefore zero discordance in the
binary subgroup says virtually nothing about source/teacher agreement across
the cohort. Do not turn this into a label-correctness or model-quality claim.

The prior stronger fitting-only analysisCPU946541 already assessed strong soft
source ranking pairs; alpha0.25 reordered86.83–100% of opposed plain-view pairs.
It did not establish an anchoring barrier, and this narrow new check does not
change that conclusion. Preserve the negative, do not repeat this screen or
change the live loss weight. Finish the paired/LR fixed-final accuracy tests.
See [the prior analysis](../../audits/fitting-teacher-anchor-tension-20261017-v1/README.md).
